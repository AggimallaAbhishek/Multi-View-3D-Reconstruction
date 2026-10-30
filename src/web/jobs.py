"""Background reconstruction jobs for the web interface, one at a time.

A job runs the pipeline on an upload folder, then writes the run folder
(``src.outputs``) and ``report.json``. Its state is what the page polls: the stage
running now, the stages finished, timings, and the error if it failed. Stages are
``"dataset"`` (preparing the images, before the pipeline's first callback), the pipeline
stages, and ``"outputs"`` (writing figures, clouds and the report). A job that fails
after some stages finished still writes their outputs, so the page can link them.

The work runs in a child process (``reconstruct_in_subprocess``): Open3D's offscreen
renderer aborts the whole process outside the main thread, and a native crash should
end one job, not the server. A thread in the server relays the child's progress. The
job function is injected, so tests can replace it.
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.config import Config
from src.data.loader import DatasetError
from src.evaluation.ground_truth import GT_CAMERAS_FILE, load_cameras_json
from src.outputs import new_run_dir, save_stage_outputs
from src.pipeline import STAGES, PipelineResult, Stage, run_pipeline
from src.run import read_json
from src.web.report import REPORT_NAME, build_report, write_report

log = logging.getLogger(__name__)

RUN_LABEL = "web_reconstruct"
JOB_STAGES = ("dataset", *STAGES, "outputs")  # what a job reports, in order

Status = Literal["queued", "running", "done", "failed"]
# progress(stage, timings): ``stage`` starts now; ``timings`` holds the finished ones
ProgressFn = Callable[[str, dict[str, float]], None]
JobFn = Callable[[Path, Config, ProgressFn], Path]  # -> run folder


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


class JobFailed(RuntimeError):
    """The job failed; ``run_dir`` holds the outputs of the stages that finished (None:
    nothing was written)."""

    def __init__(self, message: str, run_dir: Path | None = None):
        super().__init__(message)
        self.run_dir = run_dir


class JobBusy(RuntimeError):
    """A job is already queued or running."""

    def __init__(self, job_id: str):
        super().__init__(f"A reconstruction is already running (job {job_id})")
        self.job_id = job_id


@dataclass
class Job:
    id: str
    status: Status = "queued"
    stage: str | None = None  # running now, or where it failed
    stages_done: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # seconds per pipeline stage
    error: str | None = None
    run_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "stages_done": list(self.stages_done),
            "timings": dict(self.timings),
            "error": self.error,
            "run_dir": None if self.run_dir is None else str(self.run_dir),
        }

    @classmethod
    def finished_run(cls, run_dir: Path) -> Job:
        """A finished web run of an earlier server, from its ``report.json``."""
        timings = read_json(run_dir / REPORT_NAME).get("timings", {})
        return cls(
            id=run_dir.name,
            status="done",
            stages_done=[*timings, "outputs"],
            timings=timings,
            run_dir=run_dir,
        )


def reconstruct(upload_dir: Path, config: Config, progress: ProgressFn) -> Path:
    """Pipeline up to ``filter``, then the CLI's ``reconstruct`` run folder plus
    ``report.json`` (scored against ``gt_cameras.json`` when it was uploaded).

    Raises ``JobFailed`` carrying the run folder with the outputs of the stages that
    finished before the failure. Must run on a main thread (it renders); see
    ``reconstruct_in_subprocess``."""
    result: PipelineResult | None = None

    def on_stage(stage: Stage, current: PipelineResult) -> None:
        nonlocal result
        result = current  # filled in place, so on failure it holds every finished stage
        progress(stage, dict(current.timings))

    run_dir = None
    try:
        result = run_pipeline(upload_dir, config, until="filter", on_stage=on_stage)
        progress("outputs", dict(result.timings))
        run_dir = new_run_dir(result, RUN_LABEL)
        save_stage_outputs(result, run_dir, "filter")
        gt_path = upload_dir / GT_CAMERAS_FILE
        gt = load_cameras_json(gt_path) if gt_path.is_file() else None
        write_report(build_report(result, gt), run_dir)
    except Exception as exc:
        if run_dir is None and result is not None:
            run_dir = _save_partial(result)
        raise JobFailed(_describe(exc), run_dir) from exc
    return run_dir


def reconstruct_in_subprocess(upload_dir: Path, config: Config, progress: ProgressFn) -> Path:
    """``reconstruct`` in a spawned child process, relaying its progress."""
    ctx = multiprocessing.get_context("spawn")
    events = ctx.Queue()
    child = ctx.Process(
        target=_child, args=(Path(upload_dir), config, events), name="reconstruct", daemon=True
    )
    child.start()
    try:
        while True:
            try:
                kind, *payload = events.get(timeout=0.5)
            except queue.Empty:
                if child.is_alive():
                    continue
                try:  # it may have exited right after its last message
                    kind, *payload = events.get(timeout=1.0)
                except queue.Empty:
                    raise RuntimeError(
                        f"the reconstruction process crashed (exit code {child.exitcode})"
                    ) from None
            if kind == "stage":
                progress(*payload)
            elif kind == "failed":
                message, run_dir = payload
                raise JobFailed(message, None if run_dir is None else Path(run_dir))
            elif kind == "done":
                return Path(payload[0])
            else:
                raise RuntimeError(f"unexpected message from the reconstruction: {kind!r}")
    finally:
        child.join(timeout=5)
        events.close()


def _child(upload_dir: Path, config: Config, events: Any) -> None:
    """Messages: ("stage", stage, timings), then ("done", run_dir) or
    ("failed", message, run_dir | None)."""
    try:
        run_dir = reconstruct(upload_dir, config, lambda s, t: events.put(("stage", s, t)))
    except JobFailed as exc:
        traceback.print_exc()  # the server's terminal shows where it failed
        events.put(("failed", str(exc), None if exc.run_dir is None else str(exc.run_dir)))
    else:
        events.put(("done", str(run_dir)))


def _describe(exc: Exception) -> str:
    """The message the page shows: a failed dataset check lists its errors (the plain
    message only names the upload folder)."""
    report = getattr(exc, "report", None) if isinstance(exc, DatasetError) else None
    errors = (
        [issue.message for issue in report.issues if issue.severity == "error"] if report else []
    )
    return "the photos can't be used: " + "; ".join(errors) if errors else str(exc)


def _save_partial(result: PipelineResult) -> Path | None:
    """Best effort: a run folder with the outputs of the pipeline stages that finished
    (None if none did, or the folder could not be created)."""
    finished = [s for s in result.timings if s in STAGES]  # timed once a stage finishes
    if not finished:
        return None
    last_finished, run_dir = finished[-1], None
    try:
        run_dir = new_run_dir(result, RUN_LABEL)
        save_stage_outputs(result, run_dir, last_finished)
    except Exception:
        log.exception("could not write the outputs of the stages up to %s", last_finished)
    return run_dir


class JobManager:
    def __init__(self, job: JobFn = reconstruct_in_subprocess):
        self._job = job
        self._jobs: dict[str, Job] = {}  # insertion order = submission order
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def submit(self, upload_dir: Path, config: Config, job_id: str | None = None) -> Job:
        """Start a job on ``upload_dir``; raises ``JobBusy`` if one is unfinished."""
        with self._lock:
            active = self._active()
            if active is not None:
                raise JobBusy(active.id)
            job = Job(id=job_id or new_job_id())
            self._jobs[job.id] = job
            thread = threading.Thread(
                target=self._work,
                args=(job, Path(upload_dir), config),
                name=f"job-{job.id}",
                daemon=True,
            )
            self._threads[job.id] = thread
        thread.start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """Every job of this server process, newest first."""
        return list(reversed(self._jobs.values()))

    def active(self) -> Job | None:
        """The queued or running job, if any."""
        with self._lock:
            return self._active()

    def wait(self, job_id: str, timeout: float | None = None) -> None:
        """Block until the job has finished (for tests and scripts)."""
        self._threads[job_id].join(timeout)

    def _active(self) -> Job | None:
        return next((j for j in self._jobs.values() if j.status in ("queued", "running")), None)

    def _work(self, job: Job, upload_dir: Path, config: Config) -> None:
        self._update(job, status="running", stage="dataset")
        try:
            run_dir = self._job(upload_dir, config, lambda s, t: self._advance(job, s, t))
        except Exception as exc:  # the job reports it; the server keeps running
            log.exception("job %s failed at stage %s", job.id, job.stage)
            run_dir = exc.run_dir if isinstance(exc, JobFailed) else None
            self._update(job, status="failed", error=str(exc), run_dir=run_dir)
            return
        self._advance(job, None, job.timings)
        self._update(job, status="done", run_dir=run_dir)

    def _advance(self, job: Job, stage: str | None, timings: dict[str, float]) -> None:
        """Mark the current stage finished and ``stage`` running (None: all finished)."""
        with self._lock:
            if job.stage is not None:
                job.stages_done.append(job.stage)
            job.stage = stage
            job.timings = dict(timings)

    def _update(self, job: Job, **changes: Any) -> None:
        with self._lock:
            for name, value in changes.items():
                setattr(job, name, value)

"""HTTP API and page of the web interface (spec §3).

Uploads are checked before anything is written: image count and extensions (the
``dataset`` limits the pipeline loads the folder with) and total size. They are then saved
to ``data/raw/web_uploads/<job_id>/``, where the optional ``camera.yaml`` /
``gt_cameras.json`` are parsed with the loaders the job uses (the folder is removed if
either is invalid), and go to the ``JobManager``. Finished web
runs are rediscovered from ``<output.root>/<stamp>_web_reconstruct[_n]/report.json``, so
results survive a server restart; their id is the run folder name.
"""

from __future__ import annotations

import mimetypes
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, NoReturn

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import Config
from src.data.dataset import CAMERA_FILE
from src.evaluation.ground_truth import GT_CAMERAS_FILE, load_cameras_json
from src.geometry.calibration import load_camera_file
from src.run import is_run_dir_name, read_json
from src.web.jobs import (
    JOB_STAGES,
    RUN_LABEL,
    Job,
    JobBusy,
    JobManager,
    new_job_id,
    terminate_children,
)
from src.web.report import REPORT_NAME

UPLOAD_ROOT = Path("data/raw/web_uploads")
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    terminate_children()  # don't leave reconstructions running after the server stops


def create_app(config: Config, manager: JobManager | None = None) -> FastAPI:
    manager = manager or JobManager()
    output_root = Path(config.output.root)
    app = FastAPI(title="Multi-view 3D reconstruction", lifespan=_lifespan)
    previous: dict[str, Job] = {}  # finished runs never change: read each report once

    def previous_run(run_id: str) -> Job | None:
        """A finished web run of an earlier server, from its run folder."""
        if run_id not in previous:
            run_dir = output_root / run_id
            if not is_run_dir_name(run_id, RUN_LABEL) or not (run_dir / REPORT_NAME).is_file():
                return None
            previous[run_id] = Job.finished_run(run_dir)
        return previous[run_id]

    def find_job(job_id: str) -> Job:
        job = manager.get(job_id) or previous_run(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id}")
        return job

    def busy(job_id: str) -> JSONResponse:
        """409 with the running job's id, so the page can link to it."""
        return JSONResponse({"detail": str(JobBusy(job_id)), "job_id": job_id}, 409)

    @app.get("/api/config")
    def page_config() -> dict[str, Any]:
        web = config.web
        return {
            "poll_interval_ms": web.poll_interval_ms,
            "max_upload_images": web.max_upload_images,
            "max_upload_mb": web.max_upload_mb,
            "min_images": config.dataset.min_images,
            "extensions": list(config.dataset.extensions),
            "stages": list(JOB_STAGES),
        }

    @app.post("/api/jobs", status_code=201, response_model=None)
    def create_job(
        images: Annotated[list[UploadFile] | None, File()] = None,  # missing: 422 below
        camera: Annotated[UploadFile | None, File()] = None,
        ground_truth: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, str] | JSONResponse:
        if (active := manager.active()) is not None:
            return busy(active.id)
        job_id = new_job_id()
        upload_dir = UPLOAD_ROOT / job_id
        save_upload(images or [], camera, ground_truth, config, upload_dir)
        try:
            manager.submit(upload_dir, config, job_id=job_id)
        except JobBusy as exc:  # another upload got in first
            shutil.rmtree(upload_dir)
            return busy(exc.job_id)
        return {"job_id": job_id}

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        """This server's jobs (newest first), then earlier finished web runs."""
        jobs = manager.list()
        live = {job.run_dir.name for job in jobs if job.run_dir is not None}
        reports = output_root.glob(f"*_{RUN_LABEL}*/{REPORT_NAME}")
        names = sorted((p.parent.name for p in reports), reverse=True)
        jobs += [run for n in names if n not in live and (run := previous_run(n))]
        return [job.to_dict() for job in jobs]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return find_job(job_id).to_dict()

    @app.get("/api/jobs/{job_id}/report")
    def get_report(job_id: str) -> Any:
        job = find_job(job_id)
        if job.status != "done":
            raise HTTPException(409, f"job {job_id} is {job.status}")
        return read_json(job.run_dir / REPORT_NAME)

    @app.get("/api/jobs/{job_id}/files")
    def list_files(job_id: str) -> list[str]:
        """Paths (relative, sorted) of the files in the job's run folder: what a finished
        job wrote, or what a failed one wrote before failing; empty while running."""
        run_dir = find_job(job_id).run_dir
        if run_dir is None:
            return []
        return sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file())

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def get_file(job_id: str, path: str) -> FileResponse:
        """A file of the run folder: a finished job's, or what a failed job wrote."""
        job = find_job(job_id)
        if job.run_dir is None:
            raise HTTPException(409 if job.status in ("queued", "running") else 404, "no outputs")
        run_dir = job.run_dir.resolve()
        target = (run_dir / path).resolve()
        if not target.is_relative_to(run_dir) or not target.is_file():
            raise HTTPException(404, f"no file {path}")
        media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return FileResponse(target, media_type=media_type)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def save_upload(
    images: list[UploadFile],
    camera: UploadFile | None,
    ground_truth: UploadFile | None,
    config: Config,
    upload_dir: Path,
) -> None:
    """Check the upload and save it to ``upload_dir`` (camera and ground truth under the
    names the job looks for); raises ``HTTPException`` 422 naming the problem, with
    nothing left on disk."""
    web = config.web
    allowed = {ext.lower() for ext in config.dataset.extensions}
    to_save: dict[str, UploadFile] = {}
    for upload in images:
        name = Path(upload.filename or "").name  # drop any client-side directories
        if not name or name.startswith("."):
            _reject(f"{upload.filename!r} is not an image file name")
        if Path(name).suffix.lower() not in allowed:
            _reject(f"{name}: not an image ({', '.join(sorted(allowed))})")
        if name in to_save:
            _reject(f"{name} is uploaded twice")
        to_save[name] = upload
    if len(to_save) < config.dataset.min_images:
        _reject(f"{len(to_save)} image(s) uploaded; at least {config.dataset.min_images} required")
    if len(to_save) > web.max_upload_images:
        _reject(f"{len(to_save)} images uploaded; at most {web.max_upload_images} allowed")

    loaders = {CAMERA_FILE: load_camera_file, GT_CAMERAS_FILE: load_cameras_json}
    extras = {CAMERA_FILE: camera, GT_CAMERAS_FILE: ground_truth}
    to_save |= {name: upload for name, upload in extras.items() if upload is not None}
    total = sum(_size(upload) for upload in to_save.values())
    if total > web.max_upload_mb * 1024 * 1024:
        _reject(f"upload is {total / 2**20:.1f} MB; at most {web.max_upload_mb} MB allowed")

    upload_dir.mkdir(parents=True)
    for name, upload in to_save.items():
        upload.file.seek(0)
        with (upload_dir / name).open("wb") as out:
            shutil.copyfileobj(upload.file, out)
    for name, load in loaders.items():
        if name not in to_save:
            continue
        path = upload_dir / name
        try:
            load(path)
        except Exception as exc:  # any parse failure is the user's file, not ours
            shutil.rmtree(upload_dir)
            reason = f"missing {exc}" if isinstance(exc, KeyError) else str(exc)
            _reject(f"{name} is invalid: {reason.replace(f'{path}: ', '')}")


def _size(upload: UploadFile) -> int:
    if upload.size is not None:
        return upload.size
    upload.file.seek(0, 2)
    return upload.file.tell()


def _reject(detail: str) -> NoReturn:
    raise HTTPException(422, detail)

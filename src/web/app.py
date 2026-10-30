"""HTTP API and page of the web interface (spec §3).

Uploads are validated before anything is written: image count and extensions (the
``dataset`` limits the pipeline loads the folder with), total size, and the optional
``camera.yaml`` / ``gt_cameras.json`` parsed with the loaders the job uses. A valid upload
lands in ``data/raw/web_uploads/<job_id>/`` and goes to the ``JobManager``. Finished web
runs are rediscovered from ``<output.root>/*_web_reconstruct*/report.json``, so results
survive a server restart; their id is the run folder name.
"""

from __future__ import annotations

import mimetypes
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import Config, WebConfig
from src.evaluation.ground_truth import load_cameras_json
from src.geometry.calibration import load_camera_file
from src.run import read_json
from src.web.jobs import GT_NAME, RUN_LABEL, JobBusy, JobManager

UPLOAD_ROOT = Path("data/raw/web_uploads")
CAMERA_NAME = "camera.yaml"  # the name src.data.dataset looks for beside the images
REPORT_NAME = "report.json"
STATIC_DIR = Path(__file__).parent / "static"


def create_app(config: Config, manager: JobManager | None = None) -> FastAPI:
    manager = manager or JobManager()
    output_root = Path(config.output.root)
    app = FastAPI(title="Multi-view 3D reconstruction")

    def previous_run(run_id: str) -> Path | None:
        """The run folder of a finished web run from an earlier server, if any."""
        run_dir = output_root / run_id
        if RUN_LABEL in run_id and "/" not in run_id and (run_dir / REPORT_NAME).is_file():
            return run_dir
        return None

    def status(run_id: str) -> dict[str, Any]:
        job = manager.get(run_id)
        if job is not None:
            return job.to_dict()
        run_dir = previous_run(run_id)
        if run_dir is None:
            raise HTTPException(404, f"unknown job {run_id}")
        return {
            "id": run_id,
            "status": "done",
            "stage": None,
            "stages_done": [],
            "timings": read_json(run_dir / REPORT_NAME).get("timings", {}),
            "error": None,
            "run_dir": str(run_dir),
        }

    def finished_run_dir(run_id: str) -> Path:
        job = status(run_id)
        if job["status"] != "done":
            raise HTTPException(409, f"job {run_id} is {job['status']}")
        return Path(job["run_dir"])

    @app.get("/api/config")
    def page_config() -> dict[str, Any]:
        web = config.web
        return {
            "poll_interval_ms": web.poll_interval_ms,
            "max_upload_images": web.max_upload_images,
            "max_upload_mb": web.max_upload_mb,
            "min_images": config.dataset.min_images,
            "extensions": list(config.dataset.extensions),
        }

    @app.post("/api/jobs", status_code=201)
    def create_job(
        images: Annotated[list[UploadFile], File()],
        camera: Annotated[UploadFile | None, File()] = None,
        ground_truth: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, str]:
        if manager.busy():
            raise HTTPException(409, "a reconstruction is already running")
        named = validate_upload(images, camera, ground_truth, config)
        job_id = uuid.uuid4().hex[:12]
        upload_dir = UPLOAD_ROOT / job_id
        upload_dir.mkdir(parents=True)
        for name, upload in named.items():
            upload.file.seek(0)
            with (upload_dir / name).open("wb") as out:
                shutil.copyfileobj(upload.file, out)
        try:
            manager.submit(upload_dir, config, job_id=job_id)
        except JobBusy as exc:  # another upload got in first
            shutil.rmtree(upload_dir)
            raise HTTPException(409, str(exc)) from None
        return {"job_id": job_id}

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        """This server's jobs (newest first), then earlier finished web runs."""
        jobs = [job.to_dict() for job in manager.list()]
        live = {Path(job["run_dir"]).name for job in jobs if job["run_dir"]}
        names = sorted((p.parent.name for p in output_root.glob(f"*/{REPORT_NAME}")), reverse=True)
        jobs += [status(n) for n in names if n not in live and previous_run(n)]
        return jobs

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return status(job_id)

    @app.get("/api/jobs/{job_id}/report")
    def get_report(job_id: str) -> Any:
        return read_json(finished_run_dir(job_id) / REPORT_NAME)

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def get_file(job_id: str, path: str) -> FileResponse:
        run_dir = finished_run_dir(job_id).resolve()
        target = (run_dir / path).resolve()
        if not target.is_relative_to(run_dir) or not target.is_file():
            raise HTTPException(404, f"no file {path}")
        media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return FileResponse(target, media_type=media_type)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def validate_upload(
    images: list[UploadFile],
    camera: UploadFile | None,
    ground_truth: UploadFile | None,
    config: Config,
) -> dict[str, UploadFile]:
    """The files to save, by their name in the upload folder; raises ``HTTPException``
    422 naming the problem."""
    web: WebConfig = config.web
    allowed = {ext.lower() for ext in config.dataset.extensions}
    named: dict[str, UploadFile] = {}
    for upload in images:
        name = Path(upload.filename or "").name  # drop any client-side directories
        if not name or name.startswith("."):
            _reject(f"{upload.filename!r} is not an image file name")
        if Path(name).suffix.lower() not in allowed:
            _reject(f"{name}: not an image ({', '.join(sorted(allowed))})")
        if name in named:
            _reject(f"{name} is uploaded twice")
        named[name] = upload
    if len(named) < config.dataset.min_images:
        _reject(f"{len(named)} image(s) uploaded; at least {config.dataset.min_images} required")
    if len(named) > web.max_upload_images:
        _reject(f"{len(named)} images uploaded; at most {web.max_upload_images} allowed")

    extras = {CAMERA_NAME: camera, GT_NAME: ground_truth}
    total = sum(_size(u) for u in [*named.values(), *extras.values()] if u is not None)
    if total > web.max_upload_mb * 1024 * 1024:
        _reject(f"upload is {total / 2**20:.1f} MB; at most {web.max_upload_mb} MB allowed")

    loaders = {CAMERA_NAME: load_camera_file, GT_NAME: load_cameras_json}
    for name, upload in extras.items():
        if upload is None:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / name
            upload.file.seek(0)
            copy.write_bytes(upload.file.read())
            try:
                loaders[name](copy)
            except Exception as exc:  # any parse failure is the user's file, not ours
                _reject(f"{name}: {str(exc).replace(str(copy), name)}")
        named[name] = upload
    return named


def _size(upload: UploadFile) -> int:
    if upload.size is not None:
        return upload.size
    upload.file.seek(0, 2)
    return upload.file.tell()


def _reject(detail: str) -> None:
    raise HTTPException(422, detail)

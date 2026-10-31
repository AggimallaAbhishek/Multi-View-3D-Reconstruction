# Web Frontend — Design

**Date:** 2026-10-30
**Status:** approved in brainstorming, awaiting spec review
**Roadmap:** new Phase 12 — Web interface

## 1. Goal

A local web page where a user uploads photos (and optionally calibration and ground-truth
files), runs the full multi-view reconstruction, watches stage-by-stage progress, and
inspects the result: an interactive 3-D point cloud with cameras, summary metrics, the
ground-truth evaluation when available, and the diagnostic figures the pipeline already
writes.

### Decisions

| Question | Decision |
|---|---|
| What "tests" means | Reconstruction + quality report (not a pytest dashboard) |
| Intrinsics | Optional `camera.yaml` upload; otherwise the existing chain in `src/geometry/calibration.py` (file → EXIF → fallback), with the source reported and a warning on fallback |
| 3-D display | Interactive in-browser viewer (three.js) |
| Deployment | Local only, single user, one job at a time, no auth |
| Approach | FastAPI backend + one static HTML/JS page, no build step |

### Non-goals

Multi-user hosting, job queues, authentication, checkerboard calibration in the UI,
running pytest from the UI, dense reconstruction, editing config values from the UI.

## 2. Architecture

```
browser (index.html + app.js, three.js from CDN)
   │  POST /api/jobs, poll GET /api/jobs/{id}, GET report/files
   ▼
src/web/app.py (FastAPI) ──► src/web/jobs.py (JobManager thread ⇄ child process)
                                   │
                                   ├─► src/pipeline.run_pipeline(..., on_stage=cb)
                                   ├─► src/outputs.save_stage_outputs(...)    (moved from cli)
                                   └─► src/web/report.py → report.json
```

### 2.1 New package `src/web/` (optional extra `[web]`)

| Unit | Responsibility | Depends on |
|---|---|---|
| `jobs.py` | `JobManager`: accepts one job at a time; holds `Job` state (`id`, `status` ∈ queued/running/done/failed, `stage`, `stages_done`, `timings`, `error`, `run_dir`, `photos` (counted from the upload, then the report's total), `registered` (from `report.json` once done), `started` (`YYYYmmdd-HHMMSS`, the run folders' stamp)). Stages: `dataset`, the pipeline stages, `outputs`. The work (pipeline + run folder + report) runs in a **spawned child process** (`reconstruct_in_subprocess`); a server thread relays its progress through a queue. Reason: Open3D's offscreen render (`figures/filtered.png`) aborts the process (SIGTRAP) outside the main thread, and a native crash must end one job, not the server. The job function is injected so tests can use a fake. No FastAPI imports. | `run_pipeline`, `save_stage_outputs`, `src/run.py` helpers, `report.py` |
| `report.py` | `build_report(result, gt=None) -> dict`, written as `report.json` in the run folder (shape below). Works on partial results. | `src/evaluation` |
| `app.py` | `create_app(config, manager) -> FastAPI`: the endpoints in §3; mounts `static/`. | `jobs`, `report` |
| `static/index.html`, `static/app.js`, `static/viewer.js`, `static/style.css` | Upload, progress and results views (§4). three.js `OrbitControls`, pinned version from cdn.jsdelivr.net (external); PLY parsed in `viewer.js` (`PLYLoader` assumes 0–255 colours, Open3D writes 0–1). Fonts from Google Fonts (external; system fallbacks offline). | API only |
| `__main__.py` | `python -m src.web [--config CONFIG]`: starts uvicorn on `web.host:web.port` and opens the browser. | `app` |

#### `report.json`

```text
images        {total, registered, unregistered: [names]}
points        {sparse, filtered} | null            null: no reconstruction
calibration   {sources: ["file"|"exif"|"fallback", ...] (sorted, usually one),
               K: 3x3 of the first image at working resolution}
reprojection  {n, mean_px, median_px, p90_px, max_px} | null
timings       {dataset, features, ..., filter: seconds}   stages that ran, in order
total_s       seconds
warnings      [text]   fallback intrinsics, unregistered images, no reconstruction
ground_truth  absent without gt_cameras.json; otherwise either {error: text} or
              {cameras, rotation_deg: {median, p90, max}, centre: {median, p90, max},
               per_camera: [{image, rotation_deg, centre}]}
```

`centre` is the camera-centre error after similarity alignment as a **fraction of the
true cameras' extent** (RMS distance from their centroid), as in
`src/evaluation/metrics.camera_pose_errors`; the page shows it as a percentage. Only
registered images with a ground-truth camera are scored; fewer than 3 gives an error.

### 2.2 Changes to existing code (CLI behaviour unchanged)

1. `run_pipeline(images, config, until, on_stage: Callable[[Stage, PipelineResult], None] |
   None = None)` — calls `on_stage(stage, result)` just before each stage runs. Dataset
   preparation happens before the first call, so the job manager marks the job's stage
   `"dataset"` itself before calling `run_pipeline`.
2. The per-stage writers (`_save_*`, `_SAVERS`) and the loop in `cli._stage_command` move
   to a new module `src/outputs.py` (not `src/pipeline.py`, which stays free of file I/O)
   with a public `save_stage_outputs(result, run_dir, stage, draw=0) -> tuple[Path | None,
   int]`. The CLI calls it; the web job calls it. Both produce identical run folders.

### 2.3 Storage

- Uploads: `data/raw/web_uploads/<job_id>/` — photos, plus `camera.yaml` /
  `gt_cameras.json` if given, so the dataset loader's existing auto-detection finds them.
  The job manager removes the folder once the job has ended (done or failed); the run
  folder holds the results.
- Results: the normal run folder `outputs/<stamp>_web_reconstruct/` plus `report.json`.
- Both are under git-ignored directories.

### 2.4 Configuration (`configs/default.yaml`)

```yaml
web:
  host: 127.0.0.1
  port: 8000
  max_upload_mb: 500
  max_upload_images: 200
  poll_interval_ms: 1000
  open_browser: true
```

Read into a `WebConfig` dataclass in `src/config.py` like the other sections. Accepted
extensions and the minimum image count are **not** repeated here: uploads are checked
against `dataset.extensions` (case-insensitive) and `dataset.min_images`, the same values
the pipeline loads the folder with. `max_upload_images` rejects an upload; it is distinct
from `dataset.max_images`, which subsamples.

### 2.5 Dependencies

Optional extra in `pyproject.toml`: `web = [fastapi, uvicorn, python-multipart]`, each pinned with `==` to
its current release when the extra is added, matching the repo's pinning style. `httpx` (needed by `TestClient`) added to `dev`. three.js is loaded
from a CDN and labelled as external in the README and methodology docs.

## 3. API

| Method | Path | Behaviour |
|---|---|---|
| `POST` | `/api/jobs` | Multipart: `images` (≥ `dataset.min_images` files), optional `camera` (`camera.yaml`), optional `ground_truth` (`gt_cameras.json`). Validates count, extensions, total size, and parses both optional files before queueing. Returns `201 {job_id}`; `409 {detail, job_id}` (the running job) if a job is running; `422 {detail}` on invalid input (nothing written). |
| `GET` | `/api/jobs/{id}` | `{id, status, stage, stages_done, timings, error, run_dir, photos, registered, started}`; `404` if unknown. `GET /api/jobs` lists the same objects. |
| `DELETE` | `/api/jobs/{id}` | Removes a finished or failed run: its run folder and its entry; `204`. `409` while queued or running, or if the run folder is not a web run folder directly under `output.root`; `404` if unknown. |
| `GET` | `/api/jobs/{id}/report` | `report.json`; `409` if not done yet; `404` if unknown. |
| `GET` | `/api/jobs/{id}/files` | Relative paths of the files in the job's run folder (sorted; empty while running). The page shows only figures and downloads that exist, and links everything a failed job wrote. |
| `GET` | `/api/jobs/{id}/files/{path}` | Serves a file from the job's run folder (a failed job's too: the outputs of the stages that finished). The resolved path must lie inside the run folder, otherwise `404`. |
| `GET` | `/api/config` | Page-relevant settings: `poll_interval_ms`, `max_upload_images`, `max_upload_mb` from `web`; `min_images`, `extensions` from `dataset`. |
| `GET` | `/api/jobs` | In-memory jobs plus finished web runs rediscovered from `outputs/<stamp>_web_reconstruct[_n]/report.json` (newest first; `_n` when two runs start in the same second), so results survive a server restart. |

Rediscovered runs use the run folder name as their `id`.

## 4. Page

One page, three states.

1. **Upload** — drag-and-drop zone with thumbnails and count; optional pickers for
   `camera.yaml` and `gt_cameras.json`; **Run** button; "Open previous run" dropdown fed by
   `GET /api/jobs`, each run labelled with its start time and registered / total photos
   (or the photo count, and the status, while unfinished). Validation errors from the API shown inline.
2. **Running** — checklist of stages (dataset, then `src/pipeline.STAGES`: features,
   matches, verify, pose, triangulate, cloud, reconstruct, filter, then `outputs`): spinner on the current one, ✓ and time on finished
   ones. Polls every `poll_interval_ms`.
3. **Results**
   - Summary cards: registered / total images, filtered points, median reprojection
     error, calibration source(s), total time. Yellow warning for fallback intrinsics or
     partial registration (listing unregistered images).
   - 3-D viewer: `point_clouds/filtered.ply` with vertex colours and camera frustums from
     `camera_poses/cameras.json`; toggle filtered/sparse; point-size slider; "Fit to"
     Object (the cloud, default) / Cameras (the cloud and every camera); orbit controls.
   - Ground-truth panel (only with ground truth): median / max rotation error, position
     error, per-camera table.
   - Diagnostics gallery with captions: `matches/match_matrix.png`,
     `verification/verified_matrix.png`, `pose/initial_pair.png`,
     `reconstruction/reprojection_histogram.png`, `reconstruction/cameras.png`,
     `figures/filtered.png` — each shown only if present.
   - Downloads: `filtered.ply`, `cameras.json`, `report.json`.
   - **Delete this run** (results and failed views), confirmed in the page (no browser
     dialog); then back to the upload view with a notice.

Follows the system light/dark preference; usable at laptop and tablet widths.

## 5. Error handling

| Situation | Handling |
|---|---|
| Too few / too many images, bad extension, over size | `422`, inline message, nothing written |
| Malformed `camera.yaml` / `gt_cameras.json` | `422` naming the file, before queueing |
| `DatasetError` | Job `failed` with the CLI's message, which lists the validation errors (`ValidationReport.errors`) after the folder |
| Exception in a stage | Job `failed`, `stage` set to the failing stage, `error` holds the message; page marks it ✗ and links outputs already written |
| Partial registration | Job `done`; summary shows the count in yellow and lists unregistered images |
| Ground-truth names don't match uploads | GT panel says so; rest of the report unaffected |
| Second run while one is running | `409`; page links to the running job |
| Server restart mid-job | In-memory job lost; finished runs rediscovered from `report.json` |

## 6. Testing

Test-first, using the repo's ground-truth fixtures.

- `tests/test_web_jobs.py` — `JobManager` with an injected fake pipeline: status
  transitions queued → running → done, stage callback recorded in order, exceptions
  become `failed` with stage and message, second submit while running is refused.
- `tests/test_web_report.py` — `build_report` on a real reconstruction of the
  `tests/render.py` scene, with its known cameras as ground truth: registered count,
  calibration sources, ground-truth rotation and position errors near zero; without ground
  truth the GT section is absent.
- `tests/test_web_app.py` (`TestClient`) — 422 cases (too few images, bad extension,
  malformed YAML/JSON), 409 while running, 404 for unknown job, path traversal on
  `/files/` refused, and one end-to-end run on rendered images: POST → poll to done →
  report → `filtered.ply` downloaded.
- `tests/test_pipeline.py` (extend) — `on_stage` is called once per stage in order.
- `tests/test_outputs.py` — `save_stage_outputs` writes every stage's files, draws
  diagnostics for the last stage only, and matches what `cli reconstruct` writes.
- Web tests are skipped with `pytest.importorskip("fastapi")` when the `[web]` extra is
  not installed.
- Manual: in the browser, upload all of `data/raw/templeRing` with `camera.yaml` and
  `gt_cameras.json`; expect 47/47 registered and a median rotation error ≈ 0.15°, matching
  `scripts/evaluate_against_ground_truth.py`. Screenshot of the viewer recorded.
- The JS is not unit-tested; it holds no logic beyond calling the API and rendering.

## 7. Documentation

- `docs/ROADMAP.md`: new **Phase 12 — Web interface** with the exit criteria below.
- `README.md`: "Web UI" subsection in Quick Start (`uv pip install -e ".[web]"`,
  `python -m src.web`), a screenshot, three.js and FastAPI listed as external.
- `docs/architecture.md`: the web layer added to the data-flow diagram.

## 8. Exit criteria (Phase 12)

1. All new and existing tests pass (`pytest -q`).
2. CLI `reconstruct` output folder is unchanged by the refactor.
3. The templeRing manual check above reproduces the CLI's numbers in the browser.
4. Invalid uploads are rejected with clear messages; a failed stage is shown as failed.

# Web Frontend — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-10-30-web-frontend-design.md`
**Roadmap:** Phase 12 — Web interface
**Branch:** `feat/web-frontend` (off `main`)

Every task is test-first: write the failing test, run it and see it fail for the right
reason, implement, run the task's tests and then `pytest -q`, commit. Each commit message
ends with the repo's `Co-Authored-By` line. `source .venv/bin/activate` before anything.

### Deviation from the spec (applied in Task 3)

The per-stage writers (`_save_features` … `_save_filtered`, `_SAVERS`, ~330 lines of
`src/cli.py`, which draw figures and print summaries) move to a new module
**`src/outputs.py`**, not `src/pipeline.py`, so the pipeline stays free of file I/O. The
function is the same: `save_stage_outputs(result, run_dir, stage, draw=0)`. Task 3 updates
the spec's diagram and §2.2 to say so.

One small addition: `GET /api/config` returns the page-relevant `web` settings
(`poll_interval_ms`, limits) so the page doesn't hard-code them. Added to the spec's API
table in Task 6.

---

## Task 0 — Branch and dependencies

1. `git switch -c feat/web-frontend`
2. `uv pip install fastapi uvicorn python-multipart httpx`, then read the installed
   versions with `uv pip show`.
3. `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   web = ["fastapi==X", "uvicorn==Y", "python-multipart==Z"]   # versions from step 2
   dev = [..., "httpx==W"]                                      # TestClient needs it
   ```
4. `uv pip install -e ".[dev,web]"`; `python scripts/verify_setup.py`; `pytest -q` green.
5. Commit: `build: add optional web extra (fastapi, uvicorn, python-multipart)`.

## Task 1 — `WebConfig`

**Test** (`tests/test_config.py`, extend):
- `load_config().web` has the defaults from the spec §2.4 (`host == "127.0.0.1"`,
  `port == 8000`, …). (Checkpoint 1 reduced this to pinning `host`; `min_images` and
  the extensions were dropped in favour of `dataset.*`, `max_images` renamed
  `max_upload_images`.)
- `config_from_dict({"web": {"port": "x"}})` raises `ConfigError` (wrong type).
- `config_from_dict({"web": {"nope": 1}})` raises `ConfigError` (unknown key).

**Implement:**
- `src/config.py`: `@dataclass class WebConfig` with the fields of spec §2.4, typed like
  the other sections; add
  `web: WebConfig = field(default_factory=WebConfig)` to `Config`.
- `configs/default.yaml`: a commented `web:` section with the same values.

Commit: `feat(config): add web section`.

## Task 2 — `run_pipeline(..., on_stage=...)`

**Test** (`tests/test_pipeline.py`, uses the existing `scene_dir` fixture):
- `test_on_stage_called_once_per_stage_in_order`: record calls into a list with
  `until="verify"`; assert it equals `list(stages_until("verify"))`.
- `test_on_stage_sees_previous_stage_timing`: inside the callback for `"matches"`,
  `"features" in result.timings` — proves it is called *before* the stage, after the
  previous one finished. (Pass the callback a `(stage, result)` pair; see below.)
- Existing tests unchanged and green (default `None`).

**Implement** (`src/pipeline.py`):
```python
StageCallback = Callable[[Stage, PipelineResult], None]

def run_pipeline(images, config, until="triangulate", on_stage: StageCallback | None = None):
    ...
    for stage in stages_until(until):
        if on_stage is not None:
            on_stage(stage, result)
        ...
```

Commit: `feat(pipeline): optional per-stage progress callback`.

## Task 3 — `src/outputs.py` with `save_stage_outputs`

**Test** (new `tests/test_outputs.py`, module-scoped fixture: `run_pipeline(scene_dir,
config, until="filter")` on the rendered scene):
- `save_stage_outputs(result, run_dir, "filter")` writes, under `run_dir`:
  `features/feature_statistics.csv`, `matches/match_statistics.csv`,
  `verification/verification_statistics.csv`, `pose/initial_pair.json`,
  `triangulation/triangulation.json`, `reconstruction/reconstruction.json`,
  `camera_poses/cameras.json`, `point_clouds/sparse.ply`, `point_clouds/filtered.ply`;
  returns `(path_to_filtered_ply, 0)`.
- The `cloud` stage's two-view outputs are skipped when `stage != "cloud"` (no
  two-view PLY written) — same rule as today.
- CLI regression: `cli.main(["reconstruct", str(scene_dir), ...])` with the test config
  still returns 0 and produces the same set of relative file paths as calling
  `save_stage_outputs` directly (compare sorted path lists, ignoring `config.yaml`,
  `environment.json`, `dataset.json`).

**Implement:**
- Move `_save_*`, `_SAVERS`, and the helpers only they use (`_spread`,
  `_inspection_order`, `_pair_stem`, …) from `src/cli.py` to `src/outputs.py`, unchanged.
- Add:
  ```python
  def save_stage_outputs(result, run_dir, stage, draw=0) -> tuple[Path | None, int]:
      """Write the outputs of every stage up to ``stage`` (the loop formerly in cli)."""
  ```
- `cli._stage_command` calls it. `cli.py` keeps argument parsing and `_validate`,
  `_view`, `_calibrate`.
- Update spec §2 diagram and §2.2 to name `src/outputs.py`.
- Verify by hand: `python -m src.cli reconstruct data/raw/templeRing` prints the same
  summary as before and writes the same folder layout.

Commit: `refactor: move per-stage output writers from cli to src/outputs`.

## Task 4 — `src/web/report.py`

**Test** (new `tests/test_web_report.py`, module fixture: rendered scene with 6 cameras,
`run_pipeline(..., until="filter")`, ground truth = the scene's known cameras keyed by
file name):
- `build_report(result)`:
  - `images.total == 6`, `images.registered == 6`, `images.unregistered == []`
  - `points.sparse == len(rec.landmarks)`, `points.filtered == int(result.filtering.keep.sum())`
  - `calibration.sources == ["file"]`, `calibration.K` equals the first image's K
  - `reprojection.median_px` < `config.triangulation.max_reprojection_error_px`
  - `timings` has every stage key plus `"dataset"`; `"ground_truth"` key absent
- `build_report(result, gt=cameras)`: `ground_truth.rotation_deg.median < 0.5`,
  `ground_truth.centre.median < 0.01`, `ground_truth.per_camera` has 6 rows with image
  names.
- `gt` missing all image names → `ground_truth == {"error": "no matching cameras in
  gt_cameras.json"}`.
- Fewer than 3 registered → `ground_truth == {"error": "...needs at least 3 registered
  images"}` (mirrors `evaluate_reconstruction`).
- `report` round-trips through `json.dumps` (only plain types).
- An unrelated 7th image → 6/7 registered, listed in `unregistered`, one warning naming it.
  (Added at checkpoint 2; the report format is in spec §2.1 "`report.json`".)
- `calibration.sources == ["fallback"]` when the scene is run without `camera.yaml` and
  `use_exif` finds nothing → `warnings` contains a fallback-intrinsics message.

**Implement:** `build_report(result: PipelineResult, gt: dict[str, Camera] | None = None)
-> dict`, reusing `camera_pose_errors`, `ReprojectionStats.from_errors`,
`rec.unregistered`, and each image's `intrinsics_source`. `write_report(report, run_dir)`
writes `report.json` via `src/run.write_json`.

Commit: `feat(web): reconstruction report builder`.

## Task 5 — `src/web/jobs.py`  ✅

Implemented as `JobManager(job=reconstruct_in_subprocess)`: one injected job function
`job(upload_dir, config, progress) -> run_dir` (pipeline + `save_stage_outputs` +
`report.json`), run by default in a spawned child process. Found while writing the
end-to-end test: Open3D's offscreen render aborts the process (exit 133, SIGTRAP) outside
the main thread, so the in-thread design of the spec would kill the server. Stages
reported: `dataset`, the pipeline stages, `outputs`. Tests (`tests/test_web_jobs.py`):
stage order and timings, `dataset` reported first, failures at a stage (incl. while
writing outputs, `DatasetError`), one job at a time (`JobBusy`), `to_dict` JSON, errors
relayed from the child, and the real job end to end on the rendered scene with ground
truth (the crash regression).

## Task 6 — `src/web/app.py`  ✅

Done. Upload checks live in `save_upload(..., upload_dir)`: it checks names, count and
size first, then saves and parses `camera.yaml` / `gt_cameras.json` in place, removing the
folder if either is invalid (no temp copy). Also rejects duplicate and hidden file names and strips client-side directories
from upload names. `static/index.html` is a placeholder until Task 8.

**Test** (new `tests/test_web_app.py`; `pytest.importorskip("fastapi")` at top;
`TestClient(create_app(config, manager))`; upload directory and outputs inside the
per-test temp dir):

Validation (no job created, nothing written under the upload root):
- 1 image → 422 mentioning the minimum.
- A `.txt` among images → 422 naming the file.
- `IMG_0001.JPG` (upper-case extension) is accepted — matching is case-insensitive like
  `src/data/loader.discover_images`.
- `web.max_upload_images` exceeded (config override to 3, send 4) → 422.
- Fewer than `dataset.min_images`, or an extension outside `dataset.extensions` → 422
  (the same values the pipeline loads the folder with).
- Total size over `max_upload_mb` (override to a tiny value) → 422.
- `camera` that isn't valid YAML / lacks `fx` → 422 naming `camera.yaml`.
- `ground_truth` that isn't valid JSON → 422 naming `gt_cameras.json`.

Job API with a fake manager:
- POST valid → 201 `{job_id}`; files saved as `data/raw/web_uploads/<id>/...` with
  `camera.yaml` / `gt_cameras.json` under those exact names.
- POST while the fake reports busy → 409.
- `GET /api/jobs/unknown` → 404; `GET /api/jobs/{id}/report` before done → 409.
- `GET /api/jobs/{id}/files/../../etc/passwd` and an absolute path → 404;
  a real file in the run dir → 200 with the right content type (`.ply`, `.png`,
  `.json`).
- `GET /api/jobs` includes a finished run rediscovered from a hand-written
  `outputs/20261030-000000_web_reconstruct/report.json`.
- `GET /` serves `index.html`.
- `GET /api/config` returns `poll_interval_ms`, `max_upload_images`, `max_upload_mb`
  from `web` and `min_images`, `extensions` from `dataset`.

End to end (real `JobManager`, rendered scene of 6 views + `camera.yaml` +
`gt_cameras.json`, marked `@pytest.mark.slow` if the suite has such a marker, otherwise
kept small): POST → poll `GET /api/jobs/{id}` until `done` (timeout 120 s) → report has
6/6 registered and a GT section → `files/point_clouds/filtered.ply` returns 200 and
starts with `ply`.

**Implement:** `create_app(config, manager=None) -> FastAPI` with the endpoints of spec
§3; upload validation in a helper `validate_upload(files, camera, gt, web_cfg)` that
raises `HTTPException(422, ...)`; `camera.yaml` checked with
`geometry.calibration.load_camera_file` on a temp copy; `gt_cameras.json` with
`evaluation.ground_truth.load_cameras_json`. Path safety:
`(run_dir / path).resolve().is_relative_to(run_dir.resolve())`. Static files via
`StaticFiles(directory=Path(__file__).parent / "static", html=True)` mounted at `/`
after the API routes.

Commit: `feat(web): FastAPI endpoints for uploads, jobs, reports and files`.

## Task 7 — `python -m src.web`  ✅

Done. `build(argv)` returns `(app, config.web)` (not `(app, host, port)`): `main` also needs
`open_browser`. `--port` overrides `web.port` the way the CLI's dotted-key options do. The
browser opens once uvicorn reports `started`; the server starts only under `__main__`,
since spawned job processes import the module as `__mp_main__`. Smoke-tested live: one
listener during a job, 6/6 registered, the run listed again after a restart.

**Test:** `tests/test_web_app.py::test_main_parses_args` — `src.web.__main__.build(argv)`
returns `(app, host, port)` honouring `--config` and `--port`; uvicorn itself isn't
started.

**Implement:** `__main__.py` with `build(argv)` and `main()` → `uvicorn.run(app, host,
port)`; opens the browser (`webbrowser.open`) after start when `web.open_browser`.

Commit: `feat(web): command-line entry point`.

## Task 8 — The page (`src/web/static/`)  ✅

Done (`index.html`, `style.css`, `app.js`, `viewer.js`). Deviations and additions:
- PLY files are parsed in `viewer.js`, not with three's `PLYLoader`: it divides colours by
  255, and Open3D writes float colours in 0..1 (the cloud rendered black).
- `GET /api/config` also returns `stages` (`jobs.JOB_STAGES`), so the checklist isn't a
  second copy of `src.pipeline.STAGES`.
- Files that aren't photos are left out of the upload (and counted), not an error: there is
  no per-file remove, and a dropped folder often holds other files.
- A failed dataset check reports its error issues ("the photos can't be used: ..."), not
  "dataset validation failed for <upload folder>".
- Fit-to-view frames the bulk of the cloud (median centre, 90th-percentile radius) from
  the cameras' mean viewing direction.
Checked in Chrome (extension and headless): upload checks, a run end to end, progress,
results in light/dark and at 768 px, ground truth, a failed run, a run with no cloud, the
409 link, reopening previous runs. No console errors.

Load the `frontend-design` skill first. Not unit-tested (spec §6); keep logic in the API.

- `index.html`: the three states of spec §4 as sections toggled by a `data-state`
  attribute; `<script type="importmap">` for `three@<pinned>` and its addons from
  cdn.jsdelivr.net.
- `app.js` (ES module):
  - Upload: drag-and-drop + file input, thumbnails via `URL.createObjectURL`, counts,
    optional pickers, `fetch POST /api/jobs` with `FormData`; show 422 `detail` inline.
  - Progress: poll `GET /api/jobs/{id}` every `poll_interval_ms` (read from
    `GET /api/config`); render the stage checklist.
  - Results: summary cards from `report.json`; warnings in yellow; GT panel and table
    when present; diagnostics gallery (skip 404s); downloads.
  - Viewer: `PLYLoader` → `THREE.Points` with `vertexColors`; frustums from
    `cameras.json` (centre `C = -Rᵀt`, pyramid from `K` and image size, scaled to 5% of
    the cloud's extent); `OrbitControls`; filtered/sparse toggle; point-size slider;
    fit-to-view on load; flip to the viewer's Y-up convention once, documented in a
    comment.
  - "Open previous run" dropdown from `GET /api/jobs`.
- `style.css`: tokens on `:root`, dark mode via `prefers-color-scheme`, works from
  ~768 px wide.

Manual check (browser): small rendered-scene upload end to end; error display with one
image; reopening a previous run.

Commit: `feat(web): upload, progress and results page with 3-D viewer`.

## Task 9 — Real-data check

1. `python -m src.web`; upload all 47 `data/raw/templeRing` images + `camera.yaml` +
   `gt_cameras.json`.
2. Expect 47/47 registered and median rotation error ≈ 0.15°; compare against
   `python scripts/evaluate_against_ground_truth.py data/raw/templeRing`. Any mismatch
   beyond rounding → `systematic-debugging` skill before going on.
3. Screenshot of the results page (viewer visible) to `docs/images/web_ui.png`.
4. Re-run the CLI real-data check from CLAUDE.md (refactor in Task 3 touched the
   pipeline's outputs).

## Task 10 — Docs and close-out

- `docs/ROADMAP.md`: Phase 12 — Web interface, tasks, exit criteria (spec §8), ✅ status
  note with the measured templeRing numbers.
- `README.md`: Quick Start "Web UI" subsection (`uv pip install -e ".[web]"`,
  `python -m src.web`), screenshot, FastAPI / uvicorn / three.js labelled external,
  status checklist ticked, project structure gains `src/web/` and `src/outputs.py`.
- `docs/architecture.md`: web layer in the data-flow diagram.
- `pytest -q` green; commit `docs: phase 12 web interface`; open a PR to `main` on
  request.

## Order and checkpoints

0 → 1 → 2 → 3 (stop: show CLI output unchanged) → 4 → 5 → 6 → 7 → 8 (stop: show
screenshots) → 9 (stop: report the numbers) → 10.

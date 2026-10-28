# Roadmap — Phase Plans

Each phase must be **working and tested before the next starts**. Every phase follows the same loop:

1. **Plan** — reread this phase's section; confirm inputs from the previous phase exist.
2. **Test first** — write synthetic/unit tests for the new module (they should fail).
3. **Implement** — smallest code that makes tests pass; thresholds go in `configs/default.yaml`.
4. **Inspect** — generate the phase's diagnostic figures and look at them.
5. **Record** — log stats to CSV/JSON, tick the README checklist, commit.

Legend: 🧪 tests · 📦 deliverable · ✅ exit criteria · ⚠️ risks · 🧰 skills

---

## Milestone overview

```mermaid
gantt
    dateFormat  X
    axisFormat  %s
    section M1 Two-view MVP
    P0 Foundation          :p0, 0, 1
    P1 Dataset             :p1, after p0, 1
    P2 Features            :p2, after p1, 1
    P3 Matching            :p3, after p2, 1
    P4 Verification        :p4, after p3, 1
    P5 Relative pose       :p5, after p4, 1
    P6 Triangulation       :p6, after p5, 1
    P7 Two-view cloud      :p7, after p6, 1
    section M2 Multi-view
    P8 Incremental SfM     :p8, after p7, 3
    P9 Filter & export     :p9, after p8, 1
    section M3 Evaluation & delivery
    P10 Experiments        :p10, after p9, 2
    P11 Docs & demo        :p11, after p10, 1
```

Units are relative effort (≈ 1 unit ≈ a few focused days). Phase 8 is the largest and riskiest; budget for it.

---

## Phase 0 — Foundation & Environment

**Goal:** A reproducible environment, project skeleton, config system, and a synthetic ground-truth scene generator that every later geometry phase is tested against.

### Tasks

| # | Task | Detail |
|---|---|---|
| 0.1 | Python env | `uv venv --python 3.12 .venv` (system Python 3.14 has no Open3D wheels). Activate. |
| 0.2 | `pyproject.toml` | Package `src`; deps: `opencv-contrib-python`, `numpy`, `scipy`, `open3d`, `matplotlib`, `pandas`, `pyyaml`; dev extra: `pytest`, `pytest-cov`, `ruff`, `jupyter`. Pin exact versions after first successful install. |
| 0.3 | `.gitignore` | `.venv/`, `__pycache__/`, `data/raw/`, `data/processed/`, `outputs/`, `*.ply` (except `data/sample/`), `.ipynb_checkpoints/` |
| 0.4 | Scaffold | `src/__init__.py`, `src/config.py`, `tests/conftest.py`, `configs/default.yaml`, `scripts/verify_setup.py`, `docs/architecture.md`, `data/sample/.gitkeep`, `outputs/.gitkeep` |
| 0.5 | `verify_setup.py` | Print versions of OpenCV / NumPy / Open3D / SciPy; assert `cv2.SIFT_create` exists; create & discard a tiny Open3D point cloud. |
| 0.6 | Config loader | `src/config.py`: load YAML → frozen dataclasses (`FeatureConfig`, `MatchConfig`, `RansacConfig`, `TriangulationConfig`, …); unknown keys raise. |
| 0.7 | Synthetic scene | `tests/synthetic.py`: `make_scene(n_points, n_cameras, noise_px, seed)` → points on/around a shape, cameras on an arc looking at the centroid, known `K`, `R_i`, `t_i`, projected 2D observations with visibility mask. Plus `project(K, R, t, X)` helper. |
| 0.8 | Conventions doc | `docs/architecture.md`: world→camera `x_c = R X + t`; center `C = −Rᵀt`; pixel origin top-left; arrays shaped `(N, 3)` / `(N, 2)` float64; images BGR uint8 as loaded by OpenCV. |
| 0.9 | Sample data | Pick a public multi-view set for development (e.g. a small COLMAP/ETH3D sample or the OpenCV `sfm` dataset) and record its source in `data/sample/README.md`. Start capturing your own object in parallel. |

### 🧪 Tests
- `test_config.py`: default config loads; unknown key raises; values have correct types.
- `test_synthetic.py`: projected points of a noise-free scene lie inside the image; `project` of camera center direction is consistent; points have positive depth in every camera; deterministic for a fixed seed.

### 📦 Deliverables
`pyproject.toml`, `.gitignore`, scaffold, `verify_setup.py`, `config.py`, `synthetic.py`, `docs/architecture.md`, first commit.

### ✅ Exit criteria
- `python scripts/verify_setup.py` prints versions and succeeds.
- `pytest -q` green.
- A fresh clone + Quick Start reproduces the environment.

### ⚠️ Risks
Mixed OpenCV wheels (remove all `opencv-*` then reinstall one) · Open3D/Python version mismatch · Apple-silicon wheel availability — verify early.

### 🧰 Skills
`python-testing-patterns`, `tdd`.

---

## Phase 1 — Dataset Loading & Validation ✅

**Goal:** Turn a folder of photos into a validated, deterministically ordered list of images with metadata and a consistent intrinsic matrix.

> **Status — done.** Entry point `src.data.dataset.prepare_dataset(dir, config)` → `Dataset` of `PreparedImage` (working-resolution BGR image, per-image `K`, scale, intrinsics source) + `ValidationReport`. CLI: `python -m src.cli validate <dir>` and `python -m src.cli calibrate <dir>`. Decisions made while building it:
> - EXIF orientation is **applied** on load (OpenCV default), so all sizes and `K` refer to the upright image.
> - Images are decoded one at a time and only the downscaled copy is kept (bounded memory for 80 × 12 MP photos).
> - `K` rescaling uses the exact pixel-centre formula `c' = (c + 0.5)·s − 0.5`.
> - EXIF focal → `K` uses the 35 mm-equivalent on the **diagonal**, so it is correct for 4:3 phone sensors too.
> - Blur is measured at ≤ 800 px so one threshold works across cameras. Near-duplicates use a 64-bit dHash.
> - Checkerboard calibration is verified on rendered boards with known `K` (fx recovered within 1 %).
> - Remaining for real data: capture or download a scene, run the notebook, record findings in `data/sample/README.md`.

### Tasks
- `data/loader.py`: discover `.jpg/.jpeg/.png/.tif` (case-insensitive), natural sort, decode with `cv2.imread`, return `ImageRecord(id, path, image, width, height, exif_focal)`.
- `data/validation.py`: report unreadable files (warning, skip), empty dir (error), mixed resolutions (warning), near-duplicates via small-thumbnail hash, blur score via Laplacian variance (warning below threshold), minimum image count.
- `data/preprocessing.py`: grayscale, resize to `max_side`, **scale `K` accordingly** (`fx, fy, cx, cy` × scale), undistort if distortion coefficients are provided.
- `geometry/calibration.py`: load `camera.yaml`; else estimate `K` from EXIF focal length + sensor width, else fallback `f = 1.2·max(w,h)`, principal point at center. Optional `calibrate_from_checkerboard()`.
- Save `outputs/<run>/dataset.json` with metadata.

### 🧪 Tests
Valid images load; corrupt file is skipped with warning; empty dir raises; order is deterministic (`img2` < `img10`); resizing by 0.5 halves `fx, fy, cx, cy`.

### 📦 Deliverables
Loader, validator, preprocessing, calibration loader, `notebooks/01_dataset_exploration.ipynb` (thumbnails grid, blur scores, resolution summary).

### ✅ Exit criteria
Sample dataset loads; validation report is readable; `K` is consistent with working resolution.

### ⚠️ Risks
EXIF orientation rotating images silently (use `cv2.IMREAD_IGNORE_ORIENTATION` or apply orientation explicitly and consistently) · HEIC files from phones (convert to JPG).

### 🧰 Skills
`computer-vision-opencv`, `python-testing-patterns`, `jupyter-notebook`.

---

## Phase 2 — Feature Extraction

**Goal:** Keypoints + descriptors for every image, cached and inspectable.

### Tasks
- `features/extractor.py`: `Extractor` interface with `SIFTExtractor` (default, `nfeatures` from config) and `ORBExtractor`. Return `Features(points: (N,2) float32, descriptors, sizes, angles)`.
- Optional RootSIFT normalization (flag).
- Cache to `outputs/<run>/features/<image_id>.npz` keyed by image hash + config hash.
- `visualization/`: draw keypoints on sample images; `feature_statistics.csv` (image, detector, count, time_ms).

### 🧪 Tests
Textured synthetic image → >0 features with descriptor dims 128 (SIFT) / 32 bytes (ORB); blank image → 0 features without crashing; cache hit returns identical arrays.

### ✅ Exit criteria
Stable feature counts on the sample set; keypoints visibly concentrate on texture, not on sky/background noise.

### 🧰 Skills
`computer-vision-opencv`, `matplotlib`.

---

## Phase 3 — Feature Matching

**Goal:** Reliable candidate correspondences between image pairs.

### Tasks
- `features/matcher.py`: FLANN (SIFT) / BFMatcher Hamming (ORB); kNN k=2; Lowe ratio test (config, default 0.75); optional mutual nearest-neighbor check.
- Pair selection: exhaustive for ≤ 50 images, sequential window (± k) option for larger sets.
- Output `Matches(i, j, idx_i, idx_j, distances)`; `match_statistics.csv` (pair, raw, after_ratio, after_mutual).
- Match visualizations for a few pairs (subsampled lines).

### 🧪 Tests
All returned indices in range; ratio test monotonic (lower ratio ⇒ fewer matches); matching an image to itself yields near-identity matches; empty descriptor sets return empty matches.

### ✅ Exit criteria
On inspected pairs the majority of lines connect the same physical point.

### 🧰 Skills
`computer-vision-opencv`, `matplotlib`, `jupyter-notebook` (`02_feature_matching.ipynb`).

---

## Phase 4 — Geometric Verification

**Goal:** Remove matches inconsistent with two-view geometry and quantify pair quality.

### Tasks
- `geometry/epipolar.py`:
  - `estimate_fundamental(pts1, pts2)` — `cv2.findFundamentalMat` with `USAC_MAGSAC` (or `FM_RANSAC`), threshold in px.
  - `estimate_essential(pts1, pts2, K)` — `cv2.findEssentialMat`, RANSAC/USAC.
  - Sampson-distance residuals; inlier ratio; homography inlier ratio (`H/F` ratio) to flag planar/low-parallax pairs.
- Implement the **normalized 8-point algorithm** yourself as an educational reference and compare against OpenCV on synthetic data.
- Epipolar-line visualization for a pair.

### 🧪 Tests (synthetic)
Noise-free: estimated `E` equals ground-truth `[t]×R` up to scale/sign; `x₂ᵀFx₁ ≈ 0` for all inliers; with 30 % injected outliers, RANSAC recovers ≥ 95 % of true inliers; < 8 points ⇒ explicit failure result, not exception.

### ✅ Exit criteria
Real pairs show stable inlier ratios; epipolar lines pass through corresponding points.

### 🧰 Skills
`computer-vision-opencv`, `matplotlib`, `python-testing-patterns`.

---

## Phase 5 — Relative Camera Pose

**Goal:** Recover `R` and unit `t` between two calibrated views.

### Tasks
- `geometry/pose_estimation.py`: wrap `cv2.recoverPose` (four-solution decomposition + cheirality); also implement SVD decomposition of `E` manually for understanding; return pose + count of points in front of both cameras.
- Reject poses when cheirality-passing ratio < threshold or baseline/parallax too small.
- Initial-pair scoring for later SfM: many inliers **and** low `H/F` inlier ratio **and** sufficient median triangulation angle.

### 🧪 Tests (synthetic)
Noise-free rotation error < 0.1°, translation-direction error < 0.1°; with 1 px noise, < 1–2°; pure-rotation pair is flagged as degenerate.

### ✅ Exit criteria
Plausible relative pose on a real pair (camera 2 displaced in the expected direction when plotted).

### 🧰 Skills
`computer-vision-opencv`, `python-testing-patterns`, `systematic-debugging`.

---

## Phase 6 — Triangulation

**Goal:** 3D points from verified correspondences with quality filtering.

### Tasks
- `geometry/triangulation.py`: build `P = K[R|t]`; `cv2.triangulatePoints` + own linear DLT for N views (needed in Phase 8); dehomogenize safely.
- Filters: positive depth in all observing cameras; reprojection error < τ px; triangulation angle > θ_min (e.g. 1.5–2°); finite values.
- `geometry/reprojection.py`: per-observation and mean/median errors.

### 🧪 Tests (synthetic)
Noise-free: 3D error ≈ 0 (up to global scale for two-view); noisy: mean reprojection < noise level × ~1.5; points behind camera are rejected; N-view DLT agrees with two-view result.

### ✅ Exit criteria
Finite, plausible set of 3D points; reprojection histogram concentrated below ~1–2 px on real data.

### 🧰 Skills
`computer-vision-opencv`, `python-testing-patterns`.

---

## Phase 7 — Two-View Sparse Point Cloud ★ Milestone M1

**Goal:** Visible, exportable two-view reconstruction.

### Tasks
- `pointcloud/coloring.py`: color each point by sampling the image at its observation (BGR→RGB, /255).
- `pointcloud/export.py`: write PLY via Open3D; write `cameras.json` (id, K, R, t, center).
- `visualization/cameras.py` / `viewer.py`: Open3D viewer showing points + camera frusta; matplotlib top-down camera plot as static figure.
- `pipeline.py` + `cli.py two-view` command wiring Phases 1–7; each run gets `outputs/<run_id>/` with `config.yaml` and `environment.json`.
- `notebooks/03_two_view_geometry.ipynb`.

### 🧪 Tests
Integration: two-view on `data/sample/` completes, PLY reloads with same point count, `cameras.json` validates.

### ✅ Exit criteria (M1)
A real image pair produces a recognizable structure, not a random cloud. Screenshot saved for the report.

### 🧰 Skills
`open3d-python`, `matplotlib`, `jupyter-notebook`.

---

## Phase 8 — Multi-View Incremental SfM

**Goal:** Register most images into one consistent reconstruction.

### Tasks
1. **Pairwise graph** — verified matches for all candidate pairs (Phases 3–4), stored as a view graph.
2. **Tracks** — `reconstruction/tracks.py`: union-find over verified matches → feature tracks; drop tracks with conflicting observations in the same image.
3. **Data model** — `camera_registry.py` (registered cameras, poses) and `landmarks.py` (3D point ↔ list of `(image_id, keypoint_idx)` observations, color, error).
4. **Initialization** — best initial pair by Phase-5 score → two-view reconstruction.
5. **Next-best-view loop** — choose unregistered image with most 2D–3D correspondences → `cv2.solvePnPRansac` (+ `solvePnPRefineLM`) → triangulate new tracks visible in ≥ 2 registered views → filter.
6. **Bundle adjustment** — `bundle_adjustment.py` with `scipy.optimize.least_squares` (Huber/Cauchy loss, sparse Jacobian via `jac_sparsity`); rotations as Rodrigues vectors; fix first camera (and scale). Run locally after each registration and globally every k images / at the end.
7. **Re-triangulation & filtering** after BA (reprojection error, angle).
8. Optional: COLMAP fallback runner (`reconstruction/colmap_runner.py`) reading `cameras.bin/images.bin/points3D.bin` into the same data model.

### 🧪 Tests (synthetic, many cameras)
Track builder merges chains correctly; PnP recovers known pose; BA reduces reprojection error on a perturbed scene and recovers poses up to similarity; full synthetic SfM registers all cameras.

### ✅ Exit criteria
≥ 80 % images registered on the sample dataset; mean reprojection error ≲ 1–1.5 px after BA; visually coherent model.

### ⚠️ Risks
Drift, bad initial pair, scale/sign convention bugs, slow BA. Mitigate with synthetic tests per step, sparse Jacobians, and capping points per image during development.

### 🧰 Skills
`computer-vision-opencv`, `python-testing-patterns`, `systematic-debugging`, `open3d-python`.

---

## Phase 9 — Filtering, Export & Optional Dense ★ Milestone M2

**Goal:** Clean final outputs, optionally densified.

### Tasks
- `pointcloud/filtering.py`: remove non-finite, high-error, low-track-length points; Open3D statistical outlier removal and radius outlier removal; log counts before/after each filter.
- Export `sparse.ply`, `filtered.ply`, `cameras.json`, `reconstruction.json` (stats).
- Optional dense: export our poses to COLMAP format → `image_undistorter` → `patch_match_stereo` → `stereo_fusion` → `dense.ply`; optional Poisson mesh via Open3D (`meshes/reconstructed_mesh.ply`). Clearly label as external.
- `cli.py reconstruct` and `cli.py view`.

### 🧪 Tests
Filters never increase point count; exported files reload identically; COLMAP runner skipped gracefully if `colmap` is not on PATH.

### ✅ Exit criteria (M2)
End-to-end `reconstruct` on a real dataset produces a clean, coherent multi-view cloud with cameras.

### 🧰 Skills
`open3d-python`.

---

## Phase 10 — Evaluation & Experiments

**Goal:** Answer RQ1–RQ5 with reproducible numbers.

### Tasks
- `evaluation/metrics.py`: all metrics from README §12, including pose error vs synthetic/COLMAP and Chamfer/completeness after Umeyama/ICP alignment.
- `evaluation/experiments.py`: config sweeps (YAML grid) → one run dir per setting → aggregated `experiment_results.csv`.
- Experiments E1–E6 (README §12). Metric scale from one measured distance.
- `evaluation/reports.py` + `scripts/evaluate_results.py`: plots (points vs #images, inlier ratio by detector, reprojection error histograms, runtime breakdown).
- `notebooks/04_reconstruction_analysis.ipynb`, `docs/experiments.md`.

### ✅ Exit criteria
Every figure/table in the report regenerates from committed configs with one command.

### 🧰 Skills
`matplotlib`, `jupyter-notebook`, `open3d-python`.

---

## Phase 11 — Documentation & Presentation ★ Milestone M3

### Tasks
- `docs/architecture.md` (final diagram), `docs/methodology.md` (math + design choices, what is ours vs external), `docs/experiments.md` (results & discussion, failure cases, limitations).
- README: update status badge, real screenshots, final commands.
- Presentation: problem → geometry → architecture → workflow → results → limitations → conclusions; live or recorded demo.
- Final pass: `pytest` green, fresh-clone install verified, `code-review` on the final branch.

### ✅ Exit criteria
README checklist fully ticked; a classmate can reproduce a reconstruction from the README alone.

### 🧰 Skills
`code-review`, `matplotlib`.

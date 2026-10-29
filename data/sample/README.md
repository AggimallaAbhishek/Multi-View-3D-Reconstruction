# Sample data

Full datasets live in `data/raw/<name>/` (git-ignored) and are fetched by script, so the repository stays small.

| Set | Source | Images | Ground truth | Licence / terms | How to get it |
|---|---|---|---|---|---|
| **templeRing** (primary) | Middlebury Multi-View Stereo — Seitz, Curless, Diebel, Scharstein, Szeliski, *A Comparison and Evaluation of Multi-View Stereo Reconstruction Algorithms*, CVPR 2006. https://vision.middlebury.edu/mview/data/ | 47 × 640×480 PNG around a plaster temple replica; consecutive views mostly 7–8° apart, but the file sequence sweeps ~1.5 turns with a few jumps (not angular order) | `K`, `R`, `t` per image (`P = K[R\|t]`, same world→camera convention as this project) | Free for research use; cite the paper above | `python scripts/download_middlebury.py templeRing` |
| templeSparseRing | same | 16 views, 22.5° apart | same | same | `python scripts/download_middlebury.py templeSparseRing` |
| dinoRing / dinoSparseRing | same | 48 / 16 views of a plaster dinosaur | same | same | `python scripts/download_middlebury.py dinoRing` |

The download script writes `camera.yaml` (shared intrinsics, for `camera.calibration_file`) and `gt_cameras.json` (per-image ground truth) next to the images.

## Results on templeRing (Phases 1–9)

```bash
python scripts/download_middlebury.py templeRing
python -m src.cli validate data/raw/templeRing
python scripts/evaluate_against_ground_truth.py data/raw/templeRing
```

| Stage | Result |
|---|---|
| Validation | 47/47 readable; no false blur warnings; exactly one near-duplicate flagged — `templeR0001`/`templeR0030`, which have **identical** ground-truth poses (the ring closes on itself) |
| Features (SIFT, 640×480) | median 875 per image, min 597 |
| Matching (exhaustive, 1081 pairs) | 235 pairs verified with the essential matrix |
| Relative poses | 229 pairs; rotation error median **0.51°** (90 % < 1.79°, max 4.33°), translation direction median **0.49°** (90 % < 1.23°, max 4.57°) — re-measured after `E` refinement on its inliers (was 0.61° / 0.55°, max 12°) |
| Initial pair | `templeR0033`–`templeR0034`: 626 points, 8.3° parallax, errors 0.47° / 0.57° |
| Two-view cloud (Phase 7, M1) | 626 coloured points in `sparse.ply`; the close-up render shows the temple's entablature, three columns and base, as in the photos |
| Triangulation (Phase 6) | 626/626 points valid; reprojection median 0.04 px; 3-D shape error after similarity alignment median 0.094 % of scene size; scale vs true baseline 1.059 |
| Multi-view SfM (Phase 8) | **47/47 images registered**, 0 failed attempts (median 338 PnP inliers); 6454 landmarks, 27 905 observations (mean track length 4.3). After similarity alignment to ground truth: camera rotation error median **0.15°** (max 0.41°), centre error median **0.18 %** of the camera ring's extent (max 0.71 %). Reprojection mean **0.23 px**, median 0.14 px, max 1.99 px (filter at 2 px). Cameras close the full ring; the render shows the temple's block and columns |
| Filtering (Phase 9, M2) | 6365 of 6454 points kept in `filtered.ply`: mean reprojection error > 1 px 60, statistical outliers 19, radius outliers 10. Removes the stray points up to ~13 units out (99 % of points lie within 1.35 of the median); the temple's shape is unchanged |
| Runtime (Apple M4, `evaluate_against_ground_truth.py`, caches warm) | 2.0 s up to the two-view cloud, 13.6 s for the multi-view reconstruction (bundle adjustment dominates) |

Before `E` refinement, three of the 229 pairwise poses erred by 5–12°: wide baselines (23–38°) with 75–169 inliers, where E is poorly conditioned; error grew as inliers fell (median 0.32° above 250 inliers, ~1° below 120). Refinement brought the worst down to 4.3°. Incremental SfM (Phase 8) therefore registers views by PnP and refines with bundle adjustment rather than chaining pairwise poses — its camera rotations (median 0.15°) are 3× more accurate than the pairwise estimates.

Own captures: document the camera, resolution, focal length, lighting and capture path for each scene in `data/raw/<scene>/README.md`.

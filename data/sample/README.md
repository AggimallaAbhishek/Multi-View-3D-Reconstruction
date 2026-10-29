# Sample data

Full datasets live in `data/raw/<name>/` (git-ignored) and are fetched by script, so the repository stays small.

| Set | Source | Images | Ground truth | Licence / terms | How to get it |
|---|---|---|---|---|---|
| **templeRing** (primary) | Middlebury Multi-View Stereo — Seitz, Curless, Diebel, Scharstein, Szeliski, *A Comparison and Evaluation of Multi-View Stereo Reconstruction Algorithms*, CVPR 2006. https://vision.middlebury.edu/mview/data/ | 47 × 640×480 PNG, ring of views ~7.7° apart around a plaster temple replica | `K`, `R`, `t` per image (`P = K[R\|t]`, same world→camera convention as this project) | Free for research use; cite the paper above | `python scripts/download_middlebury.py templeRing` |
| templeSparseRing | same | 16 views, 22.5° apart | same | same | `python scripts/download_middlebury.py templeSparseRing` |
| dinoRing / dinoSparseRing | same | 48 / 16 views of a plaster dinosaur | same | same | `python scripts/download_middlebury.py dinoRing` |

The download script writes `camera.yaml` (shared intrinsics, for `camera.calibration_file`) and `gt_cameras.json` (per-image ground truth) next to the images.

## Results on templeRing (Phases 1–7)

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
| Relative poses | 229 pairs; rotation error median **0.61°** (90 % < 2.05°), translation direction median **0.55°** (90 % < 1.90°) |
| Initial pair | `templeR0033`–`templeR0034`: 626 points, 8.3° parallax, errors 0.47° / 0.57° |
| Two-view cloud (Phase 7, M1) | 626 coloured points in `sparse.ply`; the close-up render shows the temple's entablature, three columns and base, as in the photos |
| Triangulation (Phase 6) | 626/626 points valid; reprojection median 0.04 px; 3-D shape error after similarity alignment median 0.094 % of scene size; scale vs true baseline 1.059 |
| Runtime (M-series Mac) | 3.0 s cold, 0.8 s with caches |

Three of the 229 pairwise poses err by 5–12°: wide baselines (23–38°) with 75–169 inliers, where E is poorly conditioned. Error grows as inliers fall (median 0.32° above 250 inliers, ~1° below 120). Incremental SfM (Phase 8) should therefore register views by PnP and refine with bundle adjustment rather than chain pairwise poses.

Own captures: document the camera, resolution, focal length, lighting and capture path for each scene in `data/raw/<scene>/README.md`.

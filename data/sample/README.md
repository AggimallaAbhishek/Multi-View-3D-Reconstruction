# Sample data

Full datasets live in `data/raw/<name>/` (git-ignored) and are fetched by script, so the repository stays small.

| Set | Source | Images | Ground truth | Licence / terms | How to get it |
|---|---|---|---|---|---|
| **templeRing** (primary) | Middlebury Multi-View Stereo — Seitz, Curless, Diebel, Scharstein, Szeliski, *A Comparison and Evaluation of Multi-View Stereo Reconstruction Algorithms*, CVPR 2006. https://vision.middlebury.edu/mview/data/ | 47 × 640×480 PNG around a plaster temple replica; consecutive views mostly 7–8° apart, but the file sequence sweeps ~1.5 turns with a few jumps (not angular order) | `K`, `R`, `t` per image (`P = K[R\|t]`, same world→camera convention as this project) | Free for research use; cite the paper above | `python scripts/download_middlebury.py templeRing` |
| templeSparseRing | same | 16 views, 22.5° apart | same | same | `python scripts/download_middlebury.py templeSparseRing` |
| dinoRing / dinoSparseRing | same | 48 / 16 views of a plaster dinosaur | same | same | `python scripts/download_middlebury.py dinoRing` |
| ETH3D pipes / office / terrains (and the other training scenes) | ETH3D high-resolution multi-view benchmark — Schöps et al., *A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos*, CVPR 2017. https://www.eth3d.net/datasets | 14 / 26 / 42 undistorted DSLR photos, 6220×4141 (worked on at 1600 px) | laser-scan-registered `K`, `R`, `t` (COLMAP text format, converted) | CC BY-NC-SA 4.0 | `uv pip install -e ".[data]"` then `python scripts/download_eth3d.py pipes` |

The download script writes `camera.yaml` (shared intrinsics, for `camera.calibration_file`) and `gt_cameras.json` (per-image ground truth) next to the images.

## Results on templeRing (Phases 1–9)

*Measured during each phase with the settings of the time (SIFT contrast threshold 0.04). With the current defaults the full pipeline gives 47/47 cameras at 0.15° median / 0.32° max rotation error and 9907 filtered points; see [`docs/experiments.md`](../../docs/experiments.md).*

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

## Results on dinoRing

```bash
python scripts/download_middlebury.py dinoRing
python scripts/evaluate_against_ground_truth.py data/raw/dinoRing
```

48 views of a smooth plaster dinosaur on black, lit from one side: a low-texture case. With a fixed SIFT contrast threshold (0.04) the images had 74–221 features each, tracking their brightness (correlation 0.9), and only the 19 brightly lit views registered. With the per-image adaptive threshold (default since the fix: below 500 features, halve down to 0.005):

| | Fixed threshold 0.04 | Adaptive (default) |
|---|---|---|
| Registered | 19/48 | **48/48** |
| Landmarks | 500 | 3635 |
| Camera rotation error, median / max | 0.24° / 0.57° | **0.15° / 0.28°** |
| Camera centre error, median / max | 0.47 % / 1.00 % | 0.23 % / 0.45 % |
| Reprojection error, mean | 0.52 px | 0.44 px |
| Runtime (Apple M4, cold) | — | 26 s |

These were measured with `K` fixed in bundle adjustment. With the focal length refined (the default since E3b, [docs/experiments.md](../../docs/experiments.md#e3b--refining-the-focal-length-in-bundle-adjustment-rq3)) dinoRing is unchanged within noise: 48/48, 3636 landmarks, rotation error 0.15° / 0.30°, centre error 0.24 % / 0.47 %, reprojection mean 0.44 px.

dinoSparseRing (16 views ~22° apart) is beyond the ~20° matching limit: 5/16 registered with the adaptive threshold (4/16 with the fixed one).

Affine simulation (`features.affine_simulation`, ASIFT via OpenCV's `AffineFeature`, **external**) does not fix it. It lifts the weakest neighbour links — with `sift_contrast_threshold: 0.01`, the pairs 4–5, 5–6, 11–12 and 12–13 verify with 45 / 39 / 104 / 57 inliers instead of 13 / 12 / 22 / 21 — yet only 6/16 images register (rotation error 0.50° median). The limit is registration, not matching: PnP needs 2D–3D links, i.e. tracks the new image shares with **two** registered images, and at 22° spacing image 12 shares 104 tracks with the reconstruction but only 15 of them with two registered images. On templeRing affine simulation is harmful: 7× the landmarks but rotation error 1.22° (vs 0.15°) and 38 min (vs 12 s) — its keypoints, mapped back from warped views, are less precisely located and duplicated across views. It stays off by default.

With the relative-pose registration fallback (default on; methodology §7) dinoSparseRing registers 6/16 (rotation error median 0.19°, max 0.37°; was 5/16, 0.75°). The rest cannot be reached from the information in the images: the pose stage's own estimates for the weak neighbour pairs are 15–98° off with 0.1–7° parallax where ~22° is expected (the matches there do not determine a relative pose), and the fallback's landmark check refuses them.

Own captures: document the camera, resolution, focal length, lighting and capture path for each scene in `data/raw/<scene>/README.md`.

## Results on ETH3D (E7)

```bash
uv pip install -e ".[data]"
for s in pipes office terrains; do python scripts/download_eth3d.py $s; done
python scripts/evaluate_against_ground_truth.py data/raw/eth3d_pipes
```

| Scene | Registered | Rotation error (median / max) | Centre error (median) | Landmarks |
|---|---|---|---|---|
| pipes (14) | 14/14 | 0.06° / 0.12° | 0.21 % | 2661 |
| office (26) | 26/26 | 0.14° / 0.35° | 0.66 % | 3170 |
| terrains (42) | 42/42 | 0.18° / 0.22° | 0.21 % | 17832 |

From the original JPGs (`--distorted`, lens distortion not given), with the distortion estimated by the pipeline: pipes 14/14 at 0.17° / 0.24°, office 26/26 at 0.05° / 0.31° (E8).

With OpenCV's default SIFT contrast threshold (0.04) the undistorted scenes gave 7/14, 25/26 and 30/42; the comparison with COLMAP and the reasons are in [`docs/experiments.md`](../../docs/experiments.md#e7--other-datasets-eth3d-generalisation).

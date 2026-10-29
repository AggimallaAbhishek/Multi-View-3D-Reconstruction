# Experiments (Phase 10)

Answers to the research questions on Middlebury **templeRing** (47 views, 640×480, ground-truth cameras). All numbers below regenerate with one command:

```bash
python scripts/download_middlebury.py templeRing     # once
python scripts/run_experiments.py                     # all specs in configs/experiments/, ~9 min on an Apple M4
python scripts/evaluate_results.py outputs/experiments/<run>   # redraw figures/tables only
```

Each experiment directory holds `experiment_results.csv` (one row per setting), `results.md`, `metrics.png`, `runtime.png`, and one sub-directory per run with its exact `config.yaml`, `environment.json`, `metrics.json` and `filtered.ply`. `notebooks/04_reconstruction_analysis.ipynb` loads the newest run.

## How runs are scored

| Metric | Definition |
|---|---|
| registered fraction | registered images / input images |
| points (filtered) | landmarks in `filtered.ply` (after Phase 9 filtering) |
| mean track length | observations per landmark |
| reprojection error | mean over all observations, px — an internal consistency check, not accuracy |
| rotation / centre error | per camera, after the similarity that best maps our camera centres onto the true ones (Umeyama); centre error as a fraction of the camera ring's RMS radius |
| completeness | share of reference points with a reconstructed point within **1 %** of the reference's bounding-box diagonal (`evaluation.completeness_tolerance`) |
| accuracy | median distance from each reconstructed point to the nearest reference point, fraction of the diagonal |
| metric scale error | the scale from **one measured distance** (the true distance between the first two registered cameras), relative to the scale fitted over all cameras |

**Reference cloud (internal, not an external ground truth).** Middlebury does not publish the temple's ground-truth surface. The reference is every verified SIFT track over all 47 images, triangulated with the **ground-truth cameras** and filtered as a reconstruction would be: 6223 points. It is independent of our pose estimates but shares our features and matches, which has two consequences: (1) a run with fewer images cannot exceed the coverage of the full set; (2) *accuracy* favours SIFT runs, whose points sit on the same keypoints as the reference — compare detectors by completeness and pose error, not accuracy.

## Results

### E1 — image count (RQ1)

Evenly spaced subsets of the ring (`dataset.max_images`); spacing in multiples of the ~7.7° between neighbouring views.

| images (spacing) | registered | points | track length | completeness | rotation error |
|---|---|---|---|---|---|
| 8 (~50°) | 2 | 413 | 2.0 | – | – |
| 12 (~32°) | 4 | 499 | 2.3 | 0.08 | 0.36° |
| 16 (~23°) | 6 | 997 | 2.7 | 0.14 | 0.58° |
| 24 (~15°) | 21 | 3071 | 3.1 | 0.74 | 0.08° |
| 32 (~11°) | 27 | 4450 | 3.4 | 0.83 | 0.13° |
| 47 (7.7°) | **47** | **6365** | **4.3** | **0.92** | 0.15° |

**RQ1:** completeness is limited by *view spacing*, not by image count as such. Between 23° and 15° spacing the reconstruction jumps from 6 to 21 registered images and from 14 % to 74 % completeness: SIFT matches on this plaster object barely survive verification beyond ~20° (Phase 4 saw 3 matches at 60°), so wider-spaced views never get enough 2D–3D links to register. Above that threshold, more images add points and longer tracks with diminishing returns (24 → 47 images: +18 points of completeness). Pose accuracy is already good once images register at all. Runtime grows superlinearly with image count (0.3 s → 15.6 s), mostly bundle adjustment.

### E2 — matching thresholds (RQ2)

| ratio | mutual | verified pairs | inlier ratio | points | completeness | reprojection |
|---|---|---|---|---|---|---|
| 0.6 | on | 185 | 0.97 | 6396 | 0.894 | 0.19 px |
| 0.7 | on | 218 | 0.93 | **6453** | **0.918** | 0.22 px |
| 0.8 | on | 254 | 0.84 | 6217 | 0.909 | 0.24 px |
| 0.9 | on | 284 | 0.69 | 5664 | 0.886 | 0.25 px |
| 0.9 | off | 298 | 0.55 | 4809 | 0.832 | 0.25 px |

**RQ2:** looser ratios verify more pairs but with lower inlier ratios, and the final cloud *shrinks* (6453 → 4809 points) and loses completeness (0.918 → 0.832): the extra matches are ambiguous ones, which lower inlier ratios and — our likely explanation, not measured separately — merge unrelated keypoints into tracks that the track builder then drops as inconsistent (two keypoints of one image). Too strict (0.6) loses tracks instead (track length 4.0 vs 4.4). The optimum is 0.7–0.8; the default 0.75 sits in it. The mutual check is nearly free at strict ratios and matters most at loose ones (+18 % points at 0.9). Every setting registers all 47 images — on this dataset matching quality decides cloud quality, not registration.

### E3 — calibration accuracy (RQ3)

| focal length | rotation error | centre error | completeness | accuracy | metric scale error | runtime |
|---|---|---|---|---|---|---|
| −20 % | 2.31° | 2.4 % | 0.08 | 4.1 % | +6.1 % | 41.5 s |
| −10 % | 0.94° | 0.9 % | 0.22 | 1.8 % | +1.7 % | 36.3 s |
| −5 % | 0.46° | 0.5 % | 0.44 | 1.1 % | +0.1 % | 28.3 s |
| **true** | **0.15°** | **0.2 %** | **0.92** | **0.45 %** | −0.7 % | 15.4 s |
| +5 % | 0.28° | 0.3 % | 0.59 | 0.8 % | −1.3 % | 18.3 s |
| +10 % | 0.69° | 0.7 % | 0.31 | 1.2 % | −2.5 % | 24.8 s |
| +20 % | 1.23° | 1.2 % | 0.18 | 1.6 % | −3.5 % | 41.5 s |
| guessed (−49 %) | 20.0° | 24 % | 0.00 | 57 % | +84 % | 29.0 s |

**RQ3:** calibration is the single most important input. Every setting registers all 47 images and the reprojection error barely moves (0.23 → 0.29 px at −20 %) — **a wrong focal length is invisible in the internal diagnostics** — yet the geometry degrades steadily: a 5 % focal error halves completeness, 10 % leaves a quarter, and the fallback guess (`f = 1.2·max(w, h)` = 768 px vs the true 1520 px) produces a distorted model (20° rotation error, nothing within tolerance of the reference). A wrong focal length stretches depth relative to width, a distortion no similarity transform removes. Bundle adjustment also works harder (15 → 41 s) to absorb the inconsistency. Implication: calibrate the camera (`python -m src.cli calibrate`), or extend bundle adjustment to refine the focal length — today it keeps K fixed.

### E4 — SIFT vs ORB (RQ4)

| detector | features/image | inlier ratio | points | track length | completeness | rotation error | reprojection | runtime |
|---|---|---|---|---|---|---|---|---|
| SIFT | 869 | 0.90 | 6365 | 4.3 | **0.92** | **0.15°** | **0.23 px** | **15.5 s** |
| ORB | 3460 | 0.85 | 19409 | 3.0 | 0.57 | 0.58° | 0.42 px | 64.5 s |

Neither detector reaches the 8000-feature cap: SIFT's contrast threshold keeps ~870 features per image on the smooth plaster, ORB's FAST detector ~3500. **RQ4:** ORB yields 3× more points but shorter tracks and less reliable ones — 4× the pose error, nearly twice the reprojection error and much lower completeness (0.57 vs 0.92) despite the denser cloud; its binary descriptors are less distinctive and its keypoints less precisely localised. It is also 4× slower overall, because the extra features make matching (11 s vs 0.2 s) and bundle adjustment (50 s vs 14 s) more expensive. SIFT is the right default; ORB's only advantage on this data is raw point count. (Phase 2 measured ORB failing entirely on blurred images, where SIFT still found 265 features.)

### E5 — bundle adjustment

| bundle adjustment | rotation error (median / max) | centre error | reprojection | completeness | metric scale error | runtime |
|---|---|---|---|---|---|---|
| on | **0.15° / 0.41°** | **0.18 %** | **0.23 px** | **0.92** | −0.7 % | 15.4 s |
| off | 0.62° / 1.09° | 0.76 % | 0.43 px | 0.38 | −5.6 % | 2.3 s |

Bundle adjustment (with the structure refinement that follows it) cuts pose errors 4×, halves the reprojection error and raises completeness from 0.38 to 0.92 — without it, small PnP errors accumulate around the ring. It costs nearly 90 % of the runtime (13.5 of 15.4 s).

### Metric scale from one measured distance

A monocular reconstruction has no scale; one known distance fixes it (`evaluation.metrics.metric_scale`). Using the true distance between two neighbouring cameras, the resulting scale is within **0.7 %** of the scale fitted over all 47 cameras for the default pipeline (within 1.5 % for every E1/E2 setting that registers ≥ 3 cameras), 5.6 % without bundle adjustment, and 84 % off with guessed intrinsics. A longer reference distance (e.g. across the object) averages out more local error.

### E6 — COLMAP baseline (RQ5): not run

Needs COLMAP (`brew install colmap`), which is not installed in the development environment. RQ5 (sparse vs dense) depends on it and stays open.

## Limits of these experiments

- One dataset, one object (plaster, black background, a ring of views). Conclusions about view spacing and detector choice are specific to such low-texture objects.
- The reference shares our SIFT features and matches; it measures geometry given the tracks, not against an independent surface scan.
- Each setting runs once. The pipeline is deterministic (seeded RANSAC), so repeats give identical numbers; variation across datasets is not measured.

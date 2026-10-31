# Methodology

How each stage works, why it is built that way, and which parts are our own implementation versus external libraries. Conventions (frames, array shapes, intrinsics under resizing) are in [architecture.md](architecture.md); measured results are in [experiments.md](experiments.md) and [`data/sample/README.md`](../data/sample/README.md).

Notation: world→camera `x_c = R X + t`, camera centre `C = −Rᵀt`, projection `x ∼ K[R|t]X`, normalised coordinates `x̂ = K⁻¹x`.

---

## 1. Images and intrinsics

Images are discovered, naturally sorted (`img2` < `img10`), decoded with EXIF orientation applied, validated (unreadable, blurry, near-duplicate, mixed resolution) and downscaled to `preprocessing.max_side`.

**Intrinsics** come per image from, in order: a calibration file (`camera.yaml`, from `cli calibrate` or a dataset), the EXIF 35 mm-equivalent focal length (converted along the image diagonal), or the fallback `f = 1.2·max(w, h)` with the principal point at the image centre. Resizing by `s` rescales `K` with the pixel-centre formula `f' = s·f`, `c' = (c + 0.5)·s − 0.5`.

**Why it matters:** experiment E3 shows a 5 % focal-length error halves completeness, and the fallback guess (49 % off on templeRing) produces a distorted model, while the reprojection error barely changes. Intrinsics are the most important input; bundle adjustment keeps them fixed (§8).

## 2. Features and matching

| Step | Method | Ours / external |
|---|---|---|
| Detection + description | SIFT (default; optional RootSIFT) or ORB on the grayscale image; strict strongest-N cap | OpenCV `SIFT_create` / `ORB_create` (**external**); cap and RootSIFT ours |
| Adaptive contrast threshold | an image with fewer than 500 SIFT features is detected again with the contrast threshold halved, from 0.04 down to 0.005 | ours |
| Nearest neighbours | exact brute force, 2-NN | OpenCV `batchDistance` (**external**) |
| Ratio test | keep `d₁ < r·d₂` (Lowe 2004), default `r` = 0.75 | ours |
| Mutual check | keep only matches that are each other's nearest neighbour | ours |
| Pair selection | exhaustive up to 50 images, sequential window above | ours |

SIFT's contrast threshold is absolute, so one value cannot suit every image: on Middlebury dinoRing (smooth plaster, lit from one side) the dark half of the ring kept 74–160 features per image at 0.04 (correlation with brightness 0.9), and registration stopped at 19/48 cameras. Lowering the threshold globally to 0.005 registers all 48, but also makes templeRing 2.5× slower for no gain in coverage. Lowering it only for images below 500 features registers all 48 dinoRing cameras (rotation error median 0.15°, max 0.28°) and leaves templeRing untouched — its sparsest image already has 597.

Brute force is the default because, at 8000 SIFT features, it was as fast as FLANN and FLANN returned different matches on identical runs. E2 places the best ratio at 0.7–0.8; looser ratios verify more pairs but give fewer, worse points.

## 3. Geometric verification

For each pair, a robust two-view model separates correct matches from wrong ones.

**Model choice.** With trustworthy intrinsics on both images (file or EXIF) we fit the essential matrix `E` (5 DoF) on normalised coordinates; otherwise the fundamental matrix `F` (7 DoF) on pixels. With a known `K`, `E` is better conditioned: projecting a fitted `F` to an essential matrix gave translation errors up to ~9° where `E` gave ~1°.

$$\hat{\mathbf{x}}_j^\top E\,\hat{\mathbf{x}}_i = 0,\qquad E = [t]_\times R,\qquad F = K_j^{-\top} E\,K_i^{-1}$$

**Robust estimation** uses OpenCV's `USAC_ACCURATE` (**external**), seeded for reproducibility (measured with 30 % outliers: E error 0.005 vs 0.039 for plain RANSAC, inlier recall 0.95 vs 0.82). Our own normalised 8-point algorithm (Hartley normalisation, rank-2 enforcement) is the educational reference and is cross-checked against it in tests.

**Inliers** for either model are judged by the first-order geometric (Sampson) distance to `F`, in pixels:

$$d_S = \frac{(\mathbf{x}_j^\top F\mathbf{x}_i)^2}{(F\mathbf{x}_i)_1^2 + (F\mathbf{x}_i)_2^2 + (F^\top\mathbf{x}_j)_1^2 + (F^\top\mathbf{x}_j)_2^2}$$

**Refinement.** RANSAC only guarantees each inlier is within threshold; on wide-baseline pairs with few inliers that leaves poses degrees off. The verified `E` is therefore refined by minimising the inliers' Sampson distances over `(R, t)` — R as a Rodrigues vector, t on the unit sphere — with SciPy `least_squares` (residuals ours, solver **external**). On templeRing this cut the worst pairwise pose error from 12° to 4.3°.

**Planarity.** A homography is also fitted (OpenCV `findHomography`, **external**); the ratio of homography to epipolar inliers flags planar or rotation-only pairs, which make poor initial pairs. The homography threshold is scaled by √2·√(χ²₂/χ²₁) ≈ 1.77 so both residuals are judged at the same confidence (our choice).

## 4. Relative pose and the initial pair

**Decomposition (ours).** `E = U diag(1,1,0) Vᵀ` gives four candidate poses `(UWVᵀ, ±u₃)`, `(UWᵀVᵀ, ±u₃)`. The correct one puts points in front of both cameras (cheirality). Points farther than `max_depth` (50 baselines) count as at infinity — their depth sign is noise — so a narrow-baseline pair is not rejected for them. `cv2.recoverPose` is used only as the test oracle (agreement to 1e-4°).

**Parallax** is the rotation-compensated angle between the two viewing rays of each point. Unlike the translation, it exists without a correct `t`, so pure rotations are detected (median parallax below 1°) while their rotation is still recovered.

**Initial pair:** a valid pose, homography ratio ≤ 0.8 (not planar), median parallax ≥ 4°, and among those the most valid points. It defines the world frame — camera *i* at the origin, baseline length 1.

## 5. Triangulation

**N-view DLT (ours).** Each observation of point `X` in camera `P = K[R|t]` gives two linear equations in homogeneous `X`; on normalised coordinates, with each row scaled to unit norm, the stacked system `A X = 0` is solved for all points at once (per-point 4×4 normal matrices `AᵀA`, one batched eigendecomposition). It handles any number of views per point, which tracks and bundle adjustment need.

**Refinement (ours).** A few Gauss-Newton steps per point on the pixel reprojection error, accepted per point only if they reduce it.

**Filters.** A point is kept only if it has ≥ 2 views, finite coordinates, positive depth in every observing camera, every reprojection error ≤ 2 px, and a triangulation angle (largest angle between its rays) ≥ 1.5°. Small angles mean nearly parallel rays and unreliable depth.

On templeRing's initial pair: 626/626 points valid, reprojection median 0.04 px, 3-D shape error after similarity alignment 0.094 % of the scene size.

## 6. Feature tracks

Verified matches form a view graph; **union-find** over its (image, keypoint) nodes links chains of matches `a–b, b–c` into tracks (ours). A component that reaches two different keypoints of the same image is ambiguous — at least one match in it is wrong — and is dropped whole.

## 7. Incremental registration

Starting from the initial pair's triangulated tracks, images are added one at a time:

1. **Next best view:** the unregistered image with the most 2D–3D links (keypoints whose track already has a landmark).
2. **Pose by PnP:** `solvePnPRansac` with the SQPnP solver, seeded, then Levenberg–Marquardt refinement on the inliers (`solvePnPRefineLM`) — OpenCV, **external**. The pose is accepted if ≥ 30 links, and ≥ 25 % of them, reproject within 4 px.
3. **Grow the structure:** triangulate every track without a landmark that is now seen by ≥ 2 registered images (§5).
4. **Bundle adjustment** (§8) and **structure refinement** (§9).

An image that fails is retried after the next success (new landmarks may give it enough links). Registration uses PnP against the 3-D structure rather than chaining pairwise poses, because pairwise errors accumulate: on templeRing the registered cameras (median 0.15°) are 3× more accurate than the pairwise estimates (0.51°).

## 8. Bundle adjustment

Camera poses and points are refined jointly by minimising the robustified reprojection error (residual model, parametrisation and Jacobian structure ours; SciPy `least_squares` trust-region solver **external**):

$$\min_{\{R_c, t_c\}, \{X_p\}} \sum_{(c,p)} \rho\!\left(\left\lVert \pi\!\left(K_c (R_c X_p + t_c)\right) - \mathbf{x}_{cp} \right\rVert^2\right)$$

- **Parameters:** 6 per camera (Rodrigues rotation vector, translation) and 3 per point; in global adjustments also one **focal scale** `s` shared by every camera (`bundle_adjustment.refine_focal`, default on): `K_c → K_c · diag(s, s, 1)`, so the focal lengths (and skew) scale and the principal point stays. One `s` for all images assumes one physical camera at one zoom. Local adjustments (5 cameras) keep `K`: too few views to constrain it. A refined `s` is applied to every registered camera and to the intrinsics of images not yet registered.
- **Robust loss** `ρ` = soft-L1 with a 1 px scale: robust to wrong matches, and it converges in ~30 evaluations where SciPy's Huber crawls.
- **Exact sparse Jacobian:** each residual depends on 6 camera and 3 point parameters (and `s`). The derivatives are analytic — the chain rule through the projection, with `∂R/∂rvec` from `cv2.Rodrigues`, and `∂(u, v)/∂s = ((u, v) − c)/s` — checked against finite differences in the tests. Cost scales with observations, not with cameras × points; replacing finite differences cut the templeRing run from 16.3 to 12.0 s with identical results.
- **Gauge:** a reconstruction is defined up to a similarity (7 DoF). The first camera is held fixed (6 DoF). The scale is left free during optimisation — pinning one translation coordinate instead creates a curved valley the solver zig-zags along — and restored afterwards exactly, by scaling about the fixed camera so the next camera keeps its distance. That scaling changes no projection.
- **Schedule:** a local adjustment after every registration (the newest 5 cameras and the points they see free), a global one every 5 registrations, and at the end. The cost tolerance is 1e-6 relative (as in Ceres); 1e-8 took ~15× longer for < 0.01°.

E5: bundle adjustment cuts pose errors 4× and raises completeness from 0.38 to 0.92, at ~90 % of the runtime; the structure refinement of §9 (switchable separately, `bundle_adjustment.refine_structure`) removes every observation beyond 2 px and trims pose errors a further ~12 %.

## 9. Structure refinement after each registration (ours)

1. **Complete tracks:** attach keypoints of a landmark's track (in registered images) that now reproject within 2 px — mostly links PnP rejected at registration.
2. **Filter observations, not points:** drop single observations that reproject farther than 2 px or lie behind their camera, then landmarks left with < 2 views or a triangulation angle < 1.5°. At triangulation one bad observation rejects a whole point, which grows stricter with track length; after adjustment a long track keeps its landmark.
3. **Re-triangulate** tracks left without a landmark.

It runs after every registration (after its adjustment, when bundle adjustment is on), and the final global adjustment alternates with it until nothing changes (at most 3 rounds). It can be switched off independently (`bundle_adjustment.refine_structure`). Tested, with or without bundle adjustment: with 5 % of keypoints shifted 3.5 px (inside PnP's 4 px, outside triangulation's 2 px) every final observation is within 2 px; without refinement the shifted links stay in.

## 10. Point-cloud filtering and export

Before export (`filtered.ply`), in order, each step on the survivors of the last, with counts logged:

1. non-finite points; 2. mean reprojection error > 1 px; 3. track length below the minimum (off by default — 35 % of templeRing points are 2-view); 4. **statistical outlier removal** — mean distance to the 20 nearest neighbours more than 2 σ above average; 5. **radius outlier removal** — fewer than 4 neighbours within `5 ×` the median distance to the 4th neighbour.

Steps 4–5 use Open3D's tensor API (**external**). Because a monocular reconstruction has no scale, the radius is relative to the point spacing, and both run on a centred, unit-RMS copy, so the result is the same at any scale. Colours are the mean of bilinear samples at every observation (ours, via `cv2.remap`). Clouds are written as PLY with Open3D (**external**), cameras as JSON (ours).

## 11. Evaluation

- **Alignment.** Any comparison with ground truth first applies the least-squares similarity mapping our camera centres onto the true ones (Umeyama 1991, ours). Rotation errors are measured after it; centre errors as a fraction of the camera ring's RMS radius.
- **Reference cloud.** Middlebury's true surface is not public, so the reference is every verified SIFT track over all images triangulated with the **ground-truth cameras** — independent of our poses, but sharing our features and matches (an internal reference, not an independent scan).
- **Completeness** — share of reference points with a reconstructed point within 1 % of the reference's extent; **accuracy** — median distance from our points to the reference; **Chamfer** — mean of both directions (nearest neighbours via SciPy `cKDTree`, **external**).
- **Metric scale** from one measured distance `s = d_true / ‖a − b‖`; on templeRing it agrees within 0.7 % with the scale fitted over all cameras (a consistency check, not an independent measurement).
- **Experiments** are YAML specs (a grid or named settings of config overrides); each run records its config, environment, metrics and cloud, and reports regenerate from the CSV.

## 12. Ours vs external

| Component | Ours | External (library) |
|---|---|---|
| Dataset discovery, validation (blur, duplicates), intrinsics resolution, `K` rescaling | ✔ | decoding, resizing, undistortion: OpenCV |
| Checkerboard calibration | wrapper | `findChessboardCorners`, `cornerSubPix`, `calibrateCamera`: OpenCV |
| Feature detection / description | strict cap, RootSIFT, cache | SIFT, ORB: OpenCV |
| Matching | ratio test, mutual check, pair selection, cache | `batchDistance` (2-NN), optional FLANN: OpenCV |
| Robust E / F / H estimation | model selection, Sampson inliers, H/F ratio, E refinement residuals | USAC / RANSAC estimators, `findHomography`: OpenCV; `least_squares`: SciPy |
| Normalised 8-point algorithm | ✔ (reference implementation) | — |
| E decomposition, cheirality, parallax, initial pair | ✔ | `recoverPose` only as test oracle |
| N-view DLT, point refinement, triangulation filters | ✔ | — |
| Tracks (union-find), camera registry, landmarks | ✔ | — |
| PnP registration | next-best-view, acceptance, retry | `solvePnPRansac` (SQPnP), `solvePnPRefineLM`: OpenCV |
| Bundle adjustment | residuals, rotation parametrisation, sparsity, gauge and scale, schedule | `least_squares` (TRF): SciPy |
| Structure refinement after BA | ✔ | — |
| Point-cloud filtering | per-point filters, scale-relative radius, logging | statistical / radius outlier removal: Open3D |
| Export, viewer, renders | `cameras.json`, scene assembly | PLY IO, visualiser, offscreen renderer: Open3D |
| Evaluation, experiments, reports | ✔ | k-d tree: SciPy; plots: Matplotlib; tables: pandas |
| Web interface | endpoints, upload checks, jobs, `report.json`, page, PLY parsing, scene orientation | HTTP server: FastAPI, uvicorn; 3-D rendering and orbit controls: three.js; fonts: Google Fonts |
| Dense reconstruction, COLMAP baseline | — | not used (future work) |

No component of the pipeline is provided by COLMAP.

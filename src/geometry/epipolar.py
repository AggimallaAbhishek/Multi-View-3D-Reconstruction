"""Two-view epipolar geometry: fundamental and essential matrices, robust verification.

Conventions (docs/architecture.md): for a correspondence x_i <-> x_j in pixels,
    x_j^T F x_i = 0,    F = K_j^-T E K_i^-1,    E = [t]x R
where (R, t) maps camera-i coordinates to camera-j coordinates.

``eight_point`` is our own implementation of Hartley's normalised eight-point
algorithm, kept for teaching and as a cross-check of OpenCV. Robust estimation in
``verify_pair`` uses OpenCV's USAC/RANSAC estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from src.config import VerificationConfig

Model = Literal["essential", "fundamental"]

_ESTIMATORS = {
    "usac_accurate": cv2.USAC_ACCURATE,
    "usac_magsac": cv2.USAC_MAGSAC,
    "ransac": cv2.RANSAC,  # equals cv2.FM_RANSAC
}
_CALIBRATED_SOURCES = {"file", "exif", "checkerboard"}

# The epipolar residual (Sampson) is a 1-D distance whose spread is the pixel noise sigma;
# the homography residual is a 2-D transfer error in image j that also carries image i's
# noise (sigma * sqrt(2) per axis). Scaling by sqrt(2) and by the ratio of 95 % chi-square
# bounds for 2 vs 1 degrees of freedom makes both thresholds equally permissive, so the
# homography ratio compares like with like (as in ORB-SLAM's model selection).
HOMOGRAPHY_THRESHOLD_FACTOR = float(np.sqrt(2.0) * np.sqrt(5.991 / 3.841))


# --- linear algebra helpers --------------------------------------------------------------


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix: skew(a) @ b == np.cross(a, b)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
    """Angle in degrees between vectors a and b (row-wise for (N, 3) arrays)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cos = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def homogeneous(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((len(points), 1))])


def normalize_points(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels -> normalised image coordinates K^-1 x (N, 2)."""
    return (homogeneous(points) @ np.linalg.inv(K).T)[:, :2]


def enforce_essential(E: np.ndarray) -> np.ndarray:
    """Closest essential matrix: singular values (1, 1, 0)."""
    U, _, Vt = np.linalg.svd(E)
    return U @ np.diag([1.0, 1.0, 0.0]) @ Vt


def fundamental_from_essential(E: np.ndarray, K_i: np.ndarray, K_j: np.ndarray) -> np.ndarray:
    F = np.linalg.inv(K_j).T @ E @ np.linalg.inv(K_i)
    return F / np.linalg.norm(F)


def essential_from_fundamental(F: np.ndarray, K_i: np.ndarray, K_j: np.ndarray) -> np.ndarray:
    E = K_j.T @ F @ K_i
    return E / np.linalg.norm(E)


def sampson_distance(F: np.ndarray, x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray:
    """First-order geometric distance (pixels) of each correspondence to F.

    sqrt( (x_j^T F x_i)^2 / ((F x_i)_1^2 + (F x_i)_2^2 + (F^T x_j)_1^2 + (F^T x_j)_2^2) )
    """
    hi, hj = homogeneous(x_i), homogeneous(x_j)
    Fx = hi @ F.T  # epipolar lines in image j
    Ftx = hj @ F  # epipolar lines in image i
    algebraic = np.sum(hj * Fx, axis=1)
    denom = Fx[:, 0] ** 2 + Fx[:, 1] ** 2 + Ftx[:, 0] ** 2 + Ftx[:, 1] ** 2
    return np.abs(algebraic) / np.sqrt(np.maximum(denom, 1e-300))


# --- normalised eight-point algorithm ----------------------------------------------------


def _normalising_transform(points: np.ndarray) -> np.ndarray:
    """Similarity moving the centroid to 0 and the mean distance to sqrt(2) (Hartley 1997)."""
    centroid = points.mean(axis=0)
    mean_dist = np.mean(np.linalg.norm(points - centroid, axis=1))
    s = np.sqrt(2.0) / max(mean_dist, 1e-12)
    return np.array([[s, 0.0, -s * centroid[0]], [0.0, s, -s * centroid[1]], [0.0, 0.0, 1.0]])


def eight_point(x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray:
    """Fundamental matrix from >= 8 correspondences (least squares, no outlier handling).

    1. Normalise each point set (centroid 0, mean distance sqrt(2)) for conditioning.
    2. Each correspondence gives one row of A f = 0 from x_j^T F x_i = 0; solve by SVD.
    3. Enforce rank 2 by zeroing the smallest singular value.
    4. Undo the normalisation: F = T_j^T F_hat T_i.
    """
    if len(x_i) < 8 or len(x_i) != len(x_j):
        raise ValueError(f"eight_point needs at least 8 correspondences, got {len(x_i)}")
    T_i, T_j = _normalising_transform(x_i), _normalising_transform(x_j)
    p = homogeneous(x_i) @ T_i.T
    q = homogeneous(x_j) @ T_j.T
    A = np.column_stack(
        [
            q[:, 0] * p[:, 0], q[:, 0] * p[:, 1], q[:, 0],
            q[:, 1] * p[:, 0], q[:, 1] * p[:, 1], q[:, 1],
            p[:, 0], p[:, 1], np.ones(len(p)),
        ]
    )  # fmt: skip
    F_hat = np.linalg.svd(A)[2][-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F_hat)
    F_hat = U @ np.diag([S[0], S[1], 0.0]) @ Vt
    F = T_j.T @ F_hat @ T_i
    return F / np.linalg.norm(F)


# --- robust two-view verification --------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TwoViewGeometry:
    """Result of verifying one image pair; failures are results, not exceptions."""

    model: Model
    F: np.ndarray | None  # (3, 3), unit Frobenius norm; always set when a model was found
    E: np.ndarray | None  # (3, 3), only for the essential model
    inlier_mask: np.ndarray  # (M,) bool over the input correspondences
    residuals: np.ndarray  # (M,) Sampson distance to F in pixels (nan without a model)
    homography_inliers: int
    planar: bool  # a homography explains the inliers: planar scene or rotation only
    success: bool
    reason: str | None  # "too_few_matches" | "estimation_failed" | "too_few_inliers"

    @property
    def n_matches(self) -> int:
        return len(self.inlier_mask)

    @property
    def n_inliers(self) -> int:
        return int(self.inlier_mask.sum())

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / self.n_matches if self.n_matches else 0.0

    @property
    def median_residual(self) -> float:
        """Median Sampson distance of the inliers (pixels)."""
        return float(np.median(self.residuals[self.inlier_mask])) if self.n_inliers else np.nan

    @property
    def homography_ratio(self) -> float:
        """Homography inliers / epipolar inliers.

        Near 1: one homography explains the matches as well as F does, so the scene is
        (nearly) planar or the camera (nearly) only rotated; such pairs carry little
        depth information and make poor initial pairs for reconstruction.
        """
        return self.homography_inliers / self.n_inliers if self.n_inliers else 0.0


def select_model(method: str, source_i: str, source_j: str) -> Model:
    """Essential matrix when both images have trustworthy intrinsics, else fundamental.

    With guessed (fallback) intrinsics, an essential-matrix fit rejects good matches
    because the wrong K distorts the epipolar constraint; F does not use K at all.
    """
    if method in ("essential", "fundamental"):
        return method  # type: ignore[return-value]
    calibrated = source_i in _CALIBRATED_SOURCES and source_j in _CALIBRATED_SOURCES
    return "essential" if calibrated else "fundamental"


def verify_pair(
    x_i: np.ndarray,
    x_j: np.ndarray,
    K_i: np.ndarray,
    K_j: np.ndarray,
    config: VerificationConfig,
    model: Model,
) -> TwoViewGeometry:
    """Robustly fit E or F to pixel correspondences and classify each as inlier or outlier.

    Inliers are correspondences whose Sampson distance to the final F is at most
    ``config.ransac_threshold_px``; the same pixel criterion is used for both models.
    """
    n = len(x_i)
    nan = np.full(n, np.nan)
    none = np.zeros(n, dtype=bool)

    def failed(reason: str, F=None, E=None, residuals=nan) -> TwoViewGeometry:
        return TwoViewGeometry(model, F, E, none, residuals, 0, False, False, reason)

    if n < max(8, config.min_inliers):  # could never reach min_inliers: skip RANSAC
        return failed("too_few_matches")

    x_i = np.ascontiguousarray(x_i, dtype=np.float64)
    x_j = np.ascontiguousarray(x_j, dtype=np.float64)
    threshold = config.ransac_threshold_px
    estimator = _ESTIMATORS[config.estimator]
    cv2.setRNGSeed(0)  # RANSAC sampling uses OpenCV's RNG: make runs reproducible

    E = None
    if model == "essential":
        # Normalise with each image's own K, so pairs with different intrinsics work.
        focal = np.mean([K_i[0, 0], K_i[1, 1], K_j[0, 0], K_j[1, 1]])
        E_est, _ = cv2.findEssentialMat(
            normalize_points(x_i, K_i),
            normalize_points(x_j, K_j),
            np.eye(3),
            method=estimator,
            prob=config.confidence,
            threshold=threshold / focal,
        )
        if E_est is None or E_est.shape[0] < 3:
            return failed("estimation_failed")
        E = enforce_essential(E_est[:3])
        F = fundamental_from_essential(E, K_i, K_j)
    else:
        F_est, _ = cv2.findFundamentalMat(x_i, x_j, estimator, threshold, config.confidence)
        if F_est is None or F_est.shape[0] < 3:
            return failed("estimation_failed")
        F = F_est[:3] / np.linalg.norm(F_est[:3])

    residuals = sampson_distance(F, x_i, x_j)
    inliers = residuals <= threshold
    success = int(inliers.sum()) >= config.min_inliers

    h_inliers = 0  # the homography ratio only matters for verified pairs
    if success:
        _, h_mask = cv2.findHomography(
            x_i, x_j, cv2.RANSAC, threshold * HOMOGRAPHY_THRESHOLD_FACTOR
        )
        h_inliers = int(h_mask.sum()) if h_mask is not None else 0
    return TwoViewGeometry(
        model=model,
        F=F,
        E=E,
        inlier_mask=inliers,
        residuals=residuals,
        homography_inliers=h_inliers,
        planar=success and h_inliers > config.planar_ratio * int(inliers.sum()),
        success=success,
        reason=None if success else "too_few_inliers",
    )

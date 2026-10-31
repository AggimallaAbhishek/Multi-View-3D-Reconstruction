"""Bundle adjustment: camera poses and 3-D points refined jointly by minimising the
reprojection error, with a robust loss and an exact sparse Jacobian (scipy least_squares).

Each camera has 6 parameters (Rodrigues rotation vector, translation) and each point 3;
intrinsics stay fixed. The gauge (global rotation and translation) is removed by holding
the ``fixed`` cameras. With a single fixed camera the scale is left free during the
optimisation, where it is a harmless flat direction (pinning a translation coordinate
instead creates a curved valley the sparse solver zig-zags along), and is restored
exactly afterwards: the result is scaled about the fixed camera's centre so that the
first free camera keeps its starting distance to it. That scaling changes no projection.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csc_matrix

from src.config import BundleAdjustmentConfig
from src.geometry.camera import Camera
from src.geometry.triangulation import Observations


@dataclass(frozen=True, eq=False)
class BundleAdjustmentResult:
    cameras: list[Camera]
    points: np.ndarray  # (N, 3)
    initial_rms_px: float  # root-mean-square reprojection error before...
    final_rms_px: float  # ...and after
    iterations: int  # function evaluations
    success: bool


def rotate(points: np.ndarray, rvecs: np.ndarray) -> np.ndarray:
    """Rotate each point (K, 3) by its own Rodrigues vector (K, 3)."""
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(theta > 0, rvecs / theta, 0.0)
    cos, sin = np.cos(theta), np.sin(theta)
    dot = np.sum(k * points, axis=1, keepdims=True)
    return cos * points + sin * np.cross(k, points) + (1 - cos) * dot * k


def bundle_adjust(
    cameras: Sequence[Camera],
    obs: Observations,
    points: np.ndarray,
    config: BundleAdjustmentConfig,
    fixed: Collection[int] = (0,),
) -> BundleAdjustmentResult:
    n_cams, n_points = len(cameras), len(points)
    K = np.stack([c.K for c in cameras])
    full = np.concatenate(
        [np.concatenate([cv2.Rodrigues(c.R)[0].ravel(), c.t]) for c in cameras]
        + [np.asarray(points, float).ravel()]
    )
    free = np.ones(len(full), bool)
    for v in fixed:
        free[6 * v : 6 * v + 6] = False

    def with_free(x: np.ndarray) -> np.ndarray:
        p = full.copy()
        p[free] = x
        return p

    def rms(x: np.ndarray) -> float:
        """RMS of the per-observation pixel distance (as in reprojection_errors)."""
        if not len(obs.xy):
            return 0.0
        return float(np.sqrt(2 * np.mean(residuals(with_free(x), K, obs) ** 2)))

    x0 = full[free]
    initial = rms(x0)
    solution = least_squares(
        lambda x: residuals(with_free(x), K, obs), x0,
        jac=lambda x: jacobian(with_free(x), K, obs)[:, free],
        loss=config.loss, f_scale=config.loss_scale_px,
        max_nfev=config.max_iterations, ftol=config.function_tolerance,
        x_scale="jac", method="trf",
    )  # fmt: skip

    p = full.copy()
    p[free] = solution.x
    cam = p[: 6 * n_cams].reshape(n_cams, 6)
    refined = [
        c if v in set(fixed) else Camera(c.K, cv2.Rodrigues(cam[v, :3])[0], cam[v, 3:].copy())
        for v, c in enumerate(cameras)
    ]  # fixed cameras come back untouched (no Rodrigues round trip)
    X = p[6 * n_cams :].reshape(n_points, 3)
    free_cams = [v for v in range(n_cams) if v not in set(fixed)]
    if len(set(fixed)) == 1 and free_cams:
        refined, X = _restore_scale(refined, X, cameras, next(iter(fixed)), free_cams[0])
    return BundleAdjustmentResult(
        refined, X, initial, rms(solution.x), int(solution.nfev), bool(solution.success)
    )


def residuals(params: np.ndarray, K: np.ndarray, obs: Observations) -> np.ndarray:
    """Reprojection residuals, x then y per observation (pixels), for ``params`` =
    ``[rvec, t]`` per camera then ``X`` per point; ``K`` (C, 3, 3) per camera."""
    n_cams = len(K)
    cam = params[: 6 * n_cams].reshape(n_cams, 6)
    X = params[6 * n_cams :].reshape(-1, 3)
    X_cam = rotate(X[obs.point], cam[obs.view, :3]) + cam[obs.view, 3:]
    h = np.einsum("kij,kj->ki", K[obs.view], X_cam)
    return (h[:, :2] / h[:, 2:3] - obs.xy).ravel()


def jacobian(params: np.ndarray, K: np.ndarray, obs: Observations) -> csc_matrix:
    """Exact ``d residuals / d params`` (sparse): each residual depends on its camera's 6
    parameters and its point's 3. Rotation derivatives come from ``cv2.Rodrigues``."""
    n_cams = len(K)
    cam = params[: 6 * n_cams].reshape(n_cams, 6)
    X = params[6 * n_cams :].reshape(-1, 3)
    view, Xw = obs.view, X[obs.point]
    rodrigues = [cv2.Rodrigues(rvec) for rvec in cam[:, :3]]
    R = np.stack([r for r, _ in rodrigues])
    dR = np.stack([d.reshape(3, 3, 3) for _, d in rodrigues])  # [cam, k, i, j] = dR_ij/drvec_k
    h = np.einsum("kij,kj->ki", K[view], np.einsum("kij,kj->ki", R[view], Xw) + cam[view, 3:])
    z = h[:, 2]
    d_pixel = np.zeros((len(view), 2, 3))  # d (u, v) / d h, for (u, v) = (h0, h1) / h2
    d_pixel[:, 0, 0] = d_pixel[:, 1, 1] = 1 / z
    d_pixel[:, :, 2] = -h[:, :2] / z[:, None] ** 2
    d_cam = d_pixel @ K[view]  # d (u, v) / d X_cam
    d_rotation = d_cam @ np.einsum("nkij,nj->nik", dR[view], Xw)  # X_cam = R X + t
    blocks = np.concatenate([d_rotation, d_cam, d_cam @ R[view]], axis=2)  # (n, 2, 9)
    rows, cols = _layout(view, obs.point, n_cams)
    shape = (2 * len(view), 6 * n_cams + 3 * len(X))
    return coo_matrix((blocks.ravel(), (rows, cols)), shape=shape).tocsc()


def _layout(view: np.ndarray, point: np.ndarray, n_cams: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of each of the 9 non-zeros of each residual row, in ``jacobian``'s
    order: camera rotation (3), translation (3), point (3); x row, then y row."""
    rows = np.repeat(np.arange(2 * len(view)), 9)
    cols = np.concatenate(
        [6 * view[:, None] + np.arange(6), 6 * n_cams + 3 * point[:, None] + np.arange(3)], axis=1
    )
    return rows, np.repeat(cols, 2, axis=0).ravel()


def _restore_scale(refined, X, initial, anchor: int, reference: int):
    """Scale cameras and points about camera ``anchor``'s centre so that camera
    ``reference`` is as far from it as in ``initial``."""
    c0 = refined[anchor].centre
    s = np.linalg.norm(initial[reference].centre - initial[anchor].centre) / np.linalg.norm(
        refined[reference].centre - c0
    )
    scaled = [c if v == anchor else Camera(c.K, c.R, -c.R @ (c0 + s * (c.centre - c0)))
              for v, c in enumerate(refined)]  # fmt: skip
    return scaled, c0 + s * (X - c0)

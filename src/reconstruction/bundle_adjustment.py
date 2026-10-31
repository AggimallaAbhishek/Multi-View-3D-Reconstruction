"""Bundle adjustment: camera poses and 3-D points refined jointly by minimising the
reprojection error, with a robust loss and an exact sparse Jacobian (scipy least_squares).

Each camera has 6 parameters (Rodrigues rotation vector, translation) and each point 3.
Intrinsics stay fixed unless ``refine_focal``: then one focal scale ``s`` shared by every
camera (one physical camera) is refined too, ``K -> K @ diag(s, s, 1)`` (focal lengths
and skew scale, the principal point stays). The gauge (global rotation and translation)
is removed by holding the ``fixed`` cameras. With a single fixed camera the scale is
left free during the optimisation, where it is a harmless flat direction (pinning a
translation coordinate instead creates a curved valley the sparse solver zig-zags along),
and is restored exactly afterwards: the result is scaled about the fixed camera's centre
so that the first free camera keeps its starting distance to it. That scaling changes no
projection.
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
    focal_scale: float = 1.0  # refined focal length / starting one (1 unless refine_focal)


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
    refine_focal: bool = False,
) -> BundleAdjustmentResult:
    """Refine ``cameras`` and ``points`` (and, with ``refine_focal``, the shared focal
    length; a refined scale beyond ``config.max_focal_change`` is discarded and the
    adjustment repeated with K fixed)."""
    n_cams, n_points = len(cameras), len(points)
    K = np.stack([c.K for c in cameras])
    full = pack(cameras, points, 1.0 if refine_focal else None)
    free = np.ones(len(full), bool)
    for v in fixed:
        free[6 * v : 6 * v + 6] = False

    def with_free(x: np.ndarray) -> np.ndarray:
        p = full.copy()
        p[free] = x
        return p

    def f(x: np.ndarray) -> np.ndarray:
        return residuals(with_free(x), K, obs, refine_focal)

    def rms(r: np.ndarray) -> float:
        """RMS of the per-observation pixel distance (as in reprojection_errors)."""
        return float(np.sqrt(2 * np.mean(r**2))) if len(r) else 0.0

    x0 = full[free]
    initial = rms(f(x0))
    solution = least_squares(
        f, x0, jac=lambda x: jacobian(with_free(x), K, obs, refine_focal)[:, free],
        loss=config.loss, f_scale=config.loss_scale_px,
        max_nfev=config.max_iterations, ftol=config.function_tolerance,
        x_scale="jac", method="trf",
    )  # fmt: skip

    p = with_free(solution.x)
    s = float(p[-1]) if refine_focal else 1.0
    if not 1 / config.max_focal_change <= s <= config.max_focal_change:
        return bundle_adjust(cameras, obs, points, config, fixed)
    cam = p[: 6 * n_cams].reshape(n_cams, 6)
    refined = []
    for v, c in enumerate(cameras):
        # Fixed poses come back untouched (no Rodrigues round trip).
        R, t = (c.R, c.t) if v in fixed else (cv2.Rodrigues(cam[v, :3])[0], cam[v, 3:].copy())
        refined.append(Camera(scale_focal(c.K, s), R, t))
    X = p[6 * n_cams : 6 * n_cams + 3 * n_points].reshape(n_points, 3)
    free_cams = [v for v in range(n_cams) if v not in fixed]
    if len(set(fixed)) == 1 and free_cams:
        refined, X = _restore_scale(refined, X, cameras, next(iter(fixed)), free_cams[0])
    return BundleAdjustmentResult(
        refined, X, initial, rms(solution.fun), int(solution.nfev), bool(solution.success), s
    )


def scale_focal(K: np.ndarray, s: float) -> np.ndarray:
    """``K @ diag(s, s, 1)`` (one K or a stack): focal lengths and skew scaled, principal
    point kept."""
    return K if s == 1.0 else K @ np.diag([s, s, 1.0])


def residuals(
    params: np.ndarray, K: np.ndarray, obs: Observations, refine_focal: bool = False
) -> np.ndarray:
    """Reprojection residuals, x then y per observation (pixels), for ``params`` =
    ``[rvec, t]`` per camera, then ``X`` per point, then (``refine_focal``) the focal
    scale; ``K`` (C, 3, 3) per camera."""
    cam, X, s = _unpack(params, len(K), refine_focal)
    X_cam = rotate(X[obs.point], cam[obs.view, :3]) + cam[obs.view, 3:]
    normalised = X_cam[:, :2] / X_cam[:, 2:3]
    Kv = K[obs.view]
    return (np.einsum("kij,kj->ki", Kv[:, :2, :2], s * normalised) + Kv[:, :2, 2] - obs.xy).ravel()


def jacobian(
    params: np.ndarray, K: np.ndarray, obs: Observations, refine_focal: bool = False
) -> csc_matrix:
    """Exact ``d residuals / d params`` (sparse): each residual depends on its camera's 6
    parameters, its point's 3 and (``refine_focal``) the shared focal scale."""
    n_points = len(_unpack(params, len(K), refine_focal)[1])
    rows, cols, shape = _structure(obs.view, obs.point, len(K), n_points, refine_focal)
    values = _derivatives(params, K, obs, refine_focal)
    return coo_matrix((values, (rows, cols)), shape=shape).tocsc()


def _derivatives(
    params: np.ndarray, K: np.ndarray, obs: Observations, refine_focal: bool
) -> np.ndarray:
    """The non-zeros of ``jacobian``, in ``_structure``'s order. Rotation derivatives come
    from ``cv2.Rodrigues``."""
    cam, X, s = _unpack(params, len(K), refine_focal)
    view, Xw = obs.view, X[obs.point]
    Kv = scale_focal(K, s)[view]
    rodrigues = [cv2.Rodrigues(rvec) for rvec in cam[:, :3]]
    R = np.stack([r for r, _ in rodrigues])[view]
    dR = np.stack([d.reshape(3, 3, 3) for _, d in rodrigues])[view]  # [n, k, i, j] = dR_ij/dr_k
    h = np.einsum("kij,kj->ki", Kv, np.einsum("kij,kj->ki", R, Xw) + cam[view, 3:])
    z = h[:, 2]
    d_pixel = np.zeros((len(view), 2, 3))  # d (u, v) / d h, for (u, v) = (h0, h1) / h2
    d_pixel[:, 0, 0] = d_pixel[:, 1, 1] = 1 / z
    d_pixel[:, :, 2] = -h[:, :2] / z[:, None] ** 2
    d_cam = d_pixel @ Kv  # d (u, v) / d X_cam
    d_rotation = d_cam @ np.einsum("nkij,nj->nik", dR, Xw)  # X_cam = R X + t
    values = np.concatenate([d_rotation, d_cam, d_cam @ R], axis=2).ravel()  # (n, 2, 9)
    if not refine_focal:
        return values
    # (u, v) - c is linear in s: its derivative is that offset over s.
    d_focal = ((h[:, :2] / z[:, None] - Kv[:, :2, 2]) / s).ravel()
    return np.concatenate([values, d_focal])


def _structure(view, point, n_cams: int, n_points: int, refine_focal: bool):
    """Rows, columns and shape of ``jacobian``'s non-zeros: per residual row camera
    rotation (3), translation (3), point (3) — x row, then y row — then the focal-scale
    column for every row."""
    n_rows = 2 * len(view)
    rows = np.repeat(np.arange(n_rows), 9)
    cols = np.concatenate(
        [6 * view[:, None] + np.arange(6), 6 * n_cams + 3 * point[:, None] + np.arange(3)], axis=1
    )
    cols = np.repeat(cols, 2, axis=0).ravel()
    shape = (n_rows, 6 * n_cams + 3 * n_points + refine_focal)
    if refine_focal:
        rows = np.concatenate([rows, np.arange(n_rows)])
        cols = np.concatenate([cols, np.full(n_rows, shape[1] - 1)])
    return rows, cols, shape


def pack(cameras: Sequence[Camera], points: np.ndarray, focal_scale: float | None = None):
    """Parameter vector of ``residuals``/``jacobian``: ``[rvec, t]`` per camera, then the
    points, then the focal scale if given (the inverse of ``_unpack``)."""
    return np.concatenate(
        [np.concatenate([cv2.Rodrigues(c.R)[0].ravel(), c.t]) for c in cameras]
        + [np.asarray(points, float).ravel(), [] if focal_scale is None else [focal_scale]]
    )


def _unpack(params: np.ndarray, n_cams: int, refine_focal: bool):
    """Camera parameters (C, 6), points (N, 3) and focal scale from ``params``."""
    end = len(params) - refine_focal
    cam = params[: 6 * n_cams].reshape(n_cams, 6)
    return cam, params[6 * n_cams : end].reshape(-1, 3), float(params[-1]) if refine_focal else 1.0


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

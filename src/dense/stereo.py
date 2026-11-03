"""Depth maps from two calibrated views: rectify with the known relative pose
(``cv2.stereoRectify``), match along rows with semi-global block matching
(``cv2.StereoSGBM``) and convert the disparities back to depths in the first view's
own pixel grid. The disparity search range comes from the sparse points."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import DenseConfig
from src.data.preprocessing import to_gray
from src.geometry.camera import Camera

_NO_DISTORTION = np.zeros(5)


@dataclass(frozen=True)
class ViewPlan:
    partner: list[int | None]  # per view: its stereo partner, or None
    checks: list[list[int]]  # per view: the views its depths are checked against


def plan_views(cameras: Sequence[Camera], points: np.ndarray, config: DenseConfig) -> ViewPlan:
    """Pair every view with the view whose angle to it (seen from the scene centre, the
    median sparse point) is nearest ``target_angle_deg`` within [min, max], and list its
    ``fusion_views`` angularly nearest views for the consistency check, leaving out views
    that were paired with it."""
    centre = np.median(points, axis=0)
    rays = np.stack([c.centre - centre for c in cameras])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    angle = np.rad2deg(np.arccos(np.clip(rays @ rays.T, -1.0, 1.0)))
    np.fill_diagonal(angle, np.inf)
    partner: list[int | None] = []
    for row in angle:
        ok = np.flatnonzero((row >= config.min_angle_deg) & (row <= config.max_angle_deg))
        partner.append(int(ok[np.argmin(np.abs(row[ok] - config.target_angle_deg))])
                       if len(ok) else None)  # fmt: skip
    # A view whose depth map came from stereo with i shares i's matching errors: it
    # cannot confirm i's depths independently.
    checks = [[int(k) for k in np.argsort(row, kind="stable") if partner[k] != i and k != i]
              [: config.fusion_views] for i, row in enumerate(angle)]  # fmt: skip
    return ViewPlan(partner, checks)


def pair_depth(
    image_i: np.ndarray,
    cam_i: Camera,
    image_j: np.ndarray,
    cam_j: Camera,
    points: np.ndarray,
    config: DenseConfig,
) -> np.ndarray:
    """Depth (camera z) per pixel of view ``i`` from stereo with view ``j``; NaN where
    there is no reliable match. ``points`` (N, 3) set the disparity range."""
    h, w = image_i.shape[:2]
    R = cam_j.R @ cam_i.R.T
    T = cam_j.t - R @ cam_i.t
    R1, R2, P1, P2, *_ = cv2.stereoRectify(
        cam_i.K, _NO_DISTORTION, cam_j.K, _NO_DISTORTION, (w, h), R, T.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=config.rectify_alpha,
    )  # fmt: skip
    axis = int(abs(P2[1, 3]) > abs(P2[0, 3]))  # 1: vertical rectification
    empty = np.full((h, w), np.nan)

    # Disparities of the sparse points seen in both rectified views.
    X1 = (points @ cam_i.R.T + cam_i.t) @ R1.T
    x1 = _project(P1, X1)
    x2 = _project(P2, X1)
    seen = (X1[:, 2] > 0) & _inside(x1, w, h) & _inside(x2, w, h)
    if not seen.any():
        return empty
    d = x1[seen, axis] - x2[seen, axis]  # either sign: SGBM searches negative ones too
    q = config.depth_quantile
    lo, hi = np.quantile(d, [q, 1 - q])
    margin = config.disparity_margin_px
    min_disp = int(np.floor(lo)) - margin
    n_disp = int(min(16 * np.ceil((hi - lo + 2 * margin) / 16), config.max_disparities))

    def rectified(image, K, Rr, Pr):
        """Grey, rectified, transposed for a vertical baseline (SGBM matches along rows)."""
        mx, my = cv2.initUndistortRectifyMap(K, _NO_DISTORTION, Rr, Pr, (w, h), cv2.CV_32FC1)
        out = cv2.remap(to_gray(image), mx, my, cv2.INTER_LINEAR)
        return np.ascontiguousarray(out.T) if axis else out

    a = rectified(image_i, cam_i.K, R1, P1)
    b = rectified(image_j, cam_j.K, R2, P2)
    block = config.block_size
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=n_disp, blockSize=block,
        P1=config.sgbm_p1_factor * block**2, P2=config.sgbm_p2_factor * block**2,
        disp12MaxDiff=config.disp12_max_diff,
        uniquenessRatio=config.uniqueness_ratio, speckleWindowSize=config.speckle_window,
        speckleRange=config.speckle_range, mode=cv2.STEREO_SGBM_MODE_HH,
    )  # fmt: skip
    raw = sgbm.compute(a, b)
    disp = np.where(raw >= 16 * min_disp, raw / 16.0, np.nan)
    disp = disp.T if axis else disp  # (h, w), rectified view i

    # Back to view i's pixels: rectified coordinates of every pixel, disparity there.
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rect = cv2.undistortPoints(np.stack([u, v], -1).reshape(-1, 1, 2), cam_i.K, _NO_DISTORTION,
                             R=R1, P=P1).reshape(h, w, 2)  # fmt: skip
    disp_at = cv2.remap(disp.astype(np.float32), rect[..., 0], rect[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)  # fmt: skip
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = -P2[axis, 3] / disp_at  # depth in the rectified frame
    f, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]
    x, y = (rect[..., 0] - cx) / f, (rect[..., 1] - cy) / f
    depth = Z * (R1[0, 2] * x + R1[1, 2] * y + R1[2, 2])  # z of R1^T (Z [x, y, 1])
    valid = (depth > 0) & (local_std(image_i, config.block_size) >= config.min_texture)
    if config.edge_erosion_px > 0:  # edges fatten (background takes the object's depth)
        band = np.ones((2 * config.edge_erosion_px + 1,) * 2, np.uint8)
        valid = cv2.erode(valid.astype(np.uint8), band) > 0
    return np.where(valid, depth, np.nan)


def local_std(image: np.ndarray, window: int) -> np.ndarray:
    """Standard deviation of the grey levels in a ``window`` x ``window`` box per pixel:
    where it is low (uniform background, featureless surface) matches are unreliable."""
    grey = to_gray(image).astype(float)
    mean = cv2.blur(grey, (window, window))
    return np.sqrt(np.maximum(cv2.blur(grey * grey, (window, window)) - mean**2, 0))


def _project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    p = X @ P[:, :3].T + P[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def _inside(x: np.ndarray, w: int, h: int) -> np.ndarray:
    return (x[:, 0] >= 0) & (x[:, 0] <= w - 1) & (x[:, 1] >= 0) & (x[:, 1] <= h - 1)

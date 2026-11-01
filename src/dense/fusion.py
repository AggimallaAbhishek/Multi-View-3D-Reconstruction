"""Fusing per-view depth maps into one cloud: a pixel's 3-D point is kept if enough
other views' depth maps agree with its depth there (geometric consistency), then the
cloud is averaged on a voxel grid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.config import DenseConfig
from src.geometry.camera import Camera


@dataclass(frozen=True, eq=False)
class DenseCloud:
    points: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N, 3) RGB uint8
    views: np.ndarray  # (N, 3) unit vectors from the point towards the camera(s) that saw it


def fuse_depth_maps(
    depths: Sequence[np.ndarray | None],
    cameras: Sequence[Camera],
    images: Sequence[np.ndarray],
    checks: Sequence[Sequence[int]],
    voxel_size: float,
    config: DenseConfig,
) -> DenseCloud:
    """Back-project every view's depths and keep the points whose depth at least
    ``min_consistent_views`` of the views in ``checks[i]`` confirm within
    ``depth_tolerance`` (relative); colours from view ``i`` (BGR images)."""
    parts = []
    for i, depth in enumerate(depths):
        if depth is None:
            continue
        cam = cameras[i]
        v, u = np.nonzero(np.isfinite(depth))
        X_cam = np.column_stack([u, v, np.ones(len(u))]) @ np.linalg.inv(cam.K).T
        X = (X_cam * depth[v, u][:, None] - cam.t) @ cam.R
        agree = np.zeros(len(X), int)
        for k in checks[i]:
            if depths[k] is not None:
                agree += _confirms(X, cameras[k], depths[k], config.depth_tolerance)
        keep = agree >= config.min_consistent_views
        view = cam.centre - X[keep]
        parts.append(DenseCloud(
            X[keep], images[i][v[keep], u[keep]][:, ::-1].copy(),
            view / np.linalg.norm(view, axis=1, keepdims=True),
        ))  # fmt: skip
    if not parts:
        return DenseCloud(np.empty((0, 3)), np.empty((0, 3), np.uint8), np.empty((0, 3)))
    cloud = DenseCloud(*(np.concatenate(a) for a in zip(
        *((p.points, p.colors, p.views) for p in parts), strict=True)))  # fmt: skip
    return voxel_downsample(cloud, voxel_size) if voxel_size > 0 else cloud


def _confirms(X: np.ndarray, cam: Camera, depth: np.ndarray, tolerance: float) -> np.ndarray:
    """Whether ``depth`` (view ``cam``'s map) agrees with the points' depths in that view."""
    h, w = depth.shape
    x, z = cam.project(X)
    with np.errstate(invalid="ignore"):
        u, v = np.round(x[:, 0]), np.round(x[:, 1])
        inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    found = np.full(len(X), np.nan)
    found[inside] = depth[v[inside].astype(int), u[inside].astype(int)]
    with np.errstate(invalid="ignore"):
        return np.abs(z - found) <= tolerance * found


def voxel_downsample(cloud: DenseCloud, size: float) -> DenseCloud:
    """One point per occupied voxel of edge ``size``: the mean position, colour and
    (renormalised) viewing direction of the points in it."""
    _, cell, counts = np.unique(np.floor(cloud.points / size).astype(np.int64), axis=0,
                                return_inverse=True, return_counts=True)  # fmt: skip
    cell = cell.ravel()

    def mean(a: np.ndarray) -> np.ndarray:
        sums = np.stack([np.bincount(cell, a[:, c]) for c in range(a.shape[1])], 1)
        return sums / counts[:, None]

    views = mean(cloud.views)
    norms = np.linalg.norm(views, axis=1, keepdims=True)
    return DenseCloud(
        mean(cloud.points),
        np.round(mean(cloud.colors.astype(float))).astype(np.uint8),
        np.divide(views, norms, out=np.zeros_like(views), where=norms > 0),
    )

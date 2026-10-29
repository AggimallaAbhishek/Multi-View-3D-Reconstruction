"""Colours for 3-D points, sampled from the images that observe them."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from src.geometry.reprojection import per_point_mean
from src.geometry.triangulation import Observations


def sample_colors(image_bgr: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """RGB colours in [0, 1] at sub-pixel positions (N, 2), bilinear, clamped at borders."""
    if len(xy) == 0:
        return np.empty((0, 3))
    maps = np.ascontiguousarray(xy, dtype=np.float32).reshape(-1, 1, 2)
    sampled = cv2.remap(
        image_bgr.astype(np.float32), maps[..., 0], maps[..., 1],
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )  # fmt: skip
    return sampled.reshape(-1, 3)[:, ::-1].astype(np.float64) / 255.0


def color_points(
    images_bgr: Sequence[np.ndarray], obs: Observations, n_points: int | None = None
) -> np.ndarray:
    """Mean RGB colour of each point over all its observations, (N, 3) in [0, 1].

    ``images_bgr[v]`` is the image of view ``v`` at the resolution the observations
    use. Points without observations are black.
    """
    n = obs.n_points if n_points is None else n_points
    samples = np.empty((len(obs), 3))
    for v, idx in obs.by_view():
        samples[idx] = sample_colors(images_bgr[v], obs.xy[idx])
    return np.nan_to_num(per_point_mean(samples, obs.point, n))

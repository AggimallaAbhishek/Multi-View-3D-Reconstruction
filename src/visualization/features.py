"""Keypoint overlays."""

from __future__ import annotations

import cv2
import numpy as np

from src.features.extractor import Features
from src.visualization.common import dimmed_gray


def draw_keypoints(
    image: np.ndarray,
    features: Features,
    max_points: int | None = 500,
    color: tuple[int, int, int] = (0, 255, 0),
    background: float = 0.5,
) -> np.ndarray:
    """Keypoints drawn as circles sized by scale, with orientation, on a dimmed gray copy.

    The image is shown in grayscale at ``background`` brightness so the keypoints stand
    out on colourful scenes. Features are stored strongest first, so ``max_points`` keeps
    the strongest ones.
    """
    canvas = dimmed_gray(image, background)
    n = len(features) if max_points is None else min(len(features), max_points)
    if n == 0:
        return canvas
    keypoints = [
        cv2.KeyPoint(float(x), float(y), float(size), float(angle), float(response))
        for (x, y), size, angle, response in zip(
            features.points[:n],
            features.sizes[:n],
            features.angles[:n],
            features.responses[:n],
            strict=True,
        )
    ]
    return cv2.drawKeypoints(
        canvas, keypoints, canvas, color=color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

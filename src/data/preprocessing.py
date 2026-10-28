"""Image preprocessing that keeps the intrinsic matrix consistent with the pixels.

Pixel convention: integer coordinates are pixel centres (OpenCV). Resizing by
``s`` therefore maps a coordinate ``c`` to ``(c + 0.5) * s - 0.5``.
"""

from __future__ import annotations

import cv2
import numpy as np


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Intrinsics for an image resized by (sx, sy)."""
    K2 = np.array(K, dtype=np.float64, copy=True)
    K2[0, 0] *= sx
    K2[0, 1] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K2[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K2


def resize_max_side(
    image: np.ndarray, K: np.ndarray, max_side: int | None
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Downscale so the longest side is at most ``max_side``; never upscales.

    Returns (image, updated K, (sx, sy)). The per-axis scales are the exact
    ratios of the rounded output size, so K matches the output pixels.
    """
    height, width = image.shape[:2]
    if max_side is None or max(width, height) <= max_side:
        return image, np.array(K, dtype=np.float64, copy=True), (1.0, 1.0)

    s = max_side / max(width, height)
    new_w, new_h = max(1, round(width * s)), max(1, round(height * s))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    sx, sy = new_w / width, new_h / height
    return resized, scale_intrinsics(K, sx, sy), (sx, sy)


def undistort(image: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Remove lens distortion; the output keeps the same K with zero distortion."""
    if not np.any(dist):
        return image
    return cv2.undistort(image, K, np.asarray(dist, dtype=np.float64))


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

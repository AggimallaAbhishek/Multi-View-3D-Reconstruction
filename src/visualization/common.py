"""Shared drawing helpers."""

from __future__ import annotations

import cv2
import numpy as np


def dimmed_gray(image: np.ndarray, brightness: float) -> np.ndarray:
    """BGR grayscale copy at ``brightness`` (0-1), so coloured overlays stand out."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.convertScaleAbs(gray, alpha=brightness), cv2.COLOR_GRAY2BGR)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Two BGR images next to each other, top-aligned, padded with black."""
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    return canvas

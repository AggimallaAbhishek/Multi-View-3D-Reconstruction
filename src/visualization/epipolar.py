"""Epipolar-line overlays: the visual check that F is right."""

from __future__ import annotations

import cv2
import numpy as np

from src.geometry.epipolar import homogeneous
from src.visualization.common import dimmed_gray, side_by_side


def draw_epipolar_lines(
    image_i: np.ndarray,
    image_j: np.ndarray,
    x_i: np.ndarray,
    x_j: np.ndarray,
    F: np.ndarray,
    n_lines: int = 20,
    seed: int = 0,
    background: float = 0.6,
) -> np.ndarray:
    """Images side by side; each sampled correspondence gets its own colour.

    In image j the line is F x_i, in image i it is F^T x_j. With a correct F every
    point lies on the line of the same colour, and all lines in an image meet at the
    epipole (possibly outside the frame).
    """
    left, right = dimmed_gray(image_i, background), dimmed_gray(image_j, background)
    rng = np.random.default_rng(seed)
    chosen = np.arange(len(x_i))
    if len(chosen) > n_lines:
        chosen = np.sort(rng.choice(chosen, size=n_lines, replace=False))

    lines_j = homogeneous(x_i[chosen]) @ F.T
    lines_i = homogeneous(x_j[chosen]) @ F
    thickness = max(1, round(max(left.shape[0], right.shape[0]) / 600))
    for k, idx in enumerate(chosen):
        color = tuple(int(c) for c in rng.integers(64, 256, size=3))
        _draw_line(left, lines_i[k], color, thickness)
        _draw_line(right, lines_j[k], color, thickness)
        cv2.circle(left, _pt(x_i[idx]), 4 * thickness, color, -1, cv2.LINE_AA)
        cv2.circle(right, _pt(x_j[idx]), 4 * thickness, color, -1, cv2.LINE_AA)

    return side_by_side(left, right)


def _draw_line(image: np.ndarray, line: np.ndarray, color, thickness: int) -> None:
    """Draw a*x + b*y + c = 0 across the image."""
    a, b, c = line
    h, w = image.shape[:2]
    if abs(b) > abs(a):  # closer to horizontal: intersect the left and right borders
        p, q = (0, -c / b), (w - 1, -(c + a * (w - 1)) / b)
    else:
        p, q = (-c / a, 0), (-(c + b * (h - 1)) / a, h - 1)
    big = 1e6  # keep coordinates in int range before clipping
    p = tuple(int(np.clip(v, -big, big)) for v in p)
    q = tuple(int(np.clip(v, -big, big)) for v in q)
    ok, p, q = cv2.clipLine((0, 0, w, h), p, q)
    if ok:
        cv2.line(image, p, q, color, thickness, cv2.LINE_AA)


def _pt(x: np.ndarray) -> tuple[int, int]:
    return int(round(x[0])), int(round(x[1]))

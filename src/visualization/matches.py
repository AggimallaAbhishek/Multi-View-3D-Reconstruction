"""Side-by-side match visualisation."""

from __future__ import annotations

import cv2
import numpy as np

from src.features.extractor import Features
from src.features.matcher import Matches
from src.visualization.common import dimmed_gray, side_by_side


def draw_matches(
    image_i: np.ndarray,
    image_j: np.ndarray,
    features_i: Features,
    features_j: Features,
    matches: Matches,
    max_lines: int | None = 200,
    seed: int = 0,
    inlier_mask: np.ndarray | None = None,
    background: float = 0.6,
) -> np.ndarray:
    """Images side by side with a line per match.

    Images are shown in grayscale at ``background`` brightness so the coloured lines
    stand out. A random (seeded) subset of ``max_lines`` matches is drawn so dense
    matches stay readable. With ``inlier_mask``, inliers are green and outliers red;
    otherwise each line gets its own colour so crossings can be followed.
    """
    left = dimmed_gray(image_i, background)
    canvas = side_by_side(left, dimmed_gray(image_j, background))
    height = canvas.shape[0]

    rng = np.random.default_rng(seed)
    chosen = np.arange(len(matches))
    if max_lines is not None and len(chosen) > max_lines:
        chosen = np.sort(rng.choice(chosen, size=max_lines, replace=False))

    pi, pj = matches.points(features_i, features_j)
    offset = np.array([left.shape[1], 0.0])
    thickness = max(1, round(height / 600))
    for k in chosen:
        if inlier_mask is None:
            color = tuple(int(c) for c in rng.integers(64, 256, size=3))
        else:
            color = (0, 200, 0) if inlier_mask[k] else (0, 0, 230)
        a = tuple(np.round(pi[k]).astype(int))
        b = tuple(np.round(pj[k] + offset).astype(int))
        cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, a, 2 * thickness + 1, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, b, 2 * thickness + 1, color, thickness, cv2.LINE_AA)
    return canvas


def plot_match_matrix(
    matrix: np.ndarray,
    names: list[str],
    path=None,
    title: str = "Matches per image pair",
    label: str = "matches after ratio + mutual check",
):
    """Heatmap of match counts between every pair of images (the match graph).

    Returns the matplotlib ``Figure``; it is also saved when ``path`` is given.
    """
    import matplotlib
    from matplotlib.figure import Figure  # Figure API: no GUI backend needed

    n = len(names)
    size = min(14.0, 2.5 + 0.25 * n)
    fig = Figure(figsize=(size, size * 0.85), layout="constrained")
    ax = fig.add_subplot()
    shown = np.ma.masked_array(matrix, mask=np.eye(n, dtype=bool))  # self-pairs are not data
    cmap = matplotlib.colormaps["viridis"].with_extremes(bad="lightgray")
    image = ax.imshow(shown, cmap=cmap, interpolation="nearest")
    labels = names if n <= 40 else [str(k) for k in range(n)]
    fontsize = 8 if n <= 20 else 6
    ax.set_xticks(range(n), labels, rotation=90, fontsize=fontsize)
    ax.set_yticks(range(n), labels, fontsize=fontsize)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=label, shrink=0.8)
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig

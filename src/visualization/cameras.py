"""Static camera/point plots (top and side views)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.geometry.pose_estimation import camera_centre


def plot_two_view(
    R: np.ndarray,
    t: np.ndarray,
    points: np.ndarray,
    names: Sequence[str] = ("camera i", "camera j"),
    path=None,
    title: str = "Two-view reconstruction",
):
    """Camera i at the origin, camera j at C = -R^T t, and triangulated points.

    Left: top view (x right, z forward). Right: side view (z forward, height = -y,
    since image y points down). Axes are in baseline units. Returns the Figure.
    """
    from matplotlib.figure import Figure

    centres = np.array([np.zeros(3), camera_centre(R, t)])
    axes_z = np.array([[0.0, 0.0, 1.0], R.T @ np.array([0.0, 0.0, 1.0])])  # optical axes

    # Robust limits: ignore the most extreme 1 % of points (far, badly conditioned ones).
    if len(points):
        lo = np.percentile(points, 1, axis=0)
        hi = np.percentile(points, 99, axis=0)
        inside = np.all((points >= lo) & (points <= hi), axis=1)
        shown = points[inside]
    else:
        shown = points
    everything = np.vstack([shown, centres]) if len(shown) else centres
    span = np.ptp(everything, axis=0).max() or 1.0
    arrow = 0.15 * span

    fig = Figure(figsize=(11, 5), layout="constrained")
    fig.suptitle(title)
    for k, (ax_idx, ay_idx, sign, xlabel, ylabel) in enumerate(
        [(0, 2, 1.0, "x", "z (forward)"), (2, 1, -1.0, "z (forward)", "height (-y)")]
    ):
        ax = fig.add_subplot(1, 2, k + 1)
        if len(shown):
            ax.scatter(shown[:, ax_idx], sign * shown[:, ay_idx], s=2, c="0.55", lw=0)
        for c, (centre, axis, name) in enumerate(zip(centres, axes_z, names, strict=True)):
            colour = ("tab:blue", "tab:red")[c]
            ax.plot(centre[ax_idx], sign * centre[ay_idx], "^", ms=10, color=colour, label=name)
            ax.annotate(
                "",
                xy=(
                    centre[ax_idx] + arrow * axis[ax_idx],
                    sign * (centre[ay_idx] + arrow * axis[ay_idx]),
                ),
                xytext=(centre[ax_idx], sign * centre[ay_idx]),
                arrowprops={"arrowstyle": "->", "color": colour, "lw": 1.5},
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(loc="best", fontsize=8)
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig

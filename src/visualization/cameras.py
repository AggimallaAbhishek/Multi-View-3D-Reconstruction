"""Static camera/point plots (top and side views)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.geometry.camera import camera_centre


def plot_two_view(
    R: np.ndarray,
    t: np.ndarray,
    points: np.ndarray,
    names: Sequence[str] = ("camera i", "camera j"),
    path=None,
    title: str = "Two-view reconstruction",
):
    """Camera i at the origin, camera j at C = -R^T t, and triangulated points (axes in
    baseline units); see ``plot_cameras``. Returns the Figure."""
    return plot_cameras([np.eye(3), R], [np.zeros(3), t], points, names, path, title)


def plot_cameras(
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    points: np.ndarray,
    names: Sequence[str] | None = None,
    path=None,
    title: str = "Reconstruction",
    max_legend: int = 8,
):
    """Cameras (world -> camera R, t) as markers with their optical axes, and points.

    Left: top view (x right, z forward). Right: side view (z forward, height = -y,
    since image y points down). Two cameras are blue and red; more are coloured by
    their order. Names go into a legend if there are at most ``max_legend`` cameras.
    Returns the Figure.
    """
    from matplotlib import colormaps
    from matplotlib.figure import Figure

    n = len(rotations)
    centres = np.array([camera_centre(R, t) for R, t in zip(rotations, translations, strict=True)])
    axes_z = np.array([R.T @ np.array([0.0, 0.0, 1.0]) for R in rotations])  # optical axes
    colours = ["tab:blue", "tab:red"] if n <= 2 else colormaps["viridis"](np.linspace(0, 1, n))
    labels = list(names) if names is not None and n <= max_legend else [None] * n

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
        for centre, axis, colour, name in zip(centres, axes_z, colours, labels, strict=True):
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
        if k == 0 and labels[0] is not None:
            ax.legend(loc="best", fontsize=8)
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig

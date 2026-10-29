"""Reprojection-error histogram."""

from __future__ import annotations

import numpy as np


def plot_reprojection_histogram(
    errors: np.ndarray, threshold: float, path=None, title: str = "Reprojection error"
):
    """Histogram of per-observation errors up to 3x the threshold; the last bar collects
    everything beyond. Returns the matplotlib Figure (saved when ``path`` is given)."""
    from matplotlib.figure import Figure

    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    limit = 3.0 * threshold
    fig = Figure(figsize=(7, 3.5), layout="constrained")
    ax = fig.add_subplot()
    ax.hist(np.minimum(errors, limit), bins=60, range=(0, limit), color="tab:blue")
    ax.axvline(threshold, color="k", ls="--", lw=1, label=f"threshold {threshold:g} px")
    beyond = int(np.sum(errors > limit))
    if beyond:
        ax.annotate(
            f"{beyond} > {limit:g} px", xy=(0.98, 0.9), xycoords="axes fraction", ha="right"
        )
    ax.set_xlabel("reprojection error (px)")
    ax.set_ylabel("observations")
    median = float(np.median(errors)) if errors.size else float("nan")
    ax.set_title(f"{title}: median {median:.2f} px, {errors.size} observations", fontsize=10)
    ax.legend(loc="upper center")
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig

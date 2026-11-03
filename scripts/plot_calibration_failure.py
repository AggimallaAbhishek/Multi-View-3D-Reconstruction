"""E3 failure case: the clouds of the calibrated, -20 % and guessed-intrinsics runs, each
seen along its own principal axes at unit RMS spread, with its axis ratios.

    python scripts/plot_calibration_failure.py outputs/experiments/<run>/e3_calibration \
        --out docs/images/e3_calibration_failure.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pointcloud.export import read_ply  # noqa: E402

PANELS = [  # (run directory name, title)
    ("03_calibrated", "true focal length"),
    ("00_focal_-20", "focal length −20 %"),
    ("07_guessed_no_camera.yaml", "guessed (f = 1.2·max(w,h), −49 %)"),
]


def principal_view(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points in their principal-axis frame at unit RMS spread, and the axis ratios
    (standard deviations along the axes relative to the largest)."""
    centred = points - points.mean(axis=0)
    _, s, Vt = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ Vt.T
    projected /= np.sqrt((projected**2).sum(axis=1).mean())
    return projected, s / s[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", type=Path, help="e3_calibration experiment directory")
    parser.add_argument("--out", type=Path, default=Path("e3_calibration_failure.png"))
    args = parser.parse_args(argv)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
    for ax, (run, title) in zip(axes, PANELS, strict=True):
        points, colors = read_ply(args.experiment / run / "filtered.ply")
        view, ratios = principal_view(points)
        ax.scatter(view[:, 0], view[:, 1], s=0.5, c=colors, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}\naxis ratios 1 : {ratios[1]:.2f} : {ratios[2]:.2f}")
    fig.suptitle("E3 failure case: the same 47 photos reconstructed with different intrinsics "
                 "(principal-axis views, unit RMS spread)")  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

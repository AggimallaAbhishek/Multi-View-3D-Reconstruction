"""Compare E3 (focal length held fixed in bundle adjustment) with E3b (focal length
refined): rotation error, completeness and final focal length against the starting
focal error.

    python scripts/plot_focal_refinement.py outputs/experiments/<run>/e3_calibration \
        outputs/experiments/<run>/e3b_focal_refinement --out docs/images/e3b_focal_refinement.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _table(directory: Path) -> pd.DataFrame:
    """Rotation error (deg), completeness and final focal error (%) of every successful
    setting, indexed by setting label."""
    t = pd.read_csv(directory / "experiment_results.csv")
    t = t[t["status"] == "ok"].set_index("setting")
    return pd.DataFrame({
        "rotation": t["rotation_error_median_deg"],
        "completeness": t["completeness"],
        "final": 100 * t["focal_error"],
    })  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fixed", type=Path, help="e3_calibration experiment directory")
    parser.add_argument("refined", type=Path, help="e3b_focal_refinement experiment directory")
    parser.add_argument("--out", type=Path, default=Path("e3b_focal_refinement.png"))
    args = parser.parse_args()

    fixed = _table(args.fixed)
    start = fixed["final"].rename("start")  # K fixed: the final focal error is the start
    runs = {
        name: table.join(start, how="inner").sort_values("start")
        for name, table in [("K fixed (E3)", fixed), ("focal refined (E3b)", _table(args.refined))]
    }
    style = {"K fixed (E3)": ("#b5523b", "o", "--"), "focal refined (E3b)": ("#2f6b9a", "s", "-")}
    panels = [
        ("rotation", "median rotation error (deg)", True),
        ("completeness", "completeness", False),
        ("final", "final focal error (%)", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, (key, label, log) in zip(axes, panels, strict=True):
        for name, table in runs.items():
            color, marker, line = style[name]
            ax.plot(table["start"], table[key], color=color, marker=marker, linestyle=line,
                    label=name)  # fmt: skip
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("starting focal error (%)")
        ax.set_ylabel(label)
        ax.axvline(0, color="0.6", linewidth=0.8)
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False)
    fig.suptitle("templeRing: a wrong focal length, held fixed or refined in bundle adjustment")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

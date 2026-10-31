"""Compare E3 (focal length held fixed in bundle adjustment) with E3b (focal length
refined): rotation error, completeness and final focal length against the starting
focal error.

    python scripts/plot_focal_refinement.py outputs/experiments/<run>/e3_calibration \
        outputs/experiments/<run>/e3b_focal_refinement --out docs/images/e3b_focal_refinement.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GUESS_SCALE = 768 / 1520.4  # templeRing fallback f = 1.2 * max(w, h) vs the calibrated fx


def _rows(directory: Path) -> list[dict[str, float]]:
    """(starting focal error %, metrics) per setting, sorted by focal error."""
    rows = []
    with (directory / "experiment_results.csv").open() as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            guessed = r.get("camera.auto_calibration_file") == "False"
            scale = GUESS_SCALE if guessed else float(r["camera.focal_scale"] or 1.0)
            rows.append({
                "start": 100 * (scale - 1),
                "rotation": float(r["rotation_error_median_deg"]),
                "completeness": float(r["completeness"]),
                "final": 100 * float(r["focal_error"]),
            })  # fmt: skip
    return sorted(rows, key=lambda r: r["start"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fixed", type=Path, help="e3_calibration experiment directory")
    parser.add_argument("refined", type=Path, help="e3b_focal_refinement experiment directory")
    parser.add_argument("--out", type=Path, default=Path("e3b_focal_refinement.png"))
    args = parser.parse_args()

    runs = {"K fixed (E3)": _rows(args.fixed), "focal refined (E3b)": _rows(args.refined)}
    style = {"K fixed (E3)": ("#b5523b", "o", "--"), "focal refined (E3b)": ("#2f6b9a", "s", "-")}
    panels = [
        ("rotation", "median rotation error (deg)", True),
        ("completeness", "completeness", False),
        ("final", "final focal error (%)", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, (key, label, log) in zip(axes, panels, strict=True):
        for name, rows in runs.items():
            color, marker, line = style[name]
            ax.plot([r["start"] for r in rows], [r[key] for r in rows], color=color,
                    marker=marker, linestyle=line, label=name)  # fmt: skip
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

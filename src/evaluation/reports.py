"""Figures and a Markdown table from an experiment directory (see ``experiments``).

Everything is regenerated from ``experiment_results.csv``, ``experiment.json`` and each
run's ``reprojection_errors.npy``, so reports can be redrawn without rerunning the
pipeline. The experiment's ``plot`` entry
chooses the figure: ``x`` (a column; numeric values give lines, one per ``hue`` value),
``y`` (metrics, one panel each). Without ``x`` each setting gets a bar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.run import read_json

LABELS = {
    "registered": "registered images",
    "registered_fraction": "registered fraction",
    "landmarks": "landmarks",
    "filtered_points": "points (filtered)",
    "mean_track_length": "mean track length",
    "inlier_ratio_mean": "mean inlier ratio",
    "verified_pairs": "verified pairs",
    "features_mean": "features per image",
    "reprojection_mean_px": "reprojection error (px, mean)",
    "rotation_error_median_deg": "rotation error (deg, median)",
    "rotation_error_max_deg": "rotation error (deg, max)",
    "centre_error_median": "centre error (fraction of extent, median)",
    "centre_error_max": "centre error (fraction of extent, max)",
    "completeness": "completeness",
    "accuracy_median": "accuracy (fraction of extent, median)",
    "chamfer": "Chamfer (fraction of extent)",
    "time_total_s": "runtime (s)",
}
TABLE = (
    "registered", "filtered_points", "mean_track_length", "reprojection_mean_px",
    "rotation_error_median_deg", "centre_error_median", "completeness", "accuracy_median",
    "time_total_s",
)  # fmt: skip


def make_report(experiment_dir: str | Path) -> list[Path]:
    """Write ``metrics.png``, ``runtime.png``, ``results.md`` and, when the runs saved
    their errors, ``reprojection.png`` into the experiment directory; return the paths."""
    directory = Path(experiment_dir)
    meta = read_json(directory / "experiment.json")
    table = pd.read_csv(directory / "experiment_results.csv")
    plot = meta.get("plot") or {}
    metrics = [m for m in plot.get("y", ["registered", "completeness"]) if m in table]
    title = f"{meta['name']}: {meta.get('question', '')}".rstrip(": ")
    files = [
        _metrics_figure(table, plot, metrics, title, directory / "metrics.png"),
        _runtime_figure(table, title, directory / "runtime.png"),
        _results_table(table, meta, metrics, directory / "results.md"),
    ]
    errors = {
        run["label"]: np.load(path)
        for run in meta.get("runs", [])
        if (path := directory / run["directory"] / "reprojection_errors.npy").is_file()
    }
    if errors:
        files.append(_reprojection_figure(errors, title, directory / "reprojection.png"))
    return files


def _reprojection_figure(errors: dict[str, np.ndarray], title: str, path):
    """One step histogram (share of observations) of the final reprojection errors per
    setting, over a common range."""
    from matplotlib import colormaps
    from matplotlib.figure import Figure

    limit = max(float(np.percentile(e, 99.5)) for e in errors.values() if len(e)) or 1.0
    bins = np.linspace(0.0, limit, 60)
    colours = colormaps["viridis"](np.linspace(0, 0.9, len(errors)))
    fig = Figure(figsize=(8, 4.2), layout="constrained")
    ax = fig.add_subplot()
    for (label, e), colour in zip(errors.items(), colours, strict=True):
        weights = np.full(len(e), 1.0 / max(len(e), 1))
        median = float(np.median(e)) if len(e) else float("nan")
        ax.hist(np.minimum(e, limit), bins=bins, weights=weights, histtype="step", lw=1.6,
                color=colour, label=f"{label} (median {median:.2f} px)")  # fmt: skip
    ax.set_xlabel("reprojection error (px)")
    ax.set_ylabel("share of observations")
    ax.set_title(f"Reprojection error - {title}", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=130)
    return Path(path)


def _metrics_figure(table: pd.DataFrame, plot: dict, metrics: list[str], title: str, path):
    from matplotlib.figure import Figure

    n = max(len(metrics), 1)
    fig = Figure(figsize=(4.2 * min(n, 3), 3.4 * int(np.ceil(n / 3))), layout="constrained")
    fig.suptitle(title, fontsize=11)
    x, hue = plot.get("x"), plot.get("hue")
    numeric_x = x in table and pd.api.types.is_numeric_dtype(table[x])
    for k, metric in enumerate(metrics):
        ax = fig.add_subplot(int(np.ceil(n / 3)), min(n, 3), k + 1)
        if numeric_x:
            groups = table.groupby(hue, sort=False) if hue in table else [(None, table)]
            for value, group in groups:
                group = group.sort_values(x)
                label = None if value is None else f"{hue} = {value}"
                ax.plot(group[x], group[metric], marker="o", label=label)
            ax.set_xlabel(x)
            if hue in table:
                ax.legend(fontsize=8)
        else:
            positions = np.arange(len(table))
            ax.bar(positions, table[metric], color="tab:blue")
            ax.set_xticks(positions, table["setting"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(LABELS.get(metric, metric), fontsize=9)
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=130)
    return Path(path)


def _runtime_figure(table: pd.DataFrame, title: str, path):
    from matplotlib import colormaps
    from matplotlib.figure import Figure

    stages = [c for c in table if c.startswith("time_") and c != "time_total_s"]
    times = table[stages].fillna(0.0).to_numpy()
    fig = Figure(figsize=(8, 0.45 * len(table) + 1.6), layout="constrained")
    ax = fig.add_subplot()
    colours = colormaps["tab10"](np.arange(len(stages)) % 10)
    left = np.zeros(len(table))
    for s, (stage, colour) in enumerate(zip(stages, colours, strict=True)):
        name = stage.removeprefix("time_").removesuffix("_s")
        ax.barh(np.arange(len(table)), times[:, s], left=left, color=colour, label=name)
        left += times[:, s]
    ax.set_yticks(np.arange(len(table)), table["setting"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("runtime (s)")
    ax.set_title(f"Runtime per stage - {title}", fontsize=10)
    ax.legend(fontsize=8, ncols=min(len(stages), 5), loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(path, dpi=130)
    return Path(path)


def _results_table(table: pd.DataFrame, meta: dict, metrics: list[str], path):
    columns = ["setting", "status"] + [c for c in dict.fromkeys([*metrics, *TABLE]) if c in table]
    lines = [
        f"# {meta['name']}",
        "",
        meta.get("question", ""),
        "",
        "| " + " | ".join(LABELS.get(c, c) for c in columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_format(row[c]) for c in columns) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


def _format(value) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "-"
        return f"{value:.0f}" if value.is_integer() and abs(value) >= 10 else f"{value:.3g}"
    return str(value)

"""Experiments: one pipeline run per setting, one metrics row per run.

An experiment is a YAML file (``configs/experiments/``)::

    name: e2_matching
    question: RQ2 - how does matching quality affect the point cloud?
    base:                      # optional overrides applied to every run
      features.max_features: 4000
    vary:                      # a grid: every combination of these values ...
      matching.ratio: [0.6, 0.7, 0.8, 0.9]
      matching.mutual: [true, false]
    runs:                      # ... or named settings (exactly one of vary / runs)
      calibrated: {}
      focal +10%: {camera.focal_scale: 1.1}
    plot: {x: matching.ratio, hue: matching.mutual, y: [registered, completeness]}

Overrides are dotted config keys, validated when the file is loaded. Each run gets its
own directory (``config.yaml``, ``environment.json``, ``metrics.json``,
``filtered.ply``, ``reprojection_errors.npy``); the experiment directory holds
``experiment_results.csv`` (one row per setting: label, overrides, status, metrics) and
``experiment.json``.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config import Config, ConfigError
from src.data.loader import DatasetError
from src.evaluation.ground_truth import at_working_resolution
from src.evaluation.metrics import camera_pose_errors, cloud_errors, metric_scale
from src.geometry.camera import Camera
from src.geometry.reprojection import ReprojectionStats
from src.geometry.triangulation import Observations, triangulate_and_filter
from src.pipeline import STAGES, PipelineResult, run_pipeline
from src.pointcloud.export import write_ply
from src.pointcloud.filtering import filter_cloud
from src.reconstruction.landmarks import keypoint_pixels
from src.reconstruction.tracks import ViewGraph, build_tracks
from src.run import write_csv, write_json, write_run_metadata

TIMED = ("dataset", *STAGES)
METRICS = (
    "images", "features_mean", "pairs", "raw_matches_mean", "matches_mean", "verified_pairs",
    "inlier_ratio_mean",
    "registered", "registered_fraction", "landmarks", "observations", "mean_track_length",
    "reprojection_mean_px", "reprojection_median_px", "filtered_points", "focal_px",
    "focal_error", "rotation_error_median_deg", "rotation_error_max_deg",
    "centre_error_median", "centre_error_max", "metric_scale_error",
    "completeness", "accuracy_median", "chamfer",
    *(f"time_{stage}_s" for stage in TIMED), "time_total_s",
)  # fmt: skip


class ExperimentError(ValueError):
    pass


@dataclass(frozen=True)
class Setting:
    label: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class Experiment:
    name: str
    question: str
    settings: list[Setting]
    plot: dict[str, Any] = field(default_factory=dict)


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {"name", "question", "base", "vary", "runs", "plot"}
    if unknown:
        raise ExperimentError(f"{path}: unknown key(s) {sorted(unknown)}")
    if ("vary" in raw) == ("runs" in raw):
        raise ExperimentError(f"{path}: give exactly one of 'vary' (grid) or 'runs' (named)")
    base = dict(raw.get("base") or {})
    if "vary" in raw:
        keys, values = zip(*raw["vary"].items(), strict=True)
        settings = [
            Setting(", ".join(f"{k}={v}" for k, v in zip(keys, combo, strict=True)),
                    {**base, **dict(zip(keys, combo, strict=True))})
            for combo in itertools.product(*values)
        ]  # fmt: skip
    else:
        settings = [Setting(str(label), {**base, **(o or {})}) for label, o in raw["runs"].items()]
    for s in settings:
        try:
            Config().with_overrides(s.overrides)
        except ConfigError as exc:
            raise ExperimentError(f"{path}: setting {s.label!r}: {exc}") from exc
    return Experiment(str(raw.get("name", path.stem)), str(raw.get("question", "")),
                      settings, dict(raw.get("plot") or {}))  # fmt: skip


def reference_cloud(images: str | Path, config: Config, gt: Mapping[str, Camera]) -> np.ndarray:
    """Reference geometry from ground-truth cameras: every verified track over the
    images (as ``config`` selects them) triangulated with the true cameras, filtered
    like a reconstruction. In the ground-truth frame. It still depends on our features
    and matches; it is independent of our camera poses."""
    result = run_pipeline(images, config, until="verify")
    tracks = build_tracks(ViewGraph.from_verified(result.matches, result.geometries))
    cameras = [at_working_resolution(gt[im.name], im.scale) for im in result.images]
    obs = Observations(
        tracks.image, tracks.track, keypoint_pixels(result.features, tracks.image, tracks.keypoint)
    )
    tri = triangulate_and_filter(cameras, obs, config.triangulation, n_points=tracks.n_tracks)
    X = tri.points[tri.valid]
    return X[filter_cloud(X, config.filtering).keep]


def collect_metrics(
    result: PipelineResult,
    gt: Mapping[str, Camera] | None = None,
    reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """One row of METRICS for a run (nan where a metric does not apply).

    Pose errors need ground truth and at least 3 registered cameras; cloud errors also
    need the reference. Both align the reconstruction to ground truth through its
    camera centres. ``metric_scale_error`` is the relative error of the scale taken from
    a single measured distance (the true distance between the first two registered
    cameras) against the scale fitted to all cameras.
    """
    row: dict[str, Any] = dict.fromkeys(METRICS, np.nan)
    verified = [s for s in result.verification_stats or [] if s.success]
    row.update(
        images=len(result.names),
        features_mean=float(np.mean([s.n_features for s in result.feature_stats])),
        pairs=len(result.matches),
        raw_matches_mean=float(np.mean([s.raw for s in result.match_stats])),
        matches_mean=float(np.mean([s.final for s in result.match_stats])),
        verified_pairs=len(verified),
        inlier_ratio_mean=float(np.mean([s.inlier_ratio for s in verified])) if verified else 0.0,
        registered=0,
        registered_fraction=0.0,
    )
    for stage, seconds in result.timings.items():
        row[f"time_{stage}_s"] = seconds
    row["time_total_s"] = sum(result.timings.values())

    rec = result.reconstruction
    if rec is None:
        return row
    lm, images, X = rec.landmarks, rec.registry.images, rec.landmarks.position
    cameras = [rec.registry.camera(m) for m in images]
    rp = ReprojectionStats.from_errors(rec.reprojection_errors(result.features))
    keep = result.filtering.keep if result.filtering is not None else np.ones(len(lm), bool)
    row.update(
        registered=len(images),
        registered_fraction=len(images) / len(result.names),
        landmarks=len(lm),
        observations=rp.n,
        mean_track_length=float(lm.track_lengths().mean()) if len(lm) else np.nan,
        reprojection_mean_px=rp.mean,
        reprojection_median_px=rp.median,
        filtered_points=int(keep.sum()),
        focal_px=float(np.median([c.K[0, 0] for c in cameras])),  # refined if refine_focal
    )
    if gt is None or len(images) < 3:
        return row
    truth = [gt[result.names[m]] for m in images]
    errors = camera_pose_errors(cameras, truth)
    one = metric_scale(cameras[0].centre, cameras[1].centre,
                       np.linalg.norm(truth[1].centre - truth[0].centre))  # fmt: skip
    # Ground-truth K is at the original resolution, ours at the working one.
    true_fx = [t.K[0, 0] * result.images[m].scale[0] for t, m in zip(truth, images, strict=True)]
    row.update(
        rotation_error_median_deg=float(np.median(errors.rotation_deg)),
        rotation_error_max_deg=float(errors.rotation_deg.max()),
        centre_error_median=float(np.median(errors.centre)),
        centre_error_max=float(errors.centre.max()),
        metric_scale_error=one / errors.scale - 1.0,
        focal_error=row["focal_px"] / float(np.median(true_fx)) - 1.0,
    )
    if reference is not None:
        tolerance = result.config.evaluation.completeness_tolerance
        ce = cloud_errors(errors.similarity.apply(X[keep]), reference, tolerance)
        row.update(completeness=ce.completeness, accuracy_median=ce.accuracy_median,
                   chamfer=ce.chamfer)  # fmt: skip
    return row


def run_experiment(
    experiment: Experiment,
    images: str | Path,
    base_config: Config,
    out_dir: str | Path,
    gt: Mapping[str, Camera] | None = None,
    reference: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Run every setting through the pipeline (up to ``filter``) and write the results.

    A setting whose dataset is unusable is recorded with status ``failed: ...`` and
    zero registered images instead of stopping the experiment.
    """
    exp_dir = Path(out_dir) / experiment.name
    rows, runs = [], []
    for k, setting in enumerate(experiment.settings):
        config = base_config.with_overrides(setting.overrides)
        directory = f"{k:02d}_{_slug(setting.label)}"
        run_dir = exp_dir / directory
        run_dir.mkdir(parents=True, exist_ok=True)
        write_run_metadata(run_dir, config)
        try:
            result = run_pipeline(images, config, until="filter")
        except DatasetError as exc:
            status, metrics = f"failed: {exc}", {**dict.fromkeys(METRICS, np.nan), "registered": 0}
        else:
            metrics = collect_metrics(result, gt, reference)
            rec = result.reconstruction
            status = "ok" if rec is not None else "no_initial_pair"
            if rec is not None:
                keep = result.filtering.keep
                write_ply(run_dir / "filtered.ply", rec.landmarks.position[keep],
                          rec.landmarks.color[keep] / 255.0)  # fmt: skip
                np.save(run_dir / "reprojection_errors.npy",
                        rec.reprojection_errors(result.features).astype(np.float32))  # fmt: skip
        record = {"setting": setting.label, "status": status, "overrides": setting.overrides}
        write_json(run_dir / "metrics.json", {**record, **metrics})
        rows.append({"setting": setting.label, **setting.overrides, "status": status, **metrics})
        runs.append({"label": setting.label, "directory": directory,
                     "overrides": setting.overrides, "status": status})  # fmt: skip

    columns = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(exp_dir / "experiment_results.csv", columns,
              ([_cell(row.get(c)) for c in columns] for row in rows))  # fmt: skip
    write_json(exp_dir / "experiment.json", {
        "name": experiment.name, "question": experiment.question, "images": str(images),
        "plot": experiment.plot, "runs": runs,
    })  # fmt: skip
    return rows


def _slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", label).strip("_") or "run"


def _cell(value: Any) -> Any:
    if isinstance(value, float):
        return "" if np.isnan(value) else f"{value:.6g}"
    return value

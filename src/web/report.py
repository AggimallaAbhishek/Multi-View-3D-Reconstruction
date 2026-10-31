"""``report.json``: one summary of a pipeline run for the web interface.

Built from a ``PipelineResult`` however far it got, so a run that stopped early still
gets a report; sections whose stage did not run are ``None`` and ``warnings`` says why.
With ground-truth cameras the registered cameras are scored like
``scripts/evaluate_against_ground_truth.py`` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import camera_pose_errors
from src.geometry.camera import Camera
from src.geometry.reprojection import ReprojectionStats
from src.pipeline import PipelineResult
from src.run import write_json

REPORT_NAME = "report.json"


def build_report(result: PipelineResult, gt: dict[str, Camera] | None = None) -> dict[str, Any]:
    """Plain-JSON summary: images, points, calibration, reprojection, timings, warnings
    and, when ``gt`` (cameras by image name) is given, camera errors against it."""
    names, rec = result.names, result.reconstruction
    sources = sorted({im.intrinsics_source for im in result.images})
    warnings = []
    if "fallback" in sources:
        refined = result.config.bundle_adjustment.refine_focal
        warnings.append(
            "intrinsics from the fallback focal length (no camera.yaml or EXIF): "
            f"{'bundle adjustment refines the focal length, but ' if refined else ''}"
            "the principal point is assumed at the image centre, so poses and shape may be "
            "less accurate"
        )

    first, refined_K = 0, None  # K before and after bundle adjustment, of one image
    if rec is None:
        registered, unregistered, points, reprojection = [], list(names), None, None
        warnings.append("no reconstruction: no image pair qualified as the initial pair")
    else:
        registered = rec.registry.images
        first = registered[0]
        refined_K = rec.registry.camera(first).K.tolist()
        unregistered = [names[m] for m in rec.unregistered(len(names))]
        points = {
            "sparse": len(rec.landmarks),
            "filtered": None if result.filtering is None else int(result.filtering.keep.sum()),
        }
        errors = rec.reprojection_errors(result.features)
        reprojection = ReprojectionStats.from_errors(errors).to_dict()
        if unregistered:
            warnings.append(
                f"{len(unregistered)} image(s) not registered: {', '.join(unregistered)}"
            )

    report = {
        "images": {
            "total": len(names),
            "registered": len(registered),
            "unregistered": unregistered,
        },
        "points": points,
        "calibration": {
            "sources": sources,
            "K": result.images[first].K.tolist(),
            "refined_K": refined_K,
        },
        "reprojection": reprojection,
        "timings": dict(result.timings),
        "total_s": sum(result.timings.values()),
        "warnings": warnings,
    }
    if gt is not None:
        report["ground_truth"] = _ground_truth(result, registered, gt)
    return report


def write_report(report: dict[str, Any], run_dir: str | Path) -> Path:
    return write_json(Path(run_dir) / REPORT_NAME, report)


def _ground_truth(
    result: PipelineResult, registered: list[int], gt: dict[str, Camera]
) -> dict[str, Any]:
    names = result.names
    matched = [m for m in registered if names[m] in gt]
    if not matched:
        return {"error": "no matching cameras in gt_cameras.json"}
    if len(matched) < 3:  # a similarity alignment needs at least 3 camera centres
        return {
            "error": f"only {len(matched)} registered image(s) have a ground-truth camera; "
            "aligning to ground truth needs at least 3"
        }
    cameras = [result.reconstruction.registry.camera(m) for m in matched]
    errors = camera_pose_errors(cameras, [gt[names[m]] for m in matched])
    return {
        "cameras": len(matched),
        "rotation_deg": _percentiles(errors.rotation_deg),
        "centre": _percentiles(errors.centre),  # fraction of the true cameras' extent
        "per_camera": [
            {"image": names[m], "rotation_deg": float(r), "centre": float(c)}
            for m, r, c in zip(matched, errors.rotation_deg, errors.centre, strict=True)
        ],
    }


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }

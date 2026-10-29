"""Command-line entry point: ``python -m src.cli <command> ...``.

The pipeline itself lives in ``src.pipeline``; this module parses arguments, writes
each completed stage's outputs into a run directory and prints summaries. Diagnostic
images are drawn only for the last stage a command runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from src.config import Config, load_config
from src.data.dataset import prepare_dataset
from src.data.loader import DatasetError, ImageRecord, discover_images, iter_images
from src.data.validation import format_report
from src.features.extractor import write_feature_statistics
from src.features.matcher import (
    Matches,
    connectivity,
    match_matrix,
    save_matches,
    write_match_statistics,
)
from src.geometry.calibration import (
    CalibrationError,
    calibrate_from_checkerboard,
    save_camera_file,
)
from src.geometry.pose_estimation import write_pose_statistics
from src.geometry.verification import (
    inlier_matches,
    verification_matrix,
    write_verification_statistics,
)
from src.pipeline import PipelineResult, Stage, run_pipeline
from src.run import create_run_dir, write_run_metadata
from src.visualization.cameras import plot_two_view
from src.visualization.epipolar import draw_epipolar_lines
from src.visualization.features import draw_keypoints
from src.visualization.matches import draw_matches, plot_match_matrix


def main(argv: list[str] | None = None) -> int:
    # Shared options, attached to subcommands as argparse parents.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("images", type=Path, help="directory of input photos")
    common.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml)")
    common.add_argument("--out", type=Path, help="output root (default: config output.root)")
    detector = argparse.ArgumentParser(add_help=False)
    detector.add_argument("--detector", choices=["sift", "orb"], help="override features.detector")
    draw = argparse.ArgumentParser(add_help=False)
    draw.add_argument("--draw", type=int, default=3, help="diagnostic images to save (3)")
    method = argparse.ArgumentParser(add_help=False)
    method.add_argument(
        "--method", choices=["auto", "essential", "fundamental"], help="override model"
    )

    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "validate", parents=[common], help="load, validate and summarise an image folder"
    )
    p.set_defaults(func=_validate)

    p = sub.add_parser(
        "features", parents=[common, detector, draw], help="detect and describe keypoints"
    )
    p.set_defaults(func=_stage_command, stage="features")

    p = sub.add_parser(
        "matches", parents=[common, detector, draw], help="match features between image pairs"
    )
    p.add_argument("--ratio", type=float, help="override matching.ratio")
    p.add_argument(
        "--pairs", choices=["exhaustive", "sequential"], help="override matching.pair_mode"
    )
    p.set_defaults(func=_stage_command, stage="matches")

    p = sub.add_parser(
        "verify",
        parents=[common, detector, draw, method],
        help="match pairs and verify them with epipolar geometry",
    )
    p.add_argument("--threshold", type=float, help="override ransac_threshold_px")
    p.set_defaults(func=_stage_command, stage="verify")

    p = sub.add_parser(
        "pose",
        parents=[common, detector, method],
        help="relative pose of verified pairs; pick the initial pair",
    )
    p.set_defaults(func=_stage_command, stage="pose", draw=0)

    p = sub.add_parser("calibrate", help="calibrate a camera from checkerboard photos")
    p.add_argument("images", type=Path, help="directory of checkerboard photos")
    p.add_argument("--pattern", default="9x6", help="inner corners, COLSxROWS (9x6)")
    p.add_argument("--square", type=float, default=1.0, help="square size (any unit)")
    p.add_argument("--out", type=Path, default=Path("configs/camera.yaml"))
    p.add_argument("--config", type=Path, help="config for dataset.extensions")
    p.set_defaults(func=_calibrate)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


# --- helpers ----------------------------------------------------------------------------


def _override(config: Config, section: str, **values) -> Config:
    """Config with command-line overrides applied; ``None`` values are ignored."""
    values = {k: v for k, v in values.items() if v is not None}
    if not values:
        return config
    return dataclasses.replace(
        config, **{section: dataclasses.replace(getattr(config, section), **values)}
    )


def _config_from_args(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    config = _override(config, "features", detector=getattr(args, "detector", None))
    config = _override(
        config,
        "matching",
        ratio=getattr(args, "ratio", None),
        pair_mode=getattr(args, "pairs", None),
    )
    return _override(
        config,
        "verification",
        method=getattr(args, "method", None),
        ransac_threshold_px=getattr(args, "threshold", None),
    )


def _print_dataset_error(exc: DatasetError) -> None:
    print(f"ERROR: {exc}")
    if exc.report is not None:
        print(format_report(exc.report))


def _spread(values: Sequence[float], fmt: str = "{:.0f}", unit: str = "") -> str:
    v = np.asarray(values, dtype=float)
    return ", ".join(
        f"{name} {fmt.format(x)}{unit}"
        for name, x in (("min", v.min()), ("median", np.median(v)), ("max", v.max()))
    )


def _inspection_order(matches: Sequence[Matches]) -> list[int]:
    """Indices of pairs, neighbouring images first: the pairs most worth looking at."""
    return sorted(range(len(matches)), key=lambda k: (matches[k].j - matches[k].i, matches[k].i))


def _pair_stem(name_i: str, name_j: str) -> str:
    return f"{Path(name_i).stem}_{Path(name_j).stem}"


# --- commands -----------------------------------------------------------------------------


def _validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        dataset = prepare_dataset(args.images, config)
    except DatasetError as exc:
        _print_dataset_error(exc)
        return 1

    run_dir = create_run_dir(args.out or config.output.root, "validate")
    dataset.save(run_dir / "dataset.json")
    write_run_metadata(run_dir, config)

    print(format_report(dataset.report))
    sources = sorted({im.intrinsics_source for im in dataset.images})
    first = dataset.images[0]
    print(
        f"Working size: {first.width}x{first.height} (scale {first.scale[0]:.3f}); "
        f"intrinsics from: {', '.join(sources)}; fx = {first.K[0, 0]:.1f} px"
    )
    print(f"Wrote {run_dir / 'dataset.json'}")
    return 0


def _stage_command(args: argparse.Namespace) -> int:
    """features / matches / verify / pose: run the pipeline up to ``args.stage``."""
    config = _config_from_args(args)
    stage: Stage = args.stage
    try:
        result = run_pipeline(args.images, config, until=stage)
    except DatasetError as exc:
        _print_dataset_error(exc)
        return 1

    run_dir = create_run_dir(args.out or config.output.root, f"{stage}_{result.extractor.name}")
    write_run_metadata(run_dir, config)

    def draw_for(s: Stage) -> int:
        return max(args.draw, 0) if s == stage else 0

    written = _save_features(result, run_dir, draw_for("features"))
    if result.matches is not None:
        written = _save_matches(result, run_dir, draw_for("matches"))
    if result.geometries is not None:
        written = _save_verification(result, run_dir, draw_for("verify"))
    status = 0
    if result.poses is not None:
        written, status = _save_poses(result, run_dir)
    print(f"Wrote {written}")
    return status


# --- per-stage outputs and summaries ------------------------------------------------------


def _save_features(r: PipelineResult, run_dir: Path, draw: int) -> Path:
    out = run_dir / "features"
    csv_path = write_feature_statistics(r.feature_stats, out / "feature_statistics.csv")
    for im, f in list(zip(r.images, r.features, strict=True))[:draw]:
        cv2.imwrite(str(out / f"{Path(im.name).stem}_keypoints.png"), draw_keypoints(im.image, f))

    stats, cfg = r.feature_stats, r.config.features
    new = [s.time_ms for s in stats if not s.cached]
    print(f"Detector: {r.extractor.signature()}")
    print(
        f"Features per image: {_spread([s.n_features for s in stats])} "
        f"(cap {r.extractor.max_features})"
    )
    print(f"Grid coverage: {_spread([s.coverage for s in stats], '{:.2f}')}")
    print(
        f"Computed {len(new)} image(s)"
        + (f" at {np.mean(new):.0f} ms each" if new else "")
        + f", {len(stats) - len(new)} from cache"
    )
    for s in stats:
        if s.n_features < cfg.warn_min_features or s.coverage < cfg.warn_min_coverage:
            print(
                f"[WARNING] {s.image}: {s.n_features} features, coverage {s.coverage:.2f} "
                "- too little texture or features concentrated in one region"
            )
    return csv_path


def _save_matches(r: PipelineResult, run_dir: Path, draw: int) -> Path:
    out, names, stats, cfg = run_dir / "matches", r.names, r.match_stats, r.config.matching
    csv_path = write_match_statistics(stats, names, out / "match_statistics.csv")
    save_matches(r.matches, out / "matches.npz")
    plot_match_matrix(match_matrix(stats, len(names)), names, out / "match_matrix.png")
    for k in _inspection_order(r.matches)[:draw]:
        m = r.matches[k]
        a, b = r.images[m.i], r.images[m.j]
        canvas = draw_matches(a.image, b.image, r.features[m.i], r.features[m.j], m)
        cv2.imwrite(str(out / f"{_pair_stem(a.name, b.name)}_matches.png"), canvas)

    final = np.array([s.final for s in stats])
    good = final >= cfg.min_matches
    print(f"Matching: ratio {cfg.ratio}, mutual {cfg.mutual}, {cfg.method}")
    print(
        f"Pairs: {len(stats)} ({cfg.pair_mode}) in {sum(s.time_ms for s in stats) / 1000:.1f} s; "
        f"{good.sum()} with >= {cfg.min_matches} matches"
    )
    if good.any():
        print(f"Matches per connected pair: {_spread(final[good])}")
    best = connectivity(stats, len(names))
    for k in np.flatnonzero(best < cfg.min_matches):
        print(
            f"[WARNING] {names[k]}: best pair has only {best[k]} matches - "
            "it will probably not be registered"
        )
    return csv_path


def _save_verification(r: PipelineResult, run_dir: Path, draw: int) -> Path:
    out, names, stats, cfg = (
        run_dir / "verification", r.names, r.verification_stats, r.config.verification
    )  # fmt: skip
    csv_path = write_verification_statistics(stats, names, out / "verification_statistics.csv")
    verified = [inlier_matches(m, g) for m, g in zip(r.matches, r.geometries, strict=True)]
    save_matches(verified, out / "verified_matches.npz")
    inliers = verification_matrix(stats, len(names))
    plot_match_matrix(
        inliers, names, out / "verified_matrix.png",
        title="Verified inliers per image pair", label="geometric inliers",
    )  # fmt: skip
    for k in _inspection_order(r.matches)[:draw]:
        m, g = r.matches[k], r.geometries[k]
        a, b = r.images[m.i], r.images[m.j]
        stem = _pair_stem(a.name, b.name)
        fi, fj = r.features[m.i], r.features[m.j]
        canvas = draw_matches(a.image, b.image, fi, fj, m, inlier_mask=g.inlier_mask)
        cv2.imwrite(str(out / f"{stem}_inliers.png"), canvas)
        if g.F is not None and g.n_inliers:
            x_i, x_j = m.points(fi, fj)
            canvas = draw_epipolar_lines(
                a.image, b.image, x_i[g.inlier_mask], x_j[g.inlier_mask], g.F
            )
            cv2.imwrite(str(out / f"{stem}_epipolar.png"), canvas)

    ok = [s for s in stats if s.success]
    models = ", ".join(sorted({s.model for s in stats}))
    print(
        f"Verification: {models} ({cfg.estimator}, threshold {cfg.ransac_threshold_px} px) "
        f"in {sum(s.time_ms for s in stats) / 1000:.1f} s"
    )
    print(f"Verified pairs: {len(ok)} of {len(stats)} (>= {cfg.min_inliers} inliers)")
    if ok:
        planar = sum(s.homography_ratio > cfg.planar_ratio for s in ok)
        print(
            f"Inlier ratio: {_spread([s.inlier_ratio for s in ok], '{:.2f}')}; "
            f"median residual {np.median([s.median_residual for s in ok]):.2f} px"
        )
        print(
            f"Pairs explained by a homography (H/F > {cfg.planar_ratio}): {planar} "
            "- planar scene or rotation only, weak for initialisation"
        )
    for k in np.flatnonzero(inliers.max(axis=1) == 0):
        print(f"[WARNING] {names[k]}: no verified pair - it cannot be registered")
    return csv_path


def _save_poses(r: PipelineResult, run_dir: Path) -> tuple[Path, int]:
    out, names, stats, cfg = run_dir / "pose", r.names, r.pose_stats, r.config
    csv_path = write_pose_statistics(stats, names, out / "relative_poses.csv")

    ok = [s for s in stats if s.success]
    reasons = Counter(s.reason for s in stats if not s.success)
    failed = ", ".join(f"{reason} {n}" for reason, n in reasons.items())
    print(
        f"Relative poses: {len(ok)} of {len(stats)} verified pairs"
        + (f" (failed: {failed})" if failed else "")
    )
    if ok:
        parallax = [s.median_parallax_deg for s in ok]
        print(f"Median parallax per pair: {_spread(parallax, '{:.1f}', ' deg')}")

    init = r.initial_pair
    if init is None:
        print(
            "[WARNING] no pair qualifies as the initial pair (needs a valid pose, "
            f"H/F <= {cfg.verification.planar_ratio} and parallax >= "
            f"{cfg.pose.init_min_parallax_deg} deg). Capture views with more baseline."
        )
        return csv_path, 1

    a, b, pose, g = names[init.i], names[init.j], init.pose, init.geometry
    plot_two_view(
        pose.R, pose.t, init.points, (a, b), out / "initial_pair.png",
        title=f"Initial pair {a} - {b}: {len(init.points)} points, "
        f"parallax {pose.median_parallax_deg:.1f} deg",
    )  # fmt: skip
    (out / "initial_pair.json").write_text(
        json.dumps(init.to_dict(names), indent=2), encoding="utf-8"
    )
    print(
        f"Initial pair: {a} - {b}: {pose.n_valid} points, "
        f"parallax {pose.median_parallax_deg:.1f} deg, rotation {pose.rotation_deg:.1f} deg, "
        f"H/F {g.homography_ratio:.2f} ({g.model})"
    )
    return csv_path, 0


def _calibrate(args: argparse.Namespace) -> int:
    try:
        cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    except ValueError:
        print(f"ERROR: --pattern must look like 9x6, got {args.pattern!r}")
        return 2
    try:
        paths = discover_images(args.images, load_config(args.config).dataset.extensions)
        images = [r.image for r in iter_images(paths) if isinstance(r, ImageRecord)]
        intrinsics, rms = calibrate_from_checkerboard(images, (cols, rows), args.square)
    except (DatasetError, CalibrationError) as exc:
        print(f"ERROR: {exc}")
        return 1
    path = save_camera_file(intrinsics, args.out)
    print(f"RMS reprojection error: {rms:.3f} px")
    print(
        f"fx={intrinsics.K[0, 0]:.1f} fy={intrinsics.K[1, 1]:.1f} "
        f"cx={intrinsics.K[0, 2]:.1f} cy={intrinsics.K[1, 2]:.1f}"
    )
    print(f"Wrote {path}; set camera.calibration_file to use it")
    return 0


if __name__ == "__main__":
    sys.exit(main())

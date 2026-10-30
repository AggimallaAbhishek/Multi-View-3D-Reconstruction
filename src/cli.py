"""Command-line entry point: ``python -m src.cli <command> ...``.

The pipeline itself lives in ``src.pipeline`` and the per-stage outputs in
``src.outputs``; this module parses arguments, runs a command and reports the result.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import Config, load_config
from src.data.dataset import Dataset, prepare_dataset
from src.data.loader import DatasetError, ImageRecord, discover_images, iter_images
from src.data.validation import format_report
from src.geometry.calibration import (
    CalibrationError,
    calibrate_from_checkerboard,
    save_camera_file,
)
from src.outputs import new_run_dir, save_stage_outputs
from src.pipeline import Stage, run_pipeline
from src.pointcloud.export import read_cameras_json, read_ply
from src.run import create_run_dir, write_run_metadata


def main(argv: list[str] | None = None) -> int:
    # Shared options, attached to subcommands as argparse parents.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("images", type=Path, help="directory of input photos")
    common.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml)")
    common.add_argument("--out", dest="output.root", help="output.root (run directories go here)")
    detector = argparse.ArgumentParser(add_help=False)
    # Options that override config keys use the dotted key as their argparse ``dest``;
    # _config_from_args applies all of them through the validated Config.with_overrides.
    detector.add_argument(
        "--detector", dest="features.detector", choices=["sift", "orb"], help="features.detector"
    )
    draw = argparse.ArgumentParser(add_help=False)
    draw.add_argument("--draw", type=int, default=3, help="diagnostic images to save (3)")
    matching = argparse.ArgumentParser(add_help=False)
    matching.add_argument(
        "--ratio", dest="matching.ratio", type=float, metavar="R", help="matching.ratio"
    )
    matching.add_argument(
        "--pairs",
        dest="matching.pair_mode",
        choices=["auto", "exhaustive", "sequential"],
        help="matching.pair_mode",
    )
    # Verification options; commands that verify also match, so this includes matching.
    verification = argparse.ArgumentParser(add_help=False, parents=[matching])
    verification.add_argument(
        "--method",
        dest="verification.method",
        choices=["auto", "essential", "fundamental"],
        help="verification.method",
    )
    verification.add_argument(
        "--threshold",
        dest="verification.ransac_threshold_px",
        type=float,
        metavar="PX",
        help="verification.ransac_threshold_px",
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
        "matches",
        parents=[common, detector, draw, matching],
        help="match features between image pairs",
    )
    p.set_defaults(func=_stage_command, stage="matches")

    p = sub.add_parser(
        "verify",
        parents=[common, detector, draw, verification],
        help="match pairs and verify them with epipolar geometry",
    )
    p.set_defaults(func=_stage_command, stage="verify")

    p = sub.add_parser(
        "pose",
        parents=[common, detector, verification],
        help="relative pose of verified pairs; pick the initial pair",
    )
    p.set_defaults(func=_stage_command, stage="pose", draw=0)

    p = sub.add_parser(
        "triangulate",
        parents=[common, detector, verification],
        help="triangulate and filter the initial pair's points",
    )
    p.set_defaults(func=_stage_command, stage="triangulate", draw=0)

    p = sub.add_parser(
        "two-view",
        parents=[common, detector, verification],
        help="initial pair -> coloured sparse point cloud (PLY), cameras.json and a render",
    )
    p.set_defaults(func=_stage_command, stage="cloud", draw=0)

    p = sub.add_parser(
        "reconstruct",
        parents=[common, detector, verification],
        help="register every image (tracks, PnP, bundle adjustment) -> multi-view and "
        "filtered clouds",
    )
    p.set_defaults(func=_stage_command, stage="filter", draw=0)

    p = sub.add_parser("view", help="open a point cloud (PLY) with its cameras in a 3-D viewer")
    p.add_argument("ply", type=Path, help="point cloud, e.g. .../point_clouds/sparse.ply")
    p.add_argument("--cameras", type=Path, help="cameras.json written next to it")
    p.set_defaults(func=_view)

    p = sub.add_parser("calibrate", help="calibrate a camera from checkerboard photos")
    p.add_argument("images", type=Path, help="directory of checkerboard photos")
    p.add_argument("--pattern", default="9x6", help="inner corners, COLSxROWS (9x6)")
    p.add_argument("--square", type=float, default=1.0, help="square size (any unit)")
    p.add_argument("--out", type=Path, default=Path("configs/camera.yaml"))
    p.add_argument("--config", type=Path, help="config (dataset.extensions, calibration)")
    p.set_defaults(func=_calibrate)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


# --- helpers ----------------------------------------------------------------------------


def _config_from_args(args: argparse.Namespace) -> Config:
    """The YAML config with every dotted-key option given on the command line applied."""
    overrides = {key: value for key, value in vars(args).items() if "." in key}
    return load_config(args.config).with_overrides(overrides)


def _report_calibration(config: Config, dataset: Dataset) -> None:
    if dataset.calibration_file and not config.camera.calibration_file:
        print(f"Using calibration {dataset.calibration_file} (found next to the images)")


def _print_dataset_error(exc: DatasetError) -> None:
    print(f"ERROR: {exc}")
    if exc.report is not None:
        print(format_report(exc.report))


# --- commands -----------------------------------------------------------------------------


def _validate(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    try:
        dataset = prepare_dataset(args.images, config)
    except DatasetError as exc:
        _print_dataset_error(exc)
        return 1

    _report_calibration(config, dataset)
    run_dir = create_run_dir(config.output.root, "validate")
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
    """features, matches, verify, pose, triangulate, two-view and reconstruct: run the
    pipeline up to ``args.stage`` and write the outputs of every stage that ran (see
    ``src.outputs``)."""
    config = _config_from_args(args)
    stage: Stage = args.stage
    try:
        result = run_pipeline(args.images, config, until=stage)
    except DatasetError as exc:
        _print_dataset_error(exc)
        return 1

    _report_calibration(config, result.dataset)
    label = args.command.replace("-", "_")
    run_dir = new_run_dir(result, f"{label}_{result.extractor.name}")

    written, status = save_stage_outputs(result, run_dir, stage, draw=args.draw)
    print(f"Wrote {written}")
    return status


def _view(args: argparse.Namespace) -> int:
    from src.visualization.viewer import show

    points, colors = read_ply(args.ply)
    if args.cameras:
        cameras, _, sizes = read_cameras_json(args.cameras)
    else:
        cameras, sizes = [], []
    show(points, colors, cameras, sizes, title=args.ply.name)
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    try:
        cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    except ValueError:
        print(f"ERROR: --pattern must look like 9x6, got {args.pattern!r}")
        return 2
    try:
        config = load_config(args.config)
        paths = discover_images(args.images, config.dataset.extensions)
        images = [r.image for r in iter_images(paths) if isinstance(r, ImageRecord)]
        intrinsics, rms = calibrate_from_checkerboard(
            images, (cols, rows), args.square, config.calibration
        )
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

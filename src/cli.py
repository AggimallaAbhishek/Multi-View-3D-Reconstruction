"""Command-line entry point: ``python -m src.cli <command> ...``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.data.dataset import prepare_dataset
from src.data.loader import DatasetError, ImageRecord, discover_images, iter_images
from src.data.validation import format_report
from src.geometry.calibration import (
    CalibrationError,
    calibrate_from_checkerboard,
    save_camera_file,
)
from src.run import create_run_dir, write_run_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="load, validate and summarise an image folder")
    validate.add_argument("images", type=Path, help="directory of input photos")
    validate.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml)")
    validate.add_argument("--out", type=Path, help="output root (default: config output.root)")

    calibrate = sub.add_parser("calibrate", help="calibrate a camera from checkerboard photos")
    calibrate.add_argument("images", type=Path, help="directory of checkerboard photos")
    calibrate.add_argument("--pattern", default="9x6", help="inner corners, COLSxROWS (9x6)")
    calibrate.add_argument("--square", type=float, default=1.0, help="square size (any unit)")
    calibrate.add_argument("--out", type=Path, default=Path("configs/camera.yaml"))

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "validate":
        return _validate(args)
    if args.command == "calibrate":
        return _calibrate(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        dataset = prepare_dataset(args.images, config)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        if exc.report is not None:
            print(format_report(exc.report))
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


def _calibrate(args: argparse.Namespace) -> int:
    try:
        cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    except ValueError:
        print(f"ERROR: --pattern must look like 9x6, got {args.pattern!r}")
        return 2
    try:
        paths = discover_images(args.images, (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
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

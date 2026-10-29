"""Command-line entry point: ``python -m src.cli <command> ...``."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from src.config import Config, load_config
from src.data.dataset import Dataset, prepare_dataset
from src.data.loader import DatasetError, ImageRecord, discover_images, iter_images
from src.data.validation import format_report
from src.features.extractor import (
    FeatureCache,
    create_extractor,
    extract_features,
    write_feature_statistics,
)
from src.features.matcher import (
    DescriptorMatcher,
    Matches,
    connectivity,
    match_matrix,
    match_pairs,
    save_matches,
    select_pairs,
    write_match_statistics,
)
from src.geometry.calibration import (
    CalibrationError,
    calibrate_from_checkerboard,
    save_camera_file,
)
from src.geometry.verification import (
    inlier_matches,
    verification_matrix,
    verify_matches,
    write_verification_statistics,
)
from src.run import create_run_dir, write_run_metadata
from src.visualization.epipolar import draw_epipolar_lines
from src.visualization.features import draw_keypoints
from src.visualization.matches import draw_matches, plot_match_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="load, validate and summarise an image folder")
    validate.add_argument("images", type=Path, help="directory of input photos")
    validate.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml)")
    validate.add_argument("--out", type=Path, help="output root (default: config output.root)")

    features = sub.add_parser("features", help="detect and describe keypoints in every image")
    features.add_argument("images", type=Path, help="directory of input photos")
    features.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml)")
    features.add_argument("--out", type=Path, help="output root (default: config output.root)")
    features.add_argument("--detector", choices=["sift", "orb"], help="override features.detector")
    features.add_argument("--draw", type=int, default=3, help="keypoint images to save (3)")

    matches = sub.add_parser("matches", help="match features between image pairs")
    matches.add_argument("images", type=Path, help="directory of input photos")
    matches.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml)")
    matches.add_argument("--out", type=Path, help="output root (default: config output.root)")
    matches.add_argument("--detector", choices=["sift", "orb"], help="override features.detector")
    matches.add_argument("--ratio", type=float, help="override matching.ratio")
    matches.add_argument(
        "--pairs", choices=["exhaustive", "sequential"], help="override matching.pair_mode"
    )
    matches.add_argument("--draw", type=int, default=3, help="match images to save (3)")

    verify = sub.add_parser("verify", help="match pairs and verify them with epipolar geometry")
    verify.add_argument("images", type=Path, help="directory of input photos")
    verify.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml)")
    verify.add_argument("--out", type=Path, help="output root (default: config output.root)")
    verify.add_argument("--detector", choices=["sift", "orb"], help="override features.detector")
    verify.add_argument(
        "--method", choices=["auto", "essential", "fundamental"], help="override model"
    )
    verify.add_argument("--threshold", type=float, help="override ransac_threshold_px")
    verify.add_argument("--draw", type=int, default=3, help="pairs to draw (3)")

    calibrate = sub.add_parser("calibrate", help="calibrate a camera from checkerboard photos")
    calibrate.add_argument("images", type=Path, help="directory of checkerboard photos")
    calibrate.add_argument("--pattern", default="9x6", help="inner corners, COLSxROWS (9x6)")
    calibrate.add_argument("--square", type=float, default=1.0, help="square size (any unit)")
    calibrate.add_argument("--out", type=Path, default=Path("configs/camera.yaml"))

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "validate":
        return _validate(args)
    if args.command == "features":
        return _features(args)
    if args.command == "matches":
        return _matches(args)
    if args.command == "verify":
        return _verify(args)
    if args.command == "calibrate":
        return _calibrate(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _load_dataset(images: Path, config: Config) -> Dataset | None:
    try:
        return prepare_dataset(images, config)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        if exc.report is not None:
            print(format_report(exc.report))
        return None


def _validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dataset = _load_dataset(args.images, config)
    if dataset is None:
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


def _override(config: Config, section: str, **values) -> Config:
    """Config with command-line overrides applied; ``None`` values are ignored."""
    values = {k: v for k, v in values.items() if v is not None}
    if not values:
        return config
    return dataclasses.replace(
        config, **{section: dataclasses.replace(getattr(config, section), **values)}
    )


def _extract(dataset: Dataset, config: Config):
    extractor = create_extractor(config.features)
    cache = FeatureCache(config.features.cache_dir) if config.features.cache_dir else None
    features, stats = extract_features(dataset.images, extractor, cache)
    return extractor, features, stats


def _features(args: argparse.Namespace) -> int:
    config = _override(load_config(args.config), "features", detector=args.detector)
    dataset = _load_dataset(args.images, config)
    if dataset is None:
        return 1

    extractor, features, stats = _extract(dataset, config)

    run_dir = create_run_dir(args.out or config.output.root, f"features_{extractor.name}")
    write_run_metadata(run_dir, config)
    csv_path = write_feature_statistics(stats, run_dir / "features" / "feature_statistics.csv")
    for im, f in list(zip(dataset.images, features, strict=True))[: max(args.draw, 0)]:
        out = run_dir / "features" / f"{Path(im.name).stem}_keypoints.png"
        cv2.imwrite(str(out), draw_keypoints(im.image, f))

    counts = np.array([s.n_features for s in stats])
    coverage = np.array([s.coverage for s in stats])
    new = [s.time_ms for s in stats if not s.cached]
    print(f"Detector: {extractor.signature()}")
    print(
        f"Features per image: min {counts.min()}, median {int(np.median(counts))}, "
        f"max {counts.max()} (cap {extractor.max_features})"
    )
    print(f"Grid coverage: min {coverage.min():.2f}, median {np.median(coverage):.2f}")
    print(
        f"Computed {len(new)} image(s)"
        + (f" at {np.mean(new):.0f} ms each" if new else "")
        + f", {len(stats) - len(new)} from cache"
    )
    for s in stats:
        if s.n_features < 100 or s.coverage < 0.25:
            print(
                f"[WARNING] {s.image}: {s.n_features} features, coverage {s.coverage:.2f} "
                "- too little texture or features concentrated in one region"
            )
    print(f"Wrote {csv_path}")
    return 0


def _inspection_order(matches: list[Matches]) -> list[Matches]:
    """Neighbouring images first: they are the pairs most worth looking at."""
    return sorted(matches, key=lambda m: (m.j - m.i, m.i))


def _run_matching(args: argparse.Namespace, config: Config, label: str):
    """Shared by ``matches`` and ``verify``: features, matching, outputs and summary.

    Returns (dataset, features, matches, run_dir) or None if the dataset is unusable.
    """
    dataset = _load_dataset(args.images, config)
    if dataset is None:
        return None

    extractor, features, feature_stats = _extract(dataset, config)
    pairs = select_pairs(len(features), config.matching)
    matches, stats = match_pairs(features, pairs, DescriptorMatcher(config.matching))

    names = [im.name for im in dataset.images]
    run_dir = create_run_dir(args.out or config.output.root, f"{label}_{extractor.name}")
    write_run_metadata(run_dir, config)
    write_feature_statistics(feature_stats, run_dir / "features" / "feature_statistics.csv")
    out = run_dir / "matches"
    write_match_statistics(stats, names, out / "match_statistics.csv")
    save_matches(matches, out / "matches.npz")
    plot_match_matrix(match_matrix(stats, len(names)), names, out / "match_matrix.png")
    if label == "matches":
        for m in _inspection_order(matches)[: max(args.draw, 0)]:
            a, b = dataset.images[m.i], dataset.images[m.j]
            canvas = draw_matches(a.image, b.image, features[m.i], features[m.j], m)
            cv2.imwrite(str(out / f"{_pair_stem(a.name, b.name)}_matches.png"), canvas)

    final = np.array([s.final for s in stats])
    min_matches = config.matching.min_matches
    good = final >= min_matches
    print(f"Detector: {extractor.signature()}; ratio {config.matching.ratio}, "
          f"mutual {config.matching.mutual}, {config.matching.method}")  # fmt: skip
    print(
        f"Pairs: {len(pairs)} ({config.matching.pair_mode}) in "
        f"{sum(s.time_ms for s in stats) / 1000:.1f} s; "
        f"{good.sum()} with >= {min_matches} matches"
    )
    if good.any():
        print(
            f"Matches per connected pair: min {final[good].min()}, "
            f"median {int(np.median(final[good]))}, max {final[good].max()}"
        )
    best = connectivity(stats, len(names))
    for k in np.flatnonzero(best < min_matches):
        print(
            f"[WARNING] {names[k]}: best pair has only {best[k]} matches - "
            "it will probably not be registered"
        )
    return dataset, features, matches, run_dir


def _pair_stem(name_i: str, name_j: str) -> str:
    return f"{Path(name_i).stem}_{Path(name_j).stem}"


def _matches(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config = _override(config, "features", detector=args.detector)
    config = _override(config, "matching", ratio=args.ratio, pair_mode=args.pairs)
    result = _run_matching(args, config, "matches")
    if result is None:
        return 1
    print(f"Wrote {result[3] / 'matches' / 'match_statistics.csv'}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config = _override(config, "features", detector=args.detector)
    config = _override(
        config, "verification", method=args.method, ransac_threshold_px=args.threshold
    )
    result = _run_matching(args, config, "verify")
    if result is None:
        return 1
    dataset, features, matches, run_dir = result
    names = [im.name for im in dataset.images]

    geometries, stats = verify_matches(dataset.images, features, matches, config.verification)
    out = run_dir / "verification"
    csv_path = write_verification_statistics(stats, names, out / "verification_statistics.csv")
    verified = [inlier_matches(m, g) for m, g in zip(matches, geometries, strict=True)]
    save_matches(verified, out / "verified_matches.npz")
    plot_match_matrix(
        verification_matrix(stats, len(names)),
        names,
        out / "verified_matrix.png",
        title="Verified inliers per image pair",
        label="geometric inliers",
    )
    by_pair = {(m.i, m.j): (m, g) for m, g in zip(matches, geometries, strict=True)}
    for m in _inspection_order(matches)[: max(args.draw, 0)]:
        g = by_pair[(m.i, m.j)][1]
        a, b = dataset.images[m.i], dataset.images[m.j]
        stem = _pair_stem(a.name, b.name)
        canvas = draw_matches(
            a.image, b.image, features[m.i], features[m.j], m, inlier_mask=g.inlier_mask
        )
        cv2.imwrite(str(out / f"{stem}_inliers.png"), canvas)
        if g.F is not None and g.n_inliers:
            x_i, x_j = m.points(features[m.i], features[m.j])
            canvas = draw_epipolar_lines(
                a.image, b.image, x_i[g.inlier_mask], x_j[g.inlier_mask], g.F
            )
            cv2.imwrite(str(out / f"{stem}_epipolar.png"), canvas)

    ok = [s for s in stats if s.success]
    models = sorted({s.model for s in stats})
    v = config.verification
    print(
        f"Verification: {', '.join(models)} ({v.estimator}, threshold {v.ransac_threshold_px} px)"
        f" in {sum(s.time_ms for s in stats) / 1000:.1f} s"
    )
    print(f"Verified pairs: {len(ok)} of {len(stats)} (>= {v.min_inliers} inliers)")
    if ok:
        ratios = np.array([s.inlier_ratio for s in ok])
        residuals = np.array([s.median_residual for s in ok])
        planar = sum(s.homography_ratio > v.planar_ratio for s in ok)
        print(
            f"Inlier ratio: min {ratios.min():.2f}, median {np.median(ratios):.2f}; "
            f"median residual {np.median(residuals):.2f} px"
        )
        print(
            f"Pairs explained by a homography (H/F > {v.planar_ratio}): {planar} "
            "- planar scene or rotation only, weak for initialisation"
        )
    best = np.zeros(len(names), dtype=int)
    for s in ok:
        best[s.i] = max(best[s.i], s.n_inliers)
        best[s.j] = max(best[s.j], s.n_inliers)
    for k in np.flatnonzero(best == 0):
        print(f"[WARNING] {names[k]}: no verified pair - it cannot be registered")
    print(f"Wrote {csv_path}")
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

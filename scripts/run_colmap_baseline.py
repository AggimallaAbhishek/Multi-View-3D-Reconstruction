"""E6 baseline: our sparse reconstruction vs COLMAP's (EXTERNAL) on the same images.

    python scripts/run_colmap_baseline.py data/raw/templeRing [--config CONFIG]

Both start from the same prepared images (our loading, resizing and undistortion) and
the same intrinsics, and both refine the focal length (``--fix-intrinsics``: COLMAP's is
held fixed). Camera errors are measured
against ``gt_cameras.json`` after a similarity alignment. Writes ``comparison.json``,
COLMAP's cloud (``colmap_external.ply``) and its model (text format) into a run directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evaluation.ground_truth import (  # noqa: E402
    GT_CAMERAS_FILE,
    at_working_resolution,
    load_cameras_json,
)
from src.evaluation.metrics import camera_pose_errors  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.pointcloud.export import write_ply  # noqa: E402
from src.reconstruction.colmap_runner import ColmapError, run_colmap  # noqa: E402
from src.run import create_run_dir, write_json, write_run_metadata  # noqa: E402


def summarise(cameras: dict, points: np.ndarray, errors: np.ndarray, truth: dict, n: int,
              seconds: float) -> dict:  # fmt: skip
    names = sorted(cameras)
    pose = camera_pose_errors([cameras[k] for k in names], [truth[k] for k in names])
    return {
        "registered": f"{len(names)}/{n}",
        "rotation_error_median_deg": float(np.median(pose.rotation_deg)),
        "rotation_error_max_deg": float(np.max(pose.rotation_deg)),
        "centre_error_median": float(np.median(pose.centre)),
        "focal_px": float(np.median([cameras[k].K[0, 0] for k in names])),
        "points": len(points),
        "reprojection_mean_px": float(np.mean(errors)),
        "time_s": seconds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", type=Path)
    parser.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml)")
    parser.add_argument(
        "--fix-intrinsics",
        action="store_true",
        help="hold COLMAP's focal length fixed (default: refined, as ours)",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    gt = load_cameras_json(args.images / GT_CAMERAS_FILE)

    start = time.perf_counter()
    result = run_pipeline(args.images, config, until="filter")
    ours_s = time.perf_counter() - start
    rec = result.reconstruction
    if rec is None:
        print("our pipeline found no initial pair")
        return 1
    images = result.images
    truth = {im.name: at_working_resolution(gt[im.name], im.scale) for im in images}
    if any(not np.allclose(im.K, images[0].K) for im in images):
        print("COLMAP baseline expects one camera (identical intrinsics)")
        return 1

    run_dir = create_run_dir(config.output.root, "colmap_baseline")
    write_run_metadata(run_dir, config)
    start = time.perf_counter()
    try:
        colmap = run_colmap(
            {im.name: im.image for im in images}, images[0].K, run_dir / "colmap",
            config.features.max_features, refine_intrinsics=not args.fix_intrinsics,
        )  # fmt: skip
    except ColmapError as exc:
        print(f"COLMAP baseline not run: {exc}")
        return 1
    colmap_s = time.perf_counter() - start
    write_ply(run_dir / "colmap_external.ply", colmap.points, colmap.colors / 255.0)

    keep = result.filtering.keep
    ours_cams = {result.names[m]: rec.registry.camera(m) for m in rec.registry.images}
    colmap_label = "colmap (external)" + (", K fixed" if args.fix_intrinsics else "")
    table = {
        "ours": summarise(ours_cams, rec.landmarks.position[keep], rec.landmarks.error[keep],
                          truth, len(images), ours_s),
        colmap_label: summarise(colmap.cameras, colmap.points, colmap.errors, truth,
                                len(images), colmap_s),
    }  # fmt: skip
    write_json(run_dir / "comparison.json", table)
    rows = list(next(iter(table.values())))
    print(f"{'':28s}" + "".join(f"{name:>30s}" for name in table))
    for row in rows:
        cells = [table[name][row] for name in table]
        print(f"{row:28s}" + "".join(
            f"{c:>30}" if isinstance(c, str | int) else f"{c:>30.4g}" for c in cells))  # fmt: skip
    print(f"Wrote {run_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare the pipeline's relative poses, initial structure and multi-view
reconstruction with ground truth.

Usage:
    python scripts/evaluate_against_ground_truth.py data/raw/templeRing [--config CONFIG]

The folder must contain ``gt_cameras.json`` (see scripts/download_middlebury.py) and,
for calibrated runs, ``camera.yaml``. Prints rotation and translation-direction errors
over all successfully estimated pairs and for the initial pair, and the 3-D error of the
initial pair's triangulated points relative to the scene size, then how many images the
incremental reconstruction registers, their pose errors after similarity alignment and
the final reprojection error. Writes per-pair, per-point and per-camera CSVs into a run
directory under outputs/.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evaluation.ground_truth import GT_CAMERAS_FILE, load_cameras_json, relative_pose  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    camera_pose_errors,
    rotation_error_deg,
    translation_direction_error_deg,
    two_view_structure_errors,
)
from src.geometry.reprojection import ReprojectionStats  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.run import create_run_dir, write_csv, write_run_metadata  # noqa: E402


@dataclass(frozen=True)
class PairError:
    image_i: str
    image_j: str
    inliers: int
    parallax_deg: float
    rotation_error_deg: float
    translation_error_deg: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", type=Path)
    parser.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    gt = load_cameras_json(args.images / GT_CAMERAS_FILE)

    result = run_pipeline(args.images, config, until="reconstruct")
    names = result.names
    rows: list[PairError] = []
    for m, g, pose in zip(result.matches, result.geometries, result.poses, strict=True):
        if pose is None or not pose.success:
            continue
        R_true, t_true = relative_pose(gt[names[m.i]], gt[names[m.j]])
        rows.append(
            PairError(
                names[m.i], names[m.j], g.n_inliers, pose.median_parallax_deg,
                rotation_error_deg(pose.R, R_true),
                translation_direction_error_deg(pose.t, t_true),
            )
        )  # fmt: skip
    if not rows:
        print("no pair produced a relative pose")
        return 1

    run_dir = create_run_dir(config.output.root, "evaluate_ground_truth")
    write_run_metadata(run_dir, config)
    csv_path = write_csv(
        run_dir / "relative_pose_errors.csv",
        ["image_i", "image_j", "inliers", "parallax_deg", "rotation_error_deg",
         "translation_error_deg"],
        (
            [e.image_i, e.image_j, e.inliers, f"{e.parallax_deg:.3f}",
             f"{e.rotation_error_deg:.4f}", f"{e.translation_error_deg:.4f}"]
            for e in rows
        ),
    )  # fmt: skip

    rot = np.array([e.rotation_error_deg for e in rows])
    trans = np.array([e.translation_error_deg for e in rows])
    verified = sum(g.success for g in result.geometries)
    print(f"{len(names)} images, {len(result.matches)} pairs, {verified} verified, "
          f"{len(rows)} with a pose")  # fmt: skip
    for label, e in (("rotation", rot), ("translation direction", trans)):
        print(f"{label} error (deg): median {np.median(e):.2f}, "
              f"90th percentile {np.percentile(e, 90):.2f}, max {e.max():.2f}")  # fmt: skip
    init = result.initial_pair
    if init is not None:
        R_true, t_true = relative_pose(gt[names[init.i]], gt[names[init.j]])
        print(
            f"initial pair {names[init.i]} - {names[init.j]}: rotation error "
            f"{rotation_error_deg(init.pose.R, R_true):.2f} deg, translation direction error "
            f"{translation_direction_error_deg(init.pose.t, t_true):.2f} deg"
        )
    s = result.structure
    if s is not None:
        tri = s.triangulation
        structure = two_view_structure_errors(
            s, gt[names[s.i]], gt[names[s.j]],
            scale_i=result.images[s.i].scale, scale_j=result.images[s.j].scale,
        )  # fmt: skip
        errors = structure.errors
        counts = tri.counts
        print(
            f"initial structure: {counts['valid']} of {counts['points']} points valid; "
            f"reprojection median {tri.reprojection.median:.2f} px, "
            f"90th percentile {tri.reprojection.p90:.2f} px"
        )
        print(
            f"3-D shape error vs ground truth after similarity alignment (fraction of scene "
            f"size): median {np.median(errors):.3%}, 90th percentile "
            f"{np.percentile(errors, 90):.3%}; scale vs true baseline {structure.scale_ratio:.3f}"
        )
        write_csv(
            run_dir / "structure_errors.csv",
            ["point", "relative_3d_error", "angle_deg", "max_reprojection_px"],
            (
                [k, f"{e:.6f}", f"{a:.3f}", f"{r:.4f}"]
                for k, e, a, r in zip(
                    np.flatnonzero(tri.valid), errors, tri.angles_deg[tri.valid],
                    tri.max_errors[tri.valid], strict=True,
                )
            ),
        )  # fmt: skip
    if result.reconstruction is not None:
        evaluate_reconstruction(result, gt, run_dir)
    print(f"Wrote {csv_path}")
    return 0


def evaluate_reconstruction(result, gt, run_dir: Path) -> None:
    """Registered images, camera pose errors after aligning to ground truth, reprojection."""
    rec, names = result.reconstruction, result.names
    images = rec.registry.images
    if len(images) < 3:  # a similarity alignment needs at least 3 camera centres
        print(f"reconstruction: only {len(images)} of {len(names)} images registered")
        return
    cameras = [rec.registry.camera(m) for m in images]
    errors = camera_pose_errors(cameras, [gt[names[m]] for m in images])
    rp = ReprojectionStats.from_errors(rec.reprojection_errors(result.features))
    print(
        f"reconstruction: {len(images)} of {len(names)} images registered "
        f"({len(images) / len(names):.0%}), {len(rec.landmarks)} landmarks, {rp.n} observations"
    )
    for label, e, fmt in (
        ("camera rotation error (deg)", errors.rotation_deg, "{:.2f}"),
        ("camera centre error (fraction of camera extent)", errors.centre, "{:.2%}"),
    ):
        print(
            f"{label}: median {fmt.format(np.median(e))}, 90th percentile "
            f"{fmt.format(np.percentile(e, 90))}, max {fmt.format(e.max())}"
        )
    print(
        f"reprojection error: mean {rp.mean:.2f} px, median {rp.median:.2f} px, "
        f"90th percentile {rp.p90:.2f} px, max {rp.max:.2f} px"
    )
    missing = [names[m] for m in rec.unregistered(len(names))]
    if missing:
        print(f"not registered: {', '.join(missing)}")
    write_csv(
        run_dir / "camera_pose_errors.csv",
        ["order", "image", "rotation_error_deg", "centre_error_fraction"],
        (
            [k, names[m], f"{r:.4f}", f"{c:.6f}"]
            for k, (m, r, c) in enumerate(
                zip(images, errors.rotation_deg, errors.centre, strict=True)
            )
        ),
    )


if __name__ == "__main__":
    sys.exit(main())

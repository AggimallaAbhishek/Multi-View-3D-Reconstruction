"""Run the experiments (configs/experiments/*.yaml) and draw their reports.

Usage:
    python scripts/run_experiments.py [SPEC ...] [--images DIR] [--config CONFIG] [--out DIR]

Defaults: every spec in configs/experiments/, images data/raw/templeRing, the default
config, output under outputs/experiments/. With a ``gt_cameras.json`` next to the images,
runs are scored against ground truth: camera pose errors and, against a reference cloud
(every verified track triangulated with the true cameras), completeness and accuracy.
Each experiment gets a directory with one sub-directory per run,
experiment_results.csv, results.md, metrics.png and runtime.png.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evaluation.experiments import load_experiment, reference_cloud, run_experiment  # noqa: E402
from src.evaluation.ground_truth import load_cameras_json  # noqa: E402
from src.evaluation.reports import make_report  # noqa: E402
from src.pointcloud.export import write_ply  # noqa: E402
from src.run import create_run_dir, write_run_metadata  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("specs", nargs="*", type=Path, help="experiment YAML files")
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw/templeRing")
    parser.add_argument("--config", type=Path, help="base config (default: configs/default.yaml)")
    parser.add_argument("--out", type=Path, default=Path("outputs/experiments"))
    args = parser.parse_args(argv)

    specs = args.specs or sorted((ROOT / "configs/experiments").glob("*.yaml"))
    experiments = [load_experiment(p) for p in specs]  # fail on a bad spec before running
    config = load_config(args.config)
    out = create_run_dir(args.out, "experiments")
    write_run_metadata(out, config)

    gt_path = args.images / "gt_cameras.json"
    gt = load_cameras_json(gt_path) if gt_path.is_file() else None
    reference = None
    if gt is not None:
        reference = reference_cloud(args.images, config, gt)
        write_ply(out / "reference.ply", reference)
        print(f"Reference: {len(reference)} points from ground-truth cameras")
    else:
        print("No gt_cameras.json: internal metrics only (no pose errors, no completeness)")

    for exp in experiments:
        start = time.perf_counter()
        print(f"\n{exp.name}: {exp.question} ({len(exp.settings)} runs)")
        rows = run_experiment(exp, args.images, config, out, gt=gt, reference=reference)
        for row in rows:
            print(
                f"  {row['setting']:<40} {row['status']:<8} registered {row['registered']}"
                f"/{row['images']}, points {row['filtered_points']}, "
                f"completeness {row['completeness']:.3f}, "
                f"rotation {row['rotation_error_median_deg']:.2f} deg"
            )
        make_report(out / exp.name)
        print(f"  {time.perf_counter() - start:.0f} s -> {out / exp.name}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

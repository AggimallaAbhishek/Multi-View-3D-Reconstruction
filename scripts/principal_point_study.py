"""Would refining the principal point help? After a reconstruction, one extra global bundle
adjustment with a principal-point offset shared by all images (plus the focal scale)
free; camera rotation errors against ground truth before and after.

    python scripts/principal_point_study.py data/raw/templeRing calibrated|guessed

"guessed" ignores camera.yaml (the fallback K, principal point at the image centre).
A measurement script, not part of the pipeline (docs/experiments.md, E3b).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evaluation.ground_truth import (  # noqa: E402
    GT_CAMERAS_FILE,
    at_working_resolution,
    load_cameras_json,
)
from src.evaluation.metrics import camera_pose_errors  # noqa: E402
from src.geometry.camera import Camera  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

# Coupled to bundle_adjustment's parameter layout (cameras, points, focal scale) and
# residual order (x, y per observation).
from src.reconstruction.bundle_adjustment import (  # noqa: E402
    jacobian_structure,
    pack,
    residuals,
    unpack,
)


def refine_with_principal_point(cameras, obs, X, ba):
    """Global bundle adjustment (first camera fixed) over poses, points, the shared focal
    scale and a shared principal-point offset (dx, dy)."""
    K = np.stack([c.K for c in cameras])
    full = np.concatenate([pack(cameras, X, 1.0), [0.0, 0.0]])
    free = np.ones(len(full), bool)
    free[:6] = False

    def split(x):
        p = full.copy()
        p[free] = x
        return p[:-2], p[-2:]

    def f(x):
        p, d = split(x)
        Kd = K.copy()
        Kd[:, :2, 2] += d
        return residuals(p, Kd, obs, refine_focal=True)

    # Sparsity: the usual camera/point/focal pattern plus the offset's two columns
    # (dx for every x residual, dy for every y residual).
    rows, cols, _ = jacobian_structure(obs.view, obs.point, len(cameras), len(X), True)
    m, n = 2 * len(obs.view), len(full)
    rows = np.concatenate([rows, np.arange(0, m, 2), np.arange(1, m, 2)])
    cols = np.concatenate([cols, np.full(m // 2, n - 2), np.full(m // 2, n - 1)])
    sparsity = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, n)).tocsc()[:, free]
    solution = least_squares(
        f, full[free], jac_sparsity=sparsity, loss=ba.loss, f_scale=ba.loss_scale_px,
        x_scale="jac", method="trf", max_nfev=ba.max_iterations, ftol=ba.function_tolerance,
    )  # fmt: skip
    p, d = split(solution.x)
    cam, _, s = unpack(p, len(cameras), True)
    refined = []
    for v, c in enumerate(cameras):
        Kn = c.K @ np.diag([s, s, 1.0])
        Kn[:2, 2] += d
        refined.append(Camera(Kn, cv2.Rodrigues(cam[v, :3])[0], cam[v, 3:]))
    return refined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", type=Path)
    parser.add_argument("mode", choices=["calibrated", "guessed"])
    args = parser.parse_args(argv)

    overrides = {} if args.mode == "calibrated" else {"camera.auto_calibration_file": False}
    config = load_config().with_overrides(overrides)
    result = run_pipeline(args.images, config, until="reconstruct")
    rec = result.reconstruction
    if rec is None:
        print("no reconstruction")
        return 1
    gt = load_cameras_json(args.images / GT_CAMERAS_FILE)
    truth = [at_working_resolution(gt[result.names[m]], result.images[m].scale)
             for m in rec.registry.images]  # fmt: skip
    cameras, obs, X = rec.landmarks.as_problem(rec.registry, result.features)
    refined = refine_with_principal_point(cameras, obs, X, config.bundle_adjustment)
    print(f"true: f {truth[0].K[0, 0]:.1f}, principal point {truth[0].K[:2, 2].round(1)}")
    for label, cams in (("principal point fixed", cameras), ("principal point refined", refined)):
        errors = camera_pose_errors(cams, truth)
        print(f"{label:24s} rotation error median {np.median(errors.rotation_deg):.3f} deg, "
              f"f {cams[0].K[0, 0]:.1f}, principal point {cams[0].K[:2, 2].round(1)}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Radial lens distortion estimated from a reconstruction (self-calibration).

One global bundle adjustment over poses, points, the shared focal scale and two radial
coefficients shared by every image, OpenCV's model with the tangential terms zero:
``x_d = x (1 + k1 r^2 + k2 r^4)`` in normalised coordinates. The pipeline then
undistorts the images with the estimate and reconstructs again (``run_pipeline``):
every other stage keeps its pinhole model.

The Jacobian is the bundle adjustment's sparsity pattern plus two dense columns,
evaluated by finite differences (about 2 s on templeRing).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from src.config import BundleAdjustmentConfig
from src.geometry.camera import Camera
from src.geometry.triangulation import Observations
from src.reconstruction.bundle_adjustment import (
    free_parameters,
    jacobian_structure,
    pack,
    residuals,
    rms,
    scale_focal,
    solver_options,
)


@dataclass(frozen=True)
class RadialDistortion:
    k1: float
    k2: float

    @property
    def coefficients(self) -> np.ndarray:
        """OpenCV's ``(k1, k2, p1, p2, k3)``."""
        return np.array([self.k1, self.k2, 0.0, 0.0, 0.0])

    def apply(self, xn: np.ndarray) -> np.ndarray:
        """Distort normalised image coordinates (N, 2)."""
        r2 = np.sum(xn**2, axis=1, keepdims=True)
        return xn * (1 + self.k1 * r2 + self.k2 * r2**2)

    def corner_shift_px(self, K: np.ndarray, size: tuple[int, int]) -> float:
        """How far the distortion moves the image corner farthest from the principal
        point (pixels): the largest displacement anywhere in the image."""
        xn = _corners(K, size)
        shift = (self.apply(xn) - xn) * np.diag(K)[:2]
        return float(np.max(np.linalg.norm(shift, axis=1)))

    @staticmethod
    def field_radius(K: np.ndarray, size: tuple[int, int]) -> float:
        """Normalised radius of the image corner farthest from the principal point: how
        wide a view the distortion has to be observed over."""
        return float(np.max(np.linalg.norm(_corners(K, size), axis=1)))

    def then(self, residual: RadialDistortion) -> RadialDistortion:
        """The distortion after undistorting by ``self`` left ``residual``: to first order
        in the (small) residual the coefficients add."""
        return RadialDistortion(self.k1 + residual.k1, self.k2 + residual.k2)


def _corners(K: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Normalised coordinates (4, 2) of the four image corners."""
    w, h = size
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    return (corners - K[:2, 2]) / np.diag(K)[:2]


@dataclass(frozen=True)
class DistortionEstimate:
    distortion: RadialDistortion
    focal_scale: float  # refined / starting focal length
    initial_rms_px: float
    final_rms_px: float
    shift_px: float  # how far it moves the farthest image corner (pixels)


@dataclass(frozen=True)
class SelfCalibration:
    """What the pipeline estimated and applied."""

    estimates: list[DistortionEstimate]  # one per reconstruction, in order
    distortion: RadialDistortion  # applied to the images (zero if none was)
    focal_scale: float  # applied: the final pass's starting K / the first pass's
    passes: int  # reconstructions kept after the first (a worse pass is discarded)

    def summary(self) -> dict:
        return {
            "distortion": {"k1": self.distortion.k1, "k2": self.distortion.k2},
            "focal_scale": self.focal_scale,
            "passes": self.passes,
            "estimates": [
                {"k1": e.distortion.k1, "k2": e.distortion.k2, "focal_scale": e.focal_scale,
                 "corner_shift_px": e.shift_px, "rms_px": [e.initial_rms_px, e.final_rms_px]}
                for e in self.estimates
            ],
        }  # fmt: skip


def estimate_radial_distortion(
    cameras: Sequence[Camera],
    obs: Observations,
    points: np.ndarray,
    size: tuple[int, int],
    config: BundleAdjustmentConfig,
    fixed: Collection[int] = (0,),
) -> DistortionEstimate:
    """Refine every pose, point, one focal scale and ``(k1, k2)`` (shared by all
    cameras, starting at zero) against the observations; ``fixed`` cameras hold the
    gauge. ``size`` (working resolution) sets the reported corner shift."""
    K = np.stack([c.K for c in cameras])
    full = np.concatenate([pack(cameras, points, 1.0), [0.0, 0.0]])  # ..., s, k1, k2
    free = free_parameters(len(full), fixed)

    def with_free(x: np.ndarray) -> np.ndarray:
        p = full.copy()
        p[free] = x
        return p

    def f(x: np.ndarray) -> np.ndarray:
        p = with_free(x)
        return residuals(p[:-2], K, obs, refine_focal=True,
                         distortion=RadialDistortion(*p[-2:]).apply)  # fmt: skip

    # The bundle adjustment's pattern plus k1 and k2, on which every residual depends.
    rows, cols, _ = jacobian_structure(obs.view, obs.point, len(cameras), len(points), True)
    m, n = 2 * len(obs.view), len(full)
    rows = np.concatenate([rows, np.arange(m), np.arange(m)])
    cols = np.concatenate([cols, np.full(m, n - 2), np.full(m, n - 1)])
    sparsity = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, n)).tocsc()[:, free]
    x0 = full[free]
    solution = least_squares(f, x0, jac_sparsity=sparsity, **solver_options(config))
    *_, s, k1, k2 = with_free(solution.x)
    distortion = RadialDistortion(float(k1), float(k2))
    return DistortionEstimate(
        distortion, float(s), rms(f(x0)), rms(solution.fun),
        distortion.corner_shift_px(scale_focal(K[0], s), size),
    )  # fmt: skip

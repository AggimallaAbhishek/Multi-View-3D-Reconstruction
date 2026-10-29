"""Accuracy metrics against ground truth: relative poses and two-view structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.ground_truth import at_working_resolution, relative_pose
from src.geometry.camera import Camera
from src.geometry.epipolar import angle_between_deg
from src.geometry.pose_estimation import rotation_angle_deg
from src.geometry.triangulation import triangulate_dlt
from src.reconstruction.two_view import TwoViewStructure


def rotation_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> float:
    """Angle of the rotation taking R_est to R_true, in degrees."""
    return rotation_angle_deg(R_est.T @ R_true)


def translation_direction_error_deg(t_est: np.ndarray, t_true: np.ndarray) -> float:
    """Angle between translation directions (translation is only known up to scale)."""
    return float(angle_between_deg(np.ravel(t_est), np.ravel(t_true)))


@dataclass(frozen=True, eq=False)
class Similarity:
    """x -> scale * R @ x + t."""

    scale: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, X: np.ndarray) -> np.ndarray:
        return self.scale * X @ self.R.T + self.t


def align_similarity(source: np.ndarray, target: np.ndarray) -> Similarity:
    """Least-squares similarity mapping ``source`` onto ``target`` (Umeyama 1991), (N, 3)."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    a, b = source - mu_s, target - mu_t
    U, D, Vt = np.linalg.svd(b.T @ a / len(source))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))  # proper rotation, no reflection
    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / (a**2).sum(axis=1).mean())
    return Similarity(scale, R, mu_t - scale * R @ mu_s)


@dataclass(frozen=True, eq=False)
class StructureErrors:
    errors: np.ndarray  # (N,) per-point 3-D error after alignment, fraction of scene size
    scale_ratio: float  # aligned scale / true baseline length (1 = baseline scale correct)


def two_view_structure_errors(
    structure: TwoViewStructure,
    gt_i: Camera,
    gt_j: Camera,
    *,
    scale_i: tuple[float, float],
    scale_j: tuple[float, float],
) -> StructureErrors:
    """Shape error of the valid points, as a fraction of the scene's size.

    The same observations are triangulated with the ground-truth cameras; our points are
    mapped onto them with the best similarity transform, because a monocular two-view
    reconstruction is only defined up to rotation, translation and scale. (Fixing the
    scale to the true baseline instead would turn a sub-degree translation-direction
    error into a large uniform "error" for distant scenes.) ``scale_ratio`` reports how
    far the reconstruction's scale is from unit-baseline-times-true-baseline.

    ``gt_i``/``gt_j`` are at the original image resolution; ``scale_i``/``scale_j`` are
    the images' ``PreparedImage.scale`` (required: the observations are at working
    resolution, so ground-truth intrinsics must be rescaled to match).
    """
    cameras = [at_working_resolution(gt_i, scale_i), at_working_resolution(gt_j, scale_j)]
    valid = structure.triangulation.valid
    X_true = triangulate_dlt(cameras, structure.observations, len(valid))[valid]
    sim = align_similarity(structure.points, X_true)
    extent = np.linalg.norm(np.ptp(X_true, axis=0))
    errors = np.linalg.norm(sim.apply(structure.points) - X_true, axis=1) / extent
    _, t_true = relative_pose(gt_i, gt_j)
    return StructureErrors(errors, sim.scale / np.linalg.norm(t_true))

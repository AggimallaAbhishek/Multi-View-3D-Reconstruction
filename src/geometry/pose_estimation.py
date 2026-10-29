"""Relative camera pose from the essential matrix, and initial-pair selection.

For calibrated correspondences x_i <-> x_j and E = [t]x R, decomposing E gives four
(R, t) candidates; exactly one places the scene in front of both cameras
(cheirality). The translation is recovered only up to scale: ``t`` is a unit vector.
(R, t) maps camera-i coordinates to camera-j coordinates.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PoseConfig
from src.features.extractor import Features
from src.features.matcher import Matches
from src.geometry.epipolar import (
    TwoViewGeometry,
    angle_between_deg,
    cos_to_deg,
    enforce_essential,
    essential_factors,
    essential_from_fundamental,
    homogeneous,
    normalize_points,
)
from src.geometry.triangulation import triangulate_two_view
from src.geometry.verification import View
from src.run import write_csv

# W in Hartley & Zisserman (9.13): a 90 degree rotation about z.
# --- metrics -------------------------------------------------------------------------------


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of a rotation matrix, in degrees."""
    return float(cos_to_deg((np.trace(R) - 1.0) / 2.0))


# --- essential matrix decomposition --------------------------------------------------------


def essential_for_pose(geometry: TwoViewGeometry, K_i: np.ndarray, K_j: np.ndarray) -> np.ndarray:
    """E from a verified pair: the estimated E, or K_j^T F K_i projected onto essentials.

    The F route is only as good as K: with guessed intrinsics the pose is approximate.
    """
    if geometry.E is not None:
        return geometry.E
    if geometry.F is None:
        raise ValueError("pair has no estimated geometry")
    return enforce_essential(essential_from_fundamental(geometry.F, K_i, K_j))


def _depths(R: np.ndarray, t: np.ndarray, n_i: np.ndarray, n_j: np.ndarray):
    """Triangulate in camera-i coordinates; return (points, depth in i, depth in j)."""
    P_i = np.hstack([np.eye(3), np.zeros((3, 1))])
    P_j = np.hstack([R, t[:, None]])
    X = triangulate_two_view(P_i, P_j, n_i, n_j)
    return X, X[:, 2], X @ R[2] + t[2]


def parallax_angles_deg(R: np.ndarray, n_i: np.ndarray, n_j: np.ndarray) -> np.ndarray:
    """Rotation-compensated angle between the two viewing rays of each correspondence.

    Ray i is K_i^-1 x_i; ray j, rotated into camera i's frame, is R^T K_j^-1 x_j. For a
    correct pose this equals the triangulation angle at the 3-D point, but it does not
    depend on t, so it stays meaningful when the camera only rotated.
    """
    # (R^T r)^T = r^T R, so rotating each row of ray j is a right-multiplication by R.
    return angle_between_deg(homogeneous(n_i), homogeneous(n_j) @ R)


# --- relative pose -----------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RelativePose:
    R: np.ndarray | None  # (3, 3) camera i -> camera j
    t: np.ndarray | None  # (3,) unit translation, camera i -> camera j
    in_front_mask: np.ndarray  # (N,) positive depth in both cameras
    finite_mask: np.ndarray  # (N,) closer than max_depth in both cameras (|depth|)
    median_parallax_deg: float
    success: bool
    reason: str | None  # "too_few_points" | "low_parallax" | "cheirality"

    @property
    def valid_mask(self) -> np.ndarray:
        """Usable points: in front of both cameras and not effectively at infinity."""
        return self.in_front_mask & self.finite_mask

    @property
    def n_points(self) -> int:
        return len(self.in_front_mask)

    @property
    def n_in_front(self) -> int:
        return int(self.in_front_mask.sum())

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def cheirality_ratio(self) -> float:
        """Share of finite points in front of both cameras.

        Points beyond ``max_depth`` are excluded: their depth sign is dominated by noise,
        so they neither confirm nor contradict the pose.
        """
        n_finite = int(self.finite_mask.sum())
        return self.n_valid / n_finite if n_finite else 0.0

    @property
    def rotation_deg(self) -> float:
        return rotation_angle_deg(self.R) if self.R is not None else np.nan


def estimate_relative_pose(
    x_i: np.ndarray,
    x_j: np.ndarray,
    K_i: np.ndarray,
    K_j: np.ndarray,
    E: np.ndarray,
    config: PoseConfig,
) -> RelativePose:
    """Pick the (R, t) candidate of E that puts most points in front of both cameras.

    ``x_i``/``x_j`` are verified inlier correspondences in pixels. A point is valid if
    its depth is positive in both cameras and below ``max_depth`` baselines (|t| = 1);
    farther points are effectively at infinity and carry no information on the sign.
    The candidate is chosen by valid points (as ``cv2.recoverPose`` does); the cheirality
    ratio is then taken over finite points only. Checks, in order: enough points, enough
    parallax, enough finite points in front of both cameras.
    """
    n = len(x_i)
    if n < config.min_points:
        none = np.zeros(n, dtype=bool)
        return RelativePose(None, None, none, none.copy(), np.nan, False, "too_few_points")

    n_i, n_j = normalize_points(x_i, K_i), normalize_points(x_j, K_j)
    candidates = []
    R1, R2, t = essential_factors(E)
    for R in (R1, R2):
        with np.errstate(over="ignore", invalid="ignore"):
            _, z_i, z_j = _depths(R, t, n_i, n_j)
        finite = (np.abs(z_i) < config.max_depth) & (np.abs(z_j) < config.max_depth)
        # Flipping t negates every triangulated point (the DLT system is linear in
        # [X; 1] and in t), so both depths change sign: one triangulation per rotation.
        candidates.append((R, t, (z_i > 0) & (z_j > 0), finite))
        candidates.append((R, -t, (z_i < 0) & (z_j < 0), finite))
    # Most valid points wins; when every point is far (no valid points at all), the count
    # in front of both cameras still separates the correct candidate from the others.
    R, t, in_front, finite = max(
        candidates, key=lambda c: (int((c[2] & c[3]).sum()), int(c[2].sum()))
    )

    pose = RelativePose(
        R, t / np.linalg.norm(t), in_front, finite,
        float(np.median(parallax_angles_deg(R, n_i, n_j))), True, None,
    )  # fmt: skip
    if pose.median_parallax_deg < config.min_parallax_deg:
        return dataclasses.replace(pose, success=False, reason="low_parallax")
    if pose.cheirality_ratio < config.min_cheirality_ratio:
        return dataclasses.replace(pose, success=False, reason="cheirality")
    return pose


# --- batch over verified pairs and initial pair selection ---------------------------------


@dataclass(frozen=True)
class PoseStats:
    i: int
    j: int
    success: bool
    reason: str | None
    n_points: int
    n_in_front: int
    n_valid: int
    cheirality_ratio: float
    median_parallax_deg: float
    rotation_deg: float
    homography_ratio: float
    initial_candidate: bool


def is_initial_candidate(pose: RelativePose, planar: bool, min_parallax_deg: float) -> bool:
    """Whether a pair may start the reconstruction.

    It needs a valid pose, must not be explained by a homography (``planar``, decided
    by verification) and needs enough parallax for well-conditioned triangulation.
    """
    return pose.success and not planar and pose.median_parallax_deg >= min_parallax_deg


def estimate_pair_poses(
    views: Sequence[View],
    features: Sequence[Features],
    matches: Sequence[Matches],
    geometries: Sequence[TwoViewGeometry],
    config: PoseConfig,
) -> tuple[list[RelativePose | None], list[PoseStats]]:
    """Relative pose of every successfully verified pair (``None`` for the others).

    ``matches``/``geometries`` are aligned lists from matching and verification. Stats
    are returned for verified pairs only and record whether each is an initial-pair
    candidate.
    """
    poses: list[RelativePose | None] = []
    stats: list[PoseStats] = []
    for m, g in zip(matches, geometries, strict=True):
        if not g.success:
            poses.append(None)
            continue
        K_i, K_j = views[m.i].K, views[m.j].K
        x_i, x_j = m.points(features[m.i], features[m.j])
        pose = estimate_relative_pose(
            x_i[g.inlier_mask], x_j[g.inlier_mask], K_i, K_j, essential_for_pose(g, K_i, K_j),
            config,
        )  # fmt: skip
        poses.append(pose)
        stats.append(
            PoseStats(
                i=m.i,
                j=m.j,
                success=pose.success,
                reason=pose.reason,
                n_points=pose.n_points,
                n_in_front=pose.n_in_front,
                n_valid=pose.n_valid,
                cheirality_ratio=pose.cheirality_ratio,
                median_parallax_deg=pose.median_parallax_deg,
                rotation_deg=pose.rotation_deg,
                homography_ratio=g.homography_ratio,
                initial_candidate=is_initial_candidate(
                    pose, g.planar, config.init_min_parallax_deg
                ),
            )
        )
    return poses, stats


def select_initial_pair(stats: Sequence[PoseStats]) -> PoseStats | None:
    """The candidate pair with the most valid points; ties go to the larger parallax.

    More points means more 2D-3D correspondences to register the other images.
    """
    candidates = [s for s in stats if s.initial_candidate]
    if not candidates:
        return None
    return max(candidates, key=lambda s: (s.n_valid, s.median_parallax_deg))


@dataclass(frozen=True, eq=False)
class InitialPair:
    """The pair the reconstruction starts from, with its two-view structure."""

    i: int
    j: int
    pose: RelativePose
    geometry: TwoViewGeometry
    x_i: np.ndarray  # (M, 2) verified inlier pixels in image i
    x_j: np.ndarray  # (M, 2) the corresponding pixels in image j

    def to_dict(self, names: Sequence[str]) -> dict[str, Any]:
        p = self.pose
        return {
            "image_i": names[self.i],
            "image_j": names[self.j],
            "index_i": self.i,
            "index_j": self.j,
            "R": p.R.tolist(),
            "t": p.t.tolist(),
            "translation_scale": "unit baseline (monocular scale is unknown)",
            "valid_points": p.n_valid,
            "cheirality_ratio": p.cheirality_ratio,
            "median_parallax_deg": p.median_parallax_deg,
            "rotation_deg": p.rotation_deg,
            "homography_ratio": self.geometry.homography_ratio,
            "model": self.geometry.model,
        }


def initial_pair(
    views: Sequence[View],
    features: Sequence[Features],
    matches: Sequence[Matches],
    geometries: Sequence[TwoViewGeometry],
    poses: Sequence[RelativePose | None],
    stats: Sequence[PoseStats],
) -> InitialPair | None:
    """Select the initial pair and collect its verified correspondences (lists aligned with
    ``matches``). Its points are triangulated by ``reconstruction.two_view``."""
    best = select_initial_pair(stats)
    if best is None:
        return None
    k = next(n for n, m in enumerate(matches) if (m.i, m.j) == (best.i, best.j))
    m, g, pose = matches[k], geometries[k], poses[k]
    x_i, x_j = m.points(features[m.i], features[m.j])
    return InitialPair(m.i, m.j, pose, g, x_i[g.inlier_mask], x_j[g.inlier_mask])


def write_pose_statistics(
    stats: Sequence[PoseStats], names: Sequence[str], path: str | Path
) -> Path:
    return write_csv(
        path,
        ["image_i", "image_j", "success", "reason", "points", "in_front", "valid_points",
         "cheirality_ratio", "median_parallax_deg", "rotation_deg", "homography_ratio",
         "initial_candidate"],
        (
            [names[s.i], names[s.j], int(s.success), s.reason or "", s.n_points,
             s.n_in_front, s.n_valid, f"{s.cheirality_ratio:.4f}",
             f"{s.median_parallax_deg:.3f}", f"{s.rotation_deg:.3f}",
             f"{s.homography_ratio:.3f}", int(s.initial_candidate)]
            for s in stats
        ),
    )  # fmt: skip

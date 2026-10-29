"""The initial reconstruction: two cameras and the points they triangulate.

Camera i defines the world frame (R = I, t = 0) and the baseline has length 1, so the
structure is known up to a global similarity (rotation, translation and scale).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import TriangulationConfig
from src.geometry.camera import Camera
from src.geometry.pose_estimation import InitialPair
from src.geometry.triangulation import Observations, TriangulationResult, triangulate_and_filter


@dataclass(frozen=True, eq=False)
class TwoViewStructure:
    i: int  # image index of camera 0 (the world frame)
    j: int  # image index of camera 1
    cameras: list[Camera]
    observations: Observations  # view 0/1 = image i/j, point k = k-th inlier correspondence
    triangulation: TriangulationResult

    @property
    def points(self) -> np.ndarray:
        """Valid 3-D points, (N_valid, 3)."""
        return self.triangulation.points[self.triangulation.valid]


def build_two_view_structure(
    pair: InitialPair, K_i: np.ndarray, K_j: np.ndarray, config: TriangulationConfig
) -> TwoViewStructure:
    """Triangulate and filter the initial pair's verified correspondences."""
    cameras = [Camera(K_i, np.eye(3), np.zeros(3)), Camera(K_j, pair.pose.R, pair.pose.t)]
    n = len(pair.x_i)
    observations = Observations(
        view=np.repeat([0, 1], n),
        point=np.tile(np.arange(n), 2),
        xy=np.vstack([pair.x_i, pair.x_j]),
    )
    return TwoViewStructure(
        pair.i, pair.j, cameras, observations,
        triangulate_and_filter(cameras, observations, config, n_points=n),
    )  # fmt: skip

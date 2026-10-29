"""The initial reconstruction: two cameras and the points they triangulate.

Camera i defines the world frame (R = I, t = 0) and the baseline has length 1, so the
structure is known up to a global similarity (rotation, translation and scale).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import TriangulationConfig
from src.data.dataset import PreparedImage
from src.geometry.camera import Camera
from src.geometry.pose_estimation import InitialPair
from src.geometry.triangulation import Observations, TriangulationResult, triangulate_and_filter
from src.pointcloud.coloring import color_points
from src.pointcloud.export import write_cameras_json, write_ply


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


@dataclass(frozen=True)
class TwoViewFiles:
    ply: Path
    cameras: Path


def colored_points(
    structure: TwoViewStructure, image_i: np.ndarray, image_j: np.ndarray
) -> np.ndarray:
    """RGB colour in [0, 1] of each valid point, averaged over both images (BGR inputs)."""
    colors = color_points(
        [image_i, image_j], structure.observations, len(structure.triangulation.valid)
    )
    return colors[structure.triangulation.valid]


def save_two_view(
    structure: TwoViewStructure,
    image_i: PreparedImage,
    image_j: PreparedImage,
    directory: str | Path,
    colors: np.ndarray | None = None,
) -> TwoViewFiles:
    """Write ``point_clouds/sparse.ply`` (coloured) and ``camera_poses/cameras.json``."""
    directory = Path(directory)
    if colors is None:
        colors = colored_points(structure, image_i.image, image_j.image)
    names = [image_i.name, image_j.name]
    sizes = [(image_i.width, image_i.height), (image_j.width, image_j.height)]
    ply = write_ply(directory / "point_clouds" / "sparse.ply", structure.points, colors)
    cameras = write_cameras_json(
        directory / "camera_poses" / "cameras.json", structure.cameras, names, sizes,
        frame=f"camera of {names[0]} at the origin, baseline length 1 (scale unknown)",
    )  # fmt: skip
    return TwoViewFiles(ply, cameras)

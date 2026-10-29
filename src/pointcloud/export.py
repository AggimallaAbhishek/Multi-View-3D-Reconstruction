"""Reading and writing point clouds (PLY, via Open3D's tensor API) and cameras (JSON)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from src.geometry.camera import Camera
from src.run import read_json, write_json


def to_open3d(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.t.geometry.PointCloud:
    """Tensor point cloud (Float64 positions, Float32 colours in [0, 1]) on the CPU.

    Positions stay float64, the project's convention for 3-D points (docs/architecture.md),
    so a PLY round trip is exact.
    """
    if colors is not None and len(colors) != len(points):
        raise ValueError(f"{len(points)} points but {len(colors)} colors")
    cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.asarray(points, dtype=np.float64)))
    if colors is not None:
        cloud.point.colors = o3d.core.Tensor(np.asarray(colors, dtype=np.float32))
    return cloud


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.t.io.write_point_cloud(str(path), to_open3d(points, colors)):
        raise OSError(f"could not write {path}")
    return path


def read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """(points (N, 3), colours (N, 3) in [0, 1] or None)."""
    cloud = o3d.t.io.read_point_cloud(str(path))
    points = cloud.point.positions.numpy().astype(np.float64)
    colors = cloud.point.colors.numpy().astype(np.float64) if "colors" in cloud.point else None
    return points, colors


def write_cameras_json(
    path: str | Path,
    cameras: Sequence[Camera],
    names: Sequence[str],
    sizes: Sequence[tuple[int, int]],
    frame: str = "reconstruction frame (world -> camera: x_c = R X + t)",
) -> Path:
    data = {
        "frame": frame,
        "cameras": [
            {
                "index": k,
                "name": name,
                "width": int(size[0]),
                "height": int(size[1]),
                "K": cam.K.tolist(),
                "R": cam.R.tolist(),
                "t": cam.t.tolist(),
                "centre": cam.centre.tolist(),
            }
            for k, (cam, name, size) in enumerate(zip(cameras, names, sizes, strict=True))
        ],
    }
    return write_json(path, data)


def read_cameras_json(
    path: str | Path,
) -> tuple[list[Camera], list[str], list[tuple[int, int]]]:
    entries = read_json(path)["cameras"]
    cameras = [Camera(np.array(e["K"]), np.array(e["R"]), np.array(e["t"])) for e in entries]
    return cameras, [e["name"] for e in entries], [(e["width"], e["height"]) for e in entries]


@dataclass(frozen=True)
class SavedCloud:
    ply: Path
    cameras: Path


def save_cloud(
    directory: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None,
    cameras: Sequence[Camera],
    names: Sequence[str],
    sizes: Sequence[tuple[int, int]],
    frame: str,
    name: str = "sparse",
) -> SavedCloud:
    """``point_clouds/<name>.ply`` and ``camera_poses/cameras.json`` under ``directory``."""
    directory = Path(directory)
    return SavedCloud(
        ply=write_ply(directory / "point_clouds" / f"{name}.ply", points, colors),
        cameras=write_cameras_json(
            directory / "camera_poses" / "cameras.json", cameras, names, sizes, frame=frame
        ),
    )

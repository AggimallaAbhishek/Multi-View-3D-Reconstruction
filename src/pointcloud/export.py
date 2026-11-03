"""Reading and writing point clouds (PLY, via Open3D's tensor API) and cameras (JSON)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.geometry.camera import Camera
from src.run import read_json, write_json

if TYPE_CHECKING:  # open3d takes ~0.25 s to import; only the PLY functions need it
    import open3d as o3d


def to_open3d(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.t.geometry.PointCloud:
    """Tensor point cloud (Float64 positions, Float32 colours in [0, 1]) on the CPU.

    Positions stay float64, the project's convention for 3-D points (docs/architecture.md),
    so a PLY round trip is exact.
    """
    import open3d as o3d

    if colors is not None and len(colors) != len(points):
        raise ValueError(f"{len(points)} points but {len(colors)} colors")
    cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.asarray(points, dtype=np.float64)))
    if colors is not None:
        cloud.point.colors = o3d.core.Tensor(np.asarray(colors, dtype=np.float32))
    return cloud


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> Path:
    """Write a point cloud; an empty one becomes a valid header-only PLY (Open3D refuses
    to write zero points, but a stage that keeps none is a result, not an error)."""
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(points) == 0:
        if colors is not None and len(colors):
            raise ValueError(f"0 points but {len(colors)} colors")
        properties = ["property double x", "property double y", "property double z"]
        if colors is not None:
            properties += ["property float red", "property float green", "property float blue"]
        header = ["ply", "format ascii 1.0", "element vertex 0", *properties, "end_header"]
        path.write_text("\n".join(header) + "\n")
        return path
    if not o3d.t.io.write_point_cloud(str(path), to_open3d(points, colors)):
        raise OSError(f"could not write {path}")
    return path


def read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """(points (N, 3), colours (N, 3) in [0, 1] or None); an empty cloud gives (0, 3)."""
    import open3d as o3d

    cloud = o3d.t.io.read_point_cloud(str(path))
    if "positions" not in cloud.point:  # no vertices: Open3D reads no attributes at all
        has_colors = _ply_header_has_colors(Path(path))
        return np.empty((0, 3)), np.empty((0, 3)) if has_colors else None
    points = cloud.point.positions.numpy().astype(np.float64)
    colors = cloud.point.colors.numpy().astype(np.float64) if "colors" in cloud.point else None
    return points, colors


def _ply_header_has_colors(path: Path) -> bool:
    with path.open("rb") as f:
        for line in f:
            text = line.decode("latin1").strip()
            if text == "end_header":
                return False
            if text.startswith("property") and text.split()[-1] in ("red", "r"):
                return True
    return False


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
    data = read_json(path)
    if "cameras" not in data:
        raise ValueError(
            f"{path}: old cameras format ({{name: {{K, R, t}}}}); re-create it, for a "
            "Middlebury set with `python scripts/download_middlebury.py <name>`"
        )
    entries = data["cameras"]
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

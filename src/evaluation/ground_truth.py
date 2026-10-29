"""Ground-truth cameras for evaluation datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.preprocessing import scale_intrinsics
from src.geometry.camera import Camera
from src.pointcloud.export import read_cameras_json


def relative_pose(a: Camera, b: Camera) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) mapping camera-a coordinates to camera-b coordinates."""
    R = b.R @ a.R.T
    return R, b.t - R @ a.t


def at_working_resolution(camera: Camera, scale: tuple[float, float]) -> Camera:
    """The same camera with K rescaled to a downscaled image's pixel grid.

    Ground truth is given at the original resolution; observations are at the working
    resolution (``PreparedImage.scale``). Mixing the two silently distorts every metric.
    """
    return Camera(scale_intrinsics(camera.K, *scale), camera.R, camera.t)


def load_middlebury_par(path: str | Path) -> dict[str, Camera]:
    """Parse a Middlebury ``*_par.txt``: first line is the count, then per image
    ``name k11..k33 r11..r33 t1 t2 t3`` with projection P = K [R | t]."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    cameras = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 22:
            continue
        values = np.array(parts[1:], dtype=float)
        cameras[parts[0]] = Camera(
            K=values[:9].reshape(3, 3), R=values[9:18].reshape(3, 3), t=values[18:21]
        )
    return cameras


def load_cameras_json(path: str | Path) -> dict[str, Camera]:
    """Ground-truth cameras by image name, from a ``cameras.json`` (the same format the
    pipeline exports; see ``src.pointcloud.export.write_cameras_json``)."""
    cameras, names, _ = read_cameras_json(path)
    return dict(zip(names, cameras, strict=True))

"""Ground-truth cameras for evaluation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, eq=False)
class GroundTruthCamera:
    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3) world -> camera
    t: np.ndarray  # (3,)


def relative_pose(a: GroundTruthCamera, b: GroundTruthCamera) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) mapping camera-a coordinates to camera-b coordinates."""
    R = b.R @ a.R.T
    return R, b.t - R @ a.t


def load_middlebury_par(path: str | Path) -> dict[str, GroundTruthCamera]:
    """Parse a Middlebury ``*_par.txt``: first line is the count, then per image
    ``name k11..k33 r11..r33 t1 t2 t3`` with projection P = K [R | t]."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    cameras = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 22:
            continue
        values = np.array(parts[1:], dtype=float)
        cameras[parts[0]] = GroundTruthCamera(
            K=values[:9].reshape(3, 3), R=values[9:18].reshape(3, 3), t=values[18:21]
        )
    return cameras


def save_cameras_json(cameras: dict[str, GroundTruthCamera], path: str | Path) -> Path:
    path = Path(path)
    data = {
        name: {"K": c.K.tolist(), "R": c.R.tolist(), "t": c.t.tolist()}
        for name, c in cameras.items()
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_cameras_json(path: str | Path) -> dict[str, GroundTruthCamera]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: GroundTruthCamera(K=np.array(c["K"]), R=np.array(c["R"]), t=np.array(c["t"]))
        for name, c in data.items()
    }

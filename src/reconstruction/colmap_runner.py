"""COLMAP sparse reconstruction as a baseline (EXTERNAL: COLMAP, https://colmap.github.io/,
run as a command-line tool; nothing here is our own reconstruction).

``run_colmap`` writes the images, runs COLMAP's SIFT extraction, exhaustive matching and
incremental mapper on the CPU with the given intrinsics, and reads the largest model
back into this project's conventions (``Camera`` with world -> camera ``R, t``; pixel
centres at integer coordinates, where COLMAP puts them at +0.5).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.geometry.camera import Camera

# Parameters before the distortion terms, per COLMAP camera model: focal(s), cx, cy.
_FOCAL_MODELS = {
    "SIMPLE_PINHOLE": 1, "SIMPLE_RADIAL": 1, "RADIAL": 1, "SIMPLE_RADIAL_FISHEYE": 1,
    "PINHOLE": 2, "OPENCV": 2, "FULL_OPENCV": 2,
}  # fmt: skip


class ColmapError(RuntimeError):
    """COLMAP is missing, failed, or reconstructed nothing."""


@dataclass(frozen=True, eq=False)
class ColmapModel:
    cameras: dict[str, Camera]  # by image name, registered images only
    points: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N, 3) RGB uint8
    errors: np.ndarray  # (N,) mean reprojection error, px
    track_lengths: np.ndarray  # (N,)


def colmap_executable() -> str | None:
    return shutil.which("colmap")


def intrinsics_from_colmap(model: str, params: Sequence[float]) -> np.ndarray:
    """K (our pixel-centre convention) from a COLMAP camera; distortion is ignored."""
    if model not in _FOCAL_MODELS:
        raise ValueError(f"unsupported COLMAP camera model {model}")
    if _FOCAL_MODELS[model] == 1:
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        fx, fy, cx, cy = params[:4]
    return np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1.0]])


def read_text_model(directory: str | Path) -> ColmapModel:
    """Parse COLMAP's ``cameras.txt``, ``images.txt`` and ``points3D.txt``."""
    directory = Path(directory)
    intrinsics = {}
    for fields in _rows(directory / "cameras.txt"):
        intrinsics[fields[0]] = intrinsics_from_colmap(fields[1], [float(p) for p in fields[4:]])
    cameras = {}
    lines = [line for line in (directory / "images.txt").read_text().splitlines()
             if not line.startswith("#")]  # fmt: skip
    for header in lines[0::2]:  # every image has a header line and a 2-D points line
        fields = header.split()
        if not fields:
            continue
        qw, qx, qy, qz, *t = (float(v) for v in fields[1:8])
        cameras[fields[9]] = Camera(intrinsics[fields[8]], _rotation(qw, qx, qy, qz), np.array(t))
    rows = list(_rows(directory / "points3D.txt"))
    return ColmapModel(
        cameras,
        np.array([[float(v) for v in r[1:4]] for r in rows]).reshape(-1, 3),
        np.array([[int(v) for v in r[4:7]] for r in rows], np.uint8).reshape(-1, 3),
        np.array([float(r[7]) for r in rows]),
        np.array([(len(r) - 8) // 2 for r in rows]),
    )


def run_colmap(
    images: Mapping[str, np.ndarray],
    K: np.ndarray,
    work_dir: str | Path,
    refine_intrinsics: bool = True,
    max_features: int = 8192,
) -> ColmapModel:
    """COLMAP SfM on ``images`` (name -> BGR) taken by one camera with intrinsics ``K``
    (our convention); ``refine_intrinsics`` lets bundle adjustment refine the focal
    length (COLMAP's default). Returns the model with the most registered images."""
    exe = colmap_executable()
    if exe is None:
        raise ColmapError("colmap is not on PATH (macOS: brew install colmap)")
    work = Path(work_dir)
    image_dir, database, sparse = work / "images", work / "database.db", work / "sparse"
    image_dir.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(exist_ok=True)
    for name, image in images.items():
        if not cv2.imwrite(str(image_dir / name), image):
            raise ColmapError(f"could not write {image_dir / name}")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5
    refine = str(int(refine_intrinsics))
    steps = [
        ["feature_extractor", "--database_path", database, "--image_path", image_dir,
         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "PINHOLE",
         "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}",
         "--FeatureExtraction.use_gpu", "0",
         "--SiftExtraction.max_num_features", str(max_features)],
        ["exhaustive_matcher", "--database_path", database, "--FeatureMatching.use_gpu", "0"],
        ["mapper", "--database_path", database, "--image_path", image_dir,
         "--output_path", sparse, "--Mapper.ba_refine_focal_length", refine,
         "--Mapper.ba_refine_extra_params", "0"],
    ]  # fmt: skip
    with (work / "colmap.log").open("w") as log:
        for step in steps:
            done = subprocess.run([exe, *map(str, step)], stdout=log, stderr=subprocess.STDOUT)
            if done.returncode:
                raise ColmapError(f"colmap {step[0]} failed (see {work / 'colmap.log'})")
        models = []
        for model_dir in sorted(p for p in sparse.iterdir() if p.is_dir()):
            text = work / "text" / model_dir.name
            text.mkdir(parents=True, exist_ok=True)
            subprocess.run([exe, "model_converter", "--input_path", str(model_dir),
                            "--output_path", str(text), "--output_type", "TXT"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)  # fmt: skip
            models.append(read_text_model(text))
    if not models:
        raise ColmapError(f"colmap reconstructed nothing (see {work / 'colmap.log'})")
    return max(models, key=lambda m: len(m.cameras))


def _rows(path: Path):
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            yield line.split()


def _rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Rotation matrix of a (normalised) Hamilton quaternion, as COLMAP stores them."""
    q = np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip

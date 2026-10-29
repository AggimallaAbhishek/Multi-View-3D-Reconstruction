"""Camera intrinsics: calibration files, EXIF estimates, fallbacks and checkerboard calibration.

Priority when resolving the intrinsics of an image:
    1. calibration file (``camera.calibration_file``), rescaled to the image size
    2. EXIF 35 mm-equivalent focal length (``camera.use_exif``)
    3. fallback ``f = fallback_focal_factor * max(width, height)``
The principal point of estimated intrinsics is the image centre, ((w - 1) / 2, (h - 1) / 2).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.config import CalibrationConfig, CameraConfig
from src.data.loader import ExifInfo
from src.data.preprocessing import scale_intrinsics, to_gray

FULL_FRAME_DIAGONAL_MM = float(np.hypot(36.0, 24.0))
_CAMERA_FILE_KEYS = {"width", "height", "fx", "fy", "cx", "cy", "dist"}


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class Intrinsics:
    K: np.ndarray  # (3, 3)
    dist: np.ndarray  # OpenCV distortion coefficients (k1, k2, p1, p2[, k3, ...])
    width: int  # resolution these intrinsics refer to
    height: int
    source: str  # "file" | "exif" | "fallback" | "checkerboard"

    def scaled_to(
        self, width: int, height: int, rtol: float = CameraConfig.aspect_tolerance
    ) -> Intrinsics:
        """The same camera at another resolution with the same aspect ratio (within ``rtol``)."""
        sx, sy = width / self.width, height / self.height
        if not np.isclose(sx, sy, rtol=rtol):
            raise CalibrationError(
                f"intrinsics for {self.width}x{self.height} cannot be used for {width}x{height}: "
                "aspect ratio differs (rotated image or different camera mode?)"
            )
        return Intrinsics(
            K=scale_intrinsics(self.K, sx, sy),
            dist=self.dist.copy(),
            width=width,
            height=height,
            source=self.source,
        )


def _centred_k(focal: float, width: int, height: int) -> np.ndarray:
    return np.array(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]]
    )


def fallback_intrinsics(width: int, height: int, factor: float) -> Intrinsics:
    return Intrinsics(
        K=_centred_k(factor * max(width, height), width, height),
        dist=np.zeros(5),
        width=width,
        height=height,
        source="fallback",
    )


def intrinsics_from_exif(exif: ExifInfo, width: int, height: int) -> Intrinsics | None:
    """Focal length in pixels from the 35 mm-equivalent focal length.

    The equivalence is defined on the image diagonal, which handles any aspect ratio:
    f_px = f_35mm * diagonal_px / 43.27 mm.
    """
    if exif.focal_35mm is None:
        return None
    focal = exif.focal_35mm * float(np.hypot(width, height)) / FULL_FRAME_DIAGONAL_MM
    return Intrinsics(
        K=_centred_k(focal, width, height),
        dist=np.zeros(5),
        width=width,
        height=height,
        source="exif",
    )


def resolve_intrinsics(
    width: int,
    height: int,
    exif: ExifInfo,
    config: CameraConfig,
    calibration: Intrinsics | None = None,
) -> Intrinsics:
    """Calibration file, else EXIF, else the fallback guess; fx and fy are then
    multiplied by ``config.focal_scale`` (1 in normal use; experiments set it to
    simulate a focal-length error)."""
    if calibration is not None:
        intrinsics = calibration.scaled_to(width, height, config.aspect_tolerance)
    elif config.use_exif and (exif_K := intrinsics_from_exif(exif, width, height)) is not None:
        intrinsics = exif_K
    else:
        intrinsics = fallback_intrinsics(width, height, config.fallback_focal_factor)
    if config.focal_scale == 1.0:
        return intrinsics
    K = intrinsics.K.copy()
    K[0, 0] *= config.focal_scale
    K[1, 1] *= config.focal_scale
    return dataclasses.replace(intrinsics, K=K)


def load_camera_file(path: str | Path) -> Intrinsics:
    """Read a camera YAML file (see configs/camera.example.yaml)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - _CAMERA_FILE_KEYS
    if unknown:
        raise CalibrationError(f"{path}: unknown key(s) {sorted(unknown)}")
    missing = (_CAMERA_FILE_KEYS - {"dist"}) - set(raw)
    if missing:
        raise CalibrationError(f"{path}: missing key(s) {sorted(missing)}")
    K = np.array(
        [[raw["fx"], 0.0, raw["cx"]], [0.0, raw["fy"], raw["cy"]], [0.0, 0.0, 1.0]], dtype=float
    )
    dist = np.asarray(raw.get("dist") or [0.0] * 5, dtype=float)
    return Intrinsics(
        K=K, dist=dist, width=int(raw["width"]), height=int(raw["height"]), source="file"
    )


def save_camera_file(intrinsics: Intrinsics, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    K = intrinsics.K
    data = {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "dist": [float(d) for d in intrinsics.dist],
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(
            f"# Camera intrinsics (source: {intrinsics.source}).\n"
            "# width/height are the resolution they refer to; other sizes are rescaled.\n"
        )
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def calibrate_from_checkerboard(
    images: Sequence[np.ndarray],
    pattern_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
    config: CalibrationConfig | None = None,
) -> tuple[Intrinsics, float]:
    """Zhang calibration from checkerboard photos.

    ``pattern_size`` counts inner corners (columns, rows). Returns the intrinsics
    and the RMS reprojection error in pixels.
    """
    config = config or CalibrationConfig()
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * square_size
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        config.subpix_max_iter,
        config.subpix_eps,
    )
    window = (config.subpix_window, config.subpix_window)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    object_points, image_points = [], []
    size: tuple[int, int] | None = None
    for image in images:
        gray = to_gray(image)
        this_size = (gray.shape[1], gray.shape[0])
        if size is None:
            size = this_size
        elif this_size != size:
            raise CalibrationError("all calibration images must have the same resolution")
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, window, (-1, -1), criteria)
        object_points.append(objp)
        image_points.append(corners)

    if len(object_points) < config.min_views:
        raise CalibrationError(
            f"checkerboard {pattern_size} found in {len(object_points)} image(s); "
            f"need at least {config.min_views}"
        )
    assert size is not None
    rms, K, dist, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    intrinsics = Intrinsics(
        K=K,
        dist=np.asarray(dist, dtype=float).ravel(),
        width=size[0],
        height=size[1],
        source="checkerboard",
    )
    return intrinsics, float(rms)

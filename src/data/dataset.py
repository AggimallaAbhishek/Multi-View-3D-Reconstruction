"""Load, validate and preprocess an image directory into a ``Dataset``.

Images are processed one at a time and only the downscaled copy is kept, so
memory stays bounded for large photo sets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config
from src.data.loader import DatasetError, ImageRecord, SkippedFile, discover_images, iter_images
from src.data.preprocessing import resize_max_side, to_gray, undistort
from src.data.validation import ValidationReport, image_stats, validate
from src.geometry.calibration import Intrinsics, load_camera_file, resolve_intrinsics

log = logging.getLogger(__name__)


@dataclass
class PreparedImage:
    index: int
    name: str
    path: Path
    image: np.ndarray  # (H, W, 3) BGR uint8 at working resolution, undistorted if possible
    K: np.ndarray  # intrinsics for ``image``
    dist: np.ndarray  # zeros once undistorted
    original_size: tuple[int, int]  # (width, height) as decoded, upright
    scale: tuple[float, float]  # (sx, sy) from original to working resolution
    intrinsics_source: str

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def gray(self) -> np.ndarray:
        return to_gray(self.image)


@dataclass
class Dataset:
    directory: Path
    images: list[PreparedImage]
    report: ValidationReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "n_images": len(self.images),
            "images": [
                {
                    "index": im.index,
                    "name": im.name,
                    "original_size": list(im.original_size),
                    "size": [im.width, im.height],
                    "scale": list(im.scale),
                    "K": im.K.tolist(),
                    "dist": im.dist.tolist(),
                    "intrinsics_source": im.intrinsics_source,
                }
                for im in self.images
            ],
            "report": self.report.to_dict(),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def prepare_dataset(directory: str | Path, config: Config | None = None) -> Dataset:
    """Discover, decode, validate and preprocess every image in ``directory``.

    Raises ``DatasetError`` when no images are found or validation reports an
    error; the exception carries the report when one was produced.
    """
    config = config or Config()
    directory = Path(directory)
    paths = discover_images(directory, config.dataset.extensions)
    if not paths:
        raise DatasetError(
            f"no images with extensions {list(config.dataset.extensions)} found in {directory}"
        )

    calibration = (
        load_camera_file(config.camera.calibration_file) if config.camera.calibration_file else None
    )

    prepared, stats, skipped = [], [], []
    for item in iter_images(paths):
        if isinstance(item, SkippedFile):
            skipped.append(item)
            continue
        stats.append(image_stats(item, config.dataset.blur_working_side))
        prepared.append(_prepare_image(item, calibration, config))

    report = validate(stats, skipped, config.dataset)
    for issue in report.issues:
        log.debug("%s: %s", issue.code, issue.message)
    if not report.ok:
        raise DatasetError(f"dataset validation failed for {directory}", report=report)
    return Dataset(directory=directory, images=prepared, report=report)


def _prepare_image(
    record: ImageRecord, calibration: Intrinsics | None, config: Config
) -> PreparedImage:
    intr = resolve_intrinsics(record.width, record.height, record.exif, config.camera, calibration)
    image, K, dist = record.image, intr.K, intr.dist
    if config.preprocessing.undistort and np.any(dist):
        image = undistort(image, K, dist)
        dist = np.zeros_like(dist)
    image, K, scale = resize_max_side(image, K, config.preprocessing.max_side)
    return PreparedImage(
        index=record.index,
        name=record.name,
        path=record.path,
        image=image,
        K=K,
        dist=dist,
        original_size=(record.width, record.height),
        scale=scale,
        intrinsics_source=intr.source,
    )

"""Load, validate and preprocess an image directory into a ``Dataset``.

Images are processed one at a time. Each ``PreparedImage`` keeps only its
working-resolution grayscale image (what feature extraction needs); the colour image
is decoded again on demand with the same undistortion and resizing, so an 80-image
set holds ~150 MB instead of ~460 MB. An on-disk cache keyed by file content and the
relevant settings lets repeat runs skip decoding and validation statistics entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import Config
from src.data.loader import (
    DatasetError,
    ImageRecord,
    SkippedFile,
    decode_image,
    discover_images,
    read_exif,
)
from src.data.preprocessing import resize_max_side, to_gray, undistort
from src.data.validation import ImageStats, ValidationReport, image_stats, validate
from src.geometry.calibration import Intrinsics, load_camera_file, resolve_intrinsics
from src.run import atomic_savez, write_json

log = logging.getLogger(__name__)


@dataclass(eq=False)
class PreparedImage:
    index: int
    name: str
    path: Path
    gray: np.ndarray  # (H, W) uint8 at working resolution, undistorted if possible
    K: np.ndarray  # intrinsics at working resolution
    dist: np.ndarray  # zeros once undistorted
    original_size: tuple[int, int]  # (width, height) as decoded, upright
    scale: tuple[float, float]  # (sx, sy) from original to working resolution
    intrinsics_source: str
    # What it takes to rebuild the colour image exactly like ``gray`` was built.
    source_K: np.ndarray  # intrinsics at the original resolution
    source_dist: np.ndarray
    undistorted: bool
    max_side: int | None

    @property
    def width(self) -> int:
        return self.gray.shape[1]

    @property
    def height(self) -> int:
        return self.gray.shape[0]

    @property
    def image(self) -> np.ndarray:
        """Colour (BGR) image at working resolution, decoded from disk on each access."""
        image = decode_image(self.path.read_bytes())
        if image is None:
            raise DatasetError(f"{self.path} can no longer be decoded")
        if self.undistorted:
            image = undistort(image, self.source_K, self.source_dist)
        image, _, _ = resize_max_side(image, self.source_K, self.max_side)
        return image


@dataclass
class Dataset:
    directory: Path
    images: list[PreparedImage]
    report: ValidationReport
    calibration_file: Path | None = None  # the camera file used, if any

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
            "calibration_file": str(self.calibration_file) if self.calibration_file else None,
            "report": self.report.to_dict(),
        }

    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())


class ImageCache:
    """Prepared grayscale images and validation statistics, keyed by file content.

    The key also covers every setting that changes the result (resizing, undistortion,
    intrinsics resolution including the calibration file's content, blur and hash
    parameters, OpenCV version), so a change in any of them recomputes.
    """

    def __init__(self, directory: str | Path, config: Config, calibration: Path | None):
        self.directory = Path(directory)
        settings = {
            "preprocessing": config.to_dict()["preprocessing"],
            "camera": config.to_dict()["camera"],
            "calibration": Path(calibration).read_text("utf-8") if calibration else None,
            "blur_working_side": config.dataset.blur_working_side,
            "dhash_size": config.dataset.dhash_size,
            "opencv": cv2.__version__,
            "version": 1,
        }
        self._settings = json.dumps(settings, sort_keys=True)

    def path_for(self, name: str, data: bytes) -> Path:
        h = hashlib.sha1(self._settings.encode())
        h.update(data)
        return self.directory / f"{Path(name).stem}.{h.hexdigest()[:16]}.npz"

    def load(self, path: Path, index: int, source: Path) -> tuple[ImageStats, PreparedImage] | None:
        try:
            with np.load(path, allow_pickle=False) as d:
                original = (int(d["original_size"][0]), int(d["original_size"][1]))
                stats = ImageStats(
                    name=source.name,
                    width=original[0],
                    height=original[1],
                    blur=float(d["blur"]),
                    dhash=d["dhash"],
                )
                prepared = PreparedImage(
                    index=index,
                    name=source.name,
                    path=source,
                    gray=d["gray"],
                    K=d["K"],
                    dist=d["dist"],
                    original_size=original,
                    scale=(float(d["scale"][0]), float(d["scale"][1])),
                    intrinsics_source=str(d["intrinsics_source"]),
                    source_K=d["source_K"],
                    source_dist=d["source_dist"],
                    undistorted=bool(d["undistorted"]),
                    max_side=int(d["max_side"]) if int(d["max_side"]) > 0 else None,
                )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            log.warning("ignoring unreadable image cache entry %s", path)
            return None
        return stats, prepared

    def save(self, path: Path, stats: ImageStats, im: PreparedImage) -> None:
        atomic_savez(
            path,
            gray=im.gray,
            K=im.K,
            dist=im.dist,
            original_size=np.array(im.original_size),
            scale=np.array(im.scale),
            intrinsics_source=np.array(im.intrinsics_source),
            source_K=im.source_K,
            source_dist=im.source_dist,
            undistorted=np.array(im.undistorted),
            max_side=np.array(im.max_side or 0),
            blur=np.array(stats.blur),
            dhash=stats.dhash,
        )


def find_calibration_file(directory: str | Path, config: Config) -> Path | None:
    """The camera file to use: ``camera.calibration_file`` if configured, otherwise a
    ``camera.yaml`` next to the images (as written by ``cli calibrate`` or
    ``scripts/download_middlebury.py``), otherwise None (EXIF or fallback intrinsics)."""
    if config.camera.calibration_file:
        return Path(config.camera.calibration_file)
    beside = Path(directory) / "camera.yaml"
    return beside if beside.is_file() else None


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

    calibration_file = find_calibration_file(directory, config)
    calibration = load_camera_file(calibration_file) if calibration_file else None
    cache_dir = config.cache_path("images")
    cache = ImageCache(cache_dir, config, calibration_file) if cache_dir else None

    prepared, stats, skipped = [], [], []
    for path in paths:
        data = path.read_bytes()
        entry_path = cache.path_for(path.name, data) if cache else None
        entry = (
            cache.load(entry_path, len(prepared), path) if cache and entry_path.exists() else None
        )
        if entry is None:
            image = decode_image(data)
            if image is None:
                log.debug("skipping %s: could not decode image", path.name)
                skipped.append(SkippedFile(path, "could not decode image"))
                continue
            record = ImageRecord(len(prepared), path, image, read_exif(path))
            entry = (
                image_stats(record, config.dataset),
                _prepare_image(record, calibration, config),
            )
            if cache:
                cache.save(entry_path, *entry)
        stats.append(entry[0])
        prepared.append(replace(entry[1], index=len(prepared)))

    report = validate(stats, skipped, config.dataset)
    for issue in report.issues:
        log.debug("%s: %s", issue.code, issue.message)
    if not report.ok:
        raise DatasetError(f"dataset validation failed for {directory}", report=report)
    return Dataset(directory, prepared, report, calibration_file)


def _prepare_image(
    record: ImageRecord, calibration: Intrinsics | None, config: Config
) -> PreparedImage:
    intr = resolve_intrinsics(record.width, record.height, record.exif, config.camera, calibration)
    image, K, dist = record.image, intr.K, intr.dist
    undistorted = bool(config.preprocessing.undistort and np.any(dist))
    if undistorted:
        image = undistort(image, K, dist)
        dist = np.zeros_like(dist)
    image, K, scale = resize_max_side(image, K, config.preprocessing.max_side)
    return PreparedImage(
        index=record.index,
        name=record.name,
        path=record.path,
        gray=to_gray(image),
        K=K,
        dist=dist,
        original_size=(record.width, record.height),
        scale=scale,
        intrinsics_source=intr.source,
        source_K=intr.K,
        source_dist=intr.dist,
        undistorted=undistorted,
        max_side=config.preprocessing.max_side,
    )

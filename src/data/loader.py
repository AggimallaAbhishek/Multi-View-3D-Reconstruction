"""Discover and decode the images of a dataset directory.

Images are decoded with EXIF orientation applied, so ``width``/``height`` and all
later intrinsics refer to the upright image as a person would view it.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image

log = logging.getLogger(__name__)


class DatasetError(RuntimeError):
    """The dataset cannot be used; ``report`` holds the validation report if one exists."""

    def __init__(self, message: str, report: Any = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class ExifInfo:
    focal_mm: float | None = None
    focal_35mm: float | None = None
    make: str | None = None
    model: str | None = None


@dataclass
class ImageRecord:
    index: int
    path: Path
    image: np.ndarray  # (H, W, 3) BGR uint8, upright
    exif: ExifInfo = field(default_factory=ExifInfo)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


@dataclass(frozen=True)
class SkippedFile:
    path: Path
    reason: str


def natural_sort_key(path: Path) -> list[Any]:
    """Sort key so that image2 comes before image10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def discover_images(directory: str | Path, extensions: Iterable[str]) -> list[Path]:
    """Image files directly inside ``directory``, in natural order.

    Matching is case-insensitive; hidden files (e.g. macOS ``._*`` resource forks)
    and subdirectories are ignored.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise DatasetError(f"{directory} is not a directory")
    allowed = {ext.lower() for ext in extensions}
    paths = [
        p
        for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in allowed
    ]
    return sorted(paths, key=natural_sort_key)


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode encoded image bytes as upright BGR uint8 (EXIF orientation applied)."""
    if not data:
        return None
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image


def read_image(path: Path) -> np.ndarray | None:
    """Decode an image file, or ``None`` if it cannot be read or decoded.

    Reads bytes first, which works for non-ASCII paths, unlike ``cv2.imread``.
    """
    try:
        return decode_image(Path(path).read_bytes())
    except OSError:
        return None


def read_exif(path: Path) -> ExifInfo:
    """Focal length and camera model from EXIF; missing or unreadable tags are ``None``."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(ExifTags.IFD.Exif)
    except Exception:  # noqa: BLE001 - any decode failure just means "no EXIF"
        return ExifInfo()
    return ExifInfo(
        focal_mm=_positive(sub.get(ExifTags.Base.FocalLength)),
        focal_35mm=_positive(sub.get(ExifTags.Base.FocalLengthIn35mmFilm)),
        make=_text(exif.get(ExifTags.Base.Make)),
        model=_text(exif.get(ExifTags.Base.Model)),
    )


def iter_images(paths: Iterable[Path]) -> Iterator[ImageRecord | SkippedFile]:
    """Decode images one at a time; indices count only successfully decoded images.

    Yielding lazily keeps memory bounded: callers can downscale each image and
    drop the full-resolution array before the next one is decoded.
    """
    index = 0
    for path in paths:
        image = read_image(path)
        if image is None:
            log.debug("skipping %s: could not decode image", path.name)
            yield SkippedFile(path, "could not decode image")
            continue
        yield ImageRecord(index=index, path=path, image=image, exif=read_exif(path))
        index += 1


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip("\x00 ").strip()
    return text or None

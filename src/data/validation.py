"""Dataset checks: image count, decodability, resolution, blur and near-duplicates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

import cv2
import numpy as np

from src.config import DatasetConfig
from src.data.loader import ImageRecord, SkippedFile
from src.data.preprocessing import to_gray


@dataclass(frozen=True, eq=False)
class ImageStats:
    name: str
    width: int
    height: int
    blur: float  # variance of the Laplacian; higher is sharper
    dhash: np.ndarray  # packed difference-hash bits (see dhash)


@dataclass(frozen=True)
class ValidationIssue:
    severity: Literal["warning", "error"]
    code: str
    message: str


@dataclass
class ValidationReport:
    images: list[ImageStats]
    skipped: list[SkippedFile] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def resolutions(self) -> dict[str, int]:
        return dict(Counter(f"{s.width}x{s.height}" for s in self.images))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_images": len(self.images),
            "n_skipped": len(self.skipped),
            "resolutions": self.resolutions,
            "images": [
                {"name": s.name, "width": s.width, "height": s.height, "blur": round(s.blur, 2)}
                for s in self.images
            ],
            "skipped": [{"name": s.path.name, "reason": s.reason} for s in self.skipped],
            "issues": [
                {"severity": i.severity, "code": i.code, "message": i.message} for i in self.issues
            ],
        }


def blur_score(image: np.ndarray, working_side: int = DatasetConfig.blur_working_side) -> float:
    """Variance of the Laplacian, computed with the longest side capped at ``working_side``."""
    gray = to_gray(image)
    longest = max(gray.shape)
    if longest > working_side:
        s = working_side / longest
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(image: np.ndarray, size: int = DatasetConfig.dhash_size) -> np.ndarray:
    """Difference hash: signs of horizontal gradients on a (size+1) x size thumbnail,
    packed into size*size/8 bytes.

    The default 32x32 (1024 bits) matters: an 8x8 hash is dominated by the background
    when the object fills only part of the frame, so neighbouring views of an object on a
    black background differed by 1-2 bits, as much as true duplicates (Middlebury temple).
    At 1024 bits duplicates differ by <= ~2 % and 7.5 deg neighbours by >= ~9 %.
    """
    thumb = cv2.resize(to_gray(image), (size + 1, size), interpolation=cv2.INTER_AREA)
    thumb = thumb.astype(np.int16)
    return np.packbits((thumb[:, 1:] > thumb[:, :-1]).ravel())


def hash_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of differing hash bits: 0 for identical images, ~0.5 for unrelated ones."""
    if a.shape != b.shape:
        raise ValueError("hashes of different sizes cannot be compared")
    return float(np.unpackbits(a ^ b).mean())


def image_stats(record: ImageRecord, config: DatasetConfig) -> ImageStats:
    return ImageStats(
        name=record.name,
        width=record.width,
        height=record.height,
        blur=blur_score(record.image, config.blur_working_side),
        dhash=dhash(record.image, config.dhash_size),
    )


def validate(
    stats: list[ImageStats], skipped: list[SkippedFile], config: DatasetConfig
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    if len(stats) < config.min_images:
        issues.append(
            ValidationIssue(
                "error",
                "too_few_images",
                f"{len(stats)} readable image(s); at least {config.min_images} required",
            )
        )

    for s in skipped:
        issues.append(ValidationIssue("warning", "unreadable", f"{s.path.name}: {s.reason}"))

    resolutions = Counter((s.width, s.height) for s in stats)
    if len(resolutions) > 1:
        summary = ", ".join(f"{w}x{h} ({n})" for (w, h), n in resolutions.most_common())
        issues.append(
            ValidationIssue(
                "warning",
                "mixed_resolution",
                f"images have different resolutions: {summary}; intrinsics are per image",
            )
        )

    # Laplacian variance depends on content (smooth surfaces score low even when sharp),
    # so an image is blurry relative to the rest of its dataset, or below a low absolute
    # floor that catches a dataset that is blurry throughout.
    if stats:
        relative_limit = config.blur_relative * float(np.median([s.blur for s in stats]))
        for s in stats:
            if s.blur < config.blur_threshold or s.blur < relative_limit:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "blurry",
                        f"{s.name}: blur score {s.blur:.1f} (floor {config.blur_threshold:g}, "
                        f"{config.blur_relative:g} x dataset median = {relative_limit:.1f})",
                    )
                )

    for a, b in combinations(stats, 2):
        distance = hash_distance(a.dhash, b.dhash)
        if distance <= config.duplicate_max_fraction:
            issues.append(
                ValidationIssue(
                    "warning",
                    "near_duplicate",
                    f"{a.name} and {b.name} look nearly identical "
                    f"({distance:.1%} of hash bits differ)",
                )
            )

    return ValidationReport(images=list(stats), skipped=list(skipped), issues=issues)


def format_report(report: ValidationReport) -> str:
    lines = [
        f"Images: {len(report.images)} readable, {len(report.skipped)} skipped",
        "Resolutions: " + ", ".join(f"{r} ({n})" for r, n in report.resolutions.items()),
    ]
    if report.images:
        blur = [s.blur for s in report.images]
        lines.append(
            f"Blur score: min {min(blur):.1f}, median {float(np.median(blur)):.1f}, "
            f"max {max(blur):.1f}"
        )
    for issue in report.issues:
        lines.append(f"[{issue.severity.upper()}] {issue.code}: {issue.message}")
    lines.append("Status: " + ("OK" if report.ok else "FAILED"))
    return "\n".join(lines)

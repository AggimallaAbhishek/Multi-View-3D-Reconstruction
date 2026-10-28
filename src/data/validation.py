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

# Blur is measured at a capped size so one threshold works for any camera resolution.
BLUR_WORKING_SIDE = 800


@dataclass(frozen=True)
class ImageStats:
    name: str
    width: int
    height: int
    blur: float  # variance of the Laplacian; higher is sharper
    dhash: int  # 64-bit difference hash


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
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

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


def blur_score(image: np.ndarray) -> float:
    """Variance of the Laplacian, computed with the longest side capped at 800 px."""
    gray = to_gray(image)
    longest = max(gray.shape)
    if longest > BLUR_WORKING_SIDE:
        s = BLUR_WORKING_SIDE / longest
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(image: np.ndarray) -> int:
    """64-bit difference hash: compares horizontally adjacent pixels of a 9x8 thumbnail."""
    thumb = cv2.resize(to_gray(image), (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (thumb[:, 1:] > thumb[:, :-1]).ravel()
    return int(np.packbits(bits).view(">u8")[0])


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def image_stats(record: ImageRecord) -> ImageStats:
    return ImageStats(
        name=record.name,
        width=record.width,
        height=record.height,
        blur=blur_score(record.image),
        dhash=dhash(record.image),
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

    for s in stats:
        if s.blur < config.blur_threshold:
            issues.append(
                ValidationIssue(
                    "warning",
                    "blurry",
                    f"{s.name}: blur score {s.blur:.1f} < {config.blur_threshold:g}",
                )
            )

    for a, b in combinations(stats, 2):
        distance = hamming(a.dhash, b.dhash)
        if distance <= config.duplicate_max_distance:
            issues.append(
                ValidationIssue(
                    "warning",
                    "near_duplicate",
                    f"{a.name} and {b.name} look nearly identical (hash distance {distance})",
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

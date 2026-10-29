"""Geometric verification of every matched image pair."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.config import VerificationConfig
from src.features.extractor import Features
from src.features.matcher import Matches
from src.geometry.epipolar import TwoViewGeometry, select_model, verify_pair
from src.run import parallel_map, write_csv


class View(Protocol):
    """What verification needs from an image (``PreparedImage`` satisfies it)."""

    K: np.ndarray
    intrinsics_source: str


@dataclass(frozen=True)
class VerificationStats:
    i: int
    j: int
    model: str
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    median_residual: float  # pixels, inliers only
    homography_ratio: float
    success: bool
    reason: str | None
    time_ms: float


def verify_matches(
    views: Sequence[View],
    features: Sequence[Features],
    matches: Sequence[Matches],
    config: VerificationConfig,
    workers: int = 1,
) -> tuple[list[TwoViewGeometry], list[VerificationStats]]:
    """Fit two-view geometry to every pair of matches, in the given order.

    Pairs are independent, so they run on ``workers`` threads (0 = all cores); each
    ``verify_pair`` seeds OpenCV's per-thread RNG, so results do not depend on it.
    """

    def one(m: Matches) -> tuple[TwoViewGeometry, VerificationStats]:
        start = time.perf_counter()
        a, b = views[m.i], views[m.j]
        model = select_model(config.method, a.intrinsics_source, b.intrinsics_source)
        x_i, x_j = m.points(features[m.i], features[m.j])
        g = verify_pair(x_i, x_j, a.K, b.K, config, model=model)
        return g, VerificationStats(
            i=m.i,
            j=m.j,
            model=g.model,
            n_matches=g.n_matches,
            n_inliers=g.n_inliers,
            inlier_ratio=g.inlier_ratio,
            median_residual=g.median_residual,
            homography_ratio=g.homography_ratio,
            success=g.success,
            reason=g.reason,
            time_ms=(time.perf_counter() - start) * 1000,
        )

    results = parallel_map(one, matches, workers)
    return [g for g, _ in results], [s for _, s in results]


def unverified_images(stats: Sequence[VerificationStats], n_images: int) -> list[int]:
    """Images without any verified pair: they cannot be registered."""
    return np.flatnonzero(verification_matrix(stats, n_images).max(axis=1) == 0).tolist()


def inlier_matches(matches: Matches, geometry: TwoViewGeometry) -> Matches:
    """Matches restricted to verified inliers; empty when verification failed."""
    if not geometry.success:
        return Matches.empty(matches.i, matches.j)
    keep = geometry.inlier_mask
    return Matches(
        matches.i, matches.j, matches.idx_i[keep], matches.idx_j[keep], matches.distances[keep]
    )


def verification_matrix(stats: Sequence[VerificationStats], n_images: int) -> np.ndarray:
    """Symmetric (n, n) matrix of inlier counts for successfully verified pairs."""
    m = np.zeros((n_images, n_images), dtype=int)
    for s in stats:
        if s.success:
            m[s.i, s.j] = m[s.j, s.i] = s.n_inliers
    return m


def write_verification_statistics(
    stats: Sequence[VerificationStats], names: Sequence[str], path: str | Path
) -> Path:
    return write_csv(
        path,
        ["image_i", "image_j", "model", "matches", "inliers", "inlier_ratio",
         "median_residual_px", "homography_ratio", "success", "reason", "time_ms"],
        (
            [names[s.i], names[s.j], s.model, s.n_matches, s.n_inliers,
             f"{s.inlier_ratio:.4f}", f"{s.median_residual:.3f}",
             f"{s.homography_ratio:.3f}", int(s.success), s.reason or "",
             f"{s.time_ms:.1f}"]
            for s in stats
        ),
    )  # fmt: skip

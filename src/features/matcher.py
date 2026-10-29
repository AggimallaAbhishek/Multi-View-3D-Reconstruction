"""Descriptor matching between image pairs: k-NN, Lowe ratio test, mutual check.

Exact (brute-force) matching is the default: deterministic, and on our feature counts
as fast as FLANN. FLANN (approximate) is available for very large datasets; its
randomised k-d trees are seeded so runs are reproducible.
"""

from __future__ import annotations

import hashlib
import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from src.config import MatchConfig
from src.features.extractor import Features
from src.run import write_csv

_FLOAT_DETECTORS = {"sift", "rootsift"}
_BINARY_DETECTORS = {"orb"}


@dataclass(frozen=True, eq=False)
class Matches:
    """Correspondences between features of image ``i`` and image ``j`` (``i < j``)."""

    i: int
    j: int
    idx_i: np.ndarray  # (M,) int32 indices into features of image i, ascending
    idx_j: np.ndarray  # (M,) int32 indices into features of image j
    distances: np.ndarray  # (M,) float32 descriptor distances

    def __len__(self) -> int:
        return len(self.idx_i)

    def points(self, features_i: Features, features_j: Features) -> tuple[np.ndarray, np.ndarray]:
        """Matched pixel coordinates, each (M, 2)."""
        return features_i.points[self.idx_i], features_j.points[self.idx_j]

    @classmethod
    def empty(cls, i: int, j: int) -> Matches:
        none = np.empty(0, dtype=np.int32)
        return cls(i, j, none, none.copy(), np.empty(0, dtype=np.float32))


@dataclass(frozen=True)
class MatchStats:
    i: int
    j: int
    features_i: int
    features_j: int
    raw: int  # query features with a nearest neighbour
    after_ratio: int
    final: int  # after the mutual check (equal to after_ratio when it is disabled)
    time_ms: float
    cached: bool = False


class DescriptorMatcher:
    """k-NN matching with Lowe's ratio test and an optional mutual check.

    Exact matching uses ``cv2.batchDistance``: the same neighbours and distances as
    ``cv2.BFMatcher`` but returned as arrays, without one Python object per match.
    With ``method: flann`` (SIFT only) each image's k-d tree is built once, seeded,
    and reused for every pair the image takes part in.
    """

    def __init__(self, config: MatchConfig):
        self.config = config
        # id(descriptors) -> (descriptors, trained matcher); holding the array keeps the
        # id from being reused by another array while the index is cached.
        self._flann: dict[int, tuple[np.ndarray, cv2.FlannBasedMatcher]] = {}

    def match(
        self, features_i: Features, features_j: Features, i: int = 0, j: int = 1
    ) -> tuple[Matches, MatchStats]:
        start = time.perf_counter()
        nearest, d1, d2 = self._two_nearest(features_i, features_j)
        raw = len(nearest)
        keep = d1 < self.config.ratio * d2
        idx_i = np.flatnonzero(keep).astype(np.int32)
        idx_j = nearest[keep]
        dist = d1[keep]
        after_ratio = len(idx_i)

        if self.config.mutual and after_ratio:
            # Nearest neighbour back in image i, only for the image-j features that survived
            # the ratio test (typically 10-30 %): same result, a fraction of the work.
            used = np.unique(idx_j)
            back_used, _ = self._knn(
                features_i.detector, features_j.descriptors[used], features_i.descriptors, 1
            )
            back = np.full(len(features_j), -1, dtype=np.int32)
            back[used] = back_used[:, 0]
            mutual = back[idx_j] == idx_i
            idx_i, idx_j, dist = idx_i[mutual], idx_j[mutual], dist[mutual]

        matches = Matches(i, j, idx_i, idx_j, dist)
        stats = MatchStats(
            i=i,
            j=j,
            features_i=len(features_i),
            features_j=len(features_j),
            raw=raw,
            after_ratio=after_ratio,
            final=len(matches),
            time_ms=(time.perf_counter() - start) * 1000,
        )
        return matches, stats

    def ratios(self, features_i: Features, features_j: Features) -> np.ndarray:
        """Lowe ratio d1 / d2 for every feature of image i (the ratio test keeps < ratio).

        Correct matches concentrate at low ratios and ambiguous ones near 1, so a
        histogram of these values shows where to set ``matching.ratio``.
        """
        _, d1, d2 = self._two_nearest(features_i, features_j)
        return d1 / np.maximum(d2, 1e-12)

    def _two_nearest(
        self, features_i: Features, features_j: Features
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(nearest index in j, nearest distance, second-nearest distance) per feature of i.

        Empty when there are no query features or fewer than two features in image j,
        since the ratio test needs a second neighbour.
        """
        if features_i.detector != features_j.detector:
            raise ValueError(
                f"cannot match {features_i.detector} with {features_j.detector}: "
                "every image must use the same detector"
            )
        di, dj = features_i.descriptors, features_j.descriptors
        if len(di) == 0 or len(dj) < 2:
            empty = np.empty(0, dtype=np.float32)
            return np.empty(0, dtype=np.int32), empty, empty.copy()
        idx, dist = self._knn(features_i.detector, di, dj, 2)
        return idx[:, 0], dist[:, 0], dist[:, 1]

    def _knn(
        self, detector: str, query: np.ndarray, train: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """(N, k) indices into ``train`` and distances, nearest first."""
        if detector in _BINARY_DETECTORS:
            norm, dtype = cv2.NORM_HAMMING, cv2.CV_32S
        elif detector in _FLOAT_DETECTORS:
            if self.config.method == "flann":
                return self._flann_knn(query, train, k)
            norm, dtype = cv2.NORM_L2, cv2.CV_32F
        else:
            raise ValueError(f"no matcher for detector {detector!r}")
        dist, idx = cv2.batchDistance(
            query, train, dtype, normType=norm, K=k, update=0, crosscheck=False
        )
        return idx.astype(np.int32), dist.astype(np.float32)

    def _flann_knn(
        self, query: np.ndarray, train: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        cached = self._flann.get(id(train))
        if cached is None or cached[0] is not train:
            cv2.setRNGSeed(0)  # FLANN's randomised k-d trees use OpenCV's RNG
            index = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
            index.add([train])
            index.train()
            cached = (train, index)
            self._flann[id(train)] = cached
        knn = cached[1].knnMatch(query, k=k)
        idx = np.array([[m.trainIdx for m in row] for row in knn], dtype=np.int32)
        dist = np.array([[m.distance for m in row] for row in knn], dtype=np.float32)
        return idx, dist


class MatchCache:
    """Matches on disk, keyed by both images' descriptors and the matching settings.

    Lets a later command (``verify`` after ``matches``, or a rerun) skip matching; any
    change to features or matching settings gives new keys.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    @staticmethod
    def digest(features: Features) -> str:
        h = hashlib.sha1(f"{features.detector}{features.descriptors.shape}".encode())
        h.update(np.ascontiguousarray(features.descriptors).data)
        return h.hexdigest()

    def path_for(self, digest_i: str, digest_j: str, config: MatchConfig) -> Path:
        settings = f"{config.ratio}|{config.mutual}|{config.method}|cv{cv2.__version__}"
        key = hashlib.sha1(f"{digest_i}|{digest_j}|{settings}".encode()).hexdigest()
        return self.directory / f"{key[:24]}.npz"

    def load(self, path: Path, i: int, j: int, n_i: int, n_j: int, start: float):
        try:
            with np.load(path, allow_pickle=False) as data:
                m = Matches(i, j, data["idx_i"], data["idx_j"], data["distances"])
                raw, after_ratio = int(data["raw"]), int(data["after_ratio"])
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            return None  # unreadable entry: recompute
        stats = MatchStats(
            i, j, n_i, n_j, raw, after_ratio, len(m), (time.perf_counter() - start) * 1000,
            cached=True,
        )  # fmt: skip
        return m, stats

    def save(self, path: Path, matches: Matches, stats: MatchStats) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez(
                f,
                idx_i=matches.idx_i,
                idx_j=matches.idx_j,
                distances=matches.distances,
                raw=stats.raw,
                after_ratio=stats.after_ratio,
            )
        tmp.replace(path)


def resolve_pair_mode(n_images: int, config: MatchConfig) -> str:
    """``auto`` -> exhaustive up to ``exhaustive_max_images`` images, else sequential."""
    if config.pair_mode != "auto":
        return config.pair_mode
    return "exhaustive" if n_images <= config.exhaustive_max_images else "sequential"


def select_pairs(n_images: int, config: MatchConfig) -> list[tuple[int, int]]:
    """Image pairs (i, j), i < j, to match.

    ``exhaustive`` matches every pair (n(n-1)/2). ``sequential`` matches each image
    with the next ``sequential_window`` images, for ordered captures such as a walk
    around an object. ``auto`` picks between them by image count.
    """
    if resolve_pair_mode(n_images, config) == "exhaustive":
        return list(combinations(range(n_images), 2))
    w = config.sequential_window
    return [(i, j) for i in range(n_images) for j in range(i + 1, min(i + w, n_images - 1) + 1)]


def match_pairs(
    features: Sequence[Features],
    pairs: Sequence[tuple[int, int]],
    matcher: DescriptorMatcher,
    cache: MatchCache | None = None,
) -> tuple[list[Matches], list[MatchStats]]:
    digests = [MatchCache.digest(f) for f in features] if cache else None
    all_matches, all_stats = [], []
    for i, j in pairs:
        result = None
        if cache is not None:
            start = time.perf_counter()
            path = cache.path_for(digests[i], digests[j], matcher.config)
            if path.exists():
                result = cache.load(path, i, j, len(features[i]), len(features[j]), start)
        if result is None:
            result = matcher.match(features[i], features[j], i, j)
            if cache is not None:
                cache.save(path, *result)
        all_matches.append(result[0])
        all_stats.append(result[1])
    return all_matches, all_stats


def connectivity(stats: Sequence[MatchStats], n_images: int) -> np.ndarray:
    """For each image, the match count of its best-matching partner."""
    return match_matrix(stats, n_images).max(axis=1)


def isolated_images(stats: Sequence[MatchStats], n_images: int, min_matches: int) -> list[int]:
    """Images whose best pair has fewer than ``min_matches``: they will not register."""
    return np.flatnonzero(connectivity(stats, n_images) < min_matches).tolist()


def match_matrix(stats: Sequence[MatchStats], n_images: int) -> np.ndarray:
    """Symmetric (n, n) matrix of final match counts; unmatched pairs are 0."""
    m = np.zeros((n_images, n_images), dtype=int)
    for s in stats:
        m[s.i, s.j] = m[s.j, s.i] = s.final
    return m


def write_match_statistics(
    stats: Sequence[MatchStats], names: Sequence[str], path: str | Path
) -> Path:
    return write_csv(
        path,
        ["image_i", "image_j", "features_i", "features_j", "raw", "after_ratio",
         "after_mutual", "time_ms", "cached"],
        (
            [names[s.i], names[s.j], s.features_i, s.features_j, s.raw, s.after_ratio,
             s.final, f"{s.time_ms:.1f}", int(s.cached)]
            for s in stats
        ),
    )  # fmt: skip


def save_matches(matches: Sequence[Matches], path: str | Path) -> Path:
    """All pairs in one .npz: ``<i>_<j>`` -> (M, 2) int32 index array, plus distances."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for m in matches:
        arrays[f"idx_{m.i}_{m.j}"] = np.column_stack([m.idx_i, m.idx_j]).astype(np.int32)
        arrays[f"dist_{m.i}_{m.j}"] = m.distances
    np.savez_compressed(path, **arrays)
    return path


def load_matches(path: str | Path) -> list[Matches]:
    out = []
    with np.load(path, allow_pickle=False) as data:
        keys = sorted(
            (k for k in data.files if k.startswith("idx_")),
            key=lambda k: tuple(int(v) for v in k.split("_")[1:]),
        )
        for key in keys:
            _, i, j = key.split("_")
            idx = data[key].reshape(-1, 2)
            out.append(
                Matches(int(i), int(j), idx[:, 0].copy(), idx[:, 1].copy(), data[f"dist_{i}_{j}"])
            )
    return out

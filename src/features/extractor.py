"""Keypoint detection and description (SIFT, RootSIFT, ORB) with an on-disk cache.

Features of one image are stored as plain arrays, strongest response first:
    points       (N, 2) float64  pixel coordinates (x, y), pixel-centre convention
    descriptors  (N, 128) float32 for SIFT / RootSIFT, (N, 32) uint8 for ORB
    sizes, angles, responses  (N,) float32
"""

from __future__ import annotations

import hashlib
import logging
import time
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import FeatureConfig
from src.data.dataset import PreparedImage
from src.data.preprocessing import to_gray
from src.run import atomic_savez, write_csv

log = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class Features:
    points: np.ndarray
    descriptors: np.ndarray
    sizes: np.ndarray
    angles: np.ndarray
    responses: np.ndarray
    detector: str

    def __len__(self) -> int:
        return len(self.points)

    def subset(self, indices: np.ndarray) -> Features:
        return Features(
            points=self.points[indices],
            descriptors=self.descriptors[indices],
            sizes=self.sizes[indices],
            angles=self.angles[indices],
            responses=self.responses[indices],
            detector=self.detector,
        )

    def save(self, path: str | Path) -> Path:
        return atomic_savez(
            path,
            points=self.points,
            descriptors=self.descriptors,
            sizes=self.sizes,
            angles=self.angles,
            responses=self.responses,
            detector=np.array(self.detector),
        )

    @classmethod
    def load(cls, path: str | Path) -> Features:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                points=data["points"],
                descriptors=data["descriptors"],
                sizes=data["sizes"],
                angles=data["angles"],
                responses=data["responses"],
                detector=str(data["detector"]),
            )


class FeatureExtractor(ABC):
    name: str
    descriptor_dim: int
    descriptor_dtype: type

    def __init__(self, max_features: int = 8000):
        self.max_features = max_features

    @abstractmethod
    def _detector(self) -> cv2.Feature2D: ...

    @abstractmethod
    def signature(self) -> str:
        """Identifies everything that affects the output; used as part of the cache key."""

    def _postprocess(self, descriptors: np.ndarray) -> np.ndarray:
        return descriptors

    def detect(self, image: np.ndarray) -> Features:
        """Detect and describe keypoints in a grayscale or BGR image."""
        return self._detect_with(self._detector(), image)

    def _detect_with(self, detector: cv2.Feature2D, image: np.ndarray) -> Features:
        gray = to_gray(image)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        if not keypoints:
            return self._empty()

        responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
        # Strict cap: OpenCV's own limit is soft (ties and multi-orientation duplicates).
        order = np.argsort(-responses, kind="stable")[: self.max_features]
        keypoints = [keypoints[i] for i in order]
        return Features(
            points=np.array([kp.pt for kp in keypoints], dtype=np.float64),
            descriptors=self._postprocess(descriptors[order]),
            sizes=np.array([kp.size for kp in keypoints], dtype=np.float32),
            angles=np.array([kp.angle for kp in keypoints], dtype=np.float32),
            responses=responses[order],
            detector=self.name,
        )

    def _empty(self) -> Features:
        empty = np.empty(0, dtype=np.float32)
        return Features(
            points=np.empty((0, 2), dtype=np.float64),
            descriptors=np.empty((0, self.descriptor_dim), dtype=self.descriptor_dtype),
            sizes=empty,
            angles=empty.copy(),
            responses=empty.copy(),
            detector=self.name,
        )


class SIFTExtractor(FeatureExtractor):
    descriptor_dim = 128
    descriptor_dtype = np.float32

    def __init__(
        self,
        max_features: int = 8000,
        root_sift: bool = False,
        contrast_threshold: float = 0.04,
        min_features: int = 0,
        min_contrast_threshold: float = 0.005,
    ):
        super().__init__(max_features)
        self.root_sift = root_sift
        self.contrast_threshold = contrast_threshold
        self.min_features = min_features
        self.min_contrast_threshold = min_contrast_threshold
        self.name = "rootsift" if root_sift else "sift"

    def _detector(self, contrast_threshold: float | None = None) -> cv2.Feature2D:
        if contrast_threshold is None:
            contrast_threshold = self.contrast_threshold
        # Ask OpenCV for a little more than the cap so the strict cap picks the strongest.
        return cv2.SIFT_create(
            nfeatures=self.max_features + 64, contrastThreshold=contrast_threshold
        )

    def detect(self, image: np.ndarray) -> Features:
        """Detect and describe keypoints; while the image has fewer than ``min_features``,
        detect again with the contrast threshold halved, down to ``min_contrast_threshold``.

        SIFT's contrast threshold is absolute, so the dark side of an unevenly lit,
        low-texture object loses most of its features at a threshold that suits the rest.
        ``min_features=0`` (the constructor's default; the config's is 500) keeps the
        configured threshold for every image.
        """
        threshold = self.contrast_threshold
        features = self._detect_with(self._detector(threshold), image)
        floor = self.min_contrast_threshold * (1 - 1e-9)  # so rounding can't skip the floor
        while len(features) < self.min_features and threshold / 2 >= floor:
            threshold /= 2
            features = self._detect_with(self._detector(threshold), image)
        return features

    def _postprocess(self, descriptors: np.ndarray) -> np.ndarray:
        if not self.root_sift:
            return descriptors
        # RootSIFT: L1-normalise then take the square root, so Euclidean distance between
        # descriptors equals the Hellinger kernel distance on the original histograms.
        l1 = np.abs(descriptors).sum(axis=1, keepdims=True)
        return np.sqrt(descriptors / np.maximum(l1, 1e-12)).astype(np.float32)

    def signature(self) -> str:
        return (
            f"{self.name}-n{self.max_features}-c{self.contrast_threshold:g}"
            f"-a{self.min_features}-f{self.min_contrast_threshold:g}-cv{cv2.__version__}"
        )


class ORBExtractor(FeatureExtractor):
    name = "orb"
    descriptor_dim = 32
    descriptor_dtype = np.uint8

    def _detector(self) -> cv2.Feature2D:
        # ORB splits nfeatures across pyramid levels and then drops keypoints too close to the
        # border to describe, so it returns fewer than requested; over-request, then cap.
        return cv2.ORB_create(nfeatures=2 * self.max_features)

    def signature(self) -> str:
        return f"orb-n{self.max_features}-cv{cv2.__version__}"


def create_extractor(config: FeatureConfig) -> FeatureExtractor:
    if config.detector == "sift":
        return SIFTExtractor(
            max_features=config.max_features,
            root_sift=config.root_sift,
            contrast_threshold=config.sift_contrast_threshold,
            min_features=config.sift_adaptive_min_features,
            min_contrast_threshold=config.sift_min_contrast_threshold,
        )
    if config.detector == "orb":
        return ORBExtractor(max_features=config.max_features)
    raise ValueError(f"unknown detector {config.detector!r}")


class FeatureCache:
    """Features on disk keyed by image content and extractor signature.

    Entries are reused across runs whenever the working-resolution pixels and the
    extractor settings (including the OpenCV version) are identical.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def path_for(self, extractor: FeatureExtractor, gray: np.ndarray, label: str) -> Path:
        h = hashlib.sha1(extractor.signature().encode())
        h.update(str(gray.shape).encode())
        h.update(np.ascontiguousarray(gray).data)
        stem = Path(label).stem or "image"
        return self.directory / f"{stem}.{h.hexdigest()[:16]}.npz"

    def get_or_compute(
        self, extractor: FeatureExtractor, image: np.ndarray, label: str
    ) -> tuple[Features, bool]:
        """Returns (features, cache_hit)."""
        gray = to_gray(image)
        path = self.path_for(extractor, gray, label)
        if path.exists():
            try:
                return Features.load(path), True
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                log.warning("ignoring unreadable cache entry %s", path)
        features = extractor.detect(gray)
        features.save(path)
        return features, False


@dataclass(frozen=True)
class FeatureStats:
    image: str
    detector: str
    n_features: int
    coverage: float  # fraction of coverage-grid cells containing at least one keypoint
    time_ms: float  # extraction time (cache lookup time on a hit)
    cached: bool


def spatial_coverage(
    points: np.ndarray, width: int, height: int, grid: int = FeatureConfig.coverage_grid
) -> float:
    """Fraction of a grid x grid partition of the image that contains a keypoint.

    Low coverage means features cluster in one region, which weakens pose estimation.
    """
    if len(points) == 0:
        return 0.0
    cols = np.clip((points[:, 0] / width * grid).astype(int), 0, grid - 1)
    rows = np.clip((points[:, 1] / height * grid).astype(int), 0, grid - 1)
    return len(set(zip(rows.tolist(), cols.tolist(), strict=True))) / grid**2


def extract_features(
    images: Sequence[PreparedImage],
    extractor: FeatureExtractor,
    cache: FeatureCache | None = None,
    coverage_grid: int = FeatureConfig.coverage_grid,
) -> tuple[list[Features], list[FeatureStats]]:
    """Features for every image, in dataset order.

    Images run one after another on purpose: OpenCV's SIFT already uses all cores, and
    threading images on top of it was slower (measured 2.4 s sequential vs 3.5-5.8 s
    threaded for 13 images at 1600x1200).
    """

    def one(im: PreparedImage) -> tuple[Features, FeatureStats]:
        start = time.perf_counter()
        if cache is None:
            features, hit = extractor.detect(im.gray), False
        else:
            features, hit = cache.get_or_compute(extractor, im.gray, im.name)
        log.debug("%s: %d features (%s)", im.name, len(features), "cached" if hit else "new")
        return features, FeatureStats(
            image=im.name,
            detector=features.detector,
            n_features=len(features),
            coverage=spatial_coverage(features.points, im.width, im.height, coverage_grid),
            time_ms=(time.perf_counter() - start) * 1000,
            cached=hit,
        )

    results = [one(im) for im in images]
    return [f for f, _ in results], [s for _, s in results]


def weak_images(stats: Sequence[FeatureStats], config: FeatureConfig) -> list[FeatureStats]:
    """Images with too few features or features bunched in one region."""
    return [
        s
        for s in stats
        if s.n_features < config.warn_min_features or s.coverage < config.warn_min_coverage
    ]


def write_feature_statistics(stats: Sequence[FeatureStats], path: str | Path) -> Path:
    return write_csv(
        path,
        ["image", "detector", "n_features", "coverage", "time_ms", "cached"],
        (
            [s.image, s.detector, s.n_features, f"{s.coverage:.4f}", f"{s.time_ms:.1f}",
             int(s.cached)]
            for s in stats
        ),
    )  # fmt: skip

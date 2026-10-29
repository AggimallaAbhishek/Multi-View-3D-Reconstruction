"""The reconstruction pipeline as library code.

dataset -> features -> matches -> verify -> pose -> triangulate (initial structure) -> cloud.

Each stage reads and extends one ``PipelineResult``. The CLI and notebooks call these
functions and decide what to write or plot; nothing here touches the filesystem except
the feature cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.config import Config
from src.data.dataset import Dataset, PreparedImage, prepare_dataset
from src.features.extractor import (
    FeatureCache,
    FeatureExtractor,
    Features,
    FeatureStats,
    create_extractor,
    extract_features,
)
from src.features.matcher import (
    DescriptorMatcher,
    MatchCache,
    Matches,
    MatchStats,
    match_pairs,
    select_pairs,
)
from src.geometry.epipolar import TwoViewGeometry
from src.geometry.pose_estimation import (
    InitialPair,
    PoseStats,
    RelativePose,
    estimate_pair_poses,
    initial_pair,
)
from src.geometry.verification import VerificationStats, verify_matches
from src.pointcloud.coloring import color_points
from src.reconstruction.two_view import TwoViewStructure, build_two_view_structure

Stage = Literal["features", "matches", "verify", "pose", "triangulate", "cloud"]
STAGES: tuple[Stage, ...] = ("features", "matches", "verify", "pose", "triangulate", "cloud")


@dataclass
class PipelineResult:
    """Everything computed so far; later stages' fields stay ``None`` until they run."""

    config: Config
    dataset: Dataset
    extractor: FeatureExtractor | None = None
    features: list[Features] | None = None
    feature_stats: list[FeatureStats] | None = None
    matches: list[Matches] | None = None
    match_stats: list[MatchStats] | None = None
    geometries: list[TwoViewGeometry] | None = None
    verification_stats: list[VerificationStats] | None = None
    poses: list[RelativePose | None] | None = None
    pose_stats: list[PoseStats] | None = None
    initial_pair: InitialPair | None = None
    structure: TwoViewStructure | None = None  # set by "triangulate" if an initial pair exists
    colors: np.ndarray | None = None  # "cloud": RGB in [0, 1] of the structure's valid points

    @property
    def images(self) -> list[PreparedImage]:
        return self.dataset.images

    @property
    def names(self) -> list[str]:
        return [im.name for im in self.dataset.images]


def extract(result: PipelineResult) -> PipelineResult:
    config = result.config
    result.extractor = create_extractor(config.features)
    cache_dir = config.cache_path("features")
    result.features, result.feature_stats = extract_features(
        result.images,
        result.extractor,
        FeatureCache(cache_dir) if cache_dir else None,
        workers=config.performance.workers,
    )
    return result


def match(result: PipelineResult) -> PipelineResult:
    config = result.config
    pairs = select_pairs(len(result.features), config.matching)
    cache_dir = config.cache_path("matches")
    result.matches, result.match_stats = match_pairs(
        result.features,
        pairs,
        DescriptorMatcher(config.matching),
        MatchCache(cache_dir) if cache_dir else None,
    )
    return result


def verify(result: PipelineResult) -> PipelineResult:
    config = result.config
    result.geometries, result.verification_stats = verify_matches(
        result.images,
        result.features,
        result.matches,
        config.verification,
        workers=config.performance.workers,
    )
    return result


def estimate_poses(result: PipelineResult) -> PipelineResult:
    cfg = result.config
    result.poses, result.pose_stats = estimate_pair_poses(
        result.images,
        result.features,
        result.matches,
        result.geometries,
        cfg.pose,
    )
    result.initial_pair = initial_pair(
        result.images,
        result.features,
        result.matches,
        result.geometries,
        result.poses,
        result.pose_stats,
    )
    return result


def triangulate(result: PipelineResult) -> PipelineResult:
    pair = result.initial_pair
    if pair is not None:
        result.structure = build_two_view_structure(
            pair, result.images[pair.i].K, result.images[pair.j].K, result.config.triangulation
        )
    return result


def color_cloud(result: PipelineResult) -> PipelineResult:
    """Colour each valid point by the mean of the two images at its observations."""
    s = result.structure
    if s is not None:
        images = [result.images[s.i].image, result.images[s.j].image]  # decoded once each
        result.colors = color_points(images, s.observations, len(s.triangulation.valid))[
            s.triangulation.valid
        ]
    return result


_STEPS = {
    "features": extract,
    "matches": match,
    "verify": verify,
    "pose": estimate_poses,
    "triangulate": triangulate,
    "cloud": color_cloud,
}


def run_pipeline(
    images: str | Path, config: Config, until: Stage = "triangulate"
) -> PipelineResult:
    """Prepare the dataset and run every stage up to and including ``until``.

    Raises ``DatasetError`` if the image folder is unusable.
    """
    result = PipelineResult(config=config, dataset=prepare_dataset(images, config))
    for stage in STAGES[: STAGES.index(until) + 1]:
        _STEPS[stage](result)
    return result

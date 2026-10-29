"""The reconstruction pipeline as library code: dataset -> features -> matches -> verify -> pose.

Each stage reads and extends one ``PipelineResult``. The CLI and notebooks call these
functions and decide what to write or plot; nothing here touches the filesystem except
the feature cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

Stage = Literal["features", "matches", "verify", "pose"]
STAGES: tuple[Stage, ...] = ("features", "matches", "verify", "pose")


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

    @property
    def images(self) -> list[PreparedImage]:
        return self.dataset.images

    @property
    def names(self) -> list[str]:
        return [im.name for im in self.dataset.images]


def extract(result: PipelineResult) -> PipelineResult:
    cfg = result.config.features
    result.extractor = create_extractor(cfg)
    cache = FeatureCache(cfg.cache_dir) if cfg.cache_dir else None
    result.features, result.feature_stats = extract_features(result.images, result.extractor, cache)
    return result


def match(result: PipelineResult) -> PipelineResult:
    cfg = result.config.matching
    pairs = select_pairs(len(result.features), cfg)
    result.matches, result.match_stats = match_pairs(result.features, pairs, DescriptorMatcher(cfg))
    return result


def verify(result: PipelineResult) -> PipelineResult:
    result.geometries, result.verification_stats = verify_matches(
        result.images, result.features, result.matches, result.config.verification
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
        cfg.verification.planar_ratio,
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


_STEPS = {"features": extract, "matches": match, "verify": verify, "pose": estimate_poses}


def run_pipeline(images: str | Path, config: Config, until: Stage = "pose") -> PipelineResult:
    """Prepare the dataset and run every stage up to and including ``until``.

    Raises ``DatasetError`` if the image folder is unusable.
    """
    result = PipelineResult(config=config, dataset=prepare_dataset(images, config))
    for stage in STAGES[: STAGES.index(until) + 1]:
        _STEPS[stage](result)
    return result

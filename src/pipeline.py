"""The reconstruction pipeline as library code.

dataset -> features -> matches -> verify -> pose -> triangulate (initial structure) -> cloud
-> reconstruct (every image: tracks, incremental registration, bundle adjustment)
-> filter (outlier removal on the multi-view cloud)
-> dense (optional: stereo depth maps on the cameras, fused into a dense cloud).

Each stage reads and extends one ``PipelineResult``. The CLI and notebooks call these
functions and decide what to write or plot; nothing here touches the filesystem except
the feature cache.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import numpy as np

from src.config import Config
from src.data.dataset import Dataset, PreparedImage, prepare_dataset
from src.dense import DenseCloud, ViewPlan, fuse_depth_maps, pair_depth, plan_views
from src.features.extractor import (
    FeatureCache,
    FeatureExtractor,
    Features,
    FeatureStats,
    create_extractor,
    extract_features,
)
from src.features.matcher import (
    MatchCache,
    Matches,
    MatchStats,
    create_matcher,
    match_pairs,
    select_pairs,
)
from src.geometry.calibration import Intrinsics
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
from src.pointcloud.filtering import CloudFilterResult, filter_cloud
from src.reconstruction.bundle_adjustment import scale_focal
from src.reconstruction.distortion import (
    RadialDistortion,
    SelfCalibration,
    estimate_radial_distortion,
)
from src.reconstruction.incremental import Reconstruction, reconstruct
from src.reconstruction.tracks import Tracks, ViewGraph, build_tracks
from src.reconstruction.two_view import TwoViewStructure, build_two_view_structure
from src.run import parallel_map

Stage = Literal[
    "features",
    "matches",
    "verify",
    "pose",
    "triangulate",
    "cloud",
    "reconstruct",
    "filter",
    "dense",
]
STAGES: tuple[Stage, ...] = get_args(Stage)
CORE_STAGES: tuple[Stage, ...] = STAGES[: STAGES.index("filter") + 1]  # dense is optional


def stages_until(stage: Stage) -> tuple[Stage, ...]:
    """Every stage up to and including ``stage``, in run order."""
    return STAGES[: STAGES.index(stage) + 1]


@dataclass
class PipelineResult:
    """Everything computed so far; later stages' fields stay ``None`` until they run."""

    config: Config
    dataset: Dataset
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage, run order
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
    tracks: Tracks | None = None  # "reconstruct", from the verified matches
    reconstruction: Reconstruction | None = None  # "reconstruct", if an initial pair exists
    filtering: CloudFilterResult | None = None  # "filter": which landmarks to export
    dense: DenseResult | None = None  # "dense", if a reconstruction exists
    distortion: SelfCalibration | None = None  # "reconstruct", when self-calibrating

    @property
    def images(self) -> list[PreparedImage]:
        return self.dataset.images

    @property
    def names(self) -> list[str]:
        return [im.name for im in self.dataset.images]


def extract(result: PipelineResult) -> PipelineResult:
    config = result.config
    result.extractor = create_extractor(config.features, config.performance.torch_device)
    cache_dir = config.cache_path("features")
    result.features, result.feature_stats = extract_features(
        result.images,
        result.extractor,
        FeatureCache(cache_dir) if cache_dir else None,
        config.features.coverage_grid,
    )
    return result


def match(result: PipelineResult) -> PipelineResult:
    config = result.config
    pairs = select_pairs(len(result.features), config.matching)
    cache_dir = config.cache_path("matches")
    result.matches, result.match_stats = match_pairs(
        result.features,
        pairs,
        create_matcher(
            config.matching,
            [(im.width, im.height) for im in result.images],
            config.performance.torch_device,
        ),
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


def reconstruct_scene(result: PipelineResult) -> PipelineResult:
    """Link verified matches into tracks, grow the reconstruction from the initial pair
    and colour its landmarks (mean over their observations)."""
    result.tracks = build_tracks(ViewGraph.from_verified(result.matches, result.geometries))
    pair = result.initial_pair
    if pair is None:
        return result
    intrinsics = [im.K for im in result.images]
    rec = reconstruct(pair, intrinsics, result.tracks, result.features, result.config)
    cameras, obs, X = rec.landmarks.as_problem(rec.registry, result.features)
    images = [result.images[m].image for m in rec.registry.images]  # decoded once each
    rgb = color_points(images, obs, len(X))
    rec.landmarks.color = np.round(rgb * 255).astype(np.uint8)
    result.reconstruction = rec
    return result


def filter_scene(result: PipelineResult) -> PipelineResult:
    """Decide which landmarks of the reconstruction survive the outlier filters."""
    rec = result.reconstruction
    if rec is not None:
        lm = rec.landmarks
        result.filtering = filter_cloud(
            lm.position, result.config.filtering, errors=lm.error,
            track_lengths=lm.track_lengths(),
        )  # fmt: skip
    return result


@dataclass(frozen=True, eq=False)
class DenseResult:
    cloud: DenseCloud
    plan: ViewPlan  # indices into ``reconstruction.registry.images``
    coverage: list[float | None]  # per image: fraction of pixels with a depth (None: no partner)
    voxel_size: float


def dense_scene(result: PipelineResult) -> PipelineResult:
    """A depth map per registered image from stereo with its partner view, fused by
    multi-view consistency. The filtered sparse points set the disparity ranges, the
    scene centre and the voxel size (``dense.voxel_fraction`` of their extent)."""
    rec, cfg = result.reconstruction, result.config.dense
    if rec is None:
        return result
    images = rec.registry.images
    keep = result.filtering.keep if result.filtering is not None else slice(None)
    points = rec.landmarks.position[keep]
    if not len(points):  # no sparse points to plan stereo or set depth ranges with
        n = len(images)
        result.dense = DenseResult(
            cloud=DenseCloud.empty(),
            plan=ViewPlan(partner=[None] * n, checks=[[] for _ in range(n)]),
            coverage=[None] * n,
            voxel_size=0.0,
        )
        return result
    cameras = [rec.registry.camera(m) for m in images]
    pixels = [result.images[m].image for m in images]
    plan = plan_views(cameras, points, cfg)

    def depth(i: int) -> np.ndarray | None:
        j = plan.partner[i]
        return None if j is None else pair_depth(
            pixels[i], cameras[i], pixels[j], cameras[j], points, cfg)  # fmt: skip

    # Independent per image; OpenCV releases the GIL, so threads run them in parallel.
    depths = parallel_map(depth, range(len(images)), result.config.performance.workers)
    voxel = cfg.voxel_fraction * float(np.linalg.norm(np.ptp(points, axis=0)))
    cloud = fuse_depth_maps(depths, cameras, pixels, plan.checks, voxel, cfg)
    coverage = [None if d is None else float(np.isfinite(d).mean()) for d in depths]
    result.dense = DenseResult(cloud, plan, coverage, voxel)
    return result


_STEPS = {
    "features": extract,
    "matches": match,
    "verify": verify,
    "pose": estimate_poses,
    "triangulate": triangulate,
    "cloud": color_cloud,
    "reconstruct": reconstruct_scene,
    "filter": filter_scene,
    "dense": dense_scene,
}


StageCallback = Callable[[Stage, PipelineResult], None]


def run_pipeline(
    images: str | Path,
    config: Config,
    until: Stage = "triangulate",
    on_stage: StageCallback | None = None,
) -> PipelineResult:
    """Prepare the dataset and run every stage up to and including ``until``.

    ``on_stage(stage, result)`` is called just before each stage runs, for progress
    reporting. Raises ``DatasetError`` if the image folder is unusable.
    """
    start = time.perf_counter()
    result = PipelineResult(config=config, dataset=prepare_dataset(images, config))
    result.timings["dataset"] = time.perf_counter() - start
    stages = stages_until(until)
    if "reconstruct" in stages:
        _run_stages(result, stages_until("reconstruct"), on_stage)
        if _self_calibrates(result):
            result = _self_calibrate(result, images, on_stage)
        stages = stages[len(stages_until("reconstruct")) :]
    _run_stages(result, stages, on_stage)
    return result


def _run_stages(
    result: PipelineResult,
    stages: Sequence[Stage],
    on_stage: StageCallback | None,
    earlier: dict[str, float] | None = None,
) -> None:
    """Run ``stages`` in order, timing each (plus its time in ``earlier`` passes)."""
    for stage in stages:
        if on_stage is not None:
            on_stage(stage, result)
        start = time.perf_counter()
        _STEPS[stage](result)
        result.timings[stage] = (earlier or {}).get(stage, 0.0) + time.perf_counter() - start


def _self_calibrates(result: PipelineResult) -> bool:
    """Radial distortion is estimated when the focal length is refined (so E3's fixed-K
    runs stay fixed), the images can be undistorted, every image starts from one K, no
    distortion is known, at least ``distortion_min_images`` images are registered and
    the view is wide enough to observe it (``distortion_min_field_radius``)."""
    config, rec = result.config, result.reconstruction
    ba, first = config.bundle_adjustment, result.images[0]
    return bool(
        ba.enabled and ba.refine_focal and ba.estimate_distortion
        and config.preprocessing.undistort
        and rec is not None and len(rec.registry.images) >= ba.distortion_min_images
        and all(np.array_equal(im.K, first.K) for im in result.images)
        and not any(np.any(im.source_dist) for im in result.images)
        and RadialDistortion.field_radius(first.K, (first.width, first.height))
        >= ba.distortion_min_field_radius
    )  # fmt: skip


@dataclass(frozen=True, eq=False)
class _Pass:
    result: PipelineResult
    applied: RadialDistortion  # undistortion applied to its images
    scale: float  # its starting K / the first pass's


def _self_calibrate(
    result: PipelineResult, images: str | Path, on_stage: StageCallback | None
) -> PipelineResult:
    """Estimate radial distortion from the reconstruction; while it moves an image corner
    by more than ``distortion_min_shift_px`` (and passes remain), undistort the images
    with everything estimated so far and run every stage up to ``reconstruct`` again.
    A pass is kept if it registers more images, or as many without a higher mean
    reprojection error; otherwise it is discarded and the loop stops. An estimate whose
    focal scale leaves ``max_focal_change`` is not applied. Time spent on every pass is
    added to each stage's timing; ``on_stage`` is called again for every re-run stage."""
    ba, first = result.config.bundle_adjustment, result.images[0]
    size = (first.width, first.height)
    kept = _Pass(result, RadialDistortion(0.0, 0.0), 1.0)
    estimates, passes = [], 0
    while True:
        start = time.perf_counter()
        rec = kept.result.reconstruction
        cameras, obs, points = rec.landmarks.as_problem(rec.registry, kept.result.features)
        estimate = estimate_radial_distortion(cameras, obs, points, size, ba)
        estimates.append(estimate)
        kept.result.timings["reconstruct"] += time.perf_counter() - start
        scale = kept.scale * rec.focal_scale * estimate.focal_scale
        if (
            estimate.shift_px < ba.distortion_min_shift_px
            or len(estimates) > ba.distortion_passes
            or not 1 / ba.max_focal_change <= scale <= ba.max_focal_change
        ):
            break  # fmt: skip
        applied = kept.applied.then(estimate.distortion)
        intrinsics = Intrinsics(scale_focal(first.source_K, scale), applied.coefficients,
                                *first.original_size, first.intrinsics_source)  # fmt: skip
        config = result.config.with_overrides({"camera.focal_scale": 1.0})  # now in K
        earlier = kept.result.timings
        start = time.perf_counter()
        candidate = PipelineResult(config=config,
                                   dataset=prepare_dataset(images, config, intrinsics))  # fmt: skip
        candidate.timings["dataset"] = earlier["dataset"] + time.perf_counter() - start
        _run_stages(candidate, stages_until("reconstruct"), on_stage, earlier)
        if _quality(candidate) < _quality(kept.result):
            kept.result.timings.update(candidate.timings)  # its time was still spent
            break
        kept, passes = _Pass(candidate, applied, scale), passes + 1
    kept.result.distortion = SelfCalibration(estimates, kept.applied, kept.scale, passes)
    return kept.result


def _quality(result: PipelineResult) -> tuple[int, float]:
    """(images registered, -mean reprojection error), compared in that order: a wrong
    undistortion leaves the images registered but raised the error 4x (rendered views;
    a right one lowered it 2x on rendered views and ETH3D pipes)."""
    rec = result.reconstruction
    if rec is None:
        return 0, 0.0
    errors = rec.landmarks.error
    # no landmarks: no error to compare, and NaN would compare as "not worse"
    score = -float(np.mean(errors)) if len(errors) else -np.inf  # no landmarks: worst
    return len(rec.registry.images), score

"""Typed pipeline configuration loaded from YAML.

Each YAML section maps to a frozen dataclass. Missing keys fall back to the
dataclass defaults; unknown keys and wrongly typed values raise ``ConfigError``
so that typos in experiment configs fail loudly instead of being ignored.
"""

from __future__ import annotations

import dataclasses
import math
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file is malformed."""


@dataclass(frozen=True)
class DatasetConfig:
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    min_images: int = 2
    blur_threshold: float = 30.0
    blur_relative: float = 0.35
    blur_working_side: int = 800
    dhash_size: int = 32
    duplicate_max_fraction: float = 0.04
    max_images: int | None = None


@dataclass(frozen=True)
class PreprocessingConfig:
    max_side: int | None = 1600
    undistort: bool = True


@dataclass(frozen=True)
class CameraConfig:
    calibration_file: str | None = None
    auto_calibration_file: bool = True
    use_exif: bool = True
    fallback_focal_factor: float = 1.2
    aspect_tolerance: float = 0.01
    focal_scale: float = 1.0


@dataclass(frozen=True)
class CalibrationConfig:
    min_views: int = 3
    subpix_window: int = 11
    subpix_max_iter: int = 30
    subpix_eps: float = 0.001


@dataclass(frozen=True)
class FeatureConfig:
    detector: Literal["sift", "orb", "disk"] = "sift"
    max_features: int = 8000
    root_sift: bool = False
    sift_contrast_threshold: float = 0.02
    sift_adaptive_min_features: int = 500
    sift_min_contrast_threshold: float = 0.005
    affine_simulation: bool = False
    affine_max_tilt: int = 5
    affine_tilt_step: float = 1.4142135623730951
    affine_rotate_step_base: float = 72.0
    warn_min_features: int = 100
    warn_min_coverage: float = 0.25
    coverage_grid: int = 8


@dataclass(frozen=True)
class MatchConfig:
    ratio: float = 0.75
    mutual: bool = True
    method: Literal["bruteforce", "flann", "lightglue"] = "bruteforce"
    flann_trees: int = 5
    flann_checks: int = 64
    lightglue_filter_threshold: float = 0.7
    pair_mode: Literal["auto", "exhaustive", "sequential"] = "auto"
    exhaustive_max_images: int = 50
    sequential_window: int = 5
    min_matches: int = 30


@dataclass(frozen=True)
class VerificationConfig:
    method: Literal["auto", "essential", "fundamental"] = "auto"
    estimator: Literal["usac_accurate", "usac_magsac", "ransac"] = "usac_accurate"
    ransac_threshold_px: float = 1.5
    confidence: float = 0.999
    min_inliers: int = 30
    planar_ratio: float = 0.8
    refine: bool = True


@dataclass(frozen=True)
class PoseConfig:
    min_points: int = 30
    min_cheirality_ratio: float = 0.9
    min_parallax_deg: float = 1.0
    max_depth: float = 50.0
    init_min_parallax_deg: float = 4.0


@dataclass(frozen=True)
class TriangulationConfig:
    max_reprojection_error_px: float = 2.0
    min_angle_deg: float = 1.5
    refine: bool = True
    refine_iterations: int = 5


@dataclass(frozen=True)
class RegistrationConfig:
    reprojection_error_px: float = 4.0
    confidence: float = 0.999
    max_iterations: int = 1000
    min_inliers: int = 30
    min_inlier_ratio: float = 0.25
    refine: bool = True
    relative_pose_fallback: bool = True
    fallback_estimator: Literal["usac_accurate", "usac_magsac", "ransac"] = "usac_accurate"
    fallback_epipolar_threshold_px: float = 1.5
    fallback_min_links: int = 6
    fallback_min_pair_inliers: int = 50


@dataclass(frozen=True)
class BundleAdjustmentConfig:
    enabled: bool = True
    loss: Literal["linear", "soft_l1", "huber", "cauchy"] = "soft_l1"
    loss_scale_px: float = 1.0
    max_iterations: int = 100
    function_tolerance: float = 1e-6
    local_window: int = 5
    global_every: int = 5
    max_refinements: int = 3
    refine_structure: bool = True
    refine_focal: bool = True
    max_focal_change: float = 3.0
    estimate_distortion: bool = True
    distortion_min_shift_px: float = 2.0
    distortion_passes: int = 2
    distortion_min_images: int = 5
    distortion_min_field_radius: float = 0.3


@dataclass(frozen=True)
class FilteringConfig:
    max_reprojection_error_px: float = 1.0
    min_track_length: int = 2
    statistical_nb_neighbors: int = 20
    statistical_std_ratio: float = 2.0
    radius_nb_points: int = 4
    radius_scale: float = 5.0


@dataclass(frozen=True)
class DenseConfig:
    min_angle_deg: float = 3.0
    target_angle_deg: float = 8.0
    max_angle_deg: float = 20.0
    rectify_alpha: float = 1.0
    depth_quantile: float = 0.01
    disparity_margin_px: int = 16
    max_disparities: int = 384
    block_size: int = 3
    sgbm_p1_factor: int = 8
    sgbm_p2_factor: int = 32
    uniqueness_ratio: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    min_texture: float = 2.0
    edge_erosion_px: int = 1
    fusion_views: int = 4
    min_consistent_views: int = 2
    depth_tolerance: float = 0.01
    voxel_fraction: float = 0.001
    normal_neighbours: int = 30
    poisson_depth: int = 9
    poisson_threads: int = 1
    mesh_density_quantile: float = 0.02


@dataclass(frozen=True)
class EvaluationConfig:
    completeness_tolerance: float = 0.01
    silhouette_threshold: int = 15
    silhouette_close_px: int = 7
    silhouette_margin_px: int = 2


@dataclass(frozen=True)
class OutputConfig:
    root: str = "outputs"
    cache_dir: str | None = "outputs/cache"


@dataclass(frozen=True)
class PerformanceConfig:
    workers: int = 0
    torch_device: str = "auto"


@dataclass(frozen=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_mb: int = 500
    max_upload_images: int = 200
    poll_interval_ms: int = 1000
    open_browser: bool = True
    thumbnails: int = 12
    thumbnail_px: int = 240
    thumbnail_quality: int = 82
    showcase_frame_px: int = 960
    showcase_quality: int = 80
    showcase_photos: int = 6


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    matching: MatchConfig = field(default_factory=MatchConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    bundle_adjustment: BundleAdjustmentConfig = field(default_factory=BundleAdjustmentConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    web: WebConfig = field(default_factory=WebConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view, suitable for saving next to run outputs."""
        return _to_plain(dataclasses.asdict(self))

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Copy with dotted-key overrides, e.g. ``{"features.detector": "orb"}``.

        The result is validated like a config file: unknown keys and wrong types raise
        ``ConfigError``. ``None`` values are ignored (an option that was not given).
        """
        raw = self.to_dict()
        for key, value in overrides.items():
            if value is None:
                continue
            section, _, name = key.partition(".")
            if section not in raw or not name:
                raise ConfigError(f"override {key!r}: expected <section>.<key>")
            raw[section][name] = value
        return config_from_dict(raw, source="overrides")

    def cache_path(self, name: str) -> Path | None:
        """Directory of one cache (``features``, ``matches``, ``images``), or None if off."""
        return Path(self.output.cache_dir) / name if self.output.cache_dir else None


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config file; ``None`` loads ``configs/default.yaml``."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return config_from_dict(raw, source=str(path))


def config_from_args(args: Any) -> Config:
    """``args.config`` (None: the default config) with every command-line option whose
    argparse ``dest`` is a dotted config key (``"web.port"``) applied."""
    overrides = {key: value for key, value in vars(args).items() if "." in key}
    return load_config(args.config).with_overrides(overrides)


def config_from_dict(raw: dict[str, Any], source: str = "<dict>") -> Config:
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    config = _build(Config, raw, source)
    _check_ranges(config, source)
    if config.matching.method == "lightglue" and config.features.detector != "disk":
        raise ConfigError(
            f"{source}: matching.method lightglue needs features.detector disk "
            f"(got {config.features.detector!r})"
        )
    return config


# Validity limits (what a setting can mean at all), not tuning thresholds: those live in
# configs/default.yaml. Every number must be finite and >= 0 unless its range here
# starts below 0. key -> (low, high, low exclusive?, high exclusive?); None is unbounded.
_RANGES: dict[str, tuple[float | None, float | None, bool, bool]] = {
    "dataset.min_images": (2, None, False, False),
    "dataset.max_images": (1, None, False, False),
    "preprocessing.max_side": (1, None, False, False),
    "features.max_features": (1, None, False, False),
    "matching.ratio": (0, 1, True, False),
    "verification.confidence": (0, 1, True, True),
    "registration.confidence": (0, 1, True, True),
    "verification.ransac_threshold_px": (0, None, True, False),
    "registration.reprojection_error_px": (0, None, True, False),
    "triangulation.max_reprojection_error_px": (0, None, True, False),
    "filtering.max_reprojection_error_px": (0, None, True, False),
    "camera.focal_scale": (0, None, True, False),
    "bundle_adjustment.loss_scale_px": (0, None, True, False),
    "matching.sequential_window": (1, None, False, False),
    "calibration.min_views": (1, None, False, False),
    "calibration.subpix_window": (1, None, False, False),
    "features.coverage_grid": (1, None, False, False),
    "dense.fusion_views": (1, None, False, False),
    "web.max_upload_images": (1, None, False, False),
    "web.poll_interval_ms": (1, None, False, False),
    "web.thumbnail_quality": (0, 100, False, False),
    "web.showcase_quality": (0, 100, False, False),
    "web.port": (1, 65535, False, False),
    "dense.sgbm_p1_factor": (0, None, True, False),  # > 0: 0 would not penalise 1 px steps
    "dense.rectify_alpha": (-1, 1, False, False),  # OpenCV: -1 means "let it choose"
    "dense.poisson_threads": (-1, None, False, False),  # Open3D: -1 means all cores
}


def _check_ranges(config: Config, source: str) -> None:
    """Range-check every number, so a bad value fails here with its key instead of
    deep inside a stage (a negative max_images crashed np.linspace)."""
    for key, value in _numbers(config):
        if value is None:
            continue
        where = f"{source}: {key}"
        if not math.isfinite(value):  # NaN compares False with every bound below
            raise ConfigError(f"{where} must be a finite number (got {value})")
        low, high, low_open, high_open = _RANGES.get(key, (0, None, False, False))
        if low is not None and (value <= low if low_open else value < low):
            raise ConfigError(f"{where} must be {'>' if low_open else '>='} {low} (got {value})")
        if high is not None and (value >= high if high_open else value > high):
            raise ConfigError(f"{where} must be {'<' if high_open else '<='} {high} (got {value})")
    dataset = config.dataset
    if dataset.max_images is not None and dataset.max_images < dataset.min_images:
        raise ConfigError(
            f"{source}: dataset.max_images ({dataset.max_images}) must be >= "
            f"dataset.min_images ({dataset.min_images})"
        )
    if config.dense.sgbm_p2_factor <= config.dense.sgbm_p1_factor:  # OpenCV requires P2 > P1
        raise ConfigError(
            f"{source}: dense.sgbm_p2_factor ({config.dense.sgbm_p2_factor}) must be > "
            f"dense.sgbm_p1_factor ({config.dense.sgbm_p1_factor})"
        )
    if config.dense.block_size % 2 == 0:
        raise ConfigError(f"{source}: dense.block_size must be odd (got {config.dense.block_size})")


def _numbers(obj: Any, prefix: str = "") -> typing.Iterator[tuple[str, Any]]:
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if dataclasses.is_dataclass(value):
            yield from _numbers(value, f"{prefix}{f.name}.")
        elif value is None or (isinstance(value, int | float) and not isinstance(value, bool)):
            yield f"{prefix}{f.name}", value


def _build(cls: type, raw: dict[str, Any], where: str) -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}")

    kwargs = {}
    for name, value in raw.items():
        hint = hints[name]
        key_path = f"{where}.{name}"
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, dict):
                raise ConfigError(f"{key_path}: expected a mapping")
            kwargs[name] = _build(hint, value, key_path)
        else:
            kwargs[name] = _coerce(value, hint, key_path)
    return cls(**kwargs)


def _coerce(value: Any, hint: Any, where: str) -> Any:
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in (a for a in args if a is not type(None)):
            try:
                return _coerce(value, arg, where)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError("; ".join(errors))

    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{where}: {value!r} is not one of {list(args)}")
        return value

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list")
        return tuple(_coerce(v, args[0], where) for v in value)

    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint is int and isinstance(value, bool):
        raise ConfigError(f"{where}: expected int, got bool")
    if not isinstance(value, hint):
        raise ConfigError(f"{where}: expected {hint.__name__}, got {type(value).__name__}")
    return value


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj

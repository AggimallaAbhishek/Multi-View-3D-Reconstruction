"""Typed pipeline configuration loaded from YAML.

Each YAML section maps to a frozen dataclass. Missing keys fall back to the
dataclass defaults; unknown keys and wrongly typed values raise ``ConfigError``
so that typos in experiment configs fail loudly instead of being ignored.
"""

from __future__ import annotations

import dataclasses
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


@dataclass(frozen=True)
class PreprocessingConfig:
    max_side: int | None = 1600
    undistort: bool = True


@dataclass(frozen=True)
class CameraConfig:
    calibration_file: str | None = None
    use_exif: bool = True
    fallback_focal_factor: float = 1.2
    aspect_tolerance: float = 0.01


@dataclass(frozen=True)
class CalibrationConfig:
    min_views: int = 3
    subpix_window: int = 11
    subpix_max_iter: int = 30
    subpix_eps: float = 0.001


@dataclass(frozen=True)
class FeatureConfig:
    detector: Literal["sift", "orb"] = "sift"
    max_features: int = 8000
    root_sift: bool = False
    sift_contrast_threshold: float = 0.04
    warn_min_features: int = 100
    warn_min_coverage: float = 0.25
    coverage_grid: int = 8


@dataclass(frozen=True)
class MatchConfig:
    ratio: float = 0.75
    mutual: bool = True
    method: Literal["bruteforce", "flann"] = "bruteforce"
    flann_trees: int = 5
    flann_checks: int = 64
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


@dataclass(frozen=True)
class BundleAdjustmentConfig:
    enabled: bool = True
    loss: Literal["linear", "soft_l1", "huber", "cauchy"] = "soft_l1"
    loss_scale_px: float = 1.0
    max_iterations: int = 100
    function_tolerance: float = 1e-6
    local_window: int = 5
    global_every: int = 5


@dataclass(frozen=True)
class FilteringConfig:
    statistical_nb_neighbors: int = 20
    statistical_std_ratio: float = 2.0


@dataclass(frozen=True)
class OutputConfig:
    root: str = "outputs"
    cache_dir: str | None = "outputs/cache"


@dataclass(frozen=True)
class PerformanceConfig:
    workers: int = 0


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
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

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


def config_from_dict(raw: dict[str, Any], source: str = "<dict>") -> Config:
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    return _build(Config, raw, source)


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

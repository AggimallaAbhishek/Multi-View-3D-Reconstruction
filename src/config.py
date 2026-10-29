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
    blur_threshold: float = 100.0
    duplicate_max_distance: int = 2


@dataclass(frozen=True)
class PreprocessingConfig:
    max_side: int | None = 1600
    undistort: bool = True


@dataclass(frozen=True)
class CameraConfig:
    calibration_file: str | None = None
    use_exif: bool = True
    fallback_focal_factor: float = 1.2


@dataclass(frozen=True)
class FeatureConfig:
    detector: Literal["sift", "orb"] = "sift"
    max_features: int = 8000
    root_sift: bool = False
    sift_contrast_threshold: float = 0.04
    cache_dir: str | None = "outputs/cache/features"


@dataclass(frozen=True)
class MatchConfig:
    ratio: float = 0.75
    mutual: bool = True
    method: Literal["bruteforce", "flann"] = "bruteforce"
    pair_mode: Literal["exhaustive", "sequential"] = "exhaustive"
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


@dataclass(frozen=True)
class TriangulationConfig:
    max_reprojection_error_px: float = 2.0
    min_angle_deg: float = 1.5


@dataclass(frozen=True)
class FilteringConfig:
    statistical_nb_neighbors: int = 20
    statistical_std_ratio: float = 2.0


@dataclass(frozen=True)
class OutputConfig:
    root: str = "outputs"
    save_intermediates: bool = True


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    matching: MatchConfig = field(default_factory=MatchConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view, suitable for saving next to run outputs."""
        return _to_plain(dataclasses.asdict(self))


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

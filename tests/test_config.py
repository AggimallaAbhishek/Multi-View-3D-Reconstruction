import pytest

from src.config import (
    Config,
    ConfigError,
    FeatureConfig,
    config_from_dict,
    load_config,
)


def test_default_config_file_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.features.detector == "sift"
    assert isinstance(cfg.dataset.extensions, tuple)
    assert isinstance(cfg.verification.ransac_threshold_px, float)


def test_default_file_matches_dataclass_defaults():
    # configs/default.yaml and the dataclass defaults must not drift apart.
    assert load_config() == Config()


def test_missing_sections_use_defaults():
    cfg = config_from_dict({"features": {"detector": "orb"}})
    assert cfg.features == FeatureConfig(detector="orb")
    assert cfg.matching == Config().matching


@pytest.mark.parametrize(
    "raw",
    [
        {"featurez": {}},
        {"features": {"detectr": "sift"}},
    ],
    ids=["unknown-section", "unknown-key"],
)
def test_unknown_keys_raise(raw):
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict(raw)


@pytest.mark.parametrize(
    "raw",
    [
        {"features": {"detector": "surf"}},
        {"features": {"max_features": "many"}},
        {"features": {"max_features": True}},
        {"matching": {"ratio": "0.8"}},
        {"features": 3},
    ],
    ids=["bad-literal", "str-for-int", "bool-for-int", "str-for-float", "scalar-section"],
)
def test_invalid_values_raise(raw):
    with pytest.raises(ConfigError):
        config_from_dict(raw)


def test_int_is_accepted_for_float():
    cfg = config_from_dict({"matching": {"ratio": 1}})
    assert cfg.matching.ratio == 1.0
    assert isinstance(cfg.matching.ratio, float)


def test_optional_accepts_null():
    cfg = config_from_dict({"preprocessing": {"max_side": None}})
    assert cfg.preprocessing.max_side is None


def test_config_is_immutable():
    cfg = Config()
    with pytest.raises(AttributeError):
        cfg.features.detector = "orb"  # type: ignore[misc]


def test_to_dict_round_trips():
    cfg = load_config()
    assert config_from_dict(cfg.to_dict()) == cfg

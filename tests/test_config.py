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


def test_web_defaults():
    web = load_config().web
    assert (web.host, web.port) == ("127.0.0.1", 8000)
    assert (web.min_images, web.max_images, web.max_upload_mb) == (2, 200, 500)
    assert web.image_extensions == (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    assert web.poll_interval_ms == 1000
    assert web.open_browser is True


def test_missing_sections_use_defaults():
    cfg = config_from_dict({"features": {"detector": "orb"}})
    assert cfg.features == FeatureConfig(detector="orb")
    assert cfg.matching == Config().matching


@pytest.mark.parametrize(
    "raw",
    [
        {"featurez": {}},
        {"features": {"detectr": "sift"}},
        {"web": {"nope": 1}},
    ],
    ids=["unknown-section", "unknown-key", "unknown-web-key"],
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
        {"web": {"port": "x"}},
        {"web": {"image_extensions": ".jpg"}},
    ],
    ids=[
        "bad-literal",
        "str-for-int",
        "bool-for-int",
        "str-for-float",
        "scalar-section",
        "str-for-web-port",
        "scalar-for-web-extensions",
    ],
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


def test_with_overrides_applies_dotted_keys_and_ignores_none():
    cfg = Config().with_overrides(
        {"features.detector": "orb", "matching.ratio": 0.8, "verification.method": None}
    )
    assert cfg.features.detector == "orb"
    assert cfg.matching.ratio == 0.8
    assert cfg.verification.method == Config().verification.method


@pytest.mark.parametrize(
    "overrides",
    [
        {"features.detectr": "orb"},
        {"nosection.key": 1},
        {"features": "orb"},
        {"matching.ratio": "x"},
    ],
    ids=["unknown-key", "unknown-section", "no-dot", "wrong-type"],
)
def test_with_overrides_validates(overrides):
    with pytest.raises(ConfigError):
        Config().with_overrides(overrides)


def test_cache_path():
    assert str(Config().cache_path("features")) == "outputs/cache/features"
    assert config_from_dict({"output": {"cache_dir": None}}).cache_path("features") is None

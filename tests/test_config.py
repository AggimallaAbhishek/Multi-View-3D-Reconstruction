import pytest

from src.config import (
    DEFAULT_CONFIG_PATH,
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


def test_web_server_is_local_only_by_default():
    # The web interface has no authentication, so it must not listen beyond localhost.
    assert load_config().web.host == "127.0.0.1"


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
        {"web": {"max_upload_images": 2.5}},
    ],
    ids=[
        "bad-literal",
        "str-for-int",
        "bool-for-int",
        "str-for-float",
        "scalar-section",
        "str-for-web-port",
        "float-for-web-upload-limit",
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


def test_lightglue_requires_disk_in_the_config():
    config = load_config()
    with pytest.raises(ConfigError, match="lightglue"):
        config.with_overrides({"matching.method": "lightglue"})
    ok = config.with_overrides({"matching.method": "lightglue", "features.detector": "disk"})
    assert ok.matching.method == "lightglue"


def test_learned_config_file_selects_disk_and_lightglue():
    cfg = load_config(DEFAULT_CONFIG_PATH.with_name("learned.yaml"))
    assert (cfg.features.detector, cfg.matching.method) == ("disk", "lightglue")
    assert cfg.verification == load_config().verification  # nothing else changes


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("dataset.max_images", -1),  # crashed in np.linspace
        ("dataset.max_images", 0),
        ("dataset.min_images", 1),  # two photos at least
        ("preprocessing.max_side", 0),  # meant 1x1 fresh, 'no resize' from the cache
        ("features.max_features", 0),
        ("matching.ratio", 0.0),
        ("matching.ratio", 1.2),
        ("verification.confidence", 1.0),
        ("verification.ransac_threshold_px", 0.0),
        ("filtering.min_track_length", -2),
        ("dense.block_size", 4),  # SGBM needs an odd block
        ("web.thumbnail_quality", 101),
        ("web.port", 70000),
        ("performance.workers", -1),
        ("calibration.min_views", 0),
        ("calibration.subpix_window", 0),
        ("features.coverage_grid", 0),
        ("dense.fusion_views", 0),
        ("dense.poisson_threads", -2),
        ("web.max_upload_images", 0),
        ("web.poll_interval_ms", 0),
        ("matching.ratio", float("nan")),  # NaN compares False with every bound
        ("verification.confidence", float("nan")),
        ("camera.focal_scale", float("inf")),
    ],
)
def test_out_of_range_values_name_their_key(key, value):
    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_config().with_overrides({key: value})


def test_every_shipped_config_is_in_range():
    from pathlib import Path

    for name in ("default.yaml", "learned.yaml"):  # the pipeline configs (not camera.*.yaml)
        load_config(Path(DEFAULT_CONFIG_PATH).parent / name)  # raises if out of range
    assert load_config().with_overrides({"dense.rectify_alpha": -1.0}).dense.rectify_alpha == -1.0
    # Open3D: -1 threads means all cores, as configs/default.yaml documents
    assert load_config().with_overrides({"dense.poisson_threads": -1}).dense.poisson_threads == -1


def test_sgbm_p2_must_exceed_p1():
    with pytest.raises(ConfigError, match=r"dense\.sgbm_p2_factor"):
        load_config().with_overrides({"dense.sgbm_p1_factor": 32, "dense.sgbm_p2_factor": 32})


def test_max_images_below_min_images_is_rejected():
    with pytest.raises(ConfigError, match=r"dataset\.max_images"):
        load_config().with_overrides({"dataset.min_images": 3, "dataset.max_images": 2})

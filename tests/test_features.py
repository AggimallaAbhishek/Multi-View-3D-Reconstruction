import numpy as np
import pytest

from src.config import FeatureConfig
from src.features.extractor import (
    FeatureCache,
    Features,
    ORBExtractor,
    SIFTExtractor,
    create_extractor,
    extract_features,
    spatial_coverage,
    write_feature_statistics,
)
from src.visualization.features import draw_keypoints
from tests.images import make_natural_texture, make_texture

pytestmark = pytest.mark.filterwarnings("error")  # extraction must not emit warnings


@pytest.fixture(scope="module")
def gray():
    import cv2

    return cv2.cvtColor(make_texture(0, 640, 480), cv2.COLOR_BGR2GRAY)


BLANK = np.full((240, 320), 128, np.uint8)


# --- descriptors ----------------------------------------------------------------------


def test_sift_features_shapes_and_types(gray):
    f = SIFTExtractor(max_features=2000).detect(gray)
    assert len(f) > 100
    assert f.points.shape == (len(f), 2) and f.points.dtype == np.float64
    assert f.descriptors.shape == (len(f), 128) and f.descriptors.dtype == np.float32
    for arr in (f.sizes, f.angles, f.responses):
        assert arr.shape == (len(f),)
    assert f.detector == "sift"
    h, w = gray.shape
    assert np.all((f.points >= -0.5) & (f.points <= [w - 0.5, h - 0.5]))


def test_orb_features_shapes_and_types(gray):
    f = ORBExtractor(max_features=1000).detect(gray)
    assert len(f) > 100
    assert f.descriptors.shape == (len(f), 32) and f.descriptors.dtype == np.uint8
    assert f.detector == "orb"


@pytest.mark.parametrize(
    ("extractor", "dim", "dtype"),
    [(SIFTExtractor(), 128, np.float32), (ORBExtractor(), 32, np.uint8)],
    ids=["sift", "orb"],
)
def test_blank_image_gives_empty_features(extractor, dim, dtype):
    f = extractor.detect(BLANK)
    assert len(f) == 0
    assert f.points.shape == (0, 2)
    assert f.descriptors.shape == (0, dim) and f.descriptors.dtype == dtype


@pytest.mark.parametrize("cls", [SIFTExtractor, ORBExtractor], ids=["sift", "orb"])
def test_max_features_is_a_strict_cap_keeping_strongest(gray, cls):
    capped = cls(max_features=50).detect(gray)
    assert len(capped) == 50
    full = cls(max_features=5000).detect(gray)
    # The kept features are the strongest ones.
    assert capped.responses.min() >= np.quantile(full.responses, 0.5)


def test_features_sorted_by_descending_response(gray):
    f = SIFTExtractor(max_features=500).detect(gray)
    assert np.all(np.diff(f.responses) <= 0)


def test_extraction_is_deterministic(gray):
    a = SIFTExtractor(max_features=1000).detect(gray)
    b = SIFTExtractor(max_features=1000).detect(gray.copy())
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.descriptors, b.descriptors)


def test_root_sift_descriptors_are_unit_norm_and_nonnegative(gray):
    plain = SIFTExtractor(max_features=300).detect(gray)
    root = SIFTExtractor(max_features=300, root_sift=True).detect(gray)
    np.testing.assert_allclose(np.linalg.norm(root.descriptors, axis=1), 1.0, atol=1e-5)
    assert root.descriptors.min() >= 0
    np.testing.assert_array_equal(root.points, plain.points)  # only descriptors change
    assert root.detector == "rootsift"


def test_colour_input_is_converted(gray):
    colour = make_texture(0, 640, 480)
    a = SIFTExtractor(max_features=300).detect(colour)
    b = SIFTExtractor(max_features=300).detect(gray)
    np.testing.assert_array_equal(a.points, b.points)


# --- repeatability: SIFT should survive rotation + scale -------------------------------


def _repeatability(extractor, image, angle_deg, scale, radius=2.5, margin=40):
    """Fraction of keypoints that reappear at the transformed location."""
    import cv2

    h, w = image.shape
    A = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle_deg, scale)
    warped = cv2.warpAffine(image, A, (w, h), flags=cv2.INTER_LINEAR)
    fa, fb = extractor.detect(image), extractor.detect(warped)

    mapped = fa.points @ A[:, :2].T + A[:, 2]
    inside = np.all((mapped > margin) & (mapped < [w - margin, h - margin]), axis=1)
    mapped = mapped[inside]
    d = np.linalg.norm(mapped[:, None, :] - fb.points[None, :, :], axis=2)
    return float(np.mean(d.min(axis=1) < radius))


def test_sift_is_repeatable_under_rotation_and_scale(gray):
    assert _repeatability(SIFTExtractor(max_features=1500), gray, 30, 0.8) > 0.4


def test_orb_is_repeatable_under_rotation(gray):
    assert _repeatability(ORBExtractor(max_features=1500), gray, 30, 1.0) > 0.3


# --- adaptive contrast threshold (low-texture, unevenly lit objects) --------------------


def _dimmed(divisor: float) -> np.ndarray:
    """A grey texture with its contrast divided by ``divisor``."""
    import cv2

    texture = cv2.cvtColor(make_natural_texture(1), cv2.COLOR_BGR2GRAY).astype(np.float64)
    return np.clip(100 + (texture - texture.mean()) / divisor, 0, 255).astype(np.uint8)


def _opencv_sift_count(image: np.ndarray, contrast_threshold: float) -> int:
    """Independent reference: OpenCV's SIFT at one fixed threshold, no wrapper."""
    import cv2

    return len(cv2.SIFT_create(contrastThreshold=contrast_threshold).detect(image, None))


@pytest.fixture(scope="module")
def faint():
    """Almost no SIFT features at the default threshold (0.04), thousands at 0.005."""
    return _dimmed(12)


def test_faint_image_is_redetected_until_it_has_enough_features(faint):
    assert _opencv_sift_count(faint, 0.04) < 300  # the default threshold alone misses it
    adaptive = create_extractor(FeatureConfig(sift_adaptive_min_features=300)).detect(faint)
    assert len(adaptive) >= 300


def test_image_with_enough_features_is_detected_once_at_the_configured_threshold():
    """Half contrast: ~2200 features at 0.04, the cap (8000) at 0.02, so an extra step shows."""
    dim = _dimmed(2)
    at_default = _opencv_sift_count(dim, 0.04)
    assert at_default < _opencv_sift_count(dim, 0.02)  # a lower threshold would show
    adaptive = create_extractor(FeatureConfig(sift_adaptive_min_features=at_default)).detect(dim)
    assert len(adaptive) == at_default


def test_threshold_is_never_lowered_below_the_floor(faint):
    adaptive = FeatureConfig(sift_adaptive_min_features=10**6, sift_min_contrast_threshold=0.01)
    stopped = create_extractor(adaptive).detect(faint)  # target out of reach: 0.04 → 0.02 → 0.01
    assert len(stopped) == _opencv_sift_count(faint, 0.01) < _opencv_sift_count(faint, 0.005)


# --- factory, persistence, cache --------------------------------------------------------


def test_create_extractor_from_config():
    assert isinstance(create_extractor(FeatureConfig(detector="sift")), SIFTExtractor)
    assert isinstance(create_extractor(FeatureConfig(detector="orb")), ORBExtractor)
    root = create_extractor(FeatureConfig(detector="sift", root_sift=True))
    assert root.signature() != create_extractor(FeatureConfig()).signature()


@pytest.mark.parametrize(
    "change", [{"sift_adaptive_min_features": 0}, {"sift_min_contrast_threshold": 0.0025}]
)
def test_adaptive_settings_are_part_of_the_cache_key(change):
    default = create_extractor(FeatureConfig()).signature()
    assert create_extractor(FeatureConfig(**change)).signature() != default


@pytest.mark.parametrize("image", ["texture", "blank"])
def test_features_save_load_round_trip(tmp_path, gray, image):
    f = SIFTExtractor(max_features=200).detect(gray if image == "texture" else BLANK)
    path = f.save(tmp_path / "f.npz")
    g = Features.load(path)
    assert g.detector == f.detector
    for name in ("points", "descriptors", "sizes", "angles", "responses"):
        np.testing.assert_array_equal(getattr(g, name), getattr(f, name))
        assert getattr(g, name).dtype == getattr(f, name).dtype


def test_cache_hit_returns_identical_features(tmp_path, gray):
    cache = FeatureCache(tmp_path / "cache")
    extractor = SIFTExtractor(max_features=300)
    first, hit1 = cache.get_or_compute(extractor, gray, "img")
    second, hit2 = cache.get_or_compute(extractor, gray, "img")
    assert (hit1, hit2) == (False, True)
    np.testing.assert_array_equal(first.descriptors, second.descriptors)


def test_cache_key_depends_on_image_and_extractor(tmp_path, gray):
    cache = FeatureCache(tmp_path / "cache")
    cache.get_or_compute(SIFTExtractor(max_features=300), gray, "img")
    _, hit_other_cfg = cache.get_or_compute(SIFTExtractor(max_features=301), gray, "img")
    other_image = gray.copy()
    other_image[0, 0] ^= 1
    _, hit_other_img = cache.get_or_compute(SIFTExtractor(max_features=300), other_image, "img")
    assert not hit_other_cfg
    assert not hit_other_img


def test_corrupt_cache_entry_is_recomputed(tmp_path, gray):
    cache = FeatureCache(tmp_path / "cache")
    extractor = SIFTExtractor(max_features=300)
    cache.get_or_compute(extractor, gray, "img")
    (entry,) = (tmp_path / "cache").glob("*.npz")
    entry.write_bytes(b"garbage")
    f, hit = cache.get_or_compute(extractor, gray, "img")
    assert not hit
    assert len(f) == 300


# --- coverage, batch extraction, drawing -----------------------------------------------


def test_spatial_coverage():
    xs, ys = np.meshgrid(np.arange(5, 640, 10), np.arange(5, 480, 10))
    spread = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    assert spatial_coverage(spread, 640, 480, grid=8) == 1.0
    corner = np.array([[1.0, 1.0], [5.0, 5.0]])
    assert spatial_coverage(corner, 640, 480, grid=8) == pytest.approx(1 / 64)
    assert spatial_coverage(np.empty((0, 2)), 640, 480) == 0.0


def test_extract_features_over_dataset(tmp_path):
    from src.config import config_from_dict
    from src.data.dataset import prepare_dataset
    from tests.images import write_textured_images

    write_textured_images(tmp_path / "scene", 3, size=(640, 480))
    dataset = prepare_dataset(tmp_path / "scene", config_from_dict({}))
    cache = FeatureCache(tmp_path / "cache")
    extractor = SIFTExtractor(max_features=400)

    features, stats = extract_features(dataset.images, extractor, cache)
    assert len(features) == len(stats) == 3
    assert [s.image for s in stats] == [im.name for im in dataset.images]
    assert all(s.n_features == 400 and not s.cached for s in stats)
    assert all(0 < s.coverage <= 1 for s in stats)

    _, stats_again = extract_features(dataset.images, extractor, cache)
    assert all(s.cached for s in stats_again)

    csv_path = write_feature_statistics(stats, tmp_path / "feature_statistics.csv")
    header, *rows = csv_path.read_text().strip().splitlines()
    assert header == "image,detector,n_features,coverage,time_ms,cached"
    assert len(rows) == 3


def test_draw_keypoints(gray):
    colour = make_texture(0, 640, 480)
    f = SIFTExtractor(max_features=300).detect(gray)
    drawn = draw_keypoints(colour, f, max_points=100)
    assert drawn.shape == colour.shape and drawn.dtype == np.uint8
    assert not np.array_equal(drawn, colour)
    background_only = draw_keypoints(colour, f.subset(np.arange(0)))
    assert background_only.shape == colour.shape
    assert not np.array_equal(drawn, background_only)  # keypoints add pixels


@pytest.mark.parametrize("detector", ["sift", "orb"])
def test_cli_features_writes_statistics_and_figures(tmp_path, detector):
    from src.cli import main
    from tests.images import write_textured_images

    write_textured_images(tmp_path / "scene", 3, size=(640, 480))
    config = tmp_path / "config.yaml"
    config.write_text(
        f"output:\n  cache_dir: {tmp_path / 'cache'}\nfeatures:\n  max_features: 500\n"
    )
    out = tmp_path / "outputs"

    args = ["features", str(tmp_path / "scene"), "--config", str(config), "--out", str(out)]
    assert main([*args, "--detector", detector, "--draw", "2"]) == 0

    (run_dir,) = out.iterdir()
    assert run_dir.name.endswith(f"features_{detector}")
    assert (run_dir / "features" / "feature_statistics.csv").is_file()
    assert len(list((run_dir / "features").glob("*_keypoints.png"))) == 2
    assert len(list((tmp_path / "cache" / "features").glob("*.npz"))) == 3


def test_weak_images():
    from src.features.extractor import FeatureStats, weak_images

    cfg = FeatureConfig(warn_min_features=100, warn_min_coverage=0.25)
    stats = [
        FeatureStats("ok", "sift", 500, 0.9, 1.0, False),
        FeatureStats("few", "sift", 50, 0.9, 1.0, False),
        FeatureStats("bunched", "sift", 500, 0.1, 1.0, False),
    ]
    assert [s.image for s in weak_images(stats, cfg)] == ["few", "bunched"]

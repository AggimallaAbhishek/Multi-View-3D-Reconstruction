"""Learned features (DISK) and matching (LightGlue), both external (kornia, optional extra).

Checked against the same planar ground truth as SIFT: views of a textured plane with known
homographies. The pretrained weights are downloaded on first use; without network access
and an empty cache these tests skip.
"""

import cv2
import numpy as np
import pytest

from src.config import FeatureConfig, MatchConfig
from tests.images import make_view_sequence, relative_homography

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")

from src.features.learned import DISKExtractor, LightGlueMatcher  # noqa: E402


@pytest.fixture(scope="module")
def views():
    return make_view_sequence(4, size=(640, 480), seed=0)


@pytest.fixture(scope="module")
def extractor():
    try:
        return DISKExtractor(max_features=2000, device="cpu")
    except OSError as exc:  # weights not cached and no network
        pytest.skip(f"DISK weights unavailable: {exc}")


@pytest.fixture(scope="module")
def disk_features(views, extractor):
    return [extractor.detect(im) for im in views[0]]


@pytest.fixture(scope="module")
def matcher(views):
    sizes = [(im.shape[1], im.shape[0]) for im in views[0]]
    try:
        return LightGlueMatcher(MatchConfig(method="lightglue"), sizes, device="cpu")
    except OSError as exc:
        pytest.skip(f"LightGlue weights unavailable: {exc}")


def _precision(matches, fi, fj, H, tol=3.0):
    pi, pj = matches.points(fi, fj)
    projected = cv2.perspectiveTransform(pi[None], H)[0]
    return float(np.mean(np.linalg.norm(projected - pj, axis=1) < tol))


def test_disk_features_follow_the_feature_layout(views, disk_features):
    f = disk_features[0]
    h, w = views[0][0].shape[:2]
    assert f.detector == "disk"
    assert 500 < len(f) <= 2000
    assert f.points.dtype == np.float64 and f.points.shape == (len(f), 2)
    assert f.descriptors.dtype == np.float32 and f.descriptors.shape == (len(f), 128)
    assert np.all((f.points >= 0) & (f.points <= [w - 1, h - 1]))
    assert np.all(np.diff(f.responses) <= 0)  # strongest first, like SIFT and ORB


def test_disk_keypoints_land_on_corresponding_points(views, disk_features):
    """Repeatability: keypoints of view 1 mapped by the true homography meet keypoints of
    view 0 within 2 px far more often than chance."""
    H = relative_homography(views[1], 1, 0)
    mapped = cv2.perspectiveTransform(disk_features[1].points[None], H)[0]
    d = np.linalg.norm(mapped[:, None] - disk_features[0].points[None], axis=2).min(axis=1)
    assert np.mean(d < 2.0) > 0.4


def test_disk_extraction_is_deterministic(views, extractor, disk_features):
    again = extractor.detect(views[0][0])
    np.testing.assert_array_equal(again.points, disk_features[0].points)


@pytest.mark.parametrize("j", [1, 3])
def test_lightglue_matches_agree_with_true_homography(views, disk_features, matcher, j):
    m, stats = matcher.match(disk_features[0], disk_features[j], 0, j)
    assert len(m) > 200
    assert (
        _precision(m, disk_features[0], disk_features[j], relative_homography(views[1], 0, j))
        > 0.95
    )
    assert stats.final == len(m) and (m.i, m.j) == (0, j)
    assert np.all(np.diff(m.idx_i) > 0)  # ascending and one-to-one, like DescriptorMatcher
    assert len(np.unique(m.idx_j)) == len(m)
    assert np.all((m.distances >= 0) & (m.distances <= 1))  # 1 - matching confidence


def test_lightglue_rejects_unrelated_images(disk_features, matcher, extractor):
    noise = np.random.default_rng(5).integers(0, 256, (480, 640), dtype=np.uint8)
    unrelated = extractor.detect(cv2.GaussianBlur(noise, (0, 0), 3))
    m, _ = matcher.match(disk_features[0], unrelated, 0, 1)
    assert len(m) < 30


def test_lightglue_needs_disk_features(views, disk_features, matcher):
    from src.features.extractor import SIFTExtractor

    sift = SIFTExtractor(max_features=500).detect(views[0][1])
    with pytest.raises(ValueError, match="disk"):
        matcher.match(disk_features[0], sift, 0, 1)


def test_disk_works_with_the_descriptor_matcher(views, disk_features):
    """DISK descriptors are float vectors: the classical ratio-test matcher accepts them."""
    from src.features.matcher import DescriptorMatcher

    m, _ = DescriptorMatcher(MatchConfig()).match(disk_features[0], disk_features[1])
    assert len(m) > 100
    assert (
        _precision(m, disk_features[0], disk_features[1], relative_homography(views[1], 0, 1)) > 0.9
    )


def test_create_extractor_and_matcher_choose_the_learned_path(views):
    from src.features.extractor import create_extractor
    from src.features.matcher import create_matcher

    try:
        ex = create_extractor(FeatureConfig(detector="disk", max_features=100))
        sizes = [(640, 480)] * 2
        assert isinstance(create_matcher(MatchConfig(method="lightglue"), sizes), LightGlueMatcher)
    except OSError as exc:
        pytest.skip(f"weights unavailable: {exc}")
    assert isinstance(ex, DISKExtractor)
    assert ex.signature().startswith("disk-n100")
    assert ex.signature().endswith(f"-{ex.device.type}")  # a cache per device type


def test_confidence_threshold_keeps_only_confident_matches(views, disk_features):
    sizes = [(im.shape[1], im.shape[0]) for im in views[0]]
    try:
        loose, strict = (
            LightGlueMatcher(
                MatchConfig(method="lightglue", lightglue_filter_threshold=t), sizes, "cpu"
            )
            for t in (0.1, 0.7)
        )
    except OSError as exc:
        pytest.skip(f"LightGlue weights unavailable: {exc}")
    a, _ = loose.match(disk_features[0], disk_features[3], 0, 3)
    b, _ = strict.match(disk_features[0], disk_features[3], 0, 3)
    assert 0 < len(b) < len(a)
    assert np.all(b.distances <= 0.3 + 1e-6)
    pairs = lambda m: set(zip(m.idx_i.tolist(), m.idx_j.tolist(), strict=True))  # noqa: E731
    assert pairs(b) <= pairs(a)  # a subset, not new matches


def test_lightglue_does_not_match_a_view_rolled_upside_down(
    views, extractor, disk_features, matcher
):
    """Known limitation (templeRing's 16 upside-down views): DISK and LightGlue are trained on
    upright images, so a 180-degree roll defeats them, while rotation-invariant SIFT matches."""
    from src.features.extractor import SIFTExtractor
    from src.features.matcher import DescriptorMatcher

    rolled = cv2.rotate(views[0][1], cv2.ROTATE_180)
    m, _ = matcher.match(disk_features[0], extractor.detect(rolled), 0, 1)
    sift = SIFTExtractor(max_features=2000)
    s, _ = DescriptorMatcher(MatchConfig()).match(sift.detect(views[0][0]), sift.detect(rolled))
    assert len(m) < 30 < len(s)


def test_pipeline_with_learned_config_recovers_true_poses(tmp_path):
    """configs/learned.yaml end to end on rendered views with real parallax: DISK + LightGlue
    feed verification, registration and bundle adjustment like SIFT does."""
    from src.config import DEFAULT_CONFIG_PATH, load_config
    from src.evaluation.metrics import camera_pose_errors
    from src.geometry.camera import Camera
    from src.pipeline import run_pipeline
    from tests.render import render_box_scene, write_scene

    scene = render_box_scene(n_cameras=4, arc_deg=30.0)
    images, camera = write_scene(scene, tmp_path)
    config = load_config(DEFAULT_CONFIG_PATH.with_name("learned.yaml")).with_overrides(
        {
            "camera.calibration_file": str(camera),
            "output.cache_dir": str(tmp_path / "cache"),
            "features.max_features": 3000,
            "performance.torch_device": "cpu",
        }
    )
    try:
        result = run_pipeline(images, config, until="reconstruct")
    except OSError as exc:
        pytest.skip(f"weights unavailable: {exc}")

    assert all(f.detector == "disk" for f in result.features)
    registered = result.reconstruction.registry.images
    assert sorted(registered) == [0, 1, 2, 3]
    gt = [Camera(scene.K, scene.rotations[m], scene.translations[m]) for m in registered]
    errors = camera_pose_errors([result.reconstruction.registry.camera(m) for m in registered], gt)
    assert errors.rotation_deg.max() < 1.0
    assert errors.centre.max() < 0.05


def test_match_cache_key_changes_with_the_lightglue_version(tmp_path, monkeypatch):
    """LightGlue's weights ship with kornia: a kornia upgrade may change matches of
    unchanged descriptors, so it must not reuse cached LightGlue matches."""
    from src.features import matcher as matcher_module

    cache = matcher_module.MatchCache(tmp_path)
    cfg = MatchConfig(method="lightglue")
    before = cache.path_for("a", "b", cfg)
    monkeypatch.setattr(matcher_module, "_learned_version", lambda: "kornia-0.0.0")
    assert cache.path_for("a", "b", cfg) != before
    # The classical matchers do not depend on kornia at all.
    assert cache.path_for("a", "b", MatchConfig()) == matcher_module.MatchCache(tmp_path).path_for(
        "a", "b", MatchConfig()
    )

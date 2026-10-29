import cv2
import numpy as np
import pytest

from src.config import VerificationConfig
from src.geometry.epipolar import (
    TwoViewGeometry,
    eight_point,
    enforce_essential,
    essential_from_fundamental,
    fundamental_from_essential,
    sampson_distance,
    select_model,
    skew,
    verify_pair,
)
from tests.synthetic import intrinsics, make_scene, project

CFG = VerificationConfig(method="essential", ransac_threshold_px=1.0, min_inliers=30)


def _true_essential(scene, i=0, j=1):
    R, t = scene.relative_pose(i, j)
    return skew(t) @ R


def _same_up_to_scale(A, B, atol=1e-6):
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    return min(np.linalg.norm(A - B), np.linalg.norm(A + B)) < atol


def _with_outliers(x2, fraction, image_size, seed=1):
    rng = np.random.default_rng(seed)
    outlier = rng.random(len(x2)) < fraction
    x2 = x2.copy()
    x2[outlier] = rng.uniform([0, 0], image_size, size=(outlier.sum(), 2))
    return x2, outlier


@pytest.fixture(scope="module")
def scene():
    return make_scene(n_points=600, n_cameras=3, noise_px=0.0, seed=0)


@pytest.fixture(scope="module")
def noisy():
    return make_scene(n_points=600, n_cameras=3, noise_px=0.5, seed=0)


# --- matrix building blocks ---------------------------------------------------------------


def test_skew_is_cross_product():
    a, b = np.array([1.0, -2.0, 3.0]), np.array([0.5, 4.0, -1.0])
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))


def test_true_essential_has_two_equal_singular_values(scene):
    s = np.linalg.svd(_true_essential(scene), compute_uv=False)
    assert s[0] == pytest.approx(s[1])
    assert s[2] == pytest.approx(0.0, abs=1e-12)


def test_enforce_essential_projects_onto_essential_manifold():
    rng = np.random.default_rng(0)
    E = enforce_essential(rng.normal(size=(3, 3)))
    s = np.linalg.svd(E, compute_uv=False)
    np.testing.assert_allclose(s, [1.0, 1.0, 0.0], atol=1e-12)


def test_fundamental_essential_round_trip(scene):
    E = _true_essential(scene)
    K1, K2 = scene.K, intrinsics(800.0, 640, 480)
    F = fundamental_from_essential(E, K1, K2)
    assert _same_up_to_scale(essential_from_fundamental(F, K1, K2), E)


# --- normalised eight-point algorithm (our own implementation) -----------------------------


def test_eight_point_recovers_true_fundamental(scene):
    x1, x2 = scene.pair(0, 1)
    F = eight_point(x1, x2)
    F_true = fundamental_from_essential(_true_essential(scene), scene.K, scene.K)
    assert _same_up_to_scale(F, F_true, atol=1e-6)
    assert np.linalg.matrix_rank(F) == 2
    np.testing.assert_allclose(sampson_distance(F, x1, x2), 0.0, atol=1e-6)


def test_eight_point_with_noise_is_close_to_opencv(noisy):
    x1, x2 = noisy.pair(0, 1)
    F_ours = eight_point(x1, x2)
    F_cv, _ = cv2.findFundamentalMat(x1, x2, cv2.FM_8POINT)
    assert _same_up_to_scale(F_ours, F_cv, atol=1e-3)
    assert np.median(sampson_distance(F_ours, x1, x2)) < 0.5


def test_eight_point_with_minimal_points(scene):
    x1, x2 = scene.pair(0, 1)
    F = eight_point(x1[:8], x2[:8])
    np.testing.assert_allclose(sampson_distance(F, x1, x2), 0.0, atol=1e-4)


def test_eight_point_rejects_too_few_points(scene):
    x1, x2 = scene.pair(0, 1)
    with pytest.raises(ValueError, match="8"):
        eight_point(x1[:7], x2[:7])


# --- Sampson distance ---------------------------------------------------------------------


def test_sampson_distance_is_in_pixels(scene):
    # Moving x2 by d pixels along the normal of its epipolar line changes the algebraic
    # error to first order by d * |l|; with exact x1, Sampson distance is close to d / sqrt(2)
    # at most and d at least half of that, so it scales linearly with pixel displacement.
    x1, x2 = scene.pair(0, 1)
    F = fundamental_from_essential(_true_essential(scene), scene.K, scene.K)
    lines = np.c_[x1, np.ones(len(x1))] @ F.T
    normals = lines[:, :2] / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    d1 = sampson_distance(F, x1, x2 + 1.0 * normals)
    d3 = sampson_distance(F, x1, x2 + 3.0 * normals)
    np.testing.assert_allclose(d3 / d1, 3.0, rtol=1e-3)
    assert 0.5 < np.median(d1) <= 1.0 + 1e-9


# --- robust verification ------------------------------------------------------------------


@pytest.mark.parametrize("method", ["essential", "fundamental"])
def test_verify_noise_free_pair(scene, method):
    x1, x2 = scene.pair(0, 1)
    cfg = VerificationConfig(method=method, ransac_threshold_px=1.0)
    g = verify_pair(x1, x2, scene.K, scene.K, cfg, model=method)
    assert g.success and g.reason is None
    assert g.model == method
    assert g.inlier_mask.all()
    F_true = fundamental_from_essential(_true_essential(scene), scene.K, scene.K)
    assert _same_up_to_scale(g.F, F_true, atol=1e-3)
    if method == "essential":
        assert _same_up_to_scale(g.E, _true_essential(scene), atol=1e-3)
    else:
        assert g.E is None


@pytest.mark.parametrize("method", ["essential", "fundamental"])
def test_verify_rejects_outliers(noisy, method):
    x1, x2 = noisy.pair(0, 1)
    x2, outlier = _with_outliers(x2, 0.3, noisy.image_size)
    cfg = VerificationConfig(method=method, ransac_threshold_px=1.5)
    g = verify_pair(x1, x2, noisy.K, noisy.K, cfg, model=method)
    assert g.success
    assert np.mean(g.inlier_mask[~outlier]) > 0.95  # recall of true matches
    assert np.mean(g.inlier_mask[outlier]) < 0.03  # outliers that slip through
    assert g.n_inliers == int(g.inlier_mask.sum())
    assert g.inlier_ratio == pytest.approx(g.n_inliers / len(x1))
    assert np.all(g.residuals[g.inlier_mask] <= cfg.ransac_threshold_px)


def test_verify_with_different_intrinsics(scene):
    # Camera j has a different focal length: E must use both K matrices.
    K2 = intrinsics(900.0, 640, 480)
    both = scene.visible[0]
    x1 = scene.observations[0, both]
    x2, _ = project(K2, scene.rotations[1], scene.translations[1], scene.points3d[both])
    g = verify_pair(x1, x2, scene.K, K2, CFG, model="essential")
    assert g.success and g.inlier_mask.all()
    assert _same_up_to_scale(g.E, _true_essential(scene), atol=1e-3)


def test_verify_is_deterministic(noisy):
    x1, x2 = noisy.pair(0, 1)
    x2, _ = _with_outliers(x2, 0.3, noisy.image_size)
    a = verify_pair(x1, x2, noisy.K, noisy.K, CFG, model="essential")
    b = verify_pair(x1, x2, noisy.K, noisy.K, CFG, model="essential")
    np.testing.assert_array_equal(a.inlier_mask, b.inlier_mask)


def test_too_few_matches_is_a_failure_not_an_exception(scene):
    x1, x2 = scene.pair(0, 1)
    g = verify_pair(x1[:6], x2[:6], scene.K, scene.K, CFG, model="essential")
    assert not g.success
    assert g.reason == "too_few_matches"
    assert g.n_inliers == 0 and g.inlier_mask.shape == (6,)


def test_random_correspondences_fail_verification(scene):
    rng = np.random.default_rng(3)
    x1 = rng.uniform([0, 0], [640, 480], size=(300, 2))
    x2 = rng.uniform([0, 0], [640, 480], size=(300, 2))
    g = verify_pair(x1, x2, scene.K, scene.K, CFG, model="essential")
    assert not g.success
    assert g.reason in {"too_few_inliers", "estimation_failed"}


def test_homography_ratio_flags_planar_scenes():
    general = make_scene(n_points=600, n_cameras=3, noise_px=0.5, seed=0)
    planar = make_scene(n_points=600, n_cameras=3, noise_px=0.5, seed=0, planar=True)
    cfg = VerificationConfig(method="essential", ransac_threshold_px=1.5)
    g_general = verify_pair(*general.pair(0, 1), general.K, general.K, cfg, model="essential")
    g_planar = verify_pair(*planar.pair(0, 1), planar.K, planar.K, cfg, model="essential")
    assert g_general.homography_ratio < 0.3
    assert g_planar.homography_ratio > 0.9


def test_select_model():
    assert select_model("essential", "fallback", "fallback") == "essential"
    assert select_model("fundamental", "file", "file") == "fundamental"
    assert select_model("auto", "file", "exif") == "essential"
    assert select_model("auto", "exif", "fallback") == "fundamental"


def test_geometry_is_a_dataclass_result(scene):
    g = verify_pair(*scene.pair(0, 1), scene.K, scene.K, CFG, model="essential")
    assert isinstance(g, TwoViewGeometry)
    assert g.residuals.shape == (len(g.inlier_mask),)
    assert np.isfinite(g.median_residual)

import cv2
import numpy as np
import pytest

from src.config import BundleAdjustmentConfig
from src.evaluation.metrics import align_similarity, rotation_error_deg
from src.geometry.camera import Camera
from src.geometry.reprojection import reprojection_errors
from src.geometry.triangulation import Observations
from src.reconstruction.bundle_adjustment import (
    bundle_adjust,
    jacobian,
    pack,
    residuals,
    scale_focal,
)
from tests.synthetic import make_scene

CFG = BundleAdjustmentConfig()


def _problem(scene):
    """True cameras, every visible observation and the true points."""
    cameras = [
        Camera(scene.K, R, t) for R, t in zip(scene.rotations, scene.translations, strict=True)
    ]
    view, point = np.nonzero(scene.visible)
    obs = Observations(view, point, scene.observations[view, point])
    return cameras, obs, scene.points3d


def _perturbed(cameras, X, rng, skip=()):
    """Cameras rotated by ~1 deg and shifted ~5 %, points moved ~0.05 (scene size ~2)."""
    out = []
    for k, c in enumerate(cameras):
        if k in skip:
            out.append(c)
            continue
        dR = cv2.Rodrigues(rng.normal(0, np.deg2rad(1.0), 3))[0]
        out.append(Camera(c.K, dR @ c.R, c.t + rng.normal(0, 0.05 * np.linalg.norm(c.t), 3)))
    return out, X + rng.normal(0, 0.05, X.shape)


def _rms(cameras, obs, X):
    return float(np.sqrt(np.mean(reprojection_errors(cameras, obs, X) ** 2)))


def test_bundle_adjustment_recovers_perturbed_scene():
    scene = make_scene(n_points=300, n_cameras=6, noise_px=0.5, seed=2)
    true_cams, obs, X_true = _problem(scene)
    cams0, X0 = _perturbed(true_cams, X_true, np.random.default_rng(0), skip={0, 1})
    assert _rms(cams0, obs, X0) > 20.0

    result = bundle_adjust(cams0, obs, X0, CFG, fixed=(0, 1))
    assert result.success
    assert result.final_rms_px < result.initial_rms_px
    # Down to the noise: 0.5 px per coordinate is 0.5 * sqrt(2) px per observation.
    assert _rms(result.cameras, obs, result.points) < 0.5 * np.sqrt(2)
    assert result.final_rms_px == pytest.approx(_rms(result.cameras, obs, result.points))
    # Same optimum as adjusting from the truth (the noise-limited estimate) ...
    optimum = bundle_adjust(true_cams, obs, X_true, CFG, fixed=(0, 1))
    for est, best, true in zip(result.cameras, optimum.cameras, true_cams, strict=True):
        assert rotation_error_deg(est.R, best.R) < 1e-3
        assert np.linalg.norm(est.centre - best.centre) < 1e-4
        # ... which 0.5 px of noise leaves within ~0.1 deg / 0.01 of the true pose.
        assert rotation_error_deg(est.R, true.R) < 0.2
        assert np.linalg.norm(est.centre - true.centre) < 0.02
    assert np.median(np.linalg.norm(result.points - X_true, axis=1)) < 0.01


def test_one_fixed_camera_keeps_the_gauge_and_scale():
    scene = make_scene(n_points=300, n_cameras=6, noise_px=0.5, seed=2)
    true_cams, obs, X_true = _problem(scene)
    cams0, X0 = _perturbed(true_cams, X_true, np.random.default_rng(1), skip={0})

    result = bundle_adjust(cams0, obs, X0, CFG, fixed=(0,))
    assert result.success and result.final_rms_px < 0.5 * np.sqrt(2)
    np.testing.assert_array_equal(result.cameras[0].R, cams0[0].R)
    np.testing.assert_array_equal(result.cameras[0].t, cams0[0].t)
    baseline = np.linalg.norm(cams0[1].centre - cams0[0].centre)
    assert np.linalg.norm(result.cameras[1].centre - result.cameras[0].centre) == pytest.approx(
        baseline, rel=1e-12
    )  # the scale did not drift

    # Up to that (slightly wrong) scale the reconstruction is the true one.
    est = np.array([c.centre for c in result.cameras])
    sim = align_similarity(est, np.array([c.centre for c in true_cams]))
    assert (
        np.abs(np.linalg.norm(sim.apply(est) - [c.centre for c in true_cams], axis=1)).max() < 0.05
    )


def _max_pose_error(cameras, truth):
    return max(rotation_error_deg(e.R, t.R) for e, t in zip(cameras, truth, strict=True))


def test_robust_loss_resists_grossly_wrong_observations():
    scene = make_scene(n_points=300, n_cameras=6, noise_px=0.5, seed=2)
    true_cams, obs, X_true = _problem(scene)
    rng = np.random.default_rng(2)
    bad = rng.random(len(obs)) < 0.03  # 3 % of observations 30 px off (wrong matches)
    xy = obs.xy.copy()
    xy[bad] += rng.choice([-1.0, 1.0], size=(bad.sum(), 2)) * 30.0
    corrupted = Observations(obs.view, obs.point, xy)
    cams0, X0 = _perturbed(true_cams, X_true, np.random.default_rng(0), skip={0, 1})

    robust = bundle_adjust(cams0, corrupted, X0, CFG, fixed=(0, 1))
    plain = bundle_adjust(cams0, corrupted, X0, BundleAdjustmentConfig(loss="linear"), fixed=(0, 1))
    assert _max_pose_error(robust.cameras, true_cams) < 0.2
    assert _max_pose_error(plain.cameras, true_cams) > 3 * _max_pose_error(
        robust.cameras, true_cams
    )


@pytest.mark.parametrize("refine_focal", [False, True])
def test_jacobian_matches_finite_differences(refine_focal):
    """The exact reprojection Jacobian (camera rotation and translation, point and, with
    ``refine_focal``, the focal scale shared by every camera, ``K @ diag(s, s, 1)``)
    against SciPy's central differences, at a generic state: rotated, shifted, off the
    optimum."""
    from scipy.optimize._numdiff import approx_derivative

    scene = make_scene(n_points=40, n_cameras=4, noise_px=0.5, seed=3)
    cameras, obs, X = _problem(scene)
    cameras, X = _perturbed(cameras, X, np.random.default_rng(1))
    s = 0.93 if refine_focal else None
    params = pack(cameras, X, s)
    K = np.stack([c.K for c in cameras])

    if refine_focal:  # the residuals are those of the scaled cameras
        scaled = [Camera(scale_focal(c.K, s), c.R, c.t) for c in cameras]
        projected = [
            scaled[v].project(X[p][None])[0][0] for v, p in zip(obs.view, obs.point, strict=True)
        ]
        np.testing.assert_allclose(
            residuals(params, K, obs, True).reshape(-1, 2), np.array(projected) - obs.xy, atol=1e-9
        )
    exact = jacobian(params, K, obs, refine_focal).toarray()
    numeric = approx_derivative(lambda p: residuals(p, K, obs, refine_focal), params)
    assert exact.shape == (2 * len(obs.view), 6 * len(cameras) + 3 * len(X) + refine_focal)
    np.testing.assert_allclose(exact, numeric, rtol=1e-5, atol=1e-5 * np.abs(numeric).max())


@pytest.fixture
def wrong_focal():
    """Cameras 10 % short in focal length, perturbed (camera 0 exact), and the truth."""
    scene = make_scene(n_points=400, n_cameras=8, noise_px=0.5, seed=4, arc_deg=120.0)
    true_cams, obs, X_true = _problem(scene)
    wrong = [Camera(scale_focal(c.K, 0.9), c.R, c.t) for c in true_cams]
    cams0, X0 = _perturbed(wrong, X_true, np.random.default_rng(0), skip={0})
    return cams0, obs, X0, true_cams


def test_refine_focal_recovers_a_wrong_focal_length(wrong_focal):
    """Started 10 % off, the shared focal length converges to the truth (and is applied to
    the fixed camera too); with K held fixed the error stays and distorts the poses."""
    cams0, obs, X0, true_cams = wrong_focal

    fixed_k = bundle_adjust(cams0, obs, X0, CFG, fixed=(0,))
    refined = bundle_adjust(cams0, obs, X0, CFG, fixed=(0,), refine_focal=True)
    assert fixed_k.focal_scale == 1.0
    assert refined.focal_scale == pytest.approx(1 / 0.9, rel=2e-3)
    for c in refined.cameras:
        np.testing.assert_allclose(c.K, scale_focal(cams0[0].K, refined.focal_scale))
    np.testing.assert_array_equal(refined.cameras[0].R, cams0[0].R)
    assert refined.final_rms_px < 0.5 * np.sqrt(2) < fixed_k.final_rms_px

    def centre_error(cams):
        est = np.array([c.centre for c in cams])
        true = np.array([c.centre for c in true_cams])
        return np.linalg.norm(align_similarity(est, true).apply(est) - true, axis=1).max()

    assert centre_error(refined.cameras) < 0.02 < centre_error(fixed_k.cameras)


def test_refine_focal_off_leaves_k_unchanged():
    scene = make_scene(n_points=100, n_cameras=4, noise_px=0.5, seed=2)
    cams, obs, X = _problem(scene)
    result = bundle_adjust(cams, obs, X, CFG, fixed=(0,))
    assert result.focal_scale == 1.0
    assert all(np.array_equal(c.K, cams[0].K) for c in result.cameras)


def test_a_focal_scale_beyond_the_limit_is_discarded(wrong_focal):
    """A refined focal length that moved more than ``max_focal_change`` (here deliberately
    tight) is not trusted: the adjustment is repeated with K held fixed."""
    cams0, obs, X0, _ = wrong_focal

    tight = BundleAdjustmentConfig(max_focal_change=1.05)  # the true change is 1 / 0.9
    result = bundle_adjust(cams0, obs, X0, tight, fixed=(0,), refine_focal=True)
    fixed_k = bundle_adjust(cams0, obs, X0, tight, fixed=(0,))
    assert result.focal_scale == 1.0
    assert all(np.array_equal(c.K, cams0[0].K) for c in result.cameras)
    assert result.final_rms_px == pytest.approx(fixed_k.final_rms_px)

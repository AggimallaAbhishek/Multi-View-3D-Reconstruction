import cv2
import numpy as np
import pytest

from src.config import BundleAdjustmentConfig
from src.evaluation.metrics import align_similarity, rotation_error_deg
from src.geometry.camera import Camera
from src.geometry.reprojection import reprojection_errors
from src.geometry.triangulation import Observations
from src.reconstruction.bundle_adjustment import bundle_adjust, jacobian, residuals
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


def test_jacobian_matches_finite_differences():
    """The exact reprojection Jacobian (camera rotation and translation, point) against
    SciPy's central differences, at a generic state: rotated, shifted, off the optimum."""
    from scipy.optimize._numdiff import approx_derivative

    scene = make_scene(n_points=40, n_cameras=4, noise_px=0.5, seed=3)
    cameras, obs, X = _problem(scene)
    cameras, X = _perturbed(cameras, X, np.random.default_rng(1))
    params = np.concatenate(
        [np.concatenate([cv2.Rodrigues(c.R)[0].ravel(), c.t]) for c in cameras] + [X.ravel()]
    )
    K = np.stack([c.K for c in cameras])

    exact = jacobian(params, K, obs).toarray()
    numeric = approx_derivative(lambda p: residuals(p, K, obs), params, method="3-point")
    assert exact.shape == (2 * len(obs.view), 6 * len(cameras) + 3 * len(X))
    np.testing.assert_allclose(exact, numeric, rtol=1e-5, atol=1e-5 * np.abs(numeric).max())

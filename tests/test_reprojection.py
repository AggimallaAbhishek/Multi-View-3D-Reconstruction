import numpy as np
import pytest

from src.geometry.camera import Camera
from src.geometry.reprojection import (
    ReprojectionStats,
    per_point_mean,
    reprojection_errors,
)
from src.geometry.triangulation import Observations
from tests.synthetic import make_scene


@pytest.fixture(scope="module")
def setup():
    scene = make_scene(n_points=2000, n_cameras=3, noise_px=0.0, seed=0)
    cams = [Camera(scene.K, R, t) for R, t in zip(scene.rotations, scene.translations, strict=True)]
    n = scene.n_points
    view = np.repeat(np.arange(3), n)
    point = np.tile(np.arange(n), 3)
    return scene, cams, view, point


def test_errors_are_zero_for_exact_observations(setup):
    scene, cams, view, point = setup
    obs = Observations(view, point, scene.observations.reshape(-1, 2))
    np.testing.assert_allclose(reprojection_errors(cams, obs, scene.points3d), 0.0, atol=1e-9)


def test_errors_measure_pixel_noise(setup):
    # |N(0, s^2 I_2)| has mean s * sqrt(pi / 2).
    scene, cams, view, point = setup
    sigma = 0.8
    xy = scene.observations.reshape(-1, 2) + np.random.default_rng(1).normal(
        0, sigma, (len(view), 2)
    )
    errors = reprojection_errors(cams, Observations(view, point, xy), scene.points3d)
    assert errors.mean() == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=0.05)


def test_stats_and_per_point_mean():
    errors = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    stats = ReprojectionStats.from_errors(errors)
    assert (stats.n, stats.mean, stats.median, stats.max) == (5, 3.2, 2.0, 10.0)
    assert stats.p90 == pytest.approx(np.percentile(errors, 90))
    assert stats.to_dict()["median_px"] == 2.0

    point = np.array([0, 0, 1, 1, 1])
    np.testing.assert_allclose(per_point_mean(errors, point, 2), [0.5, 5.0])


@pytest.mark.parametrize("shape", [(0,), (0, 3)])
def test_per_point_mean_without_observations_is_nan(shape):
    # Regression: reshape(0, -1) raised, so colouring a structure with no points crashed.
    mean = per_point_mean(np.zeros(shape), np.zeros(0, dtype=int), 2)
    assert mean.shape == (2, *shape[1:]) and np.isnan(mean).all()


def test_stats_of_nothing():
    stats = ReprojectionStats.from_errors(np.empty(0))
    assert stats.n == 0 and np.isnan(stats.mean)


def test_align_similarity_recovers_known_transform():
    import cv2

    from src.evaluation.metrics import align_similarity

    rng = np.random.default_rng(0)
    A = rng.normal(size=(100, 3))
    R = cv2.Rodrigues(np.array([0.3, -0.2, 0.5]))[0]
    B = 2.5 * A @ R.T + np.array([1.0, -2.0, 0.5])
    sim = align_similarity(A, B)
    assert sim.scale == pytest.approx(2.5)
    np.testing.assert_allclose(sim.R, R, atol=1e-10)
    np.testing.assert_allclose(sim.apply(A), B, atol=1e-10)


def test_camera_pose_errors_ignore_a_global_similarity():
    import cv2

    from src.evaluation.metrics import camera_pose_errors

    scene = make_scene(n_cameras=6, seed=3)
    truth = [
        Camera(scene.K, R, t) for R, t in zip(scene.rotations, scene.translations, strict=True)
    ]
    # The same cameras in another frame: X_other = s * Q @ X + d, so R' = R Q^T, t' = s t - R' d.
    s, Q, d = 0.3, cv2.Rodrigues(np.array([0.4, 0.1, -0.7]))[0], np.array([2.0, 0.5, -1.0])
    other = [Camera(c.K, c.R @ Q.T, s * c.t - c.R @ Q.T @ d) for c in truth]
    errors = camera_pose_errors(other, truth)
    assert errors.rotation_deg.max() < 1e-5  # arccos resolution near 0
    assert errors.centre.max() < 1e-9

    # Moving one camera by 10 % of the cameras' extent shows up (mostly) on that camera.
    centres = np.array([c.centre for c in truth])
    extent = np.sqrt(((centres - centres.mean(axis=0)) ** 2).sum(axis=1).mean())
    moved = list(truth)
    c = truth[2]
    moved[2] = Camera(c.K, c.R, -c.R @ (c.centre + [0.1 * extent, 0, 0]))
    errors = camera_pose_errors(moved, truth)
    assert np.argmax(errors.centre) == 2 and 0.05 < errors.centre[2] < 0.1

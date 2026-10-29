import dataclasses

import numpy as np

from src.config import RegistrationConfig, TriangulationConfig
from src.evaluation.metrics import rotation_error_deg
from src.reconstruction.incremental import initialize
from src.reconstruction.registration import next_best_view, register_image
from tests.sfm_scenes import in_pair_frame, initial_pair, make_sfm_scene

CFG = RegistrationConfig()


def _started(s):
    """Reconstruction initialised from images 0 and 1 with their true relative pose."""
    scene = s.scene
    K = [scene.K] * scene.n_cameras
    return initialize(initial_pair(scene, 0, 1), K, s.tracks, s.features, TriangulationConfig())


def test_register_image_recovers_the_true_pose():
    s = make_sfm_scene(focal=2000.0, arc_deg=90.0)
    _, landmarks = _started(s)
    reg = register_image(2, landmarks, s.tracks, s.features, s.scene.K, CFG)
    assert reg.success and reg.reason is None
    truth = in_pair_frame(s.scene, 2)
    assert rotation_error_deg(reg.camera.R, truth.R) < 1e-6
    np.testing.assert_allclose(reg.camera.t, truth.t, atol=1e-8)
    assert len(reg.keypoints) == len(reg.landmarks) == reg.n_inliers > CFG.min_inliers


def test_register_image_rejects_wrong_links_under_noise():
    s = make_sfm_scene(focal=2000.0, arc_deg=90.0, noise_px=1.0)
    _, landmarks = _started(s)
    rng = np.random.default_rng(1)
    points = s.features[2].points.copy()
    wrong = rng.random(len(points)) < 0.3
    points[wrong] = rng.uniform([0, 0], s.scene.image_size, size=(wrong.sum(), 2))
    features = list(s.features)
    features[2] = dataclasses.replace(s.features[2], points=points)

    reg = register_image(2, landmarks, s.tracks, features, s.scene.K, CFG)
    assert reg.success
    truth = in_pair_frame(s.scene, 2)
    assert rotation_error_deg(reg.camera.R, truth.R) < 0.5
    assert np.linalg.norm(reg.camera.centre - truth.centre) < 0.02  # baseline is 1
    assert wrong[reg.keypoints].mean() < 0.05  # wrong links that slipped through


def test_too_few_links_is_a_failure_not_an_exception():
    s = make_sfm_scene(focal=2000.0, arc_deg=90.0)
    _, landmarks = _started(s)
    landmarks.keep(np.arange(len(landmarks)) < 10)
    reg = register_image(2, landmarks, s.tracks, s.features, s.scene.K, CFG)
    assert not reg.success and reg.reason == "too_few_correspondences"
    assert reg.camera is None and reg.n_inliers == 0 and reg.n_correspondences <= 10


def test_next_best_view_picks_the_image_with_most_links():
    s = make_sfm_scene(n_cameras=5, focal=2000.0, arc_deg=120.0)
    registry, landmarks = _started(s)
    links = {m: len(landmarks.correspondences(m, s.tracks)[0]) for m in (2, 3, 4)}
    assert len(set(links.values())) == 3  # a clear order to test against
    best, second = sorted(links, key=links.get, reverse=True)[:2]

    assert next_best_view(registry, landmarks, s.tracks, 5) == best
    assert next_best_view(registry, landmarks, s.tracks, 5, exclude={best}) == second
    assert next_best_view(registry, landmarks, s.tracks, 5, exclude={2, 3, 4}) is None

import dataclasses

import numpy as np

from src.config import RegistrationConfig, TriangulationConfig
from src.evaluation.metrics import rotation_error_deg
from src.geometry.camera import Camera
from src.reconstruction.camera_registry import CameraRegistry
from src.reconstruction.incremental import initialize
from src.reconstruction.landmarks import Landmarks
from src.reconstruction.registration import (
    next_best_view,
    register_from_neighbour,
    register_image,
)
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


def _few_shared_landmarks(s, target, registered, n_links):
    """True cameras of ``registered`` and landmarks at the true points of the tracks two
    of them see, except that ``target`` sees only ``n_links`` of those landmarks."""
    scene = s.scene
    registry = CameraRegistry()
    for m in registered:
        registry.register(m, Camera(scene.K, scene.rotations[m], scene.translations[m]))
    seen = s.tracks_seen_by(set(registered))
    in_target = set(s.tracks.track[s.tracks.image == target].tolist())
    shared = [k for k in seen if k in in_target]
    keep = [k for k in seen if k not in in_target] + shared[:n_links]
    landmarks = Landmarks()
    landmarks.add(keep, scene.points3d[s.truth[keep]], s.tracks, set(registered))
    return registry, landmarks


def test_relative_pose_registers_an_image_pnp_cannot():
    """Image 3 shares 8 landmarks with the reconstruction (PnP needs 30) but many tracks
    with image 2: its pose comes from the essential matrix with image 2, its scale
    (the baseline length) from the 8 landmarks."""
    s = make_sfm_scene(n_cameras=4, n_points=400, focal=1000.0, arc_deg=60.0, noise_px=0.5)
    scene = s.scene
    registry, landmarks = _few_shared_landmarks(s, 3, [0, 1, 2], n_links=8)
    pnp = register_image(3, landmarks, s.tracks, s.features, scene.K, CFG)
    assert not pnp.success and pnp.reason == "too_few_correspondences"

    reg = register_from_neighbour(3, registry, landmarks, s.tracks, s.features, scene.K, CFG)
    assert reg.success and reg.method == "relative_pose" and pnp.method == "pnp"
    assert 6 <= reg.n_inliers <= 8
    truth = Camera(scene.K, scene.rotations[3], scene.translations[3])
    assert rotation_error_deg(reg.camera.R, truth.R) < 0.2
    assert np.linalg.norm(reg.camera.centre - truth.centre) < 0.02 * np.linalg.norm(truth.centre)


def test_relative_pose_needs_enough_landmarks_for_the_scale():
    s = make_sfm_scene(n_cameras=4, n_points=400, focal=1000.0, arc_deg=60.0, noise_px=0.5)
    registry, landmarks = _few_shared_landmarks(s, 3, [0, 1, 2], n_links=3)
    reg = register_from_neighbour(3, registry, landmarks, s.tracks, s.features, s.scene.K, CFG)
    assert not reg.success and reg.reason == "too_few_links"  # fallback_min_links is 6


def _fallback_case(n_links=8):
    s = make_sfm_scene(n_cameras=4, n_points=400, focal=1000.0, arc_deg=60.0, noise_px=0.5)
    registry, landmarks = _few_shared_landmarks(s, 3, [0, 1, 2], n_links)
    return s, registry, landmarks


def test_relative_pose_rejects_links_to_the_wrong_landmarks():
    """Links whose landmarks are swapped around agree on no scale: no registration."""
    s, registry, landmarks = _fallback_case(n_links=12)
    _, ids = landmarks.correspondences(3, s.tracks)
    landmarks.position[ids] = landmarks.position[np.roll(ids, 1)]
    reg = register_from_neighbour(3, registry, landmarks, s.tracks, s.features, s.scene.K, CFG)
    assert not reg.success and reg.reason == "too_few_links"


def test_relative_pose_rejects_a_scale_behind_the_neighbour():
    """Landmarks placed so that the rays meet them only for a negative baseline length
    (camera on the far side of the neighbour): the scale is not trusted."""
    s, registry, landmarks = _fallback_case(n_links=12)
    scene = s.scene
    # Every registered image shares all 400 tracks with image 3; ties go to image 0.
    R3, t3 = scene.rotations[3], scene.translations[3]
    t0 = R3 @ scene.rotations[0].T @ scene.translations[0]
    baseline = t3 - t0  # s * t_rel with s > 0
    _, ids = landmarks.correspondences(3, s.tracks)
    X_cam = landmarks.position[ids] @ R3.T + t3  # on the rays of image 3
    # R X'' + t0 - baseline = X_cam: the rays meet X'' for s = -|baseline|.
    landmarks.position[ids] = (X_cam - t0 + baseline) @ R3
    reg = register_from_neighbour(3, registry, landmarks, s.tracks, s.features, scene.K, CFG)
    assert not reg.success


def test_relative_pose_needs_enough_shared_tracks():
    """The neighbour pair needs ``fallback_min_pair_inliers`` shared tracks (and essential-
    matrix inliers): fewer give a poorly conditioned pose (ETH3D pipes: a 40-inlier pair
    over 42 deg gave 5 deg). PnP's ``min_inliers`` does not govern it."""
    s, registry, landmarks = _fallback_case()  # 400 tracks shared with every neighbour
    K = s.scene.K
    strict = dataclasses.replace(CFG, fallback_min_pair_inliers=401)
    reg = register_from_neighbour(3, registry, landmarks, s.tracks, s.features, K, strict)
    assert not reg.success and reg.reason == "too_few_matches"
    loose_pnp = dataclasses.replace(CFG, min_inliers=10_000, fallback_min_pair_inliers=50)
    reg = register_from_neighbour(3, registry, landmarks, s.tracks, s.features, K, loose_pnp)
    assert reg.success

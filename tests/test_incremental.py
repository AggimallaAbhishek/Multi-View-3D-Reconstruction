import dataclasses

import numpy as np

from src.config import BundleAdjustmentConfig, Config, TriangulationConfig
from src.evaluation.metrics import rotation_error_deg
from src.geometry.camera import Camera
from src.geometry.reprojection import reprojection_errors
from src.geometry.triangulation import triangulation_angles_deg
from src.reconstruction.camera_registry import CameraRegistry
from src.reconstruction.incremental import (
    complete_tracks,
    filter_landmarks,
    initialize,
    reconstruct,
    triangulate_new_tracks,
)
from src.reconstruction.landmarks import Landmarks
from tests.sfm_scenes import in_pair_frame, initial_pair, make_sfm_scene

CFG = TriangulationConfig()


def test_initialize_reconstructs_the_pair_in_its_own_frame():
    s = make_sfm_scene()
    scene = s.scene
    registry, landmarks = initialize(
        initial_pair(scene, 0, 1), [scene.K] * scene.n_cameras, s.tracks, s.features, CFG
    )
    assert registry.images == [0, 1]
    np.testing.assert_allclose(registry.camera(0).R, np.eye(3))
    np.testing.assert_allclose(registry.camera(0).t, np.zeros(3))
    assert np.isclose(np.linalg.norm(registry.camera(1).centre), 1.0)  # unit baseline

    # Only tracks seen by both images become landmarks, at the true points expressed in
    # camera 0's frame and divided by the true baseline.
    assert sorted(landmarks.track.tolist()) == s.tracks_seen_by({0, 1})
    _, t_true = scene.relative_pose(0, 1)
    X = scene.points3d[s.truth[landmarks.track]]
    expected = (X @ scene.rotations[0].T + scene.translations[0]) / np.linalg.norm(t_true)
    np.testing.assert_allclose(landmarks.position, expected, atol=1e-8)


def test_new_camera_adds_only_newly_triangulable_tracks():
    s = make_sfm_scene(focal=2000.0, arc_deg=90.0)  # each camera sees only part of the scene
    scene = s.scene
    # Corrupt image 2's keypoint of one track that only images 1 and 2 share.
    only_12 = sorted(set(s.tracks_seen_by({1, 2})) - set(s.tracks_seen_by({0, 1})))
    t_bad = only_12[0]
    k = (s.tracks.track == t_bad) & (s.tracks.image == 2)
    points = s.features[2].points.copy()
    points[s.tracks.keypoint[k]] += 40.0
    features = list(s.features)
    features[2] = dataclasses.replace(s.features[2], points=points)

    registry, landmarks = initialize(
        initial_pair(scene, 0, 1), [scene.K] * scene.n_cameras, s.tracks, features, CFG
    )
    before = set(landmarks.track.tolist())
    registry.register(2, in_pair_frame(scene, 2))
    new = triangulate_new_tracks(registry, landmarks, s.tracks, features, CFG)

    added = set(landmarks.track[new].tolist())
    assert added == set(s.tracks_seen_by({0, 1, 2})) - before - {t_bad}
    b = np.linalg.norm(scene.relative_pose(0, 1)[1])
    X = scene.points3d[s.truth[landmarks.track[new]]]
    expected = (X @ scene.rotations[0].T + scene.translations[0]) / b
    np.testing.assert_allclose(landmarks.position[new], expected, atol=1e-8)
    assert np.all(landmarks.error[new] < 1e-6)
    assert len(triangulate_new_tracks(registry, landmarks, s.tracks, features, CFG)) == 0


def test_reconstruct_registers_every_camera_of_a_synthetic_scene():
    s = make_sfm_scene(n_cameras=8, n_points=600, focal=2500.0, arc_deg=150.0, noise_px=0.5)
    scene = s.scene
    assert (scene.visible.sum(axis=0) < 8).mean() > 0.5  # no camera sees everything
    rec = reconstruct(initial_pair(scene, 3, 4), [scene.K] * 8, s.tracks, s.features, Config())
    assert sorted(rec.registry.images) == list(range(8))
    for m in range(8):
        truth = in_pair_frame(scene, m, 3, 4)
        cam = rec.registry.camera(m)
        assert rotation_error_deg(cam.R, truth.R) < 0.5, m
        assert np.linalg.norm(cam.centre - truth.centre) < 0.05, m  # initial baseline is 1
    assert np.median(rec.landmarks.error) < 1.0


def _pose_errors(rec, scene, i, j):
    rot, centre = [], []
    for m in rec.registry.images:
        truth, cam = in_pair_frame(scene, m, i, j), rec.registry.camera(m)
        rot.append(rotation_error_deg(cam.R, truth.R))
        centre.append(np.linalg.norm(cam.centre - truth.centre))
    return np.array(rot), np.array(centre)


def test_bundle_adjustment_reduces_drift_in_reconstruct():
    s = make_sfm_scene(n_cameras=10, n_points=800, focal=2500.0, arc_deg=200.0, noise_px=1.0)
    scene = s.scene
    args = (initial_pair(scene, 4, 5), [scene.K] * 10, s.tracks, s.features)
    off = Config(bundle_adjustment=BundleAdjustmentConfig(enabled=False))
    without = reconstruct(*args, off)
    with_ba = reconstruct(*args, Config())
    assert len(with_ba.registry) == len(without.registry) == 10

    rot_off, centre_off = _pose_errors(without, scene, 4, 5)
    rot_on, centre_on = _pose_errors(with_ba, scene, 4, 5)
    assert centre_on.max() < 0.5 * centre_off.max()
    assert rot_on.max() < rot_off.max()
    cameras, obs, X = with_ba.landmarks.as_problem(with_ba.registry, s.features)
    errors = reprojection_errors(cameras, obs, X)
    assert np.sqrt(np.mean(errors**2)) < np.sqrt(2) * 1.0  # at the 1 px noise level
    # Stored per-landmark errors are those of the adjusted reconstruction.
    mean_error = np.bincount(obs.point, errors, len(X)) / np.bincount(obs.point, minlength=len(X))
    np.testing.assert_allclose(with_ba.landmarks.error, mean_error, rtol=1e-9)


def _true_reconstruction(s, images):
    """Registry with the true cameras of ``images`` and a landmark at the true point of
    every track that at least two of them see."""
    scene = s.scene
    registry = CameraRegistry()
    for m in images:
        registry.register(m, Camera(scene.K, scene.rotations[m], scene.translations[m]))
    landmarks = Landmarks()
    seen = s.tracks_seen_by(set(images))
    landmarks.add(seen, scene.points3d[s.truth[seen]], s.tracks, set(images))
    return registry, landmarks


def _shift_keypoint(features, image, keypoint, offset):
    points = features[image].points.copy()
    points[keypoint] += offset
    features[image] = dataclasses.replace(features[image], points=points)


def _views(landmarks, lid):
    return set(landmarks.obs_image[landmarks.obs_landmark == lid].tolist())


def test_filter_landmarks_drops_bad_observations_then_weak_landmarks():
    s = make_sfm_scene(n_cameras=4, focal=2000.0, arc_deg=90.0)  # tracks of 2 to 4 views
    features = list(s.features)
    registry, landmarks = _true_reconstruction(s, [0, 1, 2, 3])
    views = np.bincount(landmarks.obs_landmark)
    long_lm, short_lm = np.flatnonzero(views == 4)[0], np.flatnonzero(views == 2)[0]
    for lid in (long_lm, short_lm):  # one observation of each goes 5 px off
        k = np.flatnonzero(landmarks.obs_landmark == lid)[0]
        _shift_keypoint(features, landmarks.obs_image[k], landmarks.obs_keypoint[k], 5.0)
    tracks_before = landmarks.track.copy()
    n_obs = len(landmarks.obs_image)

    removed = filter_landmarks(registry, landmarks, features, CFG)
    assert removed == 3  # the two bad observations and the good one left alone
    assert len(landmarks) == len(tracks_before) - 1
    assert tracks_before[short_lm] not in landmarks.track
    kept = np.flatnonzero(landmarks.track == tracks_before[long_lm])[0]
    assert len(_views(landmarks, kept)) == 3
    assert len(landmarks.obs_image) == n_obs - 3
    cameras, obs, X = landmarks.as_problem(registry, features)
    assert reprojection_errors(cameras, obs, X).max() < 1e-6
    assert filter_landmarks(registry, landmarks, features, CFG) == 0

    # Landmarks whose rays are too parallel go as a whole.
    angles = triangulation_angles_deg(cameras, obs, X)
    cutoff = float(np.median(angles))
    strict = dataclasses.replace(CFG, min_angle_deg=cutoff)
    filter_landmarks(registry, landmarks, features, strict)
    assert len(landmarks) == (angles >= cutoff).sum()


def test_complete_tracks_reattaches_keypoints_that_fit():
    s = make_sfm_scene(n_cameras=4)
    features = list(s.features)
    registry, landmarks = _true_reconstruction(s, [0, 1, 2, 3])
    full = len(landmarks.obs_image)
    in_3 = np.flatnonzero(landmarks.obs_image == 3)
    bad = in_3[0]
    _shift_keypoint(features, 3, landmarks.obs_keypoint[bad], 5.0)
    landmarks.keep_observations(landmarks.obs_image != 3)  # as if PnP had rejected them

    added = complete_tracks(registry, landmarks, s.tracks, features, CFG)
    assert added == len(in_3) - 1  # all but the corrupted keypoint
    assert len(landmarks.obs_image) == full - 1
    cameras, obs, X = landmarks.as_problem(registry, features)
    assert reprojection_errors(cameras, obs, X).max() < 1e-6
    assert complete_tracks(registry, landmarks, s.tracks, features, CFG) == 0


def test_reconstruct_keeps_only_observations_within_the_threshold():
    s = make_sfm_scene(n_cameras=8, n_points=600, focal=2500.0, arc_deg=150.0, noise_px=0.5)
    scene = s.scene
    # 5 % of keypoints 3.5 px off: wrong enough to fail triangulation, close enough to
    # pass PnP's 4 px, so without filtering they enter the reconstruction at registration.
    features = list(s.features)
    rng = np.random.default_rng(0)
    corrupted = set()
    for m in range(8):
        ks = np.flatnonzero(rng.random(len(features[m].points)) < 0.05)
        angle = rng.uniform(0, 2 * np.pi, len(ks))
        _shift_keypoint(features, m, ks, 3.5 * np.stack([np.cos(angle), np.sin(angle)], 1))
        corrupted |= {(m, k) for k in ks.tolist()}

    config = Config()
    rec = reconstruct(initial_pair(scene, 3, 4), [scene.K] * 8, s.tracks, features, config)
    assert sorted(rec.registry.images) == list(range(8))
    lm = rec.landmarks
    cameras, obs, X = lm.as_problem(rec.registry, features)
    errors = reprojection_errors(cameras, obs, X)
    assert errors.max() <= config.triangulation.max_reprojection_error_px
    assert np.bincount(obs.point, minlength=len(lm)).min() >= 2
    min_angle = np.nanmin(triangulation_angles_deg(cameras, obs, X))
    assert min_angle >= config.triangulation.min_angle_deg
    # A shift along the epipolar line of a short track is undetectable, so some survive,
    # but far below their 5 % share of the input.
    observed = set(zip(lm.obs_image.tolist(), lm.obs_keypoint.tolist(), strict=True))
    assert len(observed & corrupted) < 0.02 * len(observed)
    for m in range(8):
        truth = in_pair_frame(scene, m, 3, 4)
        assert rotation_error_deg(rec.registry.camera(m).R, truth.R) < 0.5, m

import dataclasses

import numpy as np

from src.config import BundleAdjustmentConfig, Config, TriangulationConfig
from src.evaluation.metrics import rotation_error_deg
from src.geometry.reprojection import reprojection_errors
from src.reconstruction.incremental import initialize, reconstruct, triangulate_new_tracks
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

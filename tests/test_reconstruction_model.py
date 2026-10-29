import numpy as np
import pytest

from src.geometry.camera import Camera
from src.geometry.reprojection import reprojection_errors
from src.reconstruction.camera_registry import CameraRegistry
from src.reconstruction.landmarks import Landmarks
from src.reconstruction.tracks import Tracks
from tests.sfm_scenes import make_sfm_scene


def _camera(tx=0.0):
    return Camera(np.eye(3), np.eye(3), np.array([tx, 0.0, 0.0]))


def test_registry_keeps_registration_order_and_rejects_duplicates():
    registry = CameraRegistry()
    registry.register(4, _camera(1.0))
    registry.register(1, _camera(2.0))
    assert registry.images == [4, 1]
    assert 4 in registry and 0 not in registry
    assert registry.camera(1).t[0] == 2.0
    with pytest.raises(ValueError, match="already registered"):
        registry.register(4, _camera())
    registry.update(4, _camera(3.0))
    assert registry.camera(4).t[0] == 3.0 and registry.images == [4, 1]


def _tracks():
    # track 0: images 0, 1, 2 (keypoints 10, 11, 12); track 1: images 0, 1 (20, 21);
    # track 2: images 1, 2 (31, 32).
    return Tracks(
        track=np.array([0, 0, 0, 1, 1, 2, 2]),
        image=np.array([0, 1, 2, 0, 1, 1, 2]),
        keypoint=np.array([10, 11, 12, 20, 21, 31, 32]),
    )


def _obs(landmarks):
    return sorted(
        zip(landmarks.obs_landmark.tolist(), landmarks.obs_image.tolist(),
            landmarks.obs_keypoint.tolist(), strict=True)
    )  # fmt: skip


def test_landmarks_take_track_observations_in_registered_images():
    lm = Landmarks()
    ids = lm.add([0, 1], np.array([[0.0, 0, 1], [1.0, 0, 1]]), _tracks(), registered={0, 1})
    np.testing.assert_array_equal(ids, [0, 1])
    assert len(lm) == 2
    assert _obs(lm) == [(0, 0, 10), (0, 1, 11), (1, 0, 20), (1, 1, 21)]  # image 2 not yet

    lm.add_observations(image=2, keypoints=np.array([12]), landmarks=np.array([0]))
    assert (0, 2, 12) in _obs(lm)

    lm.keep(np.array([False, True]))
    assert len(lm) == 1
    np.testing.assert_array_equal(lm.track, [1])
    np.testing.assert_allclose(lm.position, [[1.0, 0, 1]])
    assert _obs(lm) == [(0, 0, 20), (0, 1, 21)]  # renumbered, landmark 0's gone


def test_correspondences_link_keypoints_of_a_new_image_to_their_landmarks():
    s = make_sfm_scene()
    scene, point_of, tracks, truth = s.scene, s.point_of, s.tracks, s.truth
    # Landmarks for the tracks seen by images 0 and 1, at their true positions.
    both = s.tracks_seen_by({0, 1})
    lm = Landmarks()
    lm.add(both, scene.points3d[truth[both]], tracks, registered={0, 1})

    keypoints, landmarks = lm.correspondences(2, tracks)
    np.testing.assert_allclose(lm.position[landmarks], scene.points3d[point_of[2][keypoints]])
    expected = scene.visible[0] & scene.visible[1] & scene.visible[2]
    assert len(keypoints) == expected.sum()


def test_as_problem_reprojects_an_exact_scene_with_zero_error():
    s = make_sfm_scene()
    scene, tracks, truth = s.scene, s.tracks, s.truth
    registry = CameraRegistry()
    for m in (2, 0, 3):  # registration order != image order: view indices must follow it
        registry.register(m, Camera(scene.K, scene.rotations[m], scene.translations[m]))
    lm = Landmarks()
    seen = s.tracks_seen_by({0, 2, 3})
    lm.add(seen, scene.points3d[truth[seen]], tracks, registered=set(registry.images))

    cameras, obs, X = lm.as_problem(registry, s.features)
    assert len(cameras) == 3 and cameras[0] is registry.camera(2)
    assert len(obs) == len(lm.obs_image) and X.shape == (len(lm), 3)
    assert reprojection_errors(cameras, obs, X).max() < 1e-9

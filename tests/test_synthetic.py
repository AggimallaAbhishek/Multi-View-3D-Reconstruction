import numpy as np
import pytest

from tests.synthetic import camera_center, look_at, make_scene, project


def test_rotations_are_proper(scene):
    for R in scene.rotations:
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_look_at_points_optical_axis_at_target():
    center = np.array([3.0, -1.0, -4.0])
    R, t = look_at(center, np.zeros(3))
    np.testing.assert_allclose(camera_center(R, t), center, atol=1e-12)
    # The target lies on the +z axis of the camera frame.
    target_cam = R @ np.zeros(3) + t
    np.testing.assert_allclose(target_cam[:2], 0.0, atol=1e-12)
    assert target_cam[2] > 0


def test_scene_centre_projects_to_principal_point(scene):
    for i in range(scene.n_cameras):
        pixels, _ = project(scene.K, scene.rotations[i], scene.translations[i], np.zeros((1, 3)))
        np.testing.assert_allclose(pixels[0], scene.K[:2, 2], atol=1e-9)


def test_all_points_in_front_of_and_visible_to_every_camera(scene):
    for i in range(scene.n_cameras):
        _, depths = project(scene.K, scene.rotations[i], scene.translations[i], scene.points3d)
        assert np.all(depths > 0)
    assert scene.visible.all()


def test_projection_matches_projection_matrix(scene):
    X_hom = np.hstack([scene.points3d, np.ones((scene.n_points, 1))])
    for i in range(scene.n_cameras):
        x = X_hom @ scene.projection_matrix(i).T
        np.testing.assert_allclose(x[:, :2] / x[:, 2:], scene.observations[i], atol=1e-9)


def test_points_behind_camera_are_not_visible():
    # Cameras inside the point cube: roughly half the points are behind each camera.
    behind = make_scene(n_points=200, n_cameras=2, radius=0.3, elevation=0.0, seed=1)
    _, depths = project(behind.K, behind.rotations[0], behind.translations[0], behind.points3d)
    assert np.any(depths <= 0)
    assert not behind.visible[0, depths <= 0].any()


def test_relative_pose_maps_camera_frames(scene):
    R, t = scene.relative_pose(0, 3)
    X = scene.points3d
    X0 = X @ scene.rotations[0].T + scene.translations[0]
    X3 = X @ scene.rotations[3].T + scene.translations[3]
    np.testing.assert_allclose(X0 @ R.T + t, X3, atol=1e-12)


def test_ground_truth_satisfies_epipolar_constraint(scene):
    # E = [t]x R ; x2^T F x1 = 0 for exact correspondences. Phase 4 relies on this.
    R, t = scene.relative_pose(0, 1)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    K_inv = np.linalg.inv(scene.K)
    F = K_inv.T @ (tx @ R) @ K_inv
    x1, x2 = scene.pair(0, 1)
    x1h = np.hstack([x1, np.ones((len(x1), 1))])
    x2h = np.hstack([x2, np.ones((len(x2), 1))])
    residuals = np.einsum("ni,ij,nj->n", x2h, F, x1h)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-9)


def test_noise_has_requested_scale(scene, noisy_scene):
    diff = noisy_scene.observations - scene.observations
    assert diff.std() == pytest.approx(1.0, rel=0.1)
    np.testing.assert_array_equal(noisy_scene.points3d, scene.points3d)


def test_scene_is_deterministic_for_seed():
    a = make_scene(seed=42, noise_px=0.5)
    b = make_scene(seed=42, noise_px=0.5)
    c = make_scene(seed=43, noise_px=0.5)
    np.testing.assert_array_equal(a.observations, b.observations)
    assert not np.array_equal(a.points3d, c.points3d)


def test_baseline_gives_real_parallax(scene):
    # Phase 5/6 need non-degenerate geometry: the outer cameras must be well separated.
    baseline = np.linalg.norm(scene.camera_center(0) - scene.camera_center(scene.n_cameras - 1))
    assert baseline > 3.0

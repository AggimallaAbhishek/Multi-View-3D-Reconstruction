import cv2
import numpy as np
import pytest

from src.config import TriangulationConfig, load_config
from src.geometry.camera import Camera
from src.geometry.triangulation import (
    Observations,
    refine_points,
    triangulate_and_filter,
    triangulate_dlt,
    triangulate_two_view,
    triangulation_angles_deg,
)
from tests.synthetic import make_scene, project

CFG = load_config().triangulation


def _cameras(scene):
    return [Camera(scene.K, R, t) for R, t in zip(scene.rotations, scene.translations, strict=True)]


def _observations(scene, views=None, noise_px=0.0, seed=0):
    """Every visible observation of the given views, as an observation list."""
    views = range(scene.n_cameras) if views is None else views
    view, point, xy = [], [], []
    for v in views:
        idx = np.flatnonzero(scene.visible[v])
        view.append(np.full(len(idx), v))
        point.append(idx)
        xy.append(scene.observations[v, idx])
    xy = np.concatenate(xy)
    if noise_px:
        xy = xy + np.random.default_rng(seed).normal(0, noise_px, xy.shape)
    return Observations(np.concatenate(view), np.concatenate(point), xy)


@pytest.fixture(scope="module")
def scene():
    return make_scene(n_points=300, n_cameras=5, noise_px=0.0, seed=0)


# --- camera model -------------------------------------------------------------------------


def test_camera_matches_projection_model(scene):
    cam = Camera(scene.K, scene.rotations[1], scene.translations[1])
    np.testing.assert_allclose(cam.P, scene.projection_matrix(1))
    np.testing.assert_allclose(cam.centre, scene.camera_center(1))
    pixels, depths = cam.project(scene.points3d)
    expected, expected_depths = project(
        scene.K, scene.rotations[1], scene.translations[1], scene.points3d
    )
    np.testing.assert_allclose(pixels, expected)
    np.testing.assert_allclose(depths, expected_depths)


# --- linear triangulation -------------------------------------------------------------------


def test_two_view_dlt_is_exact_and_matches_opencv(scene):
    cams = _cameras(scene)
    X = triangulate_dlt(cams, _observations(scene, views=[0, 1]))
    np.testing.assert_allclose(X, scene.points3d, atol=1e-8)

    X_h = cv2.triangulatePoints(
        cams[0].P, cams[1].P, scene.observations[0].T, scene.observations[1].T
    )
    np.testing.assert_allclose((X_h[:3] / X_h[3]).T, X, atol=1e-8)


def test_n_view_dlt_is_exact(scene):
    X = triangulate_dlt(_cameras(scene), _observations(scene))
    np.testing.assert_allclose(X, scene.points3d, atol=1e-8)


def test_n_view_agrees_with_two_view_function(scene):
    cams = _cameras(scene)
    X_n = triangulate_dlt(cams, _observations(scene, views=[0, 2]))
    X_2 = triangulate_two_view(cams[0].P, cams[2].P, scene.observations[0], scene.observations[2])
    np.testing.assert_allclose(X_n, X_2, atol=1e-8)


def test_mixed_track_lengths_in_one_call(scene):
    # Points 0-99 seen by 2 views, 100-199 by 3, 200-299 by all 5.
    obs = _observations(scene)
    keep = (
        (obs.point < 100) & (obs.view < 2)
        | (obs.point >= 100) & (obs.point < 200) & (obs.view < 3)
        | (obs.point >= 200)
    )
    X = triangulate_dlt(_cameras(scene), obs.subset(keep))
    np.testing.assert_allclose(X, scene.points3d, atol=1e-8)


def test_more_views_reduce_error_under_noise(scene):
    cams = _cameras(scene)
    err = []
    for views in ([0, 1], [0, 1, 2, 3, 4]):
        X = triangulate_dlt(cams, _observations(scene, views=views, noise_px=1.0))
        err.append(np.median(np.linalg.norm(X - scene.points3d, axis=1)))
    assert err[1] < 0.7 * err[0]


def test_point_with_single_observation_is_nan(scene):
    obs = _observations(scene, views=[0, 1])
    keep = ~((obs.point == 7) & (obs.view == 1))  # point 7 only seen by view 0
    X = triangulate_dlt(_cameras(scene), obs.subset(keep))
    assert np.isnan(X[7]).all()
    assert np.isfinite(np.delete(X, 7, axis=0)).all()


# --- nonlinear refinement ----------------------------------------------------------------


def test_refinement_reduces_reprojection_error(scene):
    from src.geometry.reprojection import reprojection_errors

    cams = _cameras(scene)
    obs = _observations(scene, noise_px=1.0)
    X_lin = triangulate_dlt(cams, obs)
    X_ref = refine_points(cams, obs, X_lin, iterations=5)
    e_lin, e_ref = reprojection_errors(cams, obs, X_lin), reprojection_errors(cams, obs, X_ref)
    assert np.sum(e_ref**2) < np.sum(e_lin**2)
    truth = scene.points3d
    assert (
        np.median(np.linalg.norm(X_ref - truth, axis=1))
        <= np.median(np.linalg.norm(X_lin - truth, axis=1)) * 1.01
    )


def test_refinement_keeps_exact_points(scene):
    cams = _cameras(scene)
    obs = _observations(scene)
    X = refine_points(cams, obs, triangulate_dlt(cams, obs), iterations=5)
    np.testing.assert_allclose(X, scene.points3d, atol=1e-8)


# --- angles and filters --------------------------------------------------------------------


def test_triangulation_angle_is_largest_pairwise_ray_angle(scene):
    cams = _cameras(scene)
    obs = _observations(scene)
    angles = triangulation_angles_deg(cams, obs, scene.points3d)
    p = 0
    rays = [scene.points3d[p] - cams[v].centre for v in range(scene.n_cameras)]
    rays = [r / np.linalg.norm(r) for r in rays]
    expected = max(np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for a in rays for b in rays)
    assert angles[p] == pytest.approx(expected)


def test_clean_scene_passes_all_filters(scene):
    result = triangulate_and_filter(_cameras(scene), _observations(scene, noise_px=0.5), CFG)
    assert result.valid.all()
    assert result.counts == {
        "points": 300, "too_few_views": 0, "non_finite": 0, "behind_camera": 0,
        "reprojection": 0, "low_angle": 0, "valid": 300,
    }  # fmt: skip
    assert result.reprojection.median < 1.0


def test_filters_reject_bad_points(scene):
    cams = _cameras(scene)
    obs = _observations(scene, views=[0, 1])
    xy = obs.xy.copy()
    # Cameras on a horizontal arc have near-horizontal epipolar lines. A horizontal error
    # moves along the epipolar line and is absorbed as a wrong depth (zero residual) -
    # undetectable from two views. A vertical error crosses the line and must be caught.
    xy[(obs.point == 3) & (obs.view == 1)] += [0.0, 25.0]
    obs = Observations(obs.view, obs.point, xy)
    result = triangulate_and_filter(cams, obs, TriangulationConfig(refine=False))
    assert not result.valid[3]
    assert result.counts["reprojection"] >= 1
    assert result.valid.sum() == result.counts["valid"]


def test_points_behind_a_camera_are_rejected():
    # Cameras inside the point cube: some points lie behind them; their projections are
    # still well defined, so linear triangulation recovers them exactly.
    scene = make_scene(n_points=200, n_cameras=2, radius=0.3, elevation=0.0, seed=1)
    cams = _cameras(scene)
    n = scene.n_points
    obs = Observations(
        np.repeat([0, 1], n), np.tile(np.arange(n), 2), scene.observations.reshape(-1, 2)
    )
    result = triangulate_and_filter(cams, obs, TriangulationConfig(refine=False))
    depths = np.stack([c.project(scene.points3d)[1] for c in cams])
    behind = (depths <= 0).any(axis=0)
    assert behind.any()
    assert not result.valid[behind].any()
    assert result.counts["behind_camera"] == behind.sum()


def test_low_angle_points_are_rejected(scene):
    # A point far along camera 0's axis is seen at a tiny angle by cameras 0 and 1.
    cams = _cameras(scene)[:2]
    far = cams[0].centre + 500.0 * cams[0].R[2]
    xy = np.stack([c.project(far[None])[0][0] for c in cams])
    obs = Observations(np.array([0, 1]), np.array([0, 0]), xy)
    result = triangulate_and_filter(cams, obs, CFG)
    assert result.angles_deg[0] < CFG.min_angle_deg
    assert not result.valid[0]
    assert result.counts["low_angle"] == 1

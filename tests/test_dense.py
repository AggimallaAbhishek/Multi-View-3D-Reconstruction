"""Dense stereo against ground truth: depth maps of the rendered box scene from known
cameras (tests/render.py), and fused clouds measured against the true surface."""

import numpy as np
import pytest

from src.config import DenseConfig
from src.dense import DenseCloud, fuse_depth_maps, pair_depth, plan_views, voxel_downsample
from src.geometry.camera import Camera
from tests.render import (
    box_scene_distance,
    box_scene_faces,
    render,
    render_box_scene,
    render_depth,
)
from tests.synthetic import intrinsics, look_at

SIZE = (640, 480)
CONFIG = DenseConfig()


@pytest.fixture(scope="module")
def scene():
    """Eight views 6.4 deg apart on a 45 deg arc."""
    return render_box_scene(n_cameras=8, size=SIZE, focal=640.0, arc_deg=45.0)


def _cameras(scene) -> list[Camera]:
    return [Camera(scene.K, R, t) for R, t in zip(scene.rotations, scene.translations, strict=True)]


def _truth(scene, i: int) -> np.ndarray:
    return render_depth(box_scene_faces(), scene.K, scene.rotations[i], scene.translations[i], SIZE)


def _surface_points(scene, n: int = 4000) -> np.ndarray:
    """Points on the scene's true surface seen by the cameras (the sparse cloud's role:
    it fixes the depth range)."""
    rng = np.random.default_rng(0)
    box = rng.uniform(-1, 1, (n, 3))
    box[np.arange(n), rng.integers(0, 3, n)] = rng.choice([-1.0, 1.0], n)
    ground = np.column_stack([rng.uniform(-3, 3, n), np.ones(n), rng.uniform(-3, 3, n)])
    return np.vstack([box, ground])


def _errors(depth: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Median relative depth error over pixels with an estimate, and the fraction of
    true-surface pixels that got one."""
    both = np.isfinite(depth) & np.isfinite(truth)
    rel = np.abs(depth[both] - truth[both]) / truth[both]
    return float(np.median(rel)), both.sum() / np.isfinite(truth).sum()


# --- pair depth -------------------------------------------------------------------------


@pytest.mark.parametrize(("i", "j"), [(3, 4), (4, 3)], ids=["partner-right", "partner-left"])
def test_pair_depth_matches_the_true_depth(scene, i, j):
    cams = _cameras(scene)
    depth = pair_depth(scene.images[i], cams[i], scene.images[j], cams[j],
                       _surface_points(scene), CONFIG)  # fmt: skip
    median, coverage = _errors(depth, _truth(scene, i))
    assert median < 0.003
    assert coverage > 0.8


def test_pair_depth_with_a_vertical_baseline():
    """The second camera moved along the first one's up axis: rectification is vertical."""
    K, faces = intrinsics(640.0, *SIZE), box_scene_faces()
    centre = np.array([0.3, -3.0, -5.5])
    up = -look_at(centre, np.zeros(3))[0][1]  # camera y points down
    cams = [Camera(K, *look_at(centre + dy * up, np.zeros(3))) for dy in (0, 0.7)]
    images = [render(faces, c.K, c.R, c.t, SIZE) for c in cams]
    depth = pair_depth(images[0], cams[0], images[1], cams[1], _surface_points(None), CONFIG)
    median, coverage = _errors(depth, render_depth(faces, K, cams[0].R, cams[0].t, SIZE))
    assert median < 0.003
    assert coverage > 0.8


def test_pair_depth_without_points_in_view_is_empty(scene):
    cams = _cameras(scene)
    behind = np.array([[0.0, 0.0, -100.0]])  # behind every camera
    depth = pair_depth(scene.images[0], cams[0], scene.images[1], cams[1], behind, CONFIG)
    assert not np.isfinite(depth).any()


# --- view planning ----------------------------------------------------------------------


def test_plan_views_pick_a_partner_in_the_angle_range_and_nearby_checks():
    """Cameras every 2 deg on a ring: the stereo partner is the view nearest the target
    angle within [min, max]; fusion checks use the nearest views."""
    K = intrinsics(500.0, *SIZE)
    angles = np.deg2rad(np.arange(0, 40, 2.0))
    cams = [Camera(K, *look_at(5 * np.array([np.sin(a), 0, -np.cos(a)]), np.zeros(3)))
            for a in angles]  # fmt: skip
    config = DenseConfig(min_angle_deg=3.0, target_angle_deg=8.0, max_angle_deg=20.0,
                         fusion_views=3)  # fmt: skip
    plan = plan_views(cams, np.zeros((1, 3)), config)
    assert plan.partner[0] == 4  # 8 deg away
    assert plan.partner[10] in (6, 14)
    assert sorted(plan.checks[10]) == [9, 11, 12] or sorted(plan.checks[10]) == [8, 9, 11]
    assert len(plan.checks[0]) == 3


def test_plan_views_without_a_partner_in_range():
    K = intrinsics(500.0, *SIZE)
    cams = [Camera(K, *look_at(np.array([x, 0, -5]), np.zeros(3))) for x in (0.0, 0.01)]
    plan = plan_views(cams, np.zeros((1, 3)), DenseConfig(min_angle_deg=3.0))
    assert plan.partner == [None, None]


# --- fusion -----------------------------------------------------------------------------


def _fused(scene, depths) -> DenseCloud:
    cams = _cameras(scene)
    plan = plan_views(cams, _surface_points(scene), CONFIG)
    return fuse_depth_maps(depths, cams, scene.images, plan.checks, 0.004, CONFIG)


def test_fusion_of_true_depths_lies_on_the_surface(scene):
    cloud = _fused(scene, [_truth(scene, i) for i in range(len(scene.images))])
    assert len(cloud.points) > 20_000
    assert np.percentile(box_scene_distance(cloud.points), 99) < 0.01
    assert cloud.colors.dtype == np.uint8 and cloud.colors.shape == cloud.points.shape


def test_fusion_drops_depths_no_other_view_confirms(scene):
    """One view's depth map is 10 % too far everywhere: none of it survives."""
    truth = [_truth(scene, i) for i in range(len(scene.images))]
    wrong = [d.copy() for d in truth]
    wrong[3] = 1.1 * truth[3]
    cams = _cameras(scene)
    plan = plan_views(cams, _surface_points(scene), CONFIG)
    full = fuse_depth_maps(truth, cams, scene.images, plan.checks, 0.004, CONFIG)
    bad = fuse_depth_maps(wrong, cams, scene.images, plan.checks, 0.004, CONFIG)
    assert np.percentile(box_scene_distance(bad.points), 99) < 0.01
    assert len(bad.points) < len(full.points)


def test_dense_pipeline_on_rendered_views(scene):
    """Pair depths from the true cameras, fused: accurate and much denser than sparse."""
    cams = _cameras(scene)
    points = _surface_points(scene)
    plan = plan_views(cams, points, CONFIG)
    depths = [None if j is None else
              pair_depth(scene.images[i], cams[i], scene.images[j], cams[j], points, CONFIG)
              for i, j in enumerate(plan.partner)]  # fmt: skip
    cloud = fuse_depth_maps(depths, cams, scene.images, plan.checks, 0.004, CONFIG)
    distance = box_scene_distance(cloud.points)
    assert len(cloud.points) > 20_000
    assert np.median(distance) < 0.005
    assert np.percentile(distance, 95) < 0.03


def test_voxel_downsample_averages_points_in_a_cell():
    points = np.array([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [5.0, 5.0, 5.0]])
    colors = np.array([[0, 0, 0], [200, 100, 50], [10, 10, 10]], np.uint8)
    cloud = voxel_downsample(DenseCloud(points, colors, np.ones((3, 3))), 1.0)
    order = np.argsort(cloud.points[:, 0])
    np.testing.assert_allclose(cloud.points[order], [[0.2, 0.2, 0.2], [5, 5, 5]])
    np.testing.assert_array_equal(cloud.colors[order], [[100, 50, 25], [10, 10, 10]])


def test_pair_depth_skips_textureless_pixels(scene):
    """A uniform patch pasted into both views gets no depth (it would match anywhere)."""
    cams = _cameras(scene)
    a, b = scene.images[3].copy(), scene.images[4].copy()
    a[100:200, 250:400] = 0
    b[100:200, 250:400] = 0
    depth = pair_depth(a, cams[3], b, cams[4], _surface_points(scene), CONFIG)
    assert not np.isfinite(depth[105:195, 255:395]).any()


def test_plan_views_do_not_check_a_depth_with_the_same_stereo_pair():
    """A view whose depth map came from stereo with view i cannot confirm i's depths
    independently (same pair, same matching errors): it is not among i's checks."""
    K = intrinsics(500.0, *SIZE)
    angles = np.deg2rad(np.arange(0, 40, 2.0))
    cams = [Camera(K, *look_at(5 * np.array([np.sin(a), 0, -np.cos(a)]), np.zeros(3)))
            for a in angles]  # fmt: skip
    plan = plan_views(cams, np.zeros((1, 3)), DenseConfig(fusion_views=6))
    for i, checks in enumerate(plan.checks):
        assert len(checks) == 6
        assert all(plan.partner[k] != i for k in checks)

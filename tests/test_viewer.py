import numpy as np
import pytest

from src.geometry.camera import Camera
from src.visualization.viewer import RenderUnavailable, render_scene, scene_geometries


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(300, 3)) * 0.5 + np.array([0, 0, 6.0])
    colors = rng.random((300, 3))
    K = np.array([[600.0, 0, 319.5], [0, 600.0, 239.5], [0, 0, 1]])
    cameras = [Camera(K, np.eye(3), np.zeros(3)), Camera(K, np.eye(3), np.array([-1.0, 0, 0]))]
    return points, colors, cameras, [(640, 480), (640, 480)]


def test_scene_geometries_contains_cloud_and_one_frustum_per_camera(scene):
    import open3d as o3d

    geometries = scene_geometries(*scene)
    assert [name for name, _ in geometries] == ["points", "camera_0", "camera_1"]
    cloud = geometries[0][1]
    assert isinstance(cloud, o3d.t.geometry.PointCloud)
    assert cloud.point.positions.shape[0] == 300
    assert all(isinstance(g, o3d.t.geometry.LineSet) for _, g in geometries[1:])


def test_frustum_sits_at_the_camera_centre(scene):
    points, colors, cameras, sizes = scene
    _, frustum = scene_geometries(points, colors, cameras, sizes)[2]
    apex = frustum.point.positions.numpy()[0]  # first vertex is the optical centre
    np.testing.assert_allclose(apex, cameras[1].centre, atol=1e-5)


def test_render_scene_writes_image(scene, tmp_path):
    try:
        path = render_scene(*scene, tmp_path / "render.png", width=320, height=240)
    except RenderUnavailable as exc:
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    import cv2

    image = cv2.imread(str(path))
    assert image.shape == (240, 320, 3)
    assert image.std() > 1  # not a blank frame

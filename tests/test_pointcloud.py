import json

import numpy as np
import pytest

from src.geometry.camera import Camera
from src.geometry.triangulation import Observations
from src.pointcloud.coloring import color_points, sample_colors
from src.pointcloud.export import read_cameras_json, read_ply, write_cameras_json, write_ply

# --- colour sampling --------------------------------------------------------------------


def _gradient_image():
    """BGR image whose red channel equals x and green channel equals y (in 0-255)."""
    h, w = 200, 256
    image = np.zeros((h, w, 3), np.uint8)
    image[..., 2] = np.arange(w)[None, :]
    image[..., 1] = np.minimum(np.arange(h), 255)[:, None]
    return image


def test_sample_colors_at_pixel_centres_are_exact():
    image = _gradient_image()
    rgb = sample_colors(image, np.array([[10.0, 20.0], [100.0, 50.0]]))
    np.testing.assert_allclose(rgb * 255, [[10, 20, 0], [100, 50, 0]], atol=1e-6)


def test_sample_colors_interpolate_between_pixels():
    rgb = sample_colors(_gradient_image(), np.array([[10.5, 20.25]]))
    np.testing.assert_allclose(rgb * 255, [[10.5, 20.25, 0]], atol=1e-3)


def test_sample_colors_clamp_outside_the_image():
    rgb = sample_colors(_gradient_image(), np.array([[-5.0, -5.0], [400.0, 10.0]]))
    np.testing.assert_allclose(rgb * 255, [[0, 0, 0], [255, 10, 0]], atol=1e-6)


def test_color_points_average_over_observations():
    a = np.zeros((10, 10, 3), np.uint8)
    b = np.full((10, 10, 3), 200, np.uint8)
    obs = Observations(view=[0, 1, 0], point=[0, 0, 1], xy=[[2, 2], [3, 3], [5, 5]])
    rgb = color_points([a, b], obs, n_points=2)
    np.testing.assert_allclose(rgb * 255, [[100, 100, 100], [0, 0, 0]], atol=1e-6)


# --- export ------------------------------------------------------------------------------


def test_ply_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    points, colors = rng.normal(size=(500, 3)), rng.random((500, 3))
    path = write_ply(tmp_path / "clouds" / "sparse.ply", points, colors)
    back_points, back_colors = read_ply(path)
    assert back_points.shape == (500, 3)
    np.testing.assert_allclose(back_points, points, atol=1e-6)
    np.testing.assert_allclose(back_colors, colors, atol=1e-6)


def test_ply_rejects_mismatched_colors(tmp_path):
    with pytest.raises(ValueError, match="colors"):
        write_ply(tmp_path / "x.ply", np.zeros((3, 3)), np.zeros((2, 3)))


def test_cameras_json_round_trip(tmp_path):
    K = np.array([[800.0, 0, 399.5], [0, 800.0, 299.5], [0, 0, 1]])
    R = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    cameras = [Camera(K, np.eye(3), np.zeros(3)), Camera(K, R, np.array([1.0, 0, 0]))]
    path = write_cameras_json(
        tmp_path / "cameras.json", cameras, ["a.png", "b.png"], [(800, 600), (800, 600)]
    )
    data = json.loads(path.read_text())
    assert [c["name"] for c in data["cameras"]] == ["a.png", "b.png"]
    np.testing.assert_allclose(data["cameras"][1]["centre"], cameras[1].centre)

    back, names, sizes = read_cameras_json(path)
    assert names == ["a.png", "b.png"] and sizes == [(800, 600), (800, 600)]
    for c, d in zip(cameras, back, strict=True):
        np.testing.assert_allclose(d.P, c.P)

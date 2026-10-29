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


def test_ply_keeps_float64_precision(tmp_path):
    # Regression: positions were cast to float32 (2e-5 error at |x| ~ 1e3).
    points = np.array([[1234.5678912345, -0.000123456789, 7.0], [1e6 + 0.123456789, 0, 0]])
    back, _ = read_ply(write_ply(tmp_path / "p.ply", points))
    assert back.dtype == np.float64
    np.testing.assert_array_equal(back, points)


def test_concurrent_cache_writes_to_one_path_are_safe(tmp_path):
    # Regression: every writer used `<path>.tmp`, so concurrent writers renamed each
    # other's temporary file away (FileNotFoundError).
    import threading

    from src.run import atomic_savez

    target, errors = tmp_path / "x.npz", []

    def work(k):
        for _ in range(30):
            try:
                atomic_savez(target, value=np.full(1000, k))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(np.load(target)["value"]) == 1000  # a complete file from one writer
    assert list(tmp_path.iterdir()) == [target]  # no temporary files left behind


def test_old_cameras_json_format_gives_a_clear_error(tmp_path):
    # gt_cameras.json used to be {name: {K, R, t}}; files from before the change must say
    # how to fix them instead of failing with KeyError: 'cameras'.
    import json

    path = tmp_path / "gt_cameras.json"
    path.write_text(json.dumps({"a.png": {"K": np.eye(3).tolist()}}))
    with pytest.raises(ValueError, match="download_middlebury"):
        read_cameras_json(path)

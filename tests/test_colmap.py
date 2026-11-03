"""COLMAP baseline runner (external tool): text-model parsing, and an end-to-end run on
rendered views with known cameras when ``colmap`` is installed."""

import cv2
import numpy as np
import pytest

from src.evaluation.metrics import camera_pose_errors
from src.reconstruction.colmap_runner import (
    colmap_executable,
    intrinsics_from_colmap,
    read_text_model,
    run_colmap,
)
from tests.render import render_box_scene, scene_cameras, view_name

CAMERAS = """# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 PINHOLE 640 480 800 810 320 240
2 SIMPLE_RADIAL 640 480 700 319.5 239.5 0.01
"""

IMAGES = """# Image list with two lines of data per image:
2 1 0 0 0 0.5 0 0 1 b.png
10 20 -1
1 0.7071067811865476 0 0.7071067811865476 0 1 2 3 2 a.png

"""

POINTS = """# 3D point list
7 1.0 2.0 3.0 255 0 10 0.5 1 0 2 0
9 -1.0 0.0 4.0 0 128 0 0.25 2 3
"""


@pytest.fixture
def model_dir(tmp_path):
    for name, text in (("cameras", CAMERAS), ("images", IMAGES), ("points3D", POINTS)):
        (tmp_path / f"{name}.txt").write_text(text)
    return tmp_path


def test_read_text_model(model_dir):
    model = read_text_model(model_dir)
    assert sorted(model.cameras) == ["a.png", "b.png"]
    a, b = model.cameras["a.png"], model.cameras["b.png"]
    # COLMAP's pixel centres are at +0.5: the principal point moves by half a pixel.
    np.testing.assert_allclose(b.K, [[800, 0, 319.5], [0, 810, 239.5], [0, 0, 1]])
    np.testing.assert_allclose(a.K, [[700, 0, 319], [0, 700, 239], [0, 0, 1]])
    np.testing.assert_allclose(b.R, np.eye(3))
    np.testing.assert_allclose(b.t, [0.5, 0, 0])
    np.testing.assert_allclose(a.R, cv2.Rodrigues(np.array([0, np.pi / 2, 0]))[0], atol=1e-12)
    np.testing.assert_allclose(a.t, [1, 2, 3])
    np.testing.assert_allclose(model.points, [[1, 2, 3], [-1, 0, 4]])
    np.testing.assert_array_equal(model.colors, [[255, 0, 10], [0, 128, 0]])
    np.testing.assert_allclose(model.errors, [0.5, 0.25])


def test_intrinsics_from_colmap_rejects_unknown_models():
    with pytest.raises(ValueError, match="FISHEYE"):
        intrinsics_from_colmap("FISHEYE", [1.0, 2.0, 3.0])


@pytest.mark.skipif(colmap_executable() is None, reason="colmap not installed")
def test_run_colmap_recovers_rendered_cameras(tmp_path):
    scene = render_box_scene(n_cameras=6, arc_deg=50.0)
    images = {view_name(k): im for k, im in enumerate(scene.images)}
    model = run_colmap(images, scene.K, tmp_path / "colmap", 8000, refine_intrinsics=False)
    truth = scene_cameras(scene)
    assert sorted(model.cameras) == sorted(truth)
    names = sorted(truth)
    errors = camera_pose_errors([model.cameras[n] for n in names], [truth[n] for n in names])
    assert np.max(errors.rotation_deg) < 0.2
    assert len(model.points) > 500

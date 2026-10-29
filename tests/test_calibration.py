import dataclasses

import numpy as np
import pytest

from src.config import CalibrationConfig, CameraConfig
from src.data.loader import ExifInfo
from src.geometry.calibration import (
    CalibrationError,
    Intrinsics,
    calibrate_from_checkerboard,
    fallback_intrinsics,
    intrinsics_from_exif,
    load_camera_file,
    resolve_intrinsics,
    save_camera_file,
)
from tests.images import checkerboard_pose, render_checkerboard

NO_EXIF = ExifInfo()


def test_fallback_intrinsics():
    intr = fallback_intrinsics(640, 480, factor=1.2)
    np.testing.assert_allclose(intr.K[0, 0], 768.0)
    np.testing.assert_allclose(intr.K[1, 1], 768.0)
    np.testing.assert_allclose(intr.K[:2, 2], [319.5, 239.5])
    np.testing.assert_array_equal(intr.dist, np.zeros(5))
    assert intr.source == "fallback"


def test_exif_35mm_equivalent_focal():
    # 35 mm equivalence is defined on the diagonal (43.27 mm for a 36x24 mm frame).
    intr = intrinsics_from_exif(ExifInfo(focal_35mm=26), 4000, 3000)
    expected = 26 * np.hypot(4000, 3000) / np.hypot(36, 24)
    np.testing.assert_allclose(intr.K[0, 0], expected)
    assert intr.source == "exif"


def test_exif_without_35mm_focal_gives_none():
    assert intrinsics_from_exif(ExifInfo(focal_mm=4.25), 4000, 3000) is None


def test_scaled_to_smaller_resolution():
    intr = Intrinsics(
        K=np.array([[3000.0, 0, 1999.5], [0, 3000.0, 1499.5], [0, 0, 1]]),
        dist=np.array([0.1, 0, 0, 0, 0]),
        width=4000,
        height=3000,
        source="file",
    )
    small = intr.scaled_to(2000, 1500)
    np.testing.assert_allclose(small.K[0, 0], 1500.0)
    np.testing.assert_allclose(small.K[:2, 2], [999.5, 749.5])
    np.testing.assert_array_equal(small.dist, intr.dist)  # distortion is resolution-free
    assert (small.width, small.height, small.source) == (2000, 1500, "file")


def test_scaled_to_rejects_different_aspect_ratio():
    intr = fallback_intrinsics(4000, 3000, 1.2)
    with pytest.raises(CalibrationError, match="aspect"):
        intr.scaled_to(3000, 4000)


def test_scaled_to_aspect_tolerance_is_configurable():
    intr = fallback_intrinsics(4000, 3000, 1.2)
    with pytest.raises(CalibrationError, match="aspect"):
        intr.scaled_to(2000, 1530)  # y scale 2 % off the x scale
    assert intr.scaled_to(2000, 1530, rtol=0.03).height == 1530
    cfg = CameraConfig(aspect_tolerance=0.03)
    assert resolve_intrinsics(2000, 1530, NO_EXIF, cfg, intr).height == 1530


def test_resolve_priority():
    cfg = CameraConfig()
    calibrated = dataclasses.replace(fallback_intrinsics(640, 480, 2.0), source="file")
    exif = ExifInfo(focal_35mm=26)

    assert resolve_intrinsics(640, 480, exif, cfg, calibrated).source == "file"
    assert resolve_intrinsics(640, 480, exif, cfg, None).source == "exif"
    assert resolve_intrinsics(640, 480, NO_EXIF, cfg, None).source == "fallback"
    no_exif_cfg = CameraConfig(use_exif=False)
    assert resolve_intrinsics(640, 480, exif, no_exif_cfg, None).source == "fallback"


def test_camera_file_round_trip(tmp_path):
    intr = Intrinsics(
        K=np.array([[1000.0, 0, 640.2], [0, 1001.0, 360.7], [0, 0, 1]]),
        dist=np.array([-0.1, 0.02, 0.001, -0.002, 0.0]),
        width=1280,
        height=720,
        source="checkerboard",
    )
    path = save_camera_file(intr, tmp_path / "camera.yaml")
    loaded = load_camera_file(path)
    np.testing.assert_allclose(loaded.K, intr.K)
    np.testing.assert_allclose(loaded.dist, intr.dist)
    assert (loaded.width, loaded.height, loaded.source) == (1280, 720, "file")


def test_camera_file_validation(tmp_path):
    path = tmp_path / "camera.yaml"
    path.write_text("width: 640\nheight: 480\nfx: 500\n")
    with pytest.raises(CalibrationError, match="missing"):
        load_camera_file(path)
    path.write_text("width: 640\nheight: 480\nfx: 500\nfy: 500\ncx: 320\ncy: 240\nfoo: 1\n")
    with pytest.raises(CalibrationError, match="unknown"):
        load_camera_file(path)


def test_checkerboard_calibration_recovers_focal_length():
    K_true = np.array([[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1]])
    tilts = [(0, 0), (20, 0), (-20, 0), (0, 20), (0, -20), (15, 15), (-15, 10)]
    images = [
        render_checkerboard(K_true, *checkerboard_pose(ax, ay, 15.0), image_size=(640, 480))
        for ax, ay in tilts
    ]
    intr, rms = calibrate_from_checkerboard(images, pattern_size=(9, 6), square_size=1.0)
    assert rms < 0.5
    np.testing.assert_allclose(intr.K[0, 0], 500.0, rtol=0.01)
    np.testing.assert_allclose(intr.K[1, 1], 500.0, rtol=0.01)
    np.testing.assert_allclose(intr.K[:2, 2], [319.5, 239.5], atol=5.0)
    assert (intr.width, intr.height, intr.source) == (640, 480, "checkerboard")


def test_checkerboard_calibration_needs_enough_views():
    blank = [np.full((480, 640), 255, np.uint8)] * 5
    with pytest.raises(CalibrationError, match="checkerboard"):
        calibrate_from_checkerboard(blank, pattern_size=(9, 6))


def test_checkerboard_min_views_comes_from_config():
    K_true = np.array([[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1]])
    images = [
        render_checkerboard(K_true, *checkerboard_pose(ax, 0, 15.0), image_size=(640, 480))
        for ax in (0, 20, -20)
    ]
    with pytest.raises(CalibrationError, match="need at least 4"):
        calibrate_from_checkerboard(images, (9, 6), config=CalibrationConfig(min_views=4))

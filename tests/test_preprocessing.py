import numpy as np
import pytest

from src.data.preprocessing import resize_max_side, scale_intrinsics, to_gray, undistort
from tests.images import make_texture

K = np.array([[600.0, 0.0, 319.5], [0.0, 610.0, 239.5], [0.0, 0.0, 1.0]])


def test_scale_intrinsics_halving():
    K2 = scale_intrinsics(K, 0.5, 0.5)
    # Focal lengths scale linearly; the principal point uses the pixel-centre convention
    # (integer coordinates at pixel centres): c' = (c + 0.5) * s - 0.5.
    np.testing.assert_allclose(K2[0, 0], 300.0)
    np.testing.assert_allclose(K2[1, 1], 305.0)
    np.testing.assert_allclose(K2[:2, 2], [159.5, 119.5])
    np.testing.assert_allclose(K2[2], [0, 0, 1])


def test_scale_intrinsics_does_not_mutate_input():
    K_copy = K.copy()
    scale_intrinsics(K, 0.25, 0.25)
    np.testing.assert_array_equal(K, K_copy)


def test_resize_caps_longest_side_and_updates_k():
    image = make_texture(0, 640, 480)
    resized, K2, scale = resize_max_side(image, K, 320)
    assert resized.shape == (240, 320, 3)
    assert scale == (0.5, 0.5)
    np.testing.assert_allclose(K2, scale_intrinsics(K, 0.5, 0.5))


def test_resize_handles_portrait_images():
    image = make_texture(0, 480, 640)
    resized, _, scale = resize_max_side(image, K, 320)
    assert resized.shape == (320, 240, 3)
    assert scale == (0.5, 0.5)


@pytest.mark.parametrize("max_side", [None, 640, 2000])
def test_resize_never_upscales(max_side):
    image = make_texture(0, 640, 480)
    resized, K2, scale = resize_max_side(image, K, max_side)
    assert resized.shape == image.shape
    assert scale == (1.0, 1.0)
    np.testing.assert_array_equal(K2, K)


def test_resized_pixels_follow_the_scaled_intrinsics():
    # A 2x2 white block centred on pixel (100.5, 60.5) must land where the scaled
    # projection model predicts: (100.5 + 0.5) * 0.5 - 0.5 = 50, (60.5 + 0.5) * 0.5 - 0.5 = 30.
    image = np.zeros((480, 640), np.uint8)
    image[60:62, 100:102] = 255
    resized, _, _ = resize_max_side(image, K, 320)
    y, x = np.unravel_index(np.argmax(resized), resized.shape)
    assert (x, y) == (50, 30)


def test_undistort_with_zero_distortion_is_identity():
    image = make_texture(0, 160, 120)
    out = undistort(image, K, np.zeros(5))
    np.testing.assert_array_equal(out, image)


def test_undistort_moves_pixels_when_distorted():
    image = make_texture(0, 640, 480)
    out = undistort(image, K, np.array([-0.3, 0.1, 0.0, 0.0, 0.0]))
    assert out.shape == image.shape
    assert not np.array_equal(out, image)


def test_to_gray():
    color = make_texture(0, 64, 48)
    gray = to_gray(color)
    assert gray.shape == (48, 64)
    assert gray.dtype == np.uint8
    assert to_gray(gray) is gray

"""Synthetic image files for dataset tests: textures, blur, EXIF, checkerboards."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image


def make_texture(seed: int, width: int = 640, height: int = 480, block: int = 8) -> np.ndarray:
    """BGR image of random coloured blocks: plenty of corners, deterministic per seed."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(max(height // block, 1), max(width // block, 1), 3))
    return cv2.resize(small.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def blur(image: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma)


def write_image(
    path: Path,
    image: np.ndarray,
    focal_35mm: int | None = None,
    focal_mm: float | None = None,
    orientation: int | None = None,
) -> Path:
    """Write a BGR image; EXIF tags are only written when any are given (JPEG via Pillow)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if focal_35mm is None and focal_mm is None and orientation is None:
        ok, buf = cv2.imencode(path.suffix, image)
        assert ok, path
        buf.tofile(path)  # tofile handles non-ASCII paths, unlike cv2.imwrite
        return path

    exif = Image.Exif()
    if orientation is not None:
        exif[ExifTags.Base.Orientation] = orientation
    sub = exif.get_ifd(ExifTags.IFD.Exif)
    if focal_35mm is not None:
        sub[ExifTags.Base.FocalLengthIn35mmFilm] = focal_35mm
    if focal_mm is not None:
        sub[ExifTags.Base.FocalLength] = focal_mm
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path, exif=exif, quality=95)
    return path


def write_textured_images(
    directory: Path,
    n: int,
    size: tuple[int, int] = (640, 480),
    seed: int = 0,
    ext: str = ".jpg",
) -> list[Path]:
    """Write ``n`` distinct textured images named image_001, image_002, ..."""
    width, height = size
    return [
        write_image(directory / f"image_{i + 1:03d}{ext}", make_texture(seed + i, width, height))
        for i in range(n)
    ]


def render_checkerboard(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: tuple[int, int],
    pattern: tuple[int, int] = (9, 6),
    square_px: int = 20,
) -> np.ndarray:
    """Render a planar checkerboard seen by camera (K, R, t).

    The board lies in the world plane Z = 0 with one square per world unit; its
    first square's top-left corner is the origin. ``pattern`` counts inner corners.
    """
    cols, rows = pattern[0] + 1, pattern[1] + 1
    margin = square_px  # white border so the detector finds the outer corners
    tex = np.full((rows * square_px + 2 * margin, cols * square_px + 2 * margin), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y0, x0 = margin + r * square_px, margin + c * square_px
                tex[y0 : y0 + square_px, x0 : x0 + square_px] = 0

    board_to_texture = np.array([[square_px, 0, margin], [0, square_px, margin], [0, 0, 1.0]])
    board_to_image = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H = board_to_image @ np.linalg.inv(board_to_texture)
    return cv2.warpPerspective(tex, H, image_size, flags=cv2.INTER_LINEAR, borderValue=128)


def checkerboard_pose(
    tilt_x_deg: float, tilt_y_deg: float, distance: float, pattern: tuple[int, int] = (9, 6)
) -> tuple[np.ndarray, np.ndarray]:
    """Pose that tilts the board about its centre and places it ``distance`` units ahead."""
    ax, ay = np.deg2rad([tilt_x_deg, tilt_y_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    R = Rx @ Ry
    board_centre = np.array([(pattern[0] + 1) / 2, (pattern[1] + 1) / 2, 0.0])
    t = -R @ board_centre + np.array([0.0, 0.0, distance])
    return R, t


def make_natural_texture(seed: int, width: int = 640, height: int = 480) -> np.ndarray:
    """BGR multi-scale noise texture: features at many scales, like real surfaces.

    Unlike ``make_texture`` (uniform 8 px blocks), keypoints here have distinctive
    neighbourhoods at several scales, which matching tests need.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((height, width, 3))
    for k in range(1, 7):
        s = 2**k
        small = rng.random((height // s + 2, width // s + 2, 3))
        up = cv2.resize(small, ((width // s + 2) * s, (height // s + 2) * s), cv2.INTER_CUBIC)
        acc += up[:height, :width] / k
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    return (acc * 255).astype(np.uint8)


def make_view_sequence(
    n: int,
    size: tuple[int, int] = (640, 480),
    seed: int = 0,
    step: float = 40.0,
    rotation_deg: float = 3.0,
    zoom: float = 0.02,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Views of one planar textured scene from a camera that pans, rotates and zooms.

    Returns (images, homographies) where ``homographies[k]`` maps scene-texture pixels
    to view-k pixels. The ground-truth mapping from view i to view j is
    ``homographies[j] @ inv(homographies[i])``.
    """
    width, height = size
    margin = int(step * n + max(width, height))
    scene = make_natural_texture(seed, width + 2 * margin, height + 2 * margin)
    centre = np.array([(width - 1) / 2, (height - 1) / 2])

    images, homographies = [], []
    for k in range(n):
        shift = np.array([[1, 0, -margin - k * step], [0, 1, -margin], [0, 0, 1.0]])
        A = cv2.getRotationMatrix2D(tuple(centre), rotation_deg * k, 1.0 + zoom * k)
        H = np.vstack([A, [0, 0, 1]]) @ shift
        H[2, :2] += np.array([2e-5, -1e-5]) * k  # mild perspective
        images.append(cv2.warpPerspective(scene, H, size, flags=cv2.INTER_LINEAR))
        homographies.append(H)
    return images, homographies


def relative_homography(homographies: list[np.ndarray], i: int, j: int) -> np.ndarray:
    return homographies[j] @ np.linalg.inv(homographies[i])

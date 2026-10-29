"""Synthetic multi-view scenes with exact ground truth.

Geometry modules are tested against these scenes before touching real photos.

Conventions (see docs/architecture.md):
    world -> camera:  x_c = R @ X + t
    camera centre:    C = -R.T @ t
    pixels:           origin at the top-left, x to the right, y down
    arrays:           points (N, 3) / (N, 2), float64
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticScene:
    K: np.ndarray  # (3, 3) shared intrinsics
    rotations: np.ndarray  # (M, 3, 3) world -> camera
    translations: np.ndarray  # (M, 3)
    points3d: np.ndarray  # (N, 3) world points
    observations: np.ndarray  # (M, N, 2) pixel coordinates, noise included
    visible: np.ndarray  # (M, N) bool: in front of the camera and inside the image
    image_size: tuple[int, int]  # (width, height)

    @property
    def n_cameras(self) -> int:
        return len(self.rotations)

    @property
    def n_points(self) -> int:
        return len(self.points3d)

    def camera_center(self, i: int) -> np.ndarray:
        return camera_center(self.rotations[i], self.translations[i])

    def projection_matrix(self, i: int) -> np.ndarray:
        """P = K [R | t], shape (3, 4)."""
        return self.K @ np.hstack([self.rotations[i], self.translations[i][:, None]])

    def pair(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Pixel correspondences of points visible in both camera i and camera j."""
        both = self.visible[i] & self.visible[j]
        return self.observations[i, both], self.observations[j, both]

    def relative_pose(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """(R, t) mapping camera-i coordinates to camera-j coordinates."""
        R_i, t_i = self.rotations[i], self.translations[i]
        R_j, t_j = self.rotations[j], self.translations[j]
        R = R_j @ R_i.T
        return R, t_j - R @ t_i

    def essential(self, i: int, j: int) -> np.ndarray:
        """True essential matrix [t]x R of the pair, with x_j^T E x_i = 0."""
        R, t = self.relative_pose(i, j)
        t_x = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
        return t_x @ R


def intrinsics(focal: float, width: int, height: int) -> np.ndarray:
    # Pixel-centre convention (docs/architecture.md): the image centre is ((w-1)/2, (h-1)/2).
    return np.array(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]]
    )


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t


def look_at(center: np.ndarray, target: np.ndarray, up: np.ndarray | None = None):
    """World -> camera (R, t) for a camera at ``center`` whose +z axis points at ``target``."""
    up = np.array([0.0, -1.0, 0.0]) if up is None else np.asarray(up, dtype=float)
    z = target - center
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)  # camera x points right when y points down
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ center


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray):
    """Project world points X (N, 3) to pixels.

    Returns (pixels (N, 2), depths (N,)). Depth is the camera-frame z coordinate;
    points with depth <= 0 are behind the camera and their pixels are meaningless.
    """
    X_cam = X @ R.T + t
    depths = X_cam[:, 2]
    x_hom = X_cam @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = x_hom[:, :2] / x_hom[:, 2:3]
    return pixels, depths


def make_scene(
    n_points: int = 200,
    n_cameras: int = 5,
    noise_px: float = 0.0,
    seed: int = 0,
    image_size: tuple[int, int] = (640, 480),
    focal: float = 600.0,
    radius: float = 6.0,
    arc_deg: float = 60.0,
    elevation: float = 1.0,
    planar: bool = False,
) -> SyntheticScene:
    """Random points in a unit cube at the origin, cameras on an arc looking at it.

    Cameras sit at distance ``radius`` from the origin, spread over ``arc_deg``
    degrees around the y axis and raised by ``elevation`` (negative y is up).
    With ``planar`` all points lie in the plane z = 0 (a degenerate scene for F).
    """
    rng = np.random.default_rng(seed)
    width, height = image_size
    K = intrinsics(focal, width, height)

    points3d = rng.uniform(-1.0, 1.0, size=(n_points, 3))
    if planar:
        points3d[:, 2] = 0.0

    angles = np.deg2rad(np.linspace(-arc_deg / 2, arc_deg / 2, n_cameras))
    rotations, translations = [], []
    for a in angles:
        center = np.array([radius * np.sin(a), -elevation, -radius * np.cos(a)])
        R, t = look_at(center, np.zeros(3))
        rotations.append(R)
        translations.append(t)
    rotations = np.stack(rotations)
    translations = np.stack(translations)

    observations = np.empty((n_cameras, n_points, 2))
    visible = np.empty((n_cameras, n_points), dtype=bool)
    for i in range(n_cameras):
        pixels, depths = project(K, rotations[i], translations[i], points3d)
        pixels = pixels + rng.normal(0.0, noise_px, size=pixels.shape) if noise_px else pixels
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        observations[i] = pixels
        visible[i] = (depths > 0) & inside

    return SyntheticScene(
        K=K,
        rotations=rotations,
        translations=translations,
        points3d=points3d,
        observations=observations,
        visible=visible,
        image_size=image_size,
    )

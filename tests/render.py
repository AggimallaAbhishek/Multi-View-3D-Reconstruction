"""Render a textured 3-D scene (a box on a ground plane) from known cameras.

Gives image-based tests real parallax with exact ground truth. Each planar face is a
textured quad warped into the image by its homography K [R | t] [U V P0]; faces
that point away from the camera are skipped (back-face culling). The box is convex
and drawn after the ground, so this painter's order is correct for cameras above
the ground plane.

World frame matches tests/synthetic.py: y points down, the ground is the plane y = 1
and the box spans [-1, 1] in x, y and z, standing on the ground.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tests.images import make_natural_texture
from tests.synthetic import intrinsics, look_at


@dataclass(frozen=True)
class Face:
    origin: np.ndarray  # P0, a corner of the quad
    u: np.ndarray  # edge vector along the texture's x axis
    v: np.ndarray  # edge vector along the texture's y axis
    texture: np.ndarray  # BGR uint8

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.u, self.v)


@dataclass(frozen=True)
class RenderedScene:
    K: np.ndarray
    rotations: np.ndarray  # (M, 3, 3) world -> camera
    translations: np.ndarray  # (M, 3)
    images: list[np.ndarray]
    image_size: tuple[int, int]

    def relative_pose(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        R = self.rotations[j] @ self.rotations[i].T
        return R, self.translations[j] - R @ self.translations[i]


def box_scene_faces(seed: int = 0) -> list[Face]:
    """Ground plane first, then the five visible-from-above faces of the box."""
    rng_seeds = iter(range(seed * 10, seed * 10 + 10))
    e = np.eye(3)
    # Outward normal = u x v (e0 x e1 = e2, e1 x e2 = e0, e2 x e0 = e1). y points down,
    # so "up" is -y.
    faces = [
        # Ground y = 1, normal up (-y): e0 x e2 = -e1.
        Face(np.array([-6.0, 1.0, -6.0]), 12 * e[0], 12 * e[2],
             make_natural_texture(next(rng_seeds), 1200, 1200)),
        # Top y = -1, normal up (-y).
        Face(np.array([-1.0, -1.0, -1.0]), 2 * e[0], 2 * e[2],
             make_natural_texture(next(rng_seeds), 400, 400)),
        # Front z = -1 (towards the default cameras), normal -z: (-e0) x e1 = -e2.
        Face(np.array([1.0, -1.0, -1.0]), -2 * e[0], 2 * e[1],
             make_natural_texture(next(rng_seeds), 400, 400)),
        # Back z = 1, normal +z: e0 x e1 = e2.
        Face(np.array([-1.0, -1.0, 1.0]), 2 * e[0], 2 * e[1],
             make_natural_texture(next(rng_seeds), 400, 400)),
        # Left x = -1, normal -x: e2 x e1 = -e0.
        Face(np.array([-1.0, -1.0, -1.0]), 2 * e[2], 2 * e[1],
             make_natural_texture(next(rng_seeds), 400, 400)),
        # Right x = 1, normal +x: (-e2) x e1 = e0.
        Face(np.array([1.0, -1.0, 1.0]), -2 * e[2], 2 * e[1],
             make_natural_texture(next(rng_seeds), 400, 400)),
    ]  # fmt: skip
    return faces


def render(
    faces: list[Face], K: np.ndarray, R: np.ndarray, t: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    width, height = size
    image = np.full((height, width, 3), 40, np.uint8)
    centre = -R.T @ t
    P = K @ np.hstack([R, t[:, None]])
    for face in faces:
        if np.dot(face.normal, centre - face.origin) <= 0:
            continue  # back-facing
        th, tw = face.texture.shape[:2]
        # Texture pixel (a, b) -> world P0 + (a + 0.5)/tw * u + (b + 0.5)/th * v.
        texture_to_world = np.column_stack(
            [face.u / tw, face.v / th, face.origin + 0.5 * face.u / tw + 0.5 * face.v / th]
        )
        H = P[:, :3] @ texture_to_world + np.outer(P[:, 3], [0.0, 0.0, 1.0])
        warped = cv2.warpPerspective(face.texture, H, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), H, size) > 127
        image[mask] = warped[mask]
    return image


def render_box_scene(
    n_cameras: int = 6,
    size: tuple[int, int] = (800, 600),
    focal: float = 800.0,
    radius: float = 5.5,
    arc_deg: float = 60.0,
    elevation: float = 3.0,
    seed: int = 0,
) -> RenderedScene:
    """Cameras on an arc around the box, raised by ``elevation``, looking at its centre."""
    width, height = size
    K = intrinsics(focal, width, height)
    faces = box_scene_faces(seed)
    rotations, translations, images = [], [], []
    for a in np.deg2rad(np.linspace(-arc_deg / 2, arc_deg / 2, n_cameras)):
        centre = np.array([radius * np.sin(a), -elevation, -radius * np.cos(a)])
        R, t = look_at(centre, np.zeros(3))
        rotations.append(R)
        translations.append(t)
        images.append(render(faces, K, R, t, size))
    return RenderedScene(K, np.stack(rotations), np.stack(translations), images, size)

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
from pathlib import Path

import cv2
import numpy as np

from src.geometry.camera import Camera
from tests.images import make_natural_texture, write_image
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


def view_name(k: int) -> str:
    """File name ``write_scene`` gives view ``k`` (0-based)."""
    return f"view_{k + 1:02d}.png"


def scene_cameras(scene: RenderedScene) -> dict[str, Camera]:
    """Ground-truth cameras keyed by the file names ``write_scene`` uses."""
    return {
        view_name(k): Camera(scene.K, R, t)
        for k, (R, t) in enumerate(zip(scene.rotations, scene.translations, strict=True))
    }


def write_scene(scene: RenderedScene, root: Path) -> tuple[Path, Path]:
    """Write the views as ``root/scene/view_01.png ...`` and the intrinsics as
    ``root/camera.yaml`` (not next to the images, so tests choose whether it is used).
    Returns (image folder, camera file)."""
    for k, im in enumerate(scene.images):
        write_image(root / "scene" / view_name(k), im)
    K, (width, height) = scene.K, scene.image_size
    camera = root / "camera.yaml"
    camera.write_text(
        f"width: {width}\nheight: {height}\n"
        f"fx: {K[0, 0]}\nfy: {K[1, 1]}\ncx: {K[0, 2]}\ncy: {K[1, 2]}\n"
    )
    return root / "scene", camera


def render_depth(
    faces: list[Face], K: np.ndarray, R: np.ndarray, t: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Ground-truth depth (camera z) per pixel centre, in ``render``'s painter's order;
    NaN where no face is hit. (Face edges can differ from ``render`` by a pixel.)"""
    width, height = size
    u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    rays = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3) @ np.linalg.inv(K).T
    d = rays @ R  # world directions; R @ d = K^-1 p, whose z is 1, so depth = lambda
    centre = -R.T @ t
    depth = np.full(len(d), np.nan)
    for face in faces:
        if np.dot(face.normal, centre - face.origin) <= 0:
            continue
        # centre + lam d = origin + a u + b v  ->  [d, -u, -v] (lam, a, b) = origin - centre
        A = np.stack([d, np.broadcast_to(-face.u, d.shape), np.broadcast_to(-face.v, d.shape)], -1)
        with np.errstate(all="ignore"):
            rhs = np.broadcast_to(face.origin - centre, d.shape)[..., None]
            lam, a, b = np.linalg.solve(A, rhs)[..., 0].T
        hit = (lam > 0) & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
        depth[hit] = lam[hit]
    return depth.reshape(height, width)


def box_scene_distance(X: np.ndarray) -> np.ndarray:
    """Distance of points (N, 3) to the surface of ``box_scene_faces``' scene (ground plane
    y = 1 or the box [-1, 1]^3)."""
    q = np.abs(X) - 1.0
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(q.max(axis=1), 0)
    return np.minimum(np.abs(X[:, 1] - 1.0), np.abs(outside + inside))


def distort_image(image: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """What a lens with OpenCV distortion ``dist`` would have recorded: every output
    (distorted) pixel samples the undistorted image where its ray lands."""
    h, w = image.shape[:2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    grid = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
    source = cv2.undistortPoints(grid, K, np.asarray(dist, float), P=K).reshape(h, w, 2)
    return cv2.remap(image, source[..., 0], source[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))  # fmt: skip

"""3-D views of a reconstruction: interactive (Open3D ``draw``) and offscreen renders.

Uses Open3D's tensor geometry and the Filament renderer (``visualization.draw``,
``rendering.OffscreenRenderer``), the current Open3D APIs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import open3d as o3d

from src.geometry.camera import Camera
from src.pointcloud.export import to_open3d

_FRUSTUM_COLOURS = [(0.12, 0.47, 0.71), (0.84, 0.15, 0.16), (0.17, 0.63, 0.17)]


class RenderUnavailable(RuntimeError):
    """Offscreen rendering is not possible here (no GPU context, headless machine)."""


def frustum_scale(points: np.ndarray, cameras: Sequence[Camera]) -> float:
    """Frustum depth: 8 % of the extent of points and camera centres together."""
    centres = np.array([c.centre for c in cameras], dtype=float).reshape(-1, 3)
    everything = np.vstack([np.asarray(points, dtype=float).reshape(-1, 3), centres])
    extent = np.linalg.norm(np.ptp(everything, axis=0)) if len(everything) else 0.0
    return 0.08 * float(extent or 1.0)


def scene_geometries(
    points: np.ndarray,
    colors: np.ndarray | None,
    cameras: Sequence[Camera],
    sizes: Sequence[tuple[int, int]],
) -> list[tuple[str, o3d.t.geometry.Geometry]]:
    """The point cloud (omitted when empty) plus one camera frustum (LineSet) per camera."""
    geometries: list[tuple[str, o3d.t.geometry.Geometry]] = []
    if len(points):  # Filament aborts the process on an empty point cloud
        geometries.append(("points", to_open3d(points, colors)))
    scale = frustum_scale(points, cameras)
    for k, (cam, (w, h)) in enumerate(zip(cameras, sizes, strict=True)):
        extrinsic = np.eye(4)
        extrinsic[:3, :3], extrinsic[:3, 3] = cam.R, cam.t
        frustum = o3d.t.geometry.LineSet.create_camera_visualization(
            int(w), int(h), o3d.core.Tensor(cam.K), o3d.core.Tensor(extrinsic), scale,
            o3d.core.Tensor(np.array(_FRUSTUM_COLOURS[k % 3], dtype=np.float32)),
        )  # fmt: skip
        geometries.append((f"camera_{k}", frustum))
    return geometries


def show(points, colors, cameras, sizes, title: str = "Reconstruction") -> None:
    """Open an interactive Open3D window (blocks until it is closed)."""
    geometries = [
        {"name": name, "geometry": g}
        for name, g in scene_geometries(points, colors, cameras, sizes)
    ]
    o3d.visualization.draw(geometries, title=title, point_size=3, show_skybox=False)


def render_scene(
    points: np.ndarray,
    colors: np.ndarray | None,
    cameras: Sequence[Camera],
    sizes: Sequence[tuple[int, int]],
    path: str | Path,
    width: int = 1280,
    height: int = 960,
    focus: str = "cameras",
) -> Path:
    """Render the scene offscreen and save it as an image.

    ``focus="cameras"``: an overview from behind and above the cameras, showing the
    cameras and the structure. ``focus="points"``: a close-up framed on the points,
    seen obliquely (from above and the side of the mean viewing direction) so their
    depth is visible.
    """
    import open3d.visualization.rendering as rendering

    try:
        renderer = rendering.OffscreenRenderer(width, height)
    except Exception as exc:  # noqa: BLE001 - any backend failure means "cannot render here"
        raise RenderUnavailable(str(exc)) from exc

    for name, geometry in scene_geometries(points, colors, cameras, sizes):
        material = rendering.MaterialRecord()
        if name == "points":
            material.shader = "defaultUnlit"
            material.point_size = 4.0 if focus == "cameras" else 7.0
        else:
            material.shader, material.line_width = "unlitLine", 2.0
        renderer.scene.add_geometry(name, geometry, material)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.view.set_post_processing(False)  # no tone mapping: true colours, white

    fov = 55.0
    if len(points):
        target = np.median(points, axis=0)
        # Robust extent of the points: diagonal of their 2nd-98th percentile box.
        spread = np.linalg.norm(np.ptp(np.percentile(points, [2, 98], axis=0), axis=0))
    else:
        target, spread = np.zeros(3), 0.0
    centres = np.array([c.centre for c in cameras])
    up = -np.mean([c.R[1] for c in cameras], axis=0)  # camera y points down in the image
    up /= np.linalg.norm(up)
    back = centres.mean(axis=0) - target
    distance = np.linalg.norm(back) or 1.0
    back /= distance
    if focus == "cameras":
        eye = target + 1.6 * distance * back + 0.6 * distance * up
    else:
        # Fit the points' spread into the field of view (no spread: stay at the cameras'
        # distance) and swing the viewpoint 25 deg sideways and 20 deg up from them.
        fit = 0.6 * spread / np.tan(np.radians(fov / 2)) or distance
        side = np.cross(up, back)
        direction = np.cos(np.radians(25)) * back + np.sin(np.radians(25)) * side
        direction = np.cos(np.radians(20)) * direction + np.sin(np.radians(20)) * up
        eye = target + fit * direction / np.linalg.norm(direction)
    renderer.setup_camera(
        fov, target.astype(np.float32), eye.astype(np.float32), up.astype(np.float32)
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_image(str(path), renderer.render_to_image()):
        raise OSError(f"could not write {path}")
    return path

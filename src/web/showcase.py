"""The intro page's showcase: one finished run as small files committed with the page.

``make_showcase(run_dir, images_dir, out_dir, config)`` writes

- ``cloud.ply`` and ``cameras.json``: the run's filtered cloud and cameras, for the live
  3-D hero;
- ``showcase.json``: the run's report (numbers shown on the page) and the frames with
  the numbers measured on them;
- five frames for "how it works", ``web.showcase_frame_px`` wide (16:10), JPEG: the
  photos, one photo's features, one verified pair's matches, the recovered cameras and
  the cloud. The features and matches are recomputed with ``config`` (the script passes
  the run's own ``config.yaml``); the last two are drawn from the reconstruction.

Everything shown is measured, never illustrative (PRODUCT.md, "real data or nothing").
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import Config
from src.data.preprocessing import resize_max_side
from src.geometry.verification import inlier_matches
from src.pipeline import run_pipeline
from src.run import read_json, write_json
from src.web.report import REPORT_NAME

FRAMES = ("photos", "features", "matches", "cameras", "cloud")
MATCH_LINES = 150  # lines drawn in the matches frame (a drawing choice, not a measurement)
# Drawing choices for the frames (layout, not measurement):
DIM = 0.45  # brightness of photos under keypoints or match lines, so the accent reads
FAINT_CLOUD = 0.55  # brightness of the cloud behind the cameras
CARD = (0.36, 0.62)  # a photo in the fan, as fractions of the frame (width, height)
VIEW_ELEVATION_DEG = 28  # the rendered views look down at the object by this much
FILL = 0.44  # half the frame's extent the rendered object fills
TICK = 0.12  # length of a camera's direction tick, as a fraction of its distance
BACKGROUND = (12, 10, 7)  # BGR of the page's viewer background (#070a0c)
ACCENT = (181, 194, 60)  # BGR of the accent (#3cc2b5)
MUTED = (169, 163, 143)  # BGR of --muted (#8fa3a9)


def make_showcase(
    run_dir: str | Path, images_dir: str | Path, out_dir: str | Path, config: Config, name: str = ""
) -> Path:
    """Write the showcase of the finished web run ``run_dir`` (photos in ``images_dir``)."""
    run_dir, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = read_json(run_dir / REPORT_NAME)
    shutil.copyfile(run_dir / "point_clouds" / "filtered.ply", out / "cloud.ply")
    cameras = read_json(run_dir / "camera_poses" / "cameras.json")
    write_json(out / "cameras.json", cameras)

    width = config.web.showcase_frame_px
    size = (width, round(width * 5 / 8))
    result = run_pipeline(images_dir, config, until="verify")
    points, colours = _read_cloud(out / "cloud.ply")
    centres = np.array([c["centre"] for c in cameras["cameras"]], dtype=np.float64)

    _save(out, "photos", _photos(result, size), config)
    _save(out, "cameras", _render(points, colours, centres, size, cameras_only=True), config)
    _save(out, "cloud", _render(points, colours, centres, size), config)
    frames = [
        {"file": "photos.jpg", "photos": len(result.images)},
        {"file": "features.jpg", **_features_frame(out, result, size, config)},
        {"file": "matches.jpg", **_matches_frame(out, result, size, config)},
        {"file": "cameras.jpg", "cameras": len(centres)},
        {"file": "cloud.jpg", "points": len(points)},
    ]
    photos = _save_photos(out, result, config)
    write_json(
        out / "showcase.json", {"name": name, "report": report, "frames": frames, "photos": photos}
    )
    return out


def _save(out: Path, name: str, image: np.ndarray, config: Config) -> None:
    cv2.imwrite(
        str(out / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, config.web.showcase_quality]
    )


def _save_photos(out: Path, result, config: Config) -> list[dict[str, str]]:
    """Up to ``web.showcase_photos`` of the photos themselves, evenly spread over the set,
    at twice the thumbnail size."""
    n, wanted = len(result.images), config.web.showcase_photos
    if n <= wanted:
        picks = list(range(n))
    else:
        picks = [round(k * (n - 1) / (wanted - 1)) for k in range(wanted)]
    photos = []
    for k, index in enumerate(picks, start=1):
        im = result.images[index]
        small, _, _ = resize_max_side(im.image, im.K, 2 * config.web.thumbnail_px)
        file = f"photo-{k:02d}.jpg"
        cv2.imwrite(str(out / file), small, [cv2.IMWRITE_JPEG_QUALITY, config.web.showcase_quality])
        photos.append({"file": file, "name": im.name})
    return photos


def _dim(image: np.ndarray, factor: float = DIM) -> np.ndarray:
    return (image * factor).astype(np.uint8)


def _canvas(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    canvas = np.empty((height, width, 3), np.uint8)
    canvas[:] = BACKGROUND
    return canvas


def _fit(image: np.ndarray, box: tuple[int, int]) -> np.ndarray:
    """``image`` scaled to fit inside ``box`` (width, height), aspect kept."""
    scale = min(box[0] / image.shape[1], box[1] / image.shape[0])
    return cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _paste(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
    h, w = image.shape[:2]
    canvas[y : y + h, x : x + w] = image


def _photos(result, size: tuple[int, int]) -> np.ndarray:
    """Up to five photos fanned out, like prints on a table."""
    canvas = _canvas(size)
    width, height = size
    shown = result.images[:5]
    card = (int(width * CARD[0]), int(height * CARD[1]))
    step = (width - card[0] - width * 0.1) / max(1, len(shown) - 1)
    for k, im in enumerate(shown):
        photo = _fit(im.image, card)
        cv2.rectangle(photo, (0, 0), (photo.shape[1] - 1, photo.shape[0] - 1), MUTED, 2)
        x = int(width * 0.05 + k * step)
        y = int((height - photo.shape[0]) / 2 + (k - (len(shown) - 1) / 2) * height * 0.04)
        _paste(canvas, photo, x, max(0, min(height - photo.shape[0], y)))
    return canvas


def _features_frame(out: Path, result, size: tuple[int, int], config: Config) -> dict[str, Any]:
    """The first photo, darkened, with every detected keypoint as a circle of its scale."""
    im, features = result.images[0], result.features[0]
    photo = _fit(im.image, (size[0] - 40, size[1] - 40))
    scale = photo.shape[1] / im.width
    photo = _dim(photo)
    for (x, y), s in zip(features.points, features.sizes, strict=True):
        cv2.circle(
            photo,
            (round(x * scale), round(y * scale)),
            max(2, round(s * scale / 2)),
            ACCENT,
            1,
            cv2.LINE_AA,
        )
    canvas = _canvas(size)
    _paste(canvas, photo, (size[0] - photo.shape[1]) // 2, (size[1] - photo.shape[0]) // 2)
    _save(out, "features", canvas, config)
    return {"photo": im.name, "keypoints": len(features)}


def _matches_frame(out: Path, result, size: tuple[int, int], config: Config) -> dict[str, Any]:
    """The verified pair of neighbouring photos with the most inliers, side by side, joined
    by their inlier matches."""
    candidates = [
        (k, s) for k, s in enumerate(result.verification_stats) if s.success and abs(s.i - s.j) == 1
    ] or [(k, s) for k, s in enumerate(result.verification_stats) if s.success]
    k, stats = max(candidates, key=lambda c: c[1].n_inliers)
    kept = inlier_matches(result.matches[k], result.geometries[k])
    a, b = result.images[stats.i], result.images[stats.j]
    half = (size[0] // 2 - 20, size[1] - 40)
    left, right = _fit(a.image, half), _fit(b.image, half)
    sa, sb = left.shape[1] / a.width, right.shape[1] / b.width
    canvas = _canvas(size)
    y0 = (size[1] - left.shape[0]) // 2
    x_left, x_right = (
        10 + (half[0] - left.shape[1]) // 2,
        size[0] // 2 + 10 + (half[0] - right.shape[1]) // 2,
    )
    _paste(canvas, _dim(left), x_left, y0)
    _paste(canvas, _dim(right), x_right, (size[1] - right.shape[0]) // 2)
    pa, pb = kept.points(result.features[stats.i], result.features[stats.j])
    y1 = (size[1] - right.shape[0]) // 2
    # an even sample, so single lines stay visible (the caption gives the full count)
    shown = np.linspace(0, len(pa) - 1, min(len(pa), MATCH_LINES)).astype(int)
    for (xa, ya), (xb, yb) in zip(pa[shown], pb[shown], strict=True):
        cv2.line(
            canvas,
            (round(x_left + xa * sa), round(y0 + ya * sa)),
            (round(x_right + xb * sb), round(y1 + yb * sb)),
            ACCENT,
            1,
            cv2.LINE_AA,
        )
    _save(out, "matches", canvas, config)
    return {"photos": [a.name, b.name], "matches": stats.n_matches, "verified": stats.n_inliers}


def _read_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d  # external: PLY reading

    cloud = o3d.t.io.read_point_cloud(str(path))
    points = cloud.point.positions.numpy().astype(np.float64)
    colours = cloud.point.colors.numpy() if "colors" in cloud.point else np.full_like(points, 0.8)
    if colours.max() > 1:
        colours = colours / 255.0
    return points, colours


def _render(
    points: np.ndarray,
    colours: np.ndarray,
    centres: np.ndarray,
    size: tuple[int, int],
    cameras_only: bool = False,
) -> np.ndarray:
    """An orthographic view from above the camera ring: the cloud (or, with
    ``cameras_only``, a faint cloud and every camera centre)."""
    width, height = size
    middle = np.median(points, axis=0)
    _, _, vt = np.linalg.svd(centres - centres.mean(axis=0))
    up = vt[2] if len(centres) >= 3 else np.array([0.0, -1.0, 0.0])
    if np.dot(up, centres.mean(axis=0) - middle) < 0:
        up = -up  # towards the side the cameras are on
    along = vt[0] - np.dot(vt[0], up) * up
    along /= np.linalg.norm(along)
    side = np.cross(up, along)
    elevation = np.deg2rad(VIEW_ELEVATION_DEG)
    view = np.cos(elevation) * side - np.sin(elevation) * up  # looking down at the object
    right = np.cross(view, up)
    right /= np.linalg.norm(right)
    screen_up = np.cross(right, view)

    def project(p: np.ndarray, scale: float) -> np.ndarray:
        d = p - middle
        return np.stack(
            [width / 2 + scale * d @ right, height / 2 - scale * d @ screen_up, d @ view], axis=1
        )

    if cameras_only:
        reach = np.abs((centres - middle) @ np.stack([right, screen_up], axis=1)).max(axis=0)
    else:
        reach = np.percentile(
            np.abs((points - middle) @ np.stack([right, screen_up], axis=1)), 98, axis=0
        )
    scale = FILL * min(width / reach[0], height / reach[1])
    canvas = _canvas(size)
    p = project(points, scale)
    order = np.argsort(-p[:, 2])  # far first
    radius = 1 if cameras_only else max(1, round(width / 480))
    bgr = (colours[:, ::-1] * 255).astype(np.uint8)
    if cameras_only:
        bgr = _dim(bgr, FAINT_CLOUD)
    for k in order:
        x, y = int(p[k, 0]), int(p[k, 1])
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), radius, tuple(int(v) for v in bgr[k]), -1, cv2.LINE_AA)
    if cameras_only:
        c = project(centres, scale)
        target = project(middle[None], scale)[0]
        for x, y, _ in c:
            tip = (int(x + TICK * (target[0] - x)), int(y + TICK * (target[1] - y)))
            cv2.line(canvas, (int(x), int(y)), tip, ACCENT, 1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), 4, ACCENT, -1, cv2.LINE_AA)
    return canvas

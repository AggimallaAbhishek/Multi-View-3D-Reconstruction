"""The intro page's showcase: a real run turned into small committed files."""

import json
import shutil

import cv2
import pytest

from src.config import config_from_dict
from src.evaluation.ground_truth import GT_CAMERAS_FILE
from src.pointcloud.export import write_cameras_json
from src.web.jobs import reconstruct
from src.web.showcase import FRAMES, make_showcase
from tests.render import render_box_scene, scene_cameras, write_scene


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A finished web run of the rendered scene (ground truth included) and its photos."""
    root = tmp_path_factory.mktemp("showcase")
    scene = render_box_scene(n_cameras=5, arc_deg=40.0)
    images, camera = write_scene(scene, root)
    shutil.copy(camera, images / "camera.yaml")
    gt = scene_cameras(scene)
    write_cameras_json(
        images / GT_CAMERAS_FILE, list(gt.values()), list(gt), [scene.image_size] * len(gt)
    )
    config = config_from_dict(
        {
            "output": {"root": str(root / "outputs"), "cache_dir": None},
            "features": {"max_features": 3000},
        }
    )
    return reconstruct(images, config, lambda s, t: None), images, config


def test_showcase_holds_the_cloud_cameras_numbers_and_frames(run, tmp_path):
    run_dir, images, config = run
    out = make_showcase(run_dir, images, tmp_path / "showcase", config, name="Rendered box")

    report = json.loads((run_dir / "report.json").read_text())
    data = json.loads((out / "showcase.json").read_text())
    assert data["name"] == "Rendered box"
    assert data["report"]["images"] == report["images"]
    assert data["report"]["stages"] == report["stages"]
    assert data["report"]["ground_truth"]["rotation_deg"] == report["ground_truth"]["rotation_deg"]
    assert (out / "cloud.ply").read_bytes() == (
        run_dir / "point_clouds" / "filtered.ply"
    ).read_bytes()
    cameras = json.loads((out / "cameras.json").read_text())["cameras"]
    assert len(cameras) == report["images"]["registered"]
    assert [f["file"] for f in data["frames"]] == [f"{name}.jpg" for name in FRAMES]


def test_frames_are_small_jpegs_of_the_configured_width(run, tmp_path):
    run_dir, images, config = run
    out = make_showcase(run_dir, images, tmp_path / "showcase", config)
    width = config.web.showcase_frame_px
    for name in FRAMES:
        path = out / f"{name}.jpg"
        frame = cv2.imread(str(path))
        assert frame.shape[1] == width and frame.shape[0] == round(width * 5 / 8), name
        assert path.stat().st_size < 200_000, name  # committed to the repository
        assert frame.std() > 4, name  # a picture, not a blank


def test_frame_numbers_come_from_the_run(run, tmp_path):
    """Each frame's caption numbers are measured on the images it shows."""
    run_dir, images, config = run
    data = json.loads(
        (make_showcase(run_dir, images, tmp_path / "s", config) / "showcase.json").read_text()
    )
    frames = {f["file"]: f for f in data["frames"]}
    assert frames["features.jpg"]["keypoints"] > 100
    matches = frames["matches.jpg"]
    assert 0 < matches["verified"] <= matches["matches"]
    assert frames["cameras.jpg"]["cameras"] == data["report"]["images"]["registered"]


def test_showcase_keeps_a_few_of_the_photos_themselves(run, tmp_path):
    """The intro shows real photos of the subject, evenly spread over the set."""
    run_dir, images, config = run
    out = make_showcase(run_dir, images, tmp_path / "s", config)
    photos = json.loads((out / "showcase.json").read_text())["photos"]
    assert len(photos) == 5  # every photo of the 5-view scene, at most web.showcase_photos (6)
    for entry in photos:
        image = cv2.imread(str(out / entry["file"]))
        assert max(image.shape[:2]) == config.web.thumbnail_px * 2
        assert (out / entry["file"]).stat().st_size < 60_000
    assert [p["name"] for p in photos] == sorted(p["name"] for p in photos)


def test_a_single_showcase_photo_is_the_first(run, tmp_path):
    run_dir, images, config = run
    one = config.with_overrides({"web.showcase_photos": 1})
    photos = json.loads(
        (make_showcase(run_dir, images, tmp_path / "s", one) / "showcase.json").read_text()
    )["photos"]
    assert [p["name"] for p in photos] == ["view_01.png"]


def _camera_at(centre, target=(0.0, 0.0, 0.0)):
    """A cameras.json entry at ``centre`` looking at ``target``."""
    import numpy as np

    c = np.asarray(centre, float)
    z = np.asarray(target, float) - c
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-9 else np.array([1.0, 0, 0])
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return {"name": "", "R": R.tolist(), "t": (-R @ c).tolist(), "centre": c.tolist()}


def test_the_most_complete_even_ring_is_chosen_for_display():
    """Cameras on a dome: the display ring is the height band whose cameras go all the
    way round, not the fuller band that leaves half the circle empty (24 cameras 15
    degrees apart: one per 10-degree sector, so all of them)."""
    import numpy as np

    from src.web.showcase import display_ring

    cameras = []
    for k in range(24):  # a complete, even ring at 30 degrees elevation
        a = 2 * np.pi * k / 24
        cameras.append(
            _camera_at([np.cos(a) * np.cos(0.52), np.sin(a) * np.cos(0.52), np.sin(0.52)])
        )
    for k in range(30):  # more cameras at 60 degrees, but only over half the circle
        a = np.pi * k / 30
        cameras.append(_camera_at([np.cos(a) * 0.5, np.sin(a) * 0.5, 0.866]))
    for k, cam in enumerate(cameras):
        cam["name"] = f"c{k:02d}"
    ring = display_ring(cameras, points=np.zeros((10, 3)))
    assert sorted(c["name"] for c in ring) == [f"c{k:02d}" for k in range(24)]


def test_a_dense_band_is_thinned_to_an_even_ring():
    """Cameras spread over a dome at every height: the ring shown is complete and even,
    one camera per sector."""
    import numpy as np

    from src.web.showcase import RING_SECTORS, display_ring

    rng = np.random.default_rng(0)
    cameras = []
    for k in range(3000):  # dense enough that every sector of every band has a camera
        a, e = rng.uniform(0, 2 * np.pi), rng.uniform(0.0, 1.4)
        cameras.append(_camera_at([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)]))
        cameras[-1]["name"] = f"c{k}"
    ring = display_ring(cameras, points=np.zeros((10, 3)))
    angles = np.sort([np.degrees(np.arctan2(c["centre"][1], c["centre"][0])) % 360 for c in ring])
    gaps = np.diff(np.concatenate([angles, [angles[0] + 360]]))
    assert len(ring) == RING_SECTORS
    assert gaps.max() < 2 * 360 / RING_SECTORS  # even: no sector skipped

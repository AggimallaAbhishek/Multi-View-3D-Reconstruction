import numpy as np
import pytest

from src.config import config_from_dict
from src.data.loader import DatasetError
from src.geometry.pose_estimation import rotation_error_deg, translation_direction_error_deg
from src.pipeline import run_pipeline
from tests.images import write_image
from tests.render import render_box_scene


@pytest.fixture(scope="module")
def scene_dir(tmp_path_factory):
    scene = render_box_scene(n_cameras=4, arc_deg=30.0)
    root = tmp_path_factory.mktemp("pipeline")
    for k, im in enumerate(scene.images):
        write_image(root / "scene" / f"view_{k + 1:02d}.png", im)
    K = scene.K
    (root / "camera.yaml").write_text(
        f"width: 800\nheight: 600\nfx: {K[0, 0]}\nfy: {K[1, 1]}\ncx: {K[0, 2]}\ncy: {K[1, 2]}\n"
    )
    config = config_from_dict(
        {
            "camera": {"calibration_file": str(root / "camera.yaml")},
            "features": {"cache_dir": str(root / "cache"), "max_features": 3000},
        }
    )
    return scene, root / "scene", config


@pytest.mark.parametrize(
    ("until", "filled", "empty"),
    [
        ("features", ["features"], ["matches", "geometries", "poses"]),
        ("matches", ["features", "matches"], ["geometries", "poses"]),
        ("verify", ["matches", "geometries"], ["poses"]),
    ],
)
def test_pipeline_stops_after_requested_stage(scene_dir, until, filled, empty):
    _, images, config = scene_dir
    result = run_pipeline(images, config, until=until)
    assert all(getattr(result, name) is not None for name in filled)
    assert all(getattr(result, name) is None for name in empty)


def test_pipeline_end_to_end_recovers_true_initial_pose(scene_dir):
    scene, images, config = scene_dir
    result = run_pipeline(images, config)

    assert result.names == [f"view_{k:02d}.png" for k in range(1, 5)]
    assert len(result.matches) == len(result.geometries) == len(result.poses) == 6
    init = result.initial_pair
    assert init is not None
    R_true, t_true = scene.relative_pose(init.i, init.j)
    assert rotation_error_deg(init.pose.R, R_true) < 0.5
    assert translation_direction_error_deg(init.pose.t, t_true) < 2.0
    assert init.points.shape == (init.pose.n_valid, 3)
    assert np.all(init.points[:, 2] > 0)  # in front of camera i
    assert init.to_dict(result.names)["image_i"] == result.names[init.i]


def test_pipeline_rejects_unusable_folder(tmp_path):
    with pytest.raises(DatasetError):
        run_pipeline(tmp_path, config_from_dict({}))

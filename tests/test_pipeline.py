import numpy as np
import pytest

from src.config import config_from_dict
from src.data.loader import DatasetError
from src.evaluation.metrics import rotation_error_deg, translation_direction_error_deg
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
            "output": {"cache_dir": str(root / "cache")},
            "features": {"max_features": 3000},
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


def test_pipeline_triangulates_accurate_initial_structure(scene_dir):
    from src.evaluation.ground_truth import GroundTruthCamera
    from src.evaluation.metrics import two_view_structure_errors

    scene, images, config = scene_dir
    result = run_pipeline(images, config, until="triangulate")
    s = result.structure
    assert (s.i, s.j) == (result.initial_pair.i, result.initial_pair.j)
    counts = s.triangulation.counts
    assert counts["valid"] > 0.8 * counts["points"]
    assert s.triangulation.reprojection.median < 0.5
    assert np.all(s.triangulation.angles_deg[s.triangulation.valid] >= 1.5)

    gt = [
        GroundTruthCamera(scene.K, R, t)
        for R, t in zip(scene.rotations, scene.translations, strict=True)
    ]
    result = two_view_structure_errors(s, gt[s.i], gt[s.j])
    assert np.median(result.errors) < 0.005  # shape error under 0.5 % of the scene size
    assert abs(result.scale_ratio - 1) < 0.05


def test_pose_stage_leaves_structure_empty(scene_dir):
    _, images, config = scene_dir
    assert run_pipeline(images, config, until="pose").structure is None


def test_cli_triangulate_writes_outputs(scene_dir, tmp_path):
    import json

    from src.cli import main

    _, images, _ = scene_dir
    camera = images.parent / "camera.yaml"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"camera:\n  calibration_file: {camera}\noutput:\n  cache_dir: {tmp_path / 'cache'}\n"
        "features:\n  max_features: 3000\n"
    )
    out = tmp_path / "outputs"
    assert main(["triangulate", str(images), "--config", str(config), "--out", str(out)]) == 0
    (run_dir,) = out.glob("*_triangulate_sift")
    tri = run_dir / "triangulation"
    for name in ["triangulation.json", "points.npz", "reprojection_histogram.png", "structure.png"]:
        assert (tri / name).is_file(), name
    summary = json.loads((tri / "triangulation.json").read_text())
    assert summary["counts"]["valid"] > 0
    assert summary["reprojection_valid_points"]["median_px"] < 0.5
    points = np.load(tri / "points.npz")
    assert points["valid"].sum() == summary["counts"]["valid"]

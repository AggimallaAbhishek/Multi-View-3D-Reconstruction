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
    assert len(init.x_i) == len(init.x_j) == init.geometry.n_inliers
    assert init.to_dict(result.names)["image_i"] == result.names[init.i]


def test_pipeline_rejects_unusable_folder(tmp_path):
    with pytest.raises(DatasetError):
        run_pipeline(tmp_path, config_from_dict({}))


def test_pipeline_triangulates_accurate_initial_structure(scene_dir):
    from src.evaluation.metrics import two_view_structure_errors
    from src.geometry.camera import Camera

    scene, images, config = scene_dir
    result = run_pipeline(images, config, until="triangulate")
    s = result.structure
    assert (s.i, s.j) == (result.initial_pair.i, result.initial_pair.j)
    counts = s.triangulation.counts
    assert counts["valid"] > 0.8 * counts["points"]
    assert s.triangulation.reprojection.median < 0.5
    assert np.all(s.triangulation.angles_deg[s.triangulation.valid] >= 1.5)

    gt = [Camera(scene.K, R, t) for R, t in zip(scene.rotations, scene.translations, strict=True)]
    result = two_view_structure_errors(s, gt[s.i], gt[s.j], scale_i=(1.0, 1.0), scale_j=(1.0, 1.0))
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
    assert len(json.loads((run_dir / "dataset.json").read_text())["images"]) == len(
        list(images.glob("*.png"))
    )


def test_cli_two_view_exports_cloud_and_cameras(scene_dir, tmp_path):
    import json

    from src.cli import main
    from src.pointcloud.export import read_cameras_json, read_ply

    scene, images, _ = scene_dir
    config = tmp_path / "config.yaml"
    config.write_text(
        f"camera:\n  calibration_file: {images.parent / 'camera.yaml'}\n"
        f"output:\n  cache_dir: {tmp_path / 'cache'}\nfeatures:\n  max_features: 3000\n"
    )
    out = tmp_path / "outputs"
    assert main(["two-view", str(images), "--config", str(config), "--out", str(out)]) == 0
    (run_dir,) = out.glob("*_two_view_sift")

    summary = json.loads((run_dir / "triangulation" / "triangulation.json").read_text())
    points, colors = read_ply(run_dir / "point_clouds" / "sparse.ply")
    assert len(points) == summary["counts"]["valid"] > 0  # PLY reloads with every valid point
    assert colors is not None and colors.std() > 0.02  # sampled from the textured images

    cameras, names, sizes = read_cameras_json(run_dir / "camera_poses" / "cameras.json")
    assert names == [summary["image_i"], summary["image_j"]]
    assert sizes == [(800, 600), (800, 600)]
    np.testing.assert_allclose(cameras[0].R, np.eye(3))
    np.testing.assert_allclose(np.linalg.norm(cameras[1].t), 1.0)  # unit baseline
    # Points lie in front of both exported cameras.
    assert all((c.project(points)[1] > 0).all() for c in cameras)


def test_structure_metric_is_resolution_independent(scene_dir):
    # Regression: ground-truth K stayed at full resolution while observations were
    # downscaled, inflating the shape error ~14x and halving the scale ratio.
    from src.evaluation.metrics import two_view_structure_errors
    from src.geometry.camera import Camera

    scene, images, config = scene_dir
    half = config.with_overrides({"preprocessing.max_side": 400})  # 800x600 -> 400x300
    result = run_pipeline(images, half, until="triangulate")
    s = result.structure
    gt = [Camera(scene.K, R, t) for R, t in zip(scene.rotations, scene.translations, strict=True)]
    metric = two_view_structure_errors(
        s, gt[s.i], gt[s.j], scale_i=result.images[s.i].scale, scale_j=result.images[s.j].scale
    )
    assert result.images[s.i].scale == (0.5, 0.5)
    assert np.median(metric.errors) < 0.005
    assert abs(metric.scale_ratio - 1) < 0.05


def test_two_view_cloud_is_available_from_the_library(scene_dir, tmp_path):
    from src.pointcloud.export import read_cameras_json, read_ply, save_cloud
    from src.reconstruction.two_view import FRAME

    _, images, config = scene_dir
    result = run_pipeline(images, config, until="cloud")
    s = result.structure
    assert result.colors.shape == s.points.shape
    a, b = result.images[s.i], result.images[s.j]
    files = save_cloud(
        tmp_path, s.points, result.colors, s.cameras, [a.name, b.name],
        [(a.width, a.height), (b.width, b.height)], FRAME,
    )  # fmt: skip
    points, colors = read_ply(files.ply)
    np.testing.assert_array_equal(points, s.points)
    np.testing.assert_allclose(colors, result.colors, atol=1e-6)
    assert read_cameras_json(files.cameras)[1] == [a.name, b.name]


def test_cli_records_out_and_camera_yaml_next_to_images_is_used(scene_dir, tmp_path, capsys):
    import shutil

    import yaml

    from src.cli import main

    _, images, _ = scene_dir
    folder = tmp_path / "scene"
    shutil.copytree(images, folder)
    shutil.copy(images.parent / "camera.yaml", folder / "camera.yaml")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"output:\n  cache_dir: {tmp_path / 'cache'}\nfeatures:\n  max_features: 2000\n"
    )
    out = tmp_path / "runs"
    assert main(["pose", str(folder), "--config", str(config), "--out", str(out)]) == 0
    assert "Using calibration" in capsys.readouterr().out

    (run_dir,) = out.glob("*_pose_sift")
    recorded = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert recorded["output"]["root"] == str(out)
    stats = (run_dir / "verification" / "verification_statistics.csv").read_text()
    assert ",essential," in stats  # calibrated intrinsics -> auto picks the essential matrix


def test_reconstruct_stage_registers_every_view(scene_dir):
    from src.evaluation.metrics import camera_pose_errors
    from src.geometry.camera import Camera
    from src.geometry.reprojection import reprojection_errors

    scene, images, config = scene_dir
    result = run_pipeline(images, config, until="reconstruct")
    rec = result.reconstruction
    registered = rec.registry.images
    assert sorted(registered) == [0, 1, 2, 3]
    assert registered[:2] == [result.initial_pair.i, result.initial_pair.j]

    gt = [Camera(scene.K, scene.rotations[m], scene.translations[m]) for m in registered]
    errors = camera_pose_errors([rec.registry.camera(m) for m in registered], gt)
    assert errors.rotation_deg.max() < 0.5
    assert errors.centre.max() < 0.02

    lm = rec.landmarks
    assert len(lm) > len(result.structure.points)  # more than the initial pair alone
    cameras, obs, X = lm.as_problem(rec.registry, result.features)
    assert np.median(reprojection_errors(cameras, obs, X)) < 0.5
    assert lm.color.shape == (len(lm), 3) and lm.color.std() > 5  # sampled from the images


def test_cli_reconstruct_exports_multi_view_cloud(scene_dir, tmp_path):
    import json

    from src.cli import main
    from src.pointcloud.export import read_cameras_json, read_ply

    _, images, _ = scene_dir
    config = tmp_path / "config.yaml"
    config.write_text(
        f"camera:\n  calibration_file: {images.parent / 'camera.yaml'}\n"
        f"output:\n  cache_dir: {tmp_path / 'cache'}\nfeatures:\n  max_features: 3000\n"
    )
    out = tmp_path / "outputs"
    assert main(["reconstruct", str(images), "--config", str(config), "--out", str(out)]) == 0
    (run_dir,) = out.glob("*_reconstruct_sift")

    rec = run_dir / "reconstruction"
    summary = json.loads((rec / "reconstruction.json").read_text())
    assert summary["registered"] == summary["images"] == 4
    assert summary["reprojection"]["median_px"] < 0.5
    for name in ["registration.csv", "reprojection_histogram.png", "cameras.png"]:
        assert (rec / name).is_file(), name

    points, colors = read_ply(run_dir / "point_clouds" / "multi_view.ply")
    assert len(points) == summary["landmarks"] > 0
    assert colors is not None and colors.std() > 0.02
    cameras, names, _ = read_cameras_json(run_dir / "camera_poses" / "multi_view_cameras.json")
    assert sorted(names) == sorted(p.name for p in images.glob("*.png"))
    assert all((c.project(points)[1] > 0).mean() > 0.9 for c in cameras)

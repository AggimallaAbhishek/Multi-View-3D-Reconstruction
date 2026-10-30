import pytest

from src.config import config_from_dict
from src.pipeline import run_pipeline
from tests.render import render_box_scene, write_scene

# Every file `reconstruct` writes besides the per-pair diagnostic images and run metadata.
RECONSTRUCT_FILES = [
    "features/feature_statistics.csv",
    "matches/match_statistics.csv",
    "matches/matches.npz",
    "matches/match_matrix.png",
    "verification/verification_statistics.csv",
    "verification/verified_matches.npz",
    "verification/verified_matrix.png",
    "pose/relative_poses.csv",
    "pose/initial_pair.json",
    "pose/initial_pair.png",
    "triangulation/triangulation.json",
    "triangulation/points.npz",
    "triangulation/reprojection_histogram.png",
    "triangulation/structure.png",
    "reconstruction/reconstruction.json",
    "reconstruction/registration.csv",
    "reconstruction/reprojection_histogram.png",
    "reconstruction/cameras.png",
    "point_clouds/sparse.ply",
    "point_clouds/filtered.ply",
    "camera_poses/cameras.json",
]


@pytest.fixture(scope="module")
def scene_dir(tmp_path_factory):
    scene = render_box_scene(n_cameras=4, arc_deg=30.0)
    root = tmp_path_factory.mktemp("outputs")
    images, camera = write_scene(scene, root)
    raw = {
        "camera": {"calibration_file": str(camera)},
        "output": {"cache_dir": str(root / "cache")},
        "features": {"max_features": 3000},
    }
    return images, raw


@pytest.fixture(scope="module")
def filtered(scene_dir):
    images, raw = scene_dir
    return run_pipeline(images, config_from_dict(raw), until="filter")


def _files(run_dir):
    return sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())


def test_save_stage_outputs_writes_every_stage_up_to_filter(filtered, tmp_path):
    from src.outputs import save_stage_outputs

    written, status = save_stage_outputs(filtered, tmp_path, "filter")
    assert (written, status) == (tmp_path / "point_clouds" / "filtered.ply", 0)
    files = _files(tmp_path)
    for name in RECONSTRUCT_FILES:
        assert name in files, name
    # the two-view cloud's renders belong to `two-view`; the multi-view cloud replaces them
    assert not any(name.startswith("figures/two_view") for name in files)
    # diagnostic images are drawn only for the last stage, and `filter` draws none
    assert not any(name.endswith(("_keypoints.png", "_matches.png")) for name in files)


def test_save_stage_outputs_draws_diagnostics_for_the_last_stage_only(filtered, tmp_path):
    from src.outputs import save_stage_outputs

    save_stage_outputs(filtered, tmp_path, "matches", draw=2)
    files = _files(tmp_path)
    assert sum(name.endswith("_matches.png") for name in files) == 2
    assert not any(name.endswith("_keypoints.png") for name in files)


def test_cli_reconstruct_writes_what_save_stage_outputs_writes(scene_dir, filtered, tmp_path):
    import yaml

    from src.cli import main
    from src.outputs import save_stage_outputs

    images, raw = scene_dir
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    out = tmp_path / "runs"
    assert main(["reconstruct", str(images), "--config", str(config), "--out", str(out)]) == 0
    (cli_dir,) = out.glob("*_reconstruct_sift")

    library_dir = tmp_path / "library"
    save_stage_outputs(filtered, library_dir, "filter")  # `reconstruct` runs to "filter"
    metadata = {"config.yaml", "environment.json", "dataset.json"}
    assert [f for f in _files(cli_dir) if f not in metadata] == _files(library_dir)

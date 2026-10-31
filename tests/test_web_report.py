import json

import numpy as np
import pytest

from src.config import config_from_dict
from src.pipeline import STAGES, run_pipeline
from src.web.report import build_report, write_report
from tests.images import make_natural_texture, write_image
from tests.render import render_box_scene, scene_cameras, write_scene

N_VIEWS = 6


@pytest.fixture(scope="module")
def scene_dir(tmp_path_factory):
    scene = render_box_scene(n_cameras=N_VIEWS, arc_deg=40.0)
    images, camera = write_scene(scene, tmp_path_factory.mktemp("report"))
    raw = {
        "output": {"cache_dir": str(images.parent / "cache")},
        "features": {"max_features": 3000},
    }
    return images, camera, raw, scene_cameras(scene)


@pytest.fixture(scope="module")
def result(scene_dir):
    images, camera, raw, _ = scene_dir
    config = config_from_dict({**raw, "camera": {"calibration_file": str(camera)}})
    return run_pipeline(images, config, until="filter")


def test_report_summarises_a_full_reconstruction(result):
    report = build_report(result)
    rec = result.reconstruction

    assert report["images"] == {
        "total": N_VIEWS,
        "registered": N_VIEWS,
        "unregistered": [],
    }
    assert report["points"] == {
        "sparse": len(rec.landmarks),
        "filtered": int(result.filtering.keep.sum()),
    }
    assert report["calibration"]["sources"] == ["file"]
    first = rec.registry.images[0]  # K before and after bundle adjustment, of one image
    np.testing.assert_allclose(report["calibration"]["K"], result.images[first].K)
    np.testing.assert_allclose(report["calibration"]["refined_K"], rec.registry.camera(first).K)
    max_px = result.config.triangulation.max_reprojection_error_px
    assert report["reprojection"]["median_px"] < max_px
    assert list(report["timings"]) == ["dataset", *STAGES]
    assert report["total_s"] == pytest.approx(sum(result.timings.values()))
    assert report["warnings"] == []
    assert "ground_truth" not in report


def test_report_scores_cameras_against_ground_truth(result, scene_dir):
    gt = scene_dir[3]
    truth = build_report(result, gt=gt)["ground_truth"]

    assert truth["cameras"] == N_VIEWS
    assert truth["rotation_deg"]["median"] < 0.5
    assert truth["rotation_deg"]["max"] >= truth["rotation_deg"]["median"]
    assert truth["centre"]["median"] < 0.01
    rows = truth["per_camera"]
    assert sorted(row["image"] for row in rows) == sorted(gt)
    assert all(row["rotation_deg"] < 1.0 for row in rows)


def test_ground_truth_without_matching_names_is_reported_not_raised(result, scene_dir):
    gt = {f"other_{name}": cam for name, cam in scene_dir[3].items()}
    report = build_report(result, gt=gt)
    assert report["ground_truth"] == {"error": "no matching cameras in gt_cameras.json"}
    assert report["images"]["registered"] == N_VIEWS  # the rest of the report is unaffected


def test_ground_truth_needs_three_registered_cameras_with_a_match(result, scene_dir):
    gt = dict(list(scene_dir[3].items())[:2])
    error = build_report(result, gt=gt)["ground_truth"]["error"]
    assert "at least 3" in error


def test_unregistered_images_are_listed_and_warned_about(scene_dir, tmp_path):
    import shutil

    images, camera, raw, _ = scene_dir
    folder = tmp_path / "scene"
    shutil.copytree(images, folder)
    # a photo of something else: no overlap with the box, so it cannot be registered
    write_image(folder / "view_99.png", make_natural_texture(seed=7, width=800, height=600))
    config = config_from_dict({**raw, "camera": {"calibration_file": str(camera)}})
    report = build_report(run_pipeline(folder, config, until="reconstruct"))

    assert report["images"] == {
        "total": N_VIEWS + 1,
        "registered": N_VIEWS,
        "unregistered": ["view_99.png"],
    }
    assert report["warnings"] == ["1 image(s) not registered: view_99.png"]


def test_report_round_trips_through_json(result, scene_dir, tmp_path):
    report = build_report(result, gt=scene_dir[3])
    path = write_report(report, tmp_path)
    assert path == tmp_path / "report.json"
    assert json.loads(path.read_text()) == report


def test_partial_run_with_fallback_intrinsics_warns(scene_dir):
    images, _, raw, _ = scene_dir  # no camera.yaml next to the images, PNGs carry no EXIF
    partial = run_pipeline(images, config_from_dict(raw), until="features")
    report = build_report(partial)

    assert report["calibration"]["sources"] == ["fallback"]
    assert report["images"] == {
        "total": N_VIEWS,
        "registered": 0,
        "unregistered": [im.name for im in partial.images],
    }
    assert report["points"] is None and report["reprojection"] is None
    assert report["calibration"]["refined_K"] is None
    assert any("fallback" in w for w in report["warnings"])
    assert any("no reconstruction" in w for w in report["warnings"])
    json.dumps(report)

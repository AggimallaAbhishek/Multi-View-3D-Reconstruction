import csv
import json

import numpy as np
import pytest

from src.config import config_from_dict
from src.evaluation.experiments import (
    ExperimentError,
    load_experiment,
    reference_cloud,
    run_experiment,
)
from src.evaluation.metrics import camera_pose_errors, cloud_errors, metric_scale
from src.geometry.camera import Camera
from tests.images import write_image
from tests.render import render_box_scene
from tests.synthetic import make_scene

# --- metrics --------------------------------------------------------------------------


def _cloud(n=2000, seed=0):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, (n, 3))


def test_cloud_errors_of_identical_clouds_are_zero():
    ref = _cloud()
    e = cloud_errors(ref, ref, tolerance=0.01)
    assert e.completeness == 1.0 and e.accuracy_median == 0.0 and e.chamfer == 0.0


def test_cloud_errors_measure_coverage_and_distance():
    ref = _cloud()
    extent = np.linalg.norm(np.ptp(ref, axis=0))
    half = ref[ref[:, 0] < 0]  # only half the reference is reconstructed
    assert cloud_errors(half, ref, tolerance=0.01).completeness == pytest.approx(0.5, abs=0.03)
    assert cloud_errors(half, ref, tolerance=0.01).accuracy_median == 0.0  # what exists is exact

    shifted = ref + [0.03 * extent, 0, 0]  # everything 3 % of the extent off
    e = cloud_errors(shifted, ref, tolerance=0.01)
    assert e.completeness < 0.05
    assert e.accuracy_median <= 0.03 + 1e-9  # nearest reference point is at most that far
    assert cloud_errors(shifted, ref, tolerance=0.05).completeness > 0.95
    assert cloud_errors(np.empty((0, 3)), ref, tolerance=0.01).completeness == 0.0


def test_metric_scale_from_one_measured_distance():
    a, b = np.array([0.0, 0, 0]), np.array([0.0, 3.0, 4.0])  # 5 units apart
    assert metric_scale(a, b, measured=0.25) == pytest.approx(0.05)


def test_pose_errors_carry_the_aligning_similarity():
    scene = make_scene(n_cameras=5, seed=1)
    truth = [
        Camera(scene.K, R, t) for R, t in zip(scene.rotations, scene.translations, strict=True)
    ]
    half = [Camera(c.K, c.R, 0.5 * c.t) for c in truth]  # same cameras at half the scale
    errors = camera_pose_errors(half, truth)
    assert errors.scale == pytest.approx(2.0)
    est = np.array([c.centre for c in half])
    np.testing.assert_allclose(errors.similarity.apply(est), [c.centre for c in truth], atol=1e-9)


# --- experiment specs -----------------------------------------------------------------


def test_grid_experiment_expands_every_combination(tmp_path):
    spec = tmp_path / "e2.yaml"
    spec.write_text(
        "name: e2\nquestion: RQ2\nbase:\n  features.max_features: 500\n"
        "vary:\n  matching.ratio: [0.6, 0.8]\n  matching.mutual: [true, false]\n"
    )
    exp = load_experiment(spec)
    assert exp.name == "e2" and exp.question == "RQ2"
    assert [s.label for s in exp.settings] == [
        "matching.ratio=0.6, matching.mutual=True",
        "matching.ratio=0.6, matching.mutual=False",
        "matching.ratio=0.8, matching.mutual=True",
        "matching.ratio=0.8, matching.mutual=False",
    ]
    assert exp.settings[1].overrides == {
        "features.max_features": 500, "matching.ratio": 0.6, "matching.mutual": False
    }  # fmt: skip


def test_named_runs_and_invalid_specs(tmp_path):
    spec = tmp_path / "e3.yaml"
    spec.write_text(
        "name: e3\nquestion: RQ3\nruns:\n  calibrated: {}\n"
        "  focal +10%:\n    camera.focal_scale: 1.1\n"
    )
    exp = load_experiment(spec)
    assert [(s.label, s.overrides) for s in exp.settings] == [
        ("calibrated", {}), ("focal +10%", {"camera.focal_scale": 1.1})
    ]  # fmt: skip

    both = tmp_path / "bad.yaml"
    both.write_text("name: x\nquestion: q\nruns: {a: {}}\nvary: {matching.ratio: [0.7]}\n")
    with pytest.raises(ExperimentError, match="exactly one"):
        load_experiment(both)
    typo = tmp_path / "typo.yaml"
    typo.write_text("name: x\nquestion: q\nvary: {matching.ratoi: [0.7]}\n")
    with pytest.raises(ExperimentError, match="ratoi"):
        load_experiment(typo)  # overrides are validated against the config up front


# --- running experiments --------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    scene = render_box_scene(n_cameras=4, arc_deg=30.0)
    root = tmp_path_factory.mktemp("experiment")
    for k, im in enumerate(scene.images):
        write_image(root / "scene" / f"view_{k + 1:02d}.png", im)
    K = scene.K
    (root / "scene" / "camera.yaml").write_text(
        f"width: 800\nheight: 600\nfx: {K[0, 0]}\nfy: {K[1, 1]}\ncx: {K[0, 2]}\ncy: {K[1, 2]}\n"
    )
    gt = {
        f"view_{k + 1:02d}.png": Camera(K, R, t)
        for k, (R, t) in enumerate(zip(scene.rotations, scene.translations, strict=True))
    }
    config = config_from_dict(
        {"output": {"cache_dir": str(root / "cache")}, "features": {"max_features": 3000}}
    )
    return root / "scene", gt, config


def test_reference_cloud_lies_on_the_true_geometry(rendered):
    images, gt, config = rendered
    ref = reference_cloud(images, config, gt)
    assert len(ref) > 300
    # Every point should lie on the ground plane y = 1 or on the box [-1, 1]^3.
    to_ground = np.abs(ref[:, 1] - 1.0)
    outside = np.maximum(np.abs(ref) - 1.0, 0.0)  # distance outside the box, per axis
    inside_gap = 1.0 - np.abs(ref).max(axis=1)  # distance to the box surface from inside
    to_box = np.where(outside.any(axis=1), np.linalg.norm(outside, axis=1), inside_gap)
    assert np.percentile(np.minimum(to_ground, to_box), 95) < 0.02


def test_run_experiment_writes_one_row_per_setting(rendered, tmp_path):
    images, gt, config = rendered
    spec = tmp_path / "e5.yaml"
    spec.write_text(
        "name: e5\nquestion: BA gain\nvary:\n  bundle_adjustment.enabled: [true, false]\n"
    )
    exp = load_experiment(spec)
    ref = reference_cloud(images, config, gt)
    rows = run_experiment(exp, images, config, tmp_path / "out", gt=gt, reference=ref)

    assert len(rows) == 2
    for row in rows:
        assert row["status"] == "ok"
        assert row["images"] == row["registered"] == 4
        assert row["rotation_error_median_deg"] < 1.0
        assert row["completeness"] > 0.5
        assert row["time_total_s"] >= row["time_reconstruct_s"] > 0
    assert rows[0]["bundle_adjustment.enabled"] is True
    assert rows[1]["bundle_adjustment.enabled"] is False

    table = tmp_path / "out" / "e5" / "experiment_results.csv"
    with table.open() as f:
        written = list(csv.DictReader(f))
    assert [r["setting"] for r in written] == [s.label for s in exp.settings]
    meta = json.loads((tmp_path / "out" / "e5" / "experiment.json").read_text())
    assert meta["question"] == "BA gain" and len(meta["runs"]) == 2
    run_dir = tmp_path / "out" / "e5" / meta["runs"][0]["directory"]
    assert (run_dir / "config.yaml").is_file() and (run_dir / "filtered.ply").is_file()


def test_failed_settings_are_recorded_not_raised(rendered, tmp_path):
    images, gt, config = rendered
    spec = tmp_path / "bad.yaml"
    spec.write_text("name: bad\nquestion: q\nvary:\n  dataset.min_images: [99]\n")
    rows = run_experiment(load_experiment(spec), images, config, tmp_path / "out", gt=gt)
    assert rows[0]["status"].startswith("failed")
    assert rows[0]["registered"] == 0


# --- reports --------------------------------------------------------------------------


def _fake_experiment(directory, plot, rows):
    from src.run import write_csv, write_json

    columns = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(directory / "experiment_results.csv", columns,
              ([r.get(c, "") for c in columns] for r in rows))  # fmt: skip
    write_json(directory / "experiment.json", {"name": directory.name, "question": "RQ?",
                                               "plot": plot, "runs": []})  # fmt: skip


def _row(setting, **values):
    times = {f"time_{s}_s": 0.1 for s in ("dataset", "features", "reconstruct")}
    return {"setting": setting, "status": "ok", "registered": 4, "completeness": 0.5,
            **times, "time_total_s": 0.3, **values}  # fmt: skip


@pytest.mark.parametrize(
    ("plot", "rows"),
    [
        (  # numeric x, one line per hue value
            {"x": "matching.ratio", "hue": "matching.mutual", "y": ["registered", "completeness"]},
            [_row(f"r={r}, m={m}", **{"matching.ratio": r, "matching.mutual": m})
             for r in (0.6, 0.8) for m in (True, False)],
        ),
        (  # categorical: one bar per setting
            {"y": ["registered"]},
            [_row("sift"), _row("orb", registered=2)],
        ),
    ],
)  # fmt: skip
def test_reports_draw_figures_and_a_results_table(tmp_path, plot, rows):
    from src.evaluation.reports import make_report

    directory = tmp_path / "e"
    directory.mkdir()
    _fake_experiment(directory, plot, rows)
    files = make_report(directory)
    names = {f.name for f in files}
    assert {"metrics.png", "runtime.png", "results.md"} <= names
    assert all(f.is_file() and f.stat().st_size > 0 for f in files)
    table = (directory / "results.md").read_text()
    assert all(r["setting"] in table for r in rows)
    assert "RQ?" in table

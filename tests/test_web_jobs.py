import json
import threading
from pathlib import Path

import pytest

from src.config import config_from_dict
from src.data.loader import DatasetError
from src.pipeline import STAGES
from src.web.jobs import JobBusy, JobFailed, JobManager, reconstruct, reconstruct_in_subprocess

CONFIG = config_from_dict({})


def fake_job(stages=("a", "b", "c"), fail_at=None, exc=None, gate=None):
    """A stand-in for the reconstruction: reports each stage, optionally fails (with
    ``exc``, default ``RuntimeError("boom")``) or waits."""
    exc = exc or RuntimeError("boom")

    def job(upload_dir, config, progress):
        timings = {"dataset": 0.1}
        for stage in (*stages, "outputs"):
            progress(stage, dict(timings))
            if gate is not None:
                gate.wait(5)
            if stage == fail_at:
                raise exc
            if stage != "outputs":
                timings[stage] = 0.2
        return Path("outputs") / f"run_{upload_dir.name}"

    return job


def test_job_runs_through_every_stage(tmp_path):
    manager = JobManager(job=fake_job())
    job = manager.submit(tmp_path / "up1", CONFIG)
    manager.wait(job.id)

    job = manager.get(job.id)
    assert job.status == "done"
    assert job.stage is None
    assert job.stages_done == ["dataset", "a", "b", "c", "outputs"]
    assert job.timings == {"dataset": 0.1, "a": 0.2, "b": 0.2, "c": 0.2}
    assert job.run_dir == Path("outputs") / "run_up1"
    assert job.error is None


def test_reports_dataset_stage_before_the_pipeline_reports_any():
    gate = threading.Event()
    seen = []

    def job(upload_dir, config, progress):
        seen.append(manager.get("j").stage)  # before any progress call
        gate.wait(5)
        return Path("outputs")

    manager = JobManager(job=job)
    manager.submit(Path("up"), CONFIG, job_id="j")
    gate.set()
    manager.wait("j")
    assert seen == ["dataset"]


def test_exception_in_a_stage_fails_the_job_at_that_stage(tmp_path):
    manager = JobManager(job=fake_job(fail_at="b"))
    job = manager.submit(tmp_path, CONFIG)
    manager.wait(job.id)

    job = manager.get(job.id)
    assert job.status == "failed"
    assert job.stage == "b"
    assert job.stages_done == ["dataset", "a"]
    assert job.error == "boom"
    assert job.run_dir is None


def test_dataset_error_message_is_kept(tmp_path):
    def job(upload_dir, config, progress):
        raise DatasetError("only 1 readable image(s); at least 2 required")

    manager = JobManager(job=job)
    job_id = manager.submit(tmp_path, CONFIG).id
    manager.wait(job_id)
    job = manager.get(job_id)
    assert (job.status, job.stage) == ("failed", "dataset")
    assert job.error == "only 1 readable image(s); at least 2 required"


def test_failure_while_writing_outputs_is_reported(tmp_path):
    manager = JobManager(job=fake_job(fail_at="outputs", exc=OSError("disk full")))
    job = manager.submit(tmp_path, CONFIG)
    manager.wait(job.id)
    job = manager.get(job.id)
    assert (job.status, job.stage, job.error) == ("failed", "outputs", "disk full")


def test_failed_job_keeps_the_outputs_written_so_far(tmp_path):
    manager = JobManager(job=fake_job(fail_at="b", exc=JobFailed("boom", tmp_path / "run")))
    job = manager.submit(tmp_path, CONFIG)
    manager.wait(job.id)
    job = manager.get(job.id)
    assert (job.status, job.error, job.run_dir) == ("failed", "boom", tmp_path / "run")


def test_only_one_job_at_a_time(tmp_path):
    gate = threading.Event()
    manager = JobManager(job=fake_job(gate=gate))
    first = manager.submit(tmp_path / "1", CONFIG)
    assert manager.active().id == first.id
    with pytest.raises(JobBusy) as busy:
        manager.submit(tmp_path / "2", CONFIG)
    assert busy.value.job_id == first.id

    gate.set()
    manager.wait(first.id)
    assert manager.active() is None
    second = manager.submit(tmp_path / "3", CONFIG, job_id="chosen")
    assert second.id == "chosen"
    manager.wait(second.id)
    assert [j.id for j in manager.list()] == ["chosen", first.id]  # newest first


def test_unknown_job_is_none():
    assert JobManager(job=fake_job()).get("missing") is None


def test_job_to_dict_is_json(tmp_path):
    manager = JobManager(job=fake_job())
    job = manager.submit(tmp_path / "up", CONFIG)
    manager.wait(job.id)
    data = json.loads(json.dumps(manager.get(job.id).to_dict()))
    assert data["status"] == "done"
    assert data["run_dir"] == str(Path("outputs") / "run_up")


# --- the real job, in a child process ---------------------------------------------------


def test_subprocess_job_relays_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    stages = []
    with pytest.raises(RuntimeError, match="no image"):
        reconstruct_in_subprocess(tmp_path / "empty", CONFIG, lambda s, t: stages.append(s))
    assert stages == []  # failed while preparing the dataset, before any stage


def test_default_job_writes_a_run_folder_with_a_report(tmp_path):
    """The real pipeline and outputs on the rendered scene, ground truth included.

    Regression: the outputs include an offscreen Open3D render, which aborts the whole
    process (SIGTRAP) outside the main thread; the job runs in a child process."""
    from src.pointcloud.export import write_cameras_json
    from tests.render import scene_cameras

    scene, images, config = rendered_upload(tmp_path)
    gt = scene_cameras(scene)
    write_cameras_json(
        images / "gt_cameras.json", list(gt.values()), list(gt), [scene.image_size] * len(gt)
    )
    manager = JobManager()
    job = manager.submit(images, config)
    manager.wait(job.id, timeout=180)

    job = manager.get(job.id)
    assert job.status == "done", job.error
    assert job.stages_done == ["dataset", *STAGES, "outputs"]
    assert job.run_dir.name.endswith("_web_reconstruct")
    assert (job.run_dir / "point_clouds" / "filtered.ply").is_file()
    report = json.loads((job.run_dir / "report.json").read_text())
    assert report["images"]["registered"] == 4
    assert report["calibration"]["sources"] == ["file"]
    assert report["ground_truth"]["rotation_deg"]["median"] < 0.5


def rendered_upload(tmp_path, n_cameras=4):
    """The rendered scene as an upload folder with ``camera.yaml``, and a config writing
    under ``tmp_path``."""
    import shutil

    from tests.render import render_box_scene, write_scene

    scene = render_box_scene(n_cameras=n_cameras, arc_deg=30.0)
    images, camera = write_scene(scene, tmp_path)
    shutil.copy(camera, images / "camera.yaml")
    config = config_from_dict(
        {
            "output": {"root": str(tmp_path / "outputs"), "cache_dir": None},
            "features": {"max_features": 3000},
        }
    )
    return scene, images, config


def test_failure_in_a_stage_writes_the_stages_before_it(tmp_path, monkeypatch):
    import src.pipeline

    def fail(result):
        raise RuntimeError("no initial pair")

    monkeypatch.setitem(src.pipeline._STEPS, "pose", fail)
    _, images, config = rendered_upload(tmp_path)
    with pytest.raises(JobFailed, match="no initial pair") as failed:
        reconstruct(images, config, lambda s, t: None)
    run_dir = failed.value.run_dir
    assert run_dir.name.endswith("_web_reconstruct")
    assert (run_dir / "dataset.json").is_file()
    assert (run_dir / "verification").is_dir()
    assert not (run_dir / "pose").exists()
    assert not (run_dir / "report.json").exists()


def test_failure_while_preparing_the_dataset_writes_nothing(tmp_path):
    (tmp_path / "empty").mkdir()
    config = config_from_dict({"output": {"root": str(tmp_path / "outputs")}})
    with pytest.raises(JobFailed) as failed:
        reconstruct(tmp_path / "empty", config, lambda s, t: None)
    assert failed.value.run_dir is None
    assert not (tmp_path / "outputs").exists()


def test_dataset_validation_failure_names_the_problems(tmp_path):
    folder = tmp_path / "upload"
    folder.mkdir()
    for name in ("a.png", "b.png"):
        (folder / name).write_bytes(b"not an image")
    config = config_from_dict({"output": {"root": str(tmp_path / "outputs")}})
    with pytest.raises(JobFailed) as failed:
        reconstruct(folder, config, lambda s, t: None)
    assert "0 readable image(s)" in str(failed.value)  # the issues, not just the folder

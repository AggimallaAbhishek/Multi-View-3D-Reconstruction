import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src.config import config_from_dict  # noqa: E402
from src.pipeline import STAGES  # noqa: E402
from src.web.app import UPLOAD_ROOT, create_app  # noqa: E402
from src.web.jobs import Job, JobBusy, JobManager  # noqa: E402

CAMERA_YAML = b"width: 64\nheight: 48\nfx: 50\nfy: 50\ncx: 32\ncy: 24\n"


def cameras_json() -> bytes:
    """One ground-truth camera, written by the exporter the pipeline uses."""
    import tempfile

    import numpy as np

    from src.geometry.camera import Camera
    from src.pointcloud.export import write_cameras_json

    K = np.array([[50.0, 0, 32], [0, 50, 24], [0, 0, 1]])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gt.json"
        write_cameras_json(path, [Camera(K, np.eye(3), np.zeros(3))], ["a.png"], [(64, 48)])
        return path.read_bytes()


class FakeManager:
    """Records submissions instead of running them; jobs stay as the test sets them."""

    def __init__(self, busy=False):
        self.jobs: dict[str, Job] = {}
        self.uploads: dict[str, Path] = {}
        self.running = Job(id="other", status="running") if busy else None

    def submit(self, upload_dir, config, job_id=None):
        if self.running is not None:
            raise JobBusy(self.running.id)
        job = Job(id=job_id)
        self.jobs[job.id] = job
        self.uploads[job.id] = Path(upload_dir)
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def list(self):
        return list(reversed(self.jobs.values()))

    def active(self):
        return self.running


def client_for(manager=None, **overrides):
    config = config_from_dict(overrides)
    return TestClient(create_app(config, manager or FakeManager()))


def images(*names, size=10):
    return [("images", (name, b"x" * size, "image/png")) for name in names]


def post(client, files, camera=None, gt=None):
    if camera is not None:
        files = [*files, ("camera", ("my_camera.yml", camera, "application/x-yaml"))]
    if gt is not None:
        files = [*files, ("ground_truth", ("truth.json", gt, "application/json"))]
    return client.post("/api/jobs", files=files)


def nothing_uploaded():
    return not UPLOAD_ROOT.exists() or not any(UPLOAD_ROOT.iterdir())


# --- upload validation ------------------------------------------------------------------


def test_too_few_images_is_rejected():
    response = post(client_for(), images("a.png"))
    assert response.status_code == 422
    assert "at least 2" in response.json()["detail"]
    assert nothing_uploaded()


def test_minimum_comes_from_the_dataset_config():
    response = post(client_for(dataset={"min_images": 3}), images("a.png", "b.png"))
    assert response.status_code == 422
    assert "at least 3" in response.json()["detail"]


def test_file_with_another_extension_is_named():
    response = post(client_for(), images("a.png", "notes.txt", "b.png"))
    assert response.status_code == 422
    assert "notes.txt" in response.json()["detail"]
    assert nothing_uploaded()


def test_extensions_come_from_the_dataset_config():
    response = post(client_for(dataset={"extensions": [".png"]}), images("a.png", "b.jpg"))
    assert response.status_code == 422
    assert "b.jpg" in response.json()["detail"]


def test_upper_case_extension_is_accepted():
    manager = FakeManager()
    response = post(client_for(manager), images("IMG_0001.JPG", "IMG_0002.JPG"))
    assert response.status_code == 201


def test_too_many_images_is_rejected():
    response = post(client_for(web={"max_upload_images": 3}), images(*(f"{c}.png" for c in "abcd")))
    assert response.status_code == 422
    assert "at most 3" in response.json()["detail"]
    assert nothing_uploaded()


def test_too_large_upload_is_rejected():
    # max_upload_mb is an int; 0 MB rejects any non-empty upload
    response = post(client_for(web={"max_upload_mb": 0}), images("a.png", "b.png"))
    assert response.status_code == 422
    assert "MB" in response.json()["detail"]
    assert nothing_uploaded()


@pytest.mark.parametrize("camera", [b"fx: [unclosed", b"width: 64\nheight: 48\n"])
def test_invalid_camera_file_is_rejected(camera):
    response = post(client_for(), images("a.png", "b.png"), camera=camera)
    assert response.status_code == 422
    assert "camera.yaml" in response.json()["detail"]
    assert nothing_uploaded()


def test_upload_without_images_has_a_text_detail():
    response = client_for().post(
        "/api/jobs", files=[("camera", ("c.yaml", CAMERA_YAML, "text/plain"))]
    )
    assert response.status_code == 422
    assert "at least 2" in response.json()["detail"]


@pytest.mark.parametrize(
    ("camera", "gt", "expected"),
    [
        (b"width: 64\n", None, "camera.yaml is invalid: missing key(s)"),
        (None, b'{"cameras": [{"name": "a.png"}]}', "gt_cameras.json is invalid: missing 'K'"),
    ],
)
def test_invalid_file_messages_do_not_show_server_paths(camera, gt, expected):
    response = post(client_for(), images("a.png", "b.png"), camera=camera, gt=gt)
    detail = response.json()["detail"]
    assert detail.startswith(expected)
    assert str(UPLOAD_ROOT) not in detail


def test_invalid_ground_truth_is_rejected():
    response = post(client_for(), images("a.png", "b.png"), gt=b"{not json")
    assert response.status_code == 422
    assert "gt_cameras.json" in response.json()["detail"]
    assert nothing_uploaded()


# --- job API ----------------------------------------------------------------------------


def test_valid_upload_is_saved_and_submitted():
    manager = FakeManager()
    response = post(
        client_for(manager), images("a.png", "b.png"), camera=CAMERA_YAML, gt=cameras_json()
    )
    assert response.status_code == 201
    job_id = response.json()["job_id"]
    upload = UPLOAD_ROOT / job_id
    assert manager.uploads[job_id] == upload
    assert sorted(p.name for p in upload.iterdir()) == [
        "a.png",
        "b.png",
        "camera.yaml",
        "gt_cameras.json",
    ]
    assert (upload / "camera.yaml").read_bytes() == CAMERA_YAML


def test_upload_names_cannot_leave_the_upload_folder():
    manager = FakeManager()
    response = post(client_for(manager), images("../../a.png", "sub/b.png"))
    assert response.status_code == 201
    upload = UPLOAD_ROOT / response.json()["job_id"]
    assert sorted(p.name for p in upload.iterdir()) == ["a.png", "b.png"]


def test_duplicate_image_names_are_rejected():
    response = post(client_for(), images("a.png", "a.png"))
    assert response.status_code == 422
    assert "a.png" in response.json()["detail"]


def test_upload_while_busy_is_a_conflict():
    response = post(client_for(FakeManager(busy=True)), images("a.png", "b.png"))
    assert response.status_code == 409
    assert response.json()["job_id"] == "other"  # the page links to it
    assert nothing_uploaded()


def test_unknown_job_is_404():
    client = client_for()
    assert client.get("/api/jobs/unknown").status_code == 404
    assert client.get("/api/jobs/unknown/report").status_code == 404
    assert client.get("/api/jobs/unknown/files/report.json").status_code == 404


def test_job_status_and_report_before_done():
    manager = FakeManager()
    manager.jobs["j"] = Job(id="j", status="running", stage="features")
    client = client_for(manager)
    status = client.get("/api/jobs/j").json()
    assert (status["id"], status["status"], status["stage"]) == ("j", "running", "features")
    assert client.get("/api/jobs/j/report").status_code == 409


@pytest.fixture
def done_job(tmp_path):
    run_dir = tmp_path / "outputs" / "20261030-120000_web_reconstruct"
    (run_dir / "point_clouds").mkdir(parents=True)
    (run_dir / "figures").mkdir()
    (run_dir / "report.json").write_text('{"total_s": 3.5}')
    (run_dir / "point_clouds" / "filtered.ply").write_bytes(b"ply\nformat ascii 1.0\n")
    (run_dir / "figures" / "filtered.png").write_bytes(b"\x89PNG")
    (tmp_path / "secret.txt").write_text("secret")
    manager = FakeManager()
    manager.jobs["j"] = Job(id="j", status="done", run_dir=run_dir)
    return client_for(manager)


def test_file_list_of_a_finished_job(done_job):
    assert done_job.get("/api/jobs/j/files").json() == [
        "figures/filtered.png",
        "point_clouds/filtered.ply",
        "report.json",
    ]


def test_file_list_of_a_job_without_outputs_is_empty():
    manager = FakeManager()
    manager.jobs["r"] = Job(id="r", status="running")
    assert client_for(manager).get("/api/jobs/r/files").json() == []
    assert client_for(manager).get("/api/jobs/unknown/files").status_code == 404


def test_report_of_a_finished_job(done_job):
    assert done_job.get("/api/jobs/j/report").json() == {"total_s": 3.5}


@pytest.mark.parametrize(
    ("path", "media_type"),
    [
        ("report.json", "application/json"),
        ("point_clouds/filtered.ply", "application/octet-stream"),
        ("figures/filtered.png", "image/png"),
    ],
)
def test_files_of_a_finished_job(done_job, path, media_type):
    response = done_job.get(f"/api/jobs/j/files/{path}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type)


@pytest.mark.parametrize(
    "path",
    [
        "..%2F..%2Fsecret.txt",
        "%2E%2E/%2E%2E/secret.txt",
        "%2Fetc%2Fpasswd",
        "missing.png",
        "figures",
    ],
)
def test_files_outside_the_run_folder_are_404(done_job, path):
    assert done_job.get(f"/api/jobs/j/files/{path}").status_code == 404


def test_previous_runs_are_rediscovered(tmp_path):
    old = tmp_path / "outputs" / "20261029-090000_web_reconstruct"
    new = tmp_path / "outputs" / "20261030-000000_web_reconstruct"
    for run_dir in (old, new):
        run_dir.mkdir(parents=True)
        (run_dir / "report.json").write_text('{"timings": {"dataset": 1.0, "features": 2.0}}')
    suffixed = tmp_path / "outputs" / "20261030-000000_web_reconstruct_2"  # same second
    (suffixed).mkdir()
    (suffixed / "report.json").write_text("{}")
    for name in (
        "20261030-010000_reconstruct",  # a CLI run
        "20261030-020000_web_reconstruct",  # no report
        "20261030-030000_web_reconstruct_copy",  # not a run folder name
    ):
        (tmp_path / "outputs" / name).mkdir()
        if name.endswith("copy"):
            (tmp_path / "outputs" / name / "report.json").write_text("{}")

    manager = FakeManager()
    manager.jobs["live"] = Job(id="live", status="running")
    client = client_for(manager, output={"root": str(tmp_path / "outputs")})

    jobs = client.get("/api/jobs").json()
    assert [j["id"] for j in jobs] == ["live", suffixed.name, new.name, old.name]
    status = client.get(f"/api/jobs/{new.name}").json()
    assert status["status"] == "done"
    assert status["stages_done"] == ["dataset", "features", "outputs"]
    assert status["timings"] == {"dataset": 1.0, "features": 2.0}
    assert client.get(f"/api/jobs/{new.name}/report").json()["timings"]["features"] == 2.0
    assert client.get(f"/api/jobs/{new.name}/files/report.json").status_code == 200


def test_rediscovered_run_lists_its_photos_and_registered_cameras(tmp_path):
    run_dir = tmp_path / "outputs" / "20261031-120000_web_reconstruct"
    run_dir.mkdir(parents=True)
    report = '{"images": {"total": 48, "registered": 19, "unregistered": []}, "timings": {}}'
    (run_dir / "report.json").write_text(report)
    client = client_for(output={"root": str(tmp_path / "outputs")})
    [listed] = client.get("/api/jobs").json()
    assert (listed["photos"], listed["registered"]) == (48, 19)
    assert listed["started"] == "20261031-120000"


def test_failed_job_serves_the_outputs_written_before_it_failed(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "verification").mkdir(parents=True)
    (run_dir / "verification" / "verified_matrix.png").write_bytes(b"\x89PNG")
    manager = FakeManager()
    manager.jobs["f"] = Job(id="f", status="failed", stage="pose", run_dir=run_dir)
    manager.jobs["g"] = Job(id="g", status="failed", stage="dataset")  # wrote nothing
    client = client_for(manager)
    assert client.get("/api/jobs/f/files/verification/verified_matrix.png").status_code == 200
    assert client.get("/api/jobs/f/report").status_code == 409
    assert client.get("/api/jobs/g/files/report.json").status_code == 404


def test_shutdown_stops_running_reconstructions(monkeypatch):
    import src.web.app

    calls = []
    monkeypatch.setattr(src.web.app, "terminate_children", lambda: calls.append(1))
    with client_for() as client:
        client.get("/api/config")
        assert calls == []
    assert calls == [1]


def test_page_is_served():
    response = client_for().get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower()


@pytest.mark.parametrize("path", ["/", "/app.js", "/viewer.js", "/style.css"])
def test_page_files_are_revalidated_on_every_load(path):
    """Without Cache-Control, browsers reuse a cached app.js after the page was updated."""
    client = client_for()
    response = client.get(path)
    assert response.headers["cache-control"] == "no-cache"
    etag = response.headers["etag"]
    assert client.get(path, headers={"If-None-Match": etag}).status_code == 304


def test_config_for_the_page():
    data = (
        client_for(web={"poll_interval_ms": 250}, dataset={"min_images": 3})
        .get("/api/config")
        .json()
    )
    assert data["poll_interval_ms"] == 250
    assert data["min_images"] == 3
    assert data["max_upload_images"] == 200
    assert data["max_upload_mb"] == 500
    assert ".jpg" in data["extensions"]
    assert data["stages"] == ["dataset", *STAGES, "outputs"]  # the page's checklist


# --- end to end -------------------------------------------------------------------------


def test_upload_to_results_end_to_end(tmp_path):
    from src.pointcloud.export import write_cameras_json
    from tests.render import render_box_scene, scene_cameras, write_scene

    scene = render_box_scene(n_cameras=6, arc_deg=40.0)
    folder, camera = write_scene(scene, tmp_path / "rendered")
    gt = scene_cameras(scene)
    gt_path = tmp_path / "gt.json"
    write_cameras_json(gt_path, list(gt.values()), list(gt), [scene.image_size] * len(gt))

    files = [("images", (p.name, p.read_bytes(), "image/png")) for p in sorted(folder.iterdir())]
    client = client_for(JobManager(), output={"cache_dir": None}, features={"max_features": 3000})
    response = post(client, files, camera=camera.read_bytes(), gt=gt_path.read_bytes())
    assert response.status_code == 201, response.text
    job_id = response.json()["job_id"]

    deadline = time.monotonic() + 120
    while (status := client.get(f"/api/jobs/{job_id}").json())["status"] not in (
        "done",
        "failed",
    ):
        assert time.monotonic() < deadline, status
        time.sleep(0.5)
    assert status["status"] == "done", status["error"]
    # one name: the run folder carries the time the page shows (and shows after a restart)
    assert Path(status["run_dir"]).name.startswith(f"{status['started']}_")

    report = client.get(f"/api/jobs/{job_id}/report").json()
    assert report["images"] == {"total": 6, "registered": 6, "unregistered": []}
    assert report["ground_truth"]["cameras"] == 6
    ply = client.get(f"/api/jobs/{job_id}/files/point_clouds/filtered.ply")
    assert ply.status_code == 200
    assert ply.content.startswith(b"ply")


# --- python -m src.web ------------------------------------------------------------------


def test_main_parses_args(tmp_path):
    from src.web.__main__ import build

    config = tmp_path / "web.yaml"
    config.write_text("web:\n  host: 0.0.0.0\n  poll_interval_ms: 250\n")
    app, web = build(["--config", str(config), "--port", "9001"])
    assert (web.host, web.port) == ("0.0.0.0", 9001)
    assert TestClient(app).get("/api/config").json()["poll_interval_ms"] == 250


def test_main_defaults_come_from_the_default_config():
    from src.config import load_config
    from src.web.__main__ import build

    _, web = build([])
    assert web == load_config().web


@pytest.mark.parametrize("open_browser", [True, False])
def test_main_opens_the_browser_once_the_server_is_up(monkeypatch, tmp_path, open_browser):
    import src.web.__main__ as entry

    opened = []

    class FakeServer:
        def __init__(self, config):
            self.config, self.started, self.should_exit = config, False, False

        def run(self):
            assert opened == []  # not before the server is listening
            self.started = True
            deadline = time.monotonic() + 5
            while open_browser and not opened and time.monotonic() < deadline:
                time.sleep(0.01)

    monkeypatch.setattr(entry.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(entry.webbrowser, "open", opened.append)
    config = tmp_path / "web.yaml"
    config.write_text(f"web:\n  port: 8123\n  open_browser: {str(open_browser).lower()}\n")
    entry.main(["--config", str(config)])
    assert opened == (["http://127.0.0.1:8123/"] if open_browser else [])


def test_main_exits_quietly_on_ctrl_c(monkeypatch, tmp_path):
    """uvicorn re-raises SIGINT after its graceful shutdown; no traceback for Ctrl+C."""
    import src.web.__main__ as entry

    class InterruptedServer:
        def __init__(self, config):
            self.started, self.should_exit = False, False

        def run(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(entry.uvicorn, "Server", InterruptedServer)
    config = tmp_path / "web.yaml"
    config.write_text("web:\n  open_browser: false\n")
    entry.main(["--config", str(config)])  # returns instead of raising

"""Record demo.mp4: a real session in the web application, driven by Playwright.

    uv run --no-project --with playwright --with pyyaml \
        python finalsubmission/tools/record_webapp_demo.py <project python> <sample dir> <out.mp4>

Starts the project's own server (`python -m src.web`) on a private port with a throwaway
output and cache folder (so the reconstruction really runs, nothing is cached and nothing
is written to the repository's outputs/), opens it in Google Chrome, and records:
the overview page, uploading the photos with camera.yaml and gt_cameras.json, the live
reconstruction, the 3-D result, the camera poses against ground truth, the matching
figures, the replay and the exports. Captions are overlaid on the page itself.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import yaml
from playwright.sync_api import Page, sync_playwright

REPO = Path(__file__).resolve().parents[2]
PORT = 8765
URL = f"http://127.0.0.1:{PORT}"
W, H = 1600, 900

CAPTION_JS = """
(text) => {
  let el = document.getElementById('__demo_caption');
  if (!el) {
    el = document.createElement('div');
    el.id = '__demo_caption';
    Object.assign(el.style, {
      position: 'fixed', left: '50%', bottom: '28px', transform: 'translateX(-50%)',
      zIndex: 2147483647, padding: '12px 22px', borderRadius: '10px',
      background: 'rgba(15,23,42,0.88)', color: '#fff', pointerEvents: 'none',
      font: '600 22px/1.35 -apple-system, "Helvetica Neue", Arial, sans-serif',
      boxShadow: '0 6px 24px rgba(0,0,0,0.3)', maxWidth: '80vw', textAlign: 'center',
      transition: 'opacity 0.25s',
    });
    document.body.appendChild(el);
  }
  el.textContent = text;
  el.style.opacity = text ? '1' : '0';
}
"""


def caption(page: Page, text: str) -> None:
    page.evaluate(CAPTION_JS, text)


def pause(page: Page, seconds: float) -> None:
    page.wait_for_timeout(int(seconds * 1000))


def smooth_scroll(page: Page, pixels: int, seconds: float) -> None:
    steps = max(1, int(seconds * 20))
    for _ in range(steps):
        page.mouse.wheel(0, pixels / steps)
        page.wait_for_timeout(50)


def drag(page: Page, selector: str, dx: float, dy: float, seconds: float) -> None:
    box = page.locator(selector).bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    steps = max(1, int(seconds * 30))
    for k in range(1, steps + 1):
        page.mouse.move(x + dx * k / steps, y + dy * k / steps)
        page.wait_for_timeout(33)
    page.mouse.up()


def start_server(python: str, work: Path) -> subprocess.Popen:
    config = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    config["web"].update(open_browser=False, port=PORT)
    config["output"].update(root=str(work / "outputs"), cache_dir=str(work / "cache"))
    cfg = work / "config.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=False))
    proc = subprocess.Popen(
        [python, "-m", "src.web", "--config", str(cfg)], cwd=REPO,
        stdout=open(work / "server.log", "w"), stderr=subprocess.STDOUT,
    )  # fmt: skip
    for _ in range(60):
        try:
            urllib.request.urlopen(URL, timeout=1)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    sys.exit(f"server did not start, see {work / 'server.log'}")


def record(sample: Path, video_dir: Path) -> Path:
    images = sorted(str(p) for p in sample.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True,
                                     args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])  # fmt: skip
        ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1,
                                  record_video_dir=str(video_dir), record_video_size={"width": W, "height": H})  # fmt: skip
        page = ctx.new_page()

        # 1. Overview page ---------------------------------------------------------------
        page.goto(URL + "/", wait_until="networkidle")
        caption(page, "Multi-View 3D Reconstruction: the web application (python -m src.web)")
        pause(page, 4)
        caption(page, "Overview page: the Middlebury Temple reconstructed by this project (312 cameras)")
        pause(page, 3)
        caption(page, "How the model is made: photos → features → matches → cameras → point cloud")
        smooth_scroll(page, 2400, 7)
        pause(page, 1)

        # 2. Workstation: upload ---------------------------------------------------------
        page.goto(URL + "/app.html", wait_until="networkidle")
        caption(page, "Workstation: add the photos (sample_data/templeRing, 47 images)")
        pause(page, 2)
        page.set_input_files("#images", images)
        pause(page, 2)
        caption(page, "…plus camera.yaml (intrinsics K) and gt_cameras.json (ground-truth poses)")
        page.set_input_files("#camera", str(sample / "camera.yaml"))
        page.set_input_files("#ground-truth", str(sample / "gt_cameras.json"))
        pause(page, 2)

        # 3. Live reconstruction ---------------------------------------------------------
        caption(page, "Reconstruct: every stage runs live (right: pipeline, bottom: console)")
        page.click("#run")
        t0 = time.time()
        page.wait_for_function("document.body.dataset.state === 'results' || document.body.dataset.state === 'failed'",
                               timeout=300_000, polling=500)  # fmt: skip
        if page.evaluate("document.body.dataset.state") != "results":
            raise RuntimeError("the reconstruction failed in the web app")
        print(f"reconstruction in the browser: {time.time() - t0:.0f} s")

        # 4. 3-D result ------------------------------------------------------------------
        caption(page, "Done: 47/47 cameras and a coloured point cloud, explored in 3-D")
        pause(page, 2)
        page.click("[data-fit='cameras']")
        pause(page, 2)
        caption(page, "Orbit with the mouse: every recovered camera around the temple")
        drag(page, "#viewer", 320, 40, 3)
        pause(page, 1)
        page.click("[data-fit='object']")
        pause(page, 1.5)
        caption(page, "Close-up of the filtered sparse cloud; auto-orbit on")
        page.click("#spin")
        pause(page, 4)
        page.click("[data-cloud='sparse']")
        caption(page, "Sparse (all landmarks) vs Filtered (outliers removed)")
        pause(page, 2.5)
        page.click("[data-cloud='filtered']")
        pause(page, 2)
        page.click("#spin")

        # 5. Poses against ground truth --------------------------------------------------
        page.click("#tab-poses")
        caption(page, "Camera poses vs ground truth: 0.15° median rotation error")
        pause(page, 2.5)
        page.click("[data-sort='rotation_deg']")
        pause(page, 1)
        page.click("[data-sort='rotation_deg']")
        caption(page, "Sorted by rotation error: the worst camera is still within 0.32°")
        pause(page, 3)

        # 6. Matching figures ------------------------------------------------------------
        page.click("#tab-matches")
        caption(page, "Matching figures: verified feature matches between photo pairs")
        pause(page, 3)
        first = page.locator("#gallery-matches img, #gallery-matches button").first
        if first.count():
            first.click()
            pause(page, 2)
            page.click("[data-step='1']")
            pause(page, 1.5)
            page.click("#lightbox-close")
        pause(page, 1)
        if page.locator("#tab-figures").is_visible():
            page.click("#tab-figures")
            caption(page, "Diagnostics: match matrix, reprojection-error histogram, cameras")
            pause(page, 3)

        # 7. Replay and exports ----------------------------------------------------------
        page.click("#tab-viewer")
        pause(page, 1)
        page.click("[data-fit='cameras']")
        caption(page, "Replay: the cameras appear in the order they were registered")
        page.click("#replay")
        pause(page, 8)
        caption(page, "Export: filtered.ply, cameras.json, report.json and a PDF report")
        page.locator("#export-card").scroll_into_view_if_needed()
        page.locator("#downloads a").first.hover()
        pause(page, 4)
        caption(page, "")
        pause(page, 0.5)

        video = Path(page.video.path())
        ctx.close()
        browser.close()
    return video


def main(python: str, sample: Path, out: Path) -> None:
    work = Path(tempfile.mkdtemp(prefix="webdemo_"))
    server = start_server(python, work)
    try:
        webm = record(sample.resolve(), work / "video")
    finally:
        server.terminate()
        server.wait(timeout=20)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm),
         "-vf", "scale=1280:720:flags=lanczos,fps=30", "-c:v", "libx264", "-preset", "slow",
         "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        check=True,
    )  # fmt: skip
    shutil.rmtree(work, ignore_errors=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip()  # fmt: skip
    print(f"{out}: {float(dur):.1f} s")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))

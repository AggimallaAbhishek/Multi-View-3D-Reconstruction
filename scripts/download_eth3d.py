"""Download an ETH3D high-resolution multi-view training scene with ground-truth cameras.

Usage: python scripts/download_eth3d.py pipes   (needs the `data` extra: py7zr)

Takes the undistorted DSLR images (pinhole, one camera per scene, 6220 x 4141) and writes
into data/raw/eth3d_<scene>/ (git-ignored):
  *.JPG             the images
  camera.yaml       the shared intrinsics, for camera.calibration_file
  gt_cameras.json   ground-truth K, R, t per image (world -> camera, P = K [R | t])

The ground truth is ETH3D's (laser-scan-registered, in COLMAP's text format), read with
src.reconstruction.colmap_runner.read_text_model, which converts COLMAP's +0.5 pixel
centres to ours. Source: https://www.eth3d.net/datasets (Schöps et al., CVPR 2017).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.ground_truth import GT_CAMERAS_FILE  # noqa: E402
from src.pointcloud.export import write_cameras_json  # noqa: E402
from src.reconstruction.colmap_runner import read_text_model  # noqa: E402

URL = "https://www.eth3d.net/data/{scene}_dslr_undistorted.7z"
SCENES = (
    "courtyard", "delivery_area", "electro", "facade", "kicker", "meadow", "office",
    "pipes", "playground", "relief", "relief_2", "terrace", "terrains",
)  # fmt: skip


def download(url: str, path: Path, attempts: int = 3, timeout: float = 60) -> bool:
    """Stream ``url`` to ``path``, retrying on network errors."""
    for attempt in range(1, attempts + 1):
        print(f"Downloading {url} (attempt {attempt}/{attempts}) ...")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, path.open("wb") as f:
                shutil.copyfileobj(response, f, length=1 << 20)
            return True
        except OSError as exc:  # URLError, timeouts and connection resets are OSErrors
            print(f"  failed: {exc}")
    return False


def main(scene: str = "pipes") -> int:
    try:
        import py7zr
    except ImportError:
        print('needs py7zr: uv pip install -e ".[data]"')
        return 2
    if scene not in SCENES:
        print(f"unknown scene {scene!r}; choose from {', '.join(SCENES)}")
        return 2
    target = ROOT / "data" / "raw" / f"eth3d_{scene}"
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive = Path(tmp) / "scene.7z"
        if not download(URL.format(scene=scene), archive):
            if not any(target.iterdir()):
                target.rmdir()
            return 1
        with py7zr.SevenZipFile(archive) as z:
            names = z.getnames()
            wanted = [n for n in names if "/dslr_images_undistorted/" in n and n.endswith(".JPG")
                      or "/dslr_calibration_undistorted/" in n and n.endswith(".txt")]  # fmt: skip
            z.extract(path=Path(tmp) / "x", targets=wanted)
        extracted = Path(tmp) / "x"
        for image in extracted.rglob("*.JPG"):
            image.replace(target / image.name)  # flattened; never trust archive paths
        calibration = next(extracted.rglob("dslr_calibration_undistorted"))
        model = read_text_model(calibration)

    cameras = {Path(name).name: camera for name, camera in model.cameras.items()}
    names = sorted(cameras)
    sizes = []
    for name in names:
        with Image.open(target / name) as im:
            sizes.append(im.size)
    write_cameras_json(
        target / GT_CAMERAS_FILE, [cameras[n] for n in names], names, sizes,
        frame="ETH3D ground truth (world -> camera: x_c = R X + t), original resolution",
    )  # fmt: skip
    K = cameras[names[0]].K
    width, height = sizes[0]
    (target / "camera.yaml").write_text(
        "# Intrinsics from the ETH3D calibration (undistorted images, shared by all).\n"
        f"width: {width}\nheight: {height}\n"
        f"fx: {K[0, 0]}\nfy: {K[1, 1]}\ncx: {K[0, 2]}\ncy: {K[1, 2]}\n"
        "dist: [0.0, 0.0, 0.0, 0.0, 0.0]\n",
        encoding="utf-8",
    )
    print(f"{len(cameras)} images with ground truth in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))

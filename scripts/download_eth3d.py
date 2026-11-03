"""Download an ETH3D high-resolution multi-view training scene with ground-truth cameras.

Usage: python scripts/download_eth3d.py pipes [--distorted]   (needs the `data` extra: py7zr)

Takes the undistorted DSLR images (pinhole, one camera per scene, 6220 x 4141) and writes
into data/raw/eth3d_<scene>/ (git-ignored) — or, with ``--distorted``, the original JPGs
with their lens distortion into data/raw/eth3d_<scene>_jpg/, with only the pinhole part of
the calibration in camera.yaml (a test of the pipeline's distortion self-calibration):
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.ground_truth import write_ground_truth  # noqa: E402
from src.reconstruction.colmap_runner import read_text_model  # noqa: E402

URL = "https://www.eth3d.net/data/{scene}_{variant}.7z"
VARIANTS = {  # name -> (archive variant, image folder, calibration folder)
    "undistorted": ("dslr_undistorted", "dslr_images_undistorted", "dslr_calibration_undistorted"),
    "distorted": ("dslr_jpg", "dslr_images", "dslr_calibration_jpg"),
}
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


def main(scene: str = "pipes", distorted: bool = False) -> int:
    try:
        import py7zr
    except ImportError:
        print('needs py7zr: uv pip install -e ".[data]"')
        return 2
    if scene not in SCENES:
        print(f"unknown scene {scene!r}; choose from {', '.join(SCENES)}")
        return 2
    variant, image_dir, calibration_dir = VARIANTS["distorted" if distorted else "undistorted"]
    target = ROOT / "data" / "raw" / f"eth3d_{scene}{'_jpg' if distorted else ''}"
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive = Path(tmp) / "scene.7z"
        if not download(URL.format(scene=scene, variant=variant), archive):
            if not any(target.iterdir()):
                target.rmdir()
            return 1
        with py7zr.SevenZipFile(archive) as z:
            names = z.getnames()
            wanted = [n for n in names if f"/{image_dir}/" in n and n.endswith(".JPG")
                      or f"/{calibration_dir}/" in n and n.endswith(".txt")]  # fmt: skip
            z.extract(path=Path(tmp) / "x", targets=wanted)
        extracted = Path(tmp) / "x"
        for image in extracted.rglob("*.JPG"):
            image.replace(target / image.name)  # flattened; never trust archive paths
        calibration = next(extracted.rglob(calibration_dir))
        model = read_text_model(calibration)

    cameras = {Path(name).name: camera for name, camera in model.cameras.items()}
    write_ground_truth(
        target, cameras,
        frame="ETH3D ground truth (world -> camera: x_c = R X + t), original resolution",
        header="ETH3D calibration's pinhole part only: the images' lens distortion\n"
        "(THIN_PRISM_FISHEYE) is left for the pipeline to estimate." if distorted
        else "Intrinsics from the ETH3D calibration (undistorted images, shared by all).",
    )  # fmt: skip
    print(f"{len(cameras)} images with ground truth in {target}")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--distorted"]
    sys.exit(main(*args[:1], distorted="--distorted" in sys.argv[1:]))

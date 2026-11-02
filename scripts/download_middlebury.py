"""Download a Middlebury multi-view dataset with ground-truth cameras.

Usage: python scripts/download_middlebury.py [templeRing|templeSparseRing|dinoRing|dinoSparseRing]

Extracts the images into data/raw/<name>/ (git-ignored) and writes, next to them:
  camera.yaml       the shared intrinsics, for camera.calibration_file
  gt_cameras.json   ground-truth K, R, t per image (world -> camera, P = K [R | t])

Source: https://vision.middlebury.edu/mview/data/ (Seitz et al., CVPR 2006).
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.ground_truth import load_middlebury_par, write_ground_truth  # noqa: E402

URL = "https://vision.middlebury.edu/mview/data/data/{name}.zip"
DATASETS = ("templeRing", "templeSparseRing", "dinoRing", "dinoSparseRing")


def main(name: str = "templeRing") -> int:
    if name not in DATASETS:
        print(f"unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
        return 2
    target = ROOT / "data" / "raw" / name
    print(f"Downloading {URL.format(name=name)} ...")
    # The server rejects urllib's default User-Agent (HTTP 403).
    request = urllib.request.Request(
        URL.format(name=name), headers={"User-Agent": "Mozilla/5.0 (mv3d dataset download)"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    target.mkdir(parents=True, exist_ok=True)
    for member in archive.infolist():
        filename = Path(member.filename).name  # flatten; never trust archive paths
        if member.is_dir() or not filename or filename.startswith("."):
            continue
        (target / filename).write_bytes(archive.read(member))

    par = next(target.glob("*_par.txt"))
    cameras = load_middlebury_par(par)
    write_ground_truth(
        target, cameras,
        frame="Middlebury ground truth (world -> camera: x_c = R X + t), original resolution",
        header="Intrinsics from the Middlebury calibration (shared by all images).",
    )  # fmt: skip
    print(f"{len(cameras)} images with ground truth in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))

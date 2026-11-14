# Multi-View 3D Reconstruction (Classical Structure from Motion)

**Group {{GROUP}} · {{COURSE}} · final project**

> [!IMPORTANT]
> 🎥 **Demo video (Google Drive, 990 MB, too large for the zip): [https://drive.google.com/drive/folders/1nIqmeL9LmYx48zrCKLDIJQHBeRR3svRe?usp=sharing](https://drive.google.com/drive/folders/1nIqmeL9LmYx48zrCKLDIJQHBeRR3svRe?usp=sharing)**  
> A screen recording of the working web application: uploading the sample photos, the live reconstruction, and exploring the 3-D result.

Photos of a static object in, then camera poses, a coloured sparse point cloud, a dense point cloud and a surface mesh out. The pipeline is classical incremental Structure from Motion: SIFT features, ratio-test matching, epipolar verification, relative pose, N-view triangulation, PnP registration, bundle adjustment, outlier filtering, stereo depth maps with multi-view fusion, and Poisson meshing. It uses Python, OpenCV, NumPy/SciPy and Open3D. There is a command-line tool and a local web app.

On the included `sample_data/templeRing` (47 photos), the pipeline recovers **47/47 cameras** with a **0.15° median rotation error** against the ground-truth cameras, a 0.21 px mean reprojection error, 9,907 filtered points, a 1.84 M-point dense cloud and a 306 k-triangle mesh, all in about 30 s.

| Member | Roll No. / ID | Email |
|---|---|---|
{{MEMBERS_TABLE}}

### The web application (`python -m src.web`)

**Overview page:** the Middlebury Temple (312 photos) as this project reconstructed it, inside a ring of its recovered cameras.

![overview page: the reconstructed temple turning inside a ring of its cameras](code/docs/images/web_intro.jpg)

**Workstation, while it runs:** the 47 photos of `sample_data/templeRing` uploaded with `camera.yaml` and `gt_cameras.json`. The pipeline column ticks through the stages live and the console logs each one.

![workstation while the reconstruction runs](code/docs/images/web_running.png)

**Workstation, the result:** 47/47 cameras and 9,907 filtered points in the 3-D view, with each stage's numbers and times, the summary, and the comparison against ground truth (0.15° median rotation error).

![workstation with the finished templeRing reconstruction](code/docs/images/web_ui.png)

| Camera poses against ground truth | Matching figures |
|---|---|
| ![camera poses tab: per-photo rotation and position error](code/docs/images/web_poses.png) | ![matches tab: match matrices and the first camera pair](code/docs/images/web_matches.png) |

---

## 1. What is in this folder

```
GroupX_3dcv_project/
├── report.pdf          IEEE two-column project report
├── README.md           this file
├── requirements.txt    Python packages (pinned versions)
├── code/               all source code
│   ├── src/            the pipeline (data, features, geometry, reconstruction, dense,
│   │                   pointcloud, evaluation, visualization, web app); CLI: src/cli.py
│   ├── configs/        default.yaml (every threshold) + experiment specs
│   ├── scripts/        dataset download, evaluation against ground truth, experiments
│   ├── tests/          579 pytest tests (synthetic and rendered ground truth)
│   ├── notebooks/      analysis notebooks
│   ├── docs/           methodology, architecture, experiments (full results), figures
│   ├── pyproject.toml
│   └── README.md       detailed project documentation (all commands and options)
└── sample_data/
    ├── templeRing/     47 photos (640x480) + camera.yaml (K) + gt_cameras.json (true poses)
    └── dinoRing/       48 photos of a dark, low-texture object, with the same files
```

The sample data is the Middlebury Multi-View Stereo benchmark (Seitz et al., CVPR 2006), which is free for research use. `camera.yaml` holds the intrinsics and `gt_cameras.json` the ground-truth poses used for evaluation.

## 2. Requirements

| | macOS | Windows |
|---|---|---|
| OS | macOS 12 or newer, Apple Silicon or Intel | Windows 10/11, 64-bit |
| Python | **3.12** (64-bit) | **3.12** (64-bit) |
| Extra system package | `libusb` (Open3D 0.20 links against it): `brew install libusb` | Microsoft Visual C++ Redistributable 2015–2022 (x64), usually already installed |
| Disk | ~1.5 GB for the virtual environment | ~1.5 GB for the virtual environment |
| RAM | 8 GB | 8 GB |
| GPU | not needed | not needed |
| Internet | to install packages; the web page also loads three.js and fonts from a CDN | same |

Python 3.13 may work, but 3.14 does not: Open3D publishes no wheels for it. All packages are pinned in `requirements.txt`. Use **one** OpenCV wheel only (`opencv-contrib-python`) and do not mix it with `opencv-python`.

---

## 3. macOS: install and run

Open **Terminal** in the unzipped `GroupX_3dcv_project` folder.

```bash
# 3.1  System prerequisites (Homebrew: https://brew.sh)
brew install python@3.12 libusb

# 3.2  Virtual environment + packages (inside code/)
cd code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r ../requirements.txt

# 3.3  Check the installation
python scripts/verify_setup.py

# 3.4  Quick demo on the sample data (about 30 s)
python -m src.cli dense ../sample_data/templeRing
#   -> outputs/<timestamp>_dense_sift/
#        point_clouds/filtered.ply   sparse cloud (cleaned)
#        point_clouds/dense.ply      dense cloud
#        meshes/reconstructed_mesh.ply
#        camera_poses/cameras.json   every recovered camera
#        reconstruction/reconstruction.json, figures/*.png

# 3.5  Accuracy against the ground-truth cameras
python scripts/evaluate_against_ground_truth.py ../sample_data/templeRing

# 3.6  Interactive 3-D viewer (Open3D window; replace <run> with the folder printed in 3.4)
python -m src.cli view outputs/<run>/point_clouds/filtered.ply \
    --cameras outputs/<run>/camera_poses/cameras.json
python -m src.cli view outputs/<run>/point_clouds/dense.ply

# 3.7  Web application -> opens http://127.0.0.1:8000/ in your browser
python -m src.web
#   Open "Workstation", drop in all files of sample_data/templeRing
#   (photos + camera.yaml + gt_cameras.json) and press Reconstruct. Ctrl+C stops the server.

# 3.8  Tests (about 3-5 min)
pytest -q
```

## 4. Windows: install and run

Install **Python 3.12 (64-bit)** from https://www.python.org/downloads/windows/ and tick *"Add python.exe to PATH"*. Alternatively, run `winget install Python.Python.3.12`. Then open **PowerShell** in the unzipped `GroupX_3dcv_project` folder.

```powershell
# 4.1  Virtual environment + packages (inside code\)
cd code
py -3.12 -m venv .venv
# Allow the activation script once per user if PowerShell blocks it:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1          # in cmd.exe instead:  .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r ..\requirements.txt

# 4.2  UTF-8 console output (the tools print symbols such as ° and ->)
$env:PYTHONUTF8 = "1"               # in cmd.exe:  set PYTHONUTF8=1

# 4.3  Check the installation
python scripts\verify_setup.py

# 4.4  Quick demo on the sample data (about 30-60 s)
python -m src.cli dense ..\sample_data\templeRing
#   -> outputs\<timestamp>_dense_sift\point_clouds\{filtered,dense}.ply,
#      meshes\reconstructed_mesh.ply, camera_poses\cameras.json

# 4.5  Accuracy against the ground-truth cameras
python scripts\evaluate_against_ground_truth.py ..\sample_data\templeRing

# 4.6  Interactive 3-D viewer (replace <run> with the folder printed in 4.4)
python -m src.cli view outputs\<run>\point_clouds\filtered.ply --cameras outputs\<run>\camera_poses\cameras.json

# 4.7  Web application -> opens http://127.0.0.1:8000/
python -m src.web

# 4.8  Tests
pytest -q
```

## 5. More commands (both systems)

Use `/` on macOS and `\` on Windows. Every command takes `--config <yaml>`, and every threshold is in `code/configs/default.yaml`.

| Goal | Command (run inside `code/`) |
|---|---|
| Sparse reconstruction only | `python -m src.cli reconstruct ../sample_data/dinoRing` |
| Your own photos (20–80 overlapping photos of a static object, about 10° apart) | `python -m src.cli dense path/to/photos` (intrinsics come from EXIF or a guess; refined automatically) |
| Calibrate your camera with a checkerboard | `python -m src.cli calibrate path/to/checkerboard --pattern 9x6 --square 0.025 --out configs/camera.yaml` |
| Individual stages with diagnostic figures | `python -m src.cli features / matches / verify / pose / triangulate / two-view <images>` |
| Use ORB instead of SIFT | `python -m src.cli reconstruct ../sample_data/templeRing --detector orb` |
| Download the full datasets | `python scripts/download_middlebury.py templeRing` (or `temple`, `dinoSparseRing`, …) and `python scripts/download_eth3d.py pipes` (needs `pip install py7zr`) |
| Reproduce all experiments of the report (about 14 min) | `python scripts/download_middlebury.py templeRing` then `python scripts/run_experiments.py` |
| Dense quality check (silhouettes) | `python scripts/evaluate_dense.py ../sample_data/templeRing` |

Outputs go to `code/outputs/`. Features and matches are cached in `code/outputs/cache/`, so a repeated run is faster. Delete that folder to start clean.

## 6. Expected results on the sample data

| | templeRing | dinoRing |
|---|---|---|
| Cameras registered | 47 / 47 | 48 / 48 |
| Rotation error vs ground truth (median / max) | 0.15° / 0.32° | 0.15° / 0.30° |
| Mean reprojection error | 0.21 px | 0.44 px |
| Filtered sparse points | ~9,900 | ~3,500 |
| Dense points | ~1.84 M | ~0.7 M |
| Runtime (Apple M4, first run) | ~30 s | ~30 s |

Runtimes on other machines may be 2 to 4 times longer. The full results tables (9 experiments, COLMAP comparison, ETH3D) are in `report.pdf` and `code/docs/experiments.md`.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| macOS: `Library not loaded: …libusb-1.0.0.dylib` on `import open3d` | `brew install libusb` |
| `No matching distribution found for open3d==0.20.0` | You are on the wrong Python version (for example 3.14) or on 32-bit Python. Recreate the venv with Python 3.12, 64-bit. |
| Windows: `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or use `cmd.exe` and `.venv\Scripts\activate.bat` |
| Windows: `UnicodeEncodeError` in the console | `$env:PYTHONUTF8 = "1"` (PowerShell) or `set PYTHONUTF8=1` (cmd) |
| Windows: `DLL load failed` when importing `cv2` or `open3d` | Install the Microsoft Visual C++ Redistributable (x64) and make sure only `opencv-contrib-python` is installed (`pip list \| findstr opencv`) |
| `cv2` errors after installing other packages | `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python` then `pip install opencv-contrib-python==5.0.0.93` |
| No PNG renders (`Render unavailable`), for example on a remote or headless machine | Harmless. The PLY files are still written. Open them with the `view` command, MeshLab or CloudCompare. |
| Web app: port 8000 already in use | `python -m src.web --port 8010` |
| Commands cannot find `configs/default.yaml` or `src` | Run every command from inside the `code/` folder. |

## 8. External code, libraries and credits

This project was **not** built on an existing reconstruction repository (it is not based on COLMAP, OpenMVG, Nerfstudio or similar). The pipeline logic is our own: E decomposition, cheirality, initial-pair selection, N-view triangulation, tracks, incremental registration and its fallback, the bundle adjustment residuals and Jacobian, structure refinement, filtering, depth-map fusion, evaluation and the web app. The report and the code mark external parts with **external**. They are:

| External component | Used for |
|---|---|
| OpenCV (`opencv-contrib-python`) | image I/O, SIFT/ORB, 2-NN distances, USAC/RANSAC estimators, `solvePnPRansac`, `stereoRectify`, `StereoSGBM`, checkerboard calibration |
| SciPy | `least_squares` solver (bundle adjustment, E refinement), k-d trees |
| Open3D | PLY I/O, statistical/radius outlier removal, Poisson meshing, viewer and offscreen renderer |
| FastAPI, uvicorn, three.js (CDN), Google Fonts | web app server and 3-D view in the browser |
| COLMAP (optional, not installed by `requirements.txt`) | **baseline only** for the comparison in the report (`scripts/run_colmap_baseline.py`) |
| DISK + LightGlue via kornia/torch (optional) | learned-feature experiment E9. kornia downloads the authors' pretrained weights on first use, so no weights are included or needed for the demo. |

No model weights or large datasets are included in this archive. The full datasets are downloaded by `code/scripts/download_middlebury.py` and `download_eth3d.py`. Datasets: Middlebury MVS (Seitz et al., 2006) and ETH3D (Schöps et al., 2017, CC BY-NC-SA 4.0).

## 9. Report and video

- `report.pdf`: IEEE conference format. It covers the introduction, problem statement, literature review, objectives, methodology, results and discussion (including ablations and failure cases), conclusion, author contributions and references.
- **Demo video: [▶ Watch on Google Drive](https://drive.google.com/drive/folders/1nIqmeL9LmYx48zrCKLDIJQHBeRR3svRe?usp=sharing)** (<https://drive.google.com/drive/folders/1nIqmeL9LmYx48zrCKLDIJQHBeRR3svRe?usp=sharing>). It is a screen recording of the working web application (`python -m src.web`): uploading `sample_data/templeRing`, the live reconstruction, and exploring the 3-D result. At 990 MB it is above the archive's 200 MB limit, so it is linked here instead of being included as `demo.mp4`.

# Final submission: how to build it

This folder produces `GroupX_3dcv_project/` and `GroupX_3dcv_project.zip` (X = your group number) for the course upload.

## 1. Fill in your group (once)

Edit `group.json`:
- `group`: your group number. It replaces X in the folder and zip name.
- `department`, `university`: one short line each.
- `members`: 1 to 5 entries, each with `name`, `id` (roll number), `email` and `contribution` (what that person worked on). This becomes the Author Contributions table at the end of the report. Delete any unused entries.

## 2. Build (about 2 minutes)

```bash
source .venv/bin/activate                          # the project venv
python finalsubmission/build_submission.py         # code, sample data, demo.mp4, report.pdf, README, zip
python finalsubmission/build_submission.py --skip-demo   # faster: reuse build/demo.mp4
```

Requirements: `tectonic` (LaTeX) and `ffmpeg` on PATH (`brew install tectonic ffmpeg`), plus `data/raw/templeRing` and `data/raw/dinoRing` (`python scripts/download_middlebury.py templeRing`, likewise `dinoRing`).

## 3. What is where

| Path | What |
|---|---|
| `group.json` | group number, members, contributions: the only file you need to edit |
| `report/main.tex` | the IEEE report source. `authors.tex` and `contributions.tex` are generated from `group.json`. It compiles on Overleaf too: upload `main.tex`, the two generated `.tex` files and `figures/`. |
| `templates/README.md`, `templates/requirements.txt` | the README (macOS and Windows steps) and requirements for the zip |
| `demo.mp4` | the 1-2 minute web-app video (the original screen recording at 1.9x speed, 1080p: `ffmpeg -i <recording> -vf "setpts=PTS/1.9,fps=30,scale=1920:-2" -an -c:v libx264 -crf 24 -pix_fmt yuv420p demo.mp4`); the build copies it into the package |
| `tools/record_webapp_demo.py` | optional: records a scripted web-app session automatically (Playwright + Chrome) |
| `tools/make_demo.py` | renders `build/renders.mp4` from a `dense` run (Open3D offscreen); only the report's sparse/dense/mesh stills come from it |
| `build/` | the demo run and the video (generated) |
| `GroupX_3dcv_project/`, `*.zip` | the deliverable (generated) |

## 4. Before uploading

1. Unzip the zip into a new folder, then follow its `README.md` on `sample_data/`. This was verified on macOS with a fresh Python 3.12 venv: the install, `verify_setup.py`, `dense` and `evaluate_against_ground_truth.py` on templeRing, `reconstruct` on dinoRing, the web app, and `pytest`.
2. Open `report.pdf`. Check the author block and the Author Contributions table.
3. Play `demo.mp4`.
4. Tag every group member on the portal.

# CLAUDE.md

Multi-view 3D reconstruction (classical SfM) in Python with OpenCV, NumPy/SciPy, and Open3D. See `README.md` for the overview and `docs/ROADMAP.md` for phase plans.

## Working rules

- Work one phase at a time, in order. Check `docs/ROADMAP.md` for the current phase's tasks and exit criteria; tick the README checklist when a phase is done.
- Geometry code is test-first against `tests/synthetic.py` (known `K`, `R`, `t`, 3D points). Don't touch real photos for a module until its synthetic tests pass.
- All thresholds live in `configs/default.yaml`, never hard-coded.
- Conventions: world→camera `x_c = R X + t`; camera center `C = -R.T @ t`; points `(N, 3)` / `(N, 2)` float64; images BGR uint8. Resizing an image must rescale `K`.
- Python 3.12 venv in `.venv` (managed with `uv`). Only `opencv-contrib-python`, never mixed OpenCV wheels.
- Don't commit `data/raw/`, `data/processed/`, or `outputs/`.
- Clearly label anything provided by COLMAP or other external tools as external.

## Skills to load

| Task | Skill |
|---|---|
| OpenCV features, matching, RANSAC, PnP | `computer-vision-opencv` |
| Anything importing `open3d` | `open3d-python` (prefer tensor API; legacy only when needed) |
| Writing tests | `python-testing-patterns`, `tdd` |
| Diagnostic figures / evaluation plots | `matplotlib` |
| Notebooks in `notebooks/` | `jupyter-notebook` (its helper script lives in `.claude/skills/jupyter-notebook/scripts/new_notebook.py`) |
| Wrong poses, scattered clouds, failing tests | `systematic-debugging` |

## Commands

```bash
source .venv/bin/activate
python scripts/verify_setup.py
pytest -q
```

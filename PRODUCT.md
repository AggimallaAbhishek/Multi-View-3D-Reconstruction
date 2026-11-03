# Product

Product truth for design work on the web interface (read by the `impeccable` design
skill). Facts confirmed with the author in brainstorming on 2026-11-03; see
`docs/superpowers/plans/2026-11-03-web-redesign.md`.

## Platform

web

## Stack

FastAPI serving static HTML, CSS and ES modules; three.js 0.160 from a CDN for the 3-D
viewer; Google Fonts (Fraunces, IBM Plex Sans, IBM Plex Mono). No build
step, no frontend framework. Local only: `python -m src.web` on 127.0.0.1, one job at a
time, no accounts.

## Users

1. **An audience at a presentation** (viva, portfolio review, talk): watches the author
   drive the page on a projector or a laptop screen. Needs to understand in minutes what
   the project does, how, and how well. Primary user of the intro page.
2. **The author, live during the demo and day to day**: uploads photos, runs the
   reconstruction, inspects the result and its quality numbers. Primary user of the
   workstation.

## Product Purpose

Turn ordinary overlapping photos of an object into a 3-D point cloud and recover every
camera position, with a classical structure-from-motion pipeline written from scratch in
Python (OpenCV, NumPy/SciPy, Open3D), and show that it works: against ground truth and
against COLMAP.

## Positioning

A research and teaching project, not a product for sale. Its credibility comes from
measured results on public benchmarks (Middlebury templeRing / dinoRing, ETH3D) and from
being explicit about what is its own work and what is external.

## Operating Context

- Two surfaces: an **intro page** (Persuade / Experience: explain the process, the
  architecture and the results, then send the visitor to the workstation) and a
  **workstation** (Operate: upload, run, inspect).
- Often projected: large type, strong contrast, the 3-D result readable from a distance.
- Runs offline except fonts and three.js; the featured reconstruction is committed.

## Capabilities and Constraints

- Shows only what the pipeline really computes: registered cameras, points, reprojection
  error, per-stage counts and timings, ground-truth camera errors, diagnostic figures.
  No invented metrics (no coverage %, GPU meters, projects, accounts).
- Real data is a sparse cloud (~10 k points for templeRing) with the cameras on a ring
  ~10× wider than the object; the design must flatter that, not a photo-real model.
- Progress arrives per stage (polled), not per camera.

## Brand Commitments

- Graphite + survey teal: near-black graphite surfaces, one teal accent (#3CC2B5) used
  for cameras, the primary action and active states only.
- Survey-instrument language: fiducial crosses at viewport corners, monospace data
  strips, hairline rules.
- Fraunces (display serif, a nod to the classical subject), IBM Plex Sans (body), IBM Plex Mono (numbers).

## Evidence on Hand

- `docs/frontend-design/*.png`: two generated workstation mock-ups (teal and indigo).
- `.superpowers/demo/`: working prototype of the workstation with real templeRing data.
- `docs/images/`: figures and screenshots of results.

## Product Principles

1. Real data or nothing: every number on screen comes from a run.
2. Label external work (COLMAP, OpenCV, Open3D, three.js, learned models) as external.
3. The reconstruction is the hero; the interface recedes around it.

## Accessibility & Inclusion

WCAG 2.2 AA contrast for all text; keyboard operable (viewer focusable, every control a
button or input); visible focus; `prefers-reduced-motion` stops auto-orbit and replay
animation; usable at 390 px wide.

# Frontend design references

Generated mock-up images for the web interface redesign (Phase 12 follow-up). These are
references for look and feel, not specifications; the brainstorming decisions are:

- **Audience:** presentation / showcase first.
- **Layout H1:** full-screen hero with the featured templeRing reconstruction (point cloud
  and its ring of recovered cameras), survey-frame details (fiducial crosses, monospace
  data strip), then a five-frame "how it was made" strip below.
- **Showcase data:** committed under `src/web/static/showcase/`.
- **Reach:** the whole app (landing, upload, running, results) in one design system.
- **Colour:** to be chosen — graphite + survey teal, or warm ink + sandstone amber.

Drop the generated images here, named by screen, e.g. `01-landing.png`, `02-upload.png`,
`03-running.png`, `04-results.png`.

## Prompts

Paste the **base prompt** first, then add one **screen prompt**. Image models render text
poorly, so judge layout, colour and mood, not the exact words.

### Base prompt (shared by every screen)

> High-fidelity UI design of a desktop web app, 1440×900, dark theme. The product is
> "Multi-View 3D Reconstruction": a scientific tool that turns ordinary overlapping photos
> of an object into a 3-D point cloud and recovers every camera position (classical
> structure from motion). Mood: precise, technical, calm, like an aerial-survey instrument
> or a photogrammetry lab, not a gaming UI. Visual language: near-black graphite
> background (#0B0F12), slightly lighter panels (#121A1F), thin 1px hairline borders
> (#24313A), off-white text (#E6EDEF), grey-blue secondary text (#8FA3A9), and a single
> teal accent (#3CC2B5) used only for camera markers, the primary button and active
> states. Small fiducial crosshair marks (+) at the corners of the main viewport, like a
> survey camera frame. Typography: a wide, heavy grotesque for headlines (Archivo
> Expanded style), a highly legible sans for body text, and a monospace font for numbers
> and labels in small uppercase with letter-spacing. Lots of negative space, strict grid,
> no gradients except a faint radial glow behind the 3-D object, no stock photos, no
> emoji, no 3-D glossy buttons. Flat, crisp, editorial, Dribbble/Awwwards quality.

For the amber variant, replace the colour sentence with:

> warm ink-black background (#0E0C0A), panels (#17140F), hairline borders (#2E2820),
> warm off-white text (#F3EDE3), taupe secondary text (#A89C8A), and a single sandstone
> amber accent (#E9A84B) taken from the colour of the reconstructed temple

### Screen prompts

**01 — Landing hero (first screen of a presentation)**

> Full-bleed hero filling the viewport: in the centre, a sparse 3-D point cloud of a
> small sandstone Greek temple model (beige and brown points, about 10,000 dots) floating
> in dark space, surrounded by a large tilted elliptical ring of 47 small glowing teal
> camera markers (tiny wireframe camera frustums pointing at the temple). Fiducial
> crosshairs at the four corners of the hero. Top bar: small monospace uppercase label
> "MULTI-VIEW 3D RECONSTRUCTION" on the left, "FEATURED · TEMPLERING · DRAG TO ORBIT" on the
> right. Bottom-left: huge bold headline in two lines "47 photos in. A temple out." with
> the word "temple" in the accent colour, and a one-line subtitle beneath. Along the bottom
> edge, a thin data strip of monospace stat cells separated by hairlines: CAMERAS 47/47 ·
> POINTS 10 076 · ROTATION ERROR 0.15° · REPROJECTION 0.13 px, and at the right end a solid
> accent button "Reconstruct your photos →".

**01b — "How it was made" strip (below the hero)**

> Section below the hero titled in small monospace caps "HOW IT WAS MADE · FIVE FRAMES".
> Five equal cards in a row like a photographer's contact sheet, each with a hairline
> border, a tiny accent label "FRAME 01"…"FRAME 05", a thumbnail and two lines of text:
> 01 Photos (a fan of five overlapping photos of the temple), 02 Features (a dark photo
> with hundreds of small teal keypoint circles), 03 Matches (two photos side by side
> joined by many thin parallel teal lines), 04 Cameras (the elliptical ring of camera
> markers around the object), 05 Cloud (the finished coloured point cloud).

**02 — Upload**

> Upload screen in the same style. A large dashed-border drop zone with fiducial
> crosshairs at its corners and the line "Drop overlapping photos of one object or
> scene", "or choose files" as an accent link, and a short hint about two-thirds overlap.
> Right column: "Optional files" with two compact file pickers (camera.yaml for camera
> intrinsics, gt_cameras.json for ground-truth poses), each with one line of help text,
> and a full-width accent button "Reconstruct". Below: a grid of small photo thumbnails
> with filenames in monospace and a "24 photos selected · Clear" bar.

**03 — Running**

> Progress screen in the same style. Title "Reconstructing" with a monospace run id.
> A vertical list of ten pipeline stages (Load photos, Detect features, Match features,
> Verify matches, First camera pair, Triangulate, Two-view cloud, Register photos + bundle
> adjustment, Remove outliers, Write results), each row with a status mark (accent tick
> when done, a small spinner on the current one, grey when waiting) and a monospace time
> on the right. To the side, a small live preview of the point cloud growing and a
> thin overall progress bar.

**04 — Results**

> Results screen in the same style. A wide dark 3-D viewer with fiducial corners showing
> the coloured point cloud and teal camera frustums, small segmented controls floating on
> top (Filtered / Sparse, a point-size slider, Fit to Object / Cameras). Under the viewer,
> the monospace data strip (Photos registered 47/47, Points, Reprojection error, Focal
> length, Features SIFT · ratio test, Time). Below: a panel "Against ground truth" with
> three summary numbers and a compact sortable table of cameras with thin inline accent
> error bars and the worst row highlighted; then a gallery of diagnostic figures (match
> matrix heat map, reprojection histogram, camera plot) as white-background cards; then
> download chips for the point cloud, camera poses and report.

## Tips

- Generate 3–4 variations per screen and keep only the ones you like.
- Tools that follow layout instructions well: ChatGPT image generation, Midjourney
  (`--ar 16:10 --style raw`), Ideogram (best with text), or Google's Imagen.
- Note in the file name what you like about an image, e.g.
  `01-landing-love-the-ring.png`, or add a line below in this README.

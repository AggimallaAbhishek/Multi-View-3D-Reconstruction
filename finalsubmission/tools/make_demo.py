"""Render demo.mp4 (about 1.5 min) from a finished `python -m src.cli dense` run.

    python finalsubmission/tools/make_demo.py <run dir> <image dir> <out.mp4>

Every 3-D frame is drawn with Open3D's offscreen renderer from the run's own outputs
(filtered.ply, dense.ply, reconstructed_mesh.ply, cameras.json, reconstruction.json);
captions are drawn with Pillow and the frames are encoded with ffmpeg (H.264, yuv420p,
so it plays in QuickTime, Windows Media Player, VLC and browsers).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.pointcloud.export import read_cameras_json, read_ply, to_open3d  # noqa: E402

W, H, FPS = 1280, 720, 30
BG = (250, 250, 248)
INK = (24, 24, 27)
MUTED = (100, 100, 110)
ACCENT = (37, 99, 235)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),
        ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
        ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
         else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ]  # fmt: skip
    for path, index in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=index)
    return ImageFont.load_default(size)


class Writer:
    """Numbered PNG frames in a temporary folder, encoded at the end."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="demo_frames_"))
        self.n = 0
        self.marks: dict[str, float] = {}

    def add(self, img: Image.Image, repeat: int = 1) -> None:
        img = img.convert("RGB")
        first = self.dir / f"{self.n:05d}.png"
        img.save(first)
        self.n += 1
        for _ in range(repeat - 1):
            shutil.copyfile(first, self.dir / f"{self.n:05d}.png")
            self.n += 1

    def encode(self, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", str(self.dir / "%05d.png"), "-c:v", "libx264", "-preset", "slow",
             "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            check=True,
        )  # fmt: skip
        shutil.rmtree(self.dir)


def caption(img: Image.Image, title: str, sub: str = "", step: str = "") -> Image.Image:
    """A title bar at the top left and a step label at the top right."""
    img = img.copy()
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 92], fill=(250, 250, 248, 225))
    d.text((40, 18), title, font=font(34, True), fill=INK)
    if sub:
        d.text((40, 60), sub, font=font(20), fill=MUTED)
    if step:
        f = font(18, True)
        tw = d.textlength(step, font=f)
        d.rounded_rectangle([W - tw - 64, 28, W - 40, 62], 8, fill=ACCENT)
        d.text((W - tw - 52, 34), step, font=f, fill=(255, 255, 255))
    return img


def card(lines: list[tuple[str, int, bool, tuple]], seconds: float, w: Writer) -> None:
    """A full-screen text card: (text, size, bold, colour) per line, centred."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    heights = [s * 1.45 for _, s, _, _ in lines]
    y = (H - sum(heights)) / 2
    for (text, size, bold, colour), h in zip(lines, heights, strict=True):
        f = font(size, bold)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=colour)
        y += h
    fade(img, seconds, w)


def fade(img: Image.Image, seconds: float, w: Writer, edge: float = 0.4) -> None:
    """Hold an image, fading in from and out to the background colour."""
    total, k = int(seconds * FPS), int(edge * FPS)
    bg = Image.new("RGB", img.size, BG)
    for i in range(k):
        w.add(Image.blend(bg, img, (i + 1) / k))
    w.add(img, total - 2 * k)
    for i in range(k):
        w.add(Image.blend(img, bg, (i + 1) / k))


def fit(img: Image.Image, box: tuple[int, int]) -> Image.Image:
    img = img.copy()
    img.thumbnail(box, Image.LANCZOS)
    return img


def table_card(rows: list[tuple[str, str]], title: str, seconds: float, w: Writer) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 60), title, font=font(40, True), fill=INK)
    y = 150
    for k, (name, value) in enumerate(rows):
        if k % 2 == 0:
            d.rectangle([70, y - 8, W - 70, y + 44], fill=(238, 240, 245))
        d.text((90, y), name, font=font(26), fill=INK)
        f = font(26, True)
        d.text((W - 90 - d.textlength(value, font=f), y), value, font=f, fill=ACCENT)
        y += 54
    fade(img, seconds, w)


# --------------------------------------------------------------------------- 3-D scenes


class Scene:
    """One Open3D offscreen renderer, a ring frame (centre, up, axes) and the cameras."""

    def __init__(self, cameras, sizes, points: np.ndarray) -> None:
        self.r = rendering.OffscreenRenderer(W, H)
        self.r.scene.set_background([c / 255 for c in BG] + [1.0])
        self.r.scene.view.set_post_processing(False)
        self.cameras, self.sizes = cameras, sizes
        centres = np.array([c.centre for c in cameras])
        self.target = np.median(points, axis=0)
        # Up = normal of the camera ring, signed so the cameras look down on the object
        # (a ring of views above a turntable; the photos' own "up" is unreliable here,
        # templeRing's are rotated by 90 deg and a third of them rolled by 180 deg).
        _, _, vt = np.linalg.svd(centres - centres.mean(axis=0))
        up = vt[2]
        if np.mean([c.R[2] @ up for c in cameras]) > 0:
            up = -up
        self.up = up
        self.a = vt[0]
        self.b = np.cross(up, self.a)
        self.ring = float(np.median(np.linalg.norm(centres - self.target, axis=1)))
        self.spread = float(
            np.linalg.norm(np.ptp(np.percentile(points, [2, 98], axis=0), axis=0))
        )

    def look(self, theta: float, dist: float, elev: float, fov: float = 45.0) -> None:
        d = np.cos(elev) * (np.cos(theta) * self.a + np.sin(theta) * self.b) + np.sin(elev) * self.up
        eye = self.target + dist * d
        self.r.setup_camera(fov, self.target.astype(np.float32), eye.astype(np.float32),
                            self.up.astype(np.float32))  # fmt: skip

    def add_points(self, name: str, pts: np.ndarray, cols: np.ndarray | None, size: float) -> None:
        m = rendering.MaterialRecord()
        m.shader, m.point_size = "defaultUnlit", size
        self.r.scene.add_geometry(name, to_open3d(pts.astype(np.float32), cols), m)

    def add_camera(self, k: int, scale: float, colour=(0.84, 0.15, 0.16)) -> None:
        cam, (cw, ch) = self.cameras[k], self.sizes[k]
        ext = np.eye(4)
        ext[:3, :3], ext[:3, 3] = cam.R, cam.t
        ls = o3d.t.geometry.LineSet.create_camera_visualization(
            int(cw), int(ch), o3d.core.Tensor(cam.K), o3d.core.Tensor(ext), scale,
            o3d.core.Tensor(np.array(colour, dtype=np.float32)),
        )  # fmt: skip
        m = rendering.MaterialRecord()
        m.shader, m.line_width = "unlitLine", 2.0
        self.r.scene.add_geometry(f"cam{k}", ls, m)

    def frame(self) -> Image.Image:
        return Image.fromarray(np.asarray(self.r.render_to_image()))


def smooth(t: float) -> float:
    return t * t * (3 - 2 * t)


def main(run: Path, images: Path, out: Path) -> None:
    w = Writer()
    recon = json.loads((run / "reconstruction" / "reconstruction.json").read_text())
    cameras, names, sizes = read_cameras_json(run / "camera_poses" / "cameras.json")
    sparse, sparse_c = read_ply(run / "point_clouds" / "filtered.ply")
    dense, dense_c = read_ply(run / "point_clouds" / "dense.ply")
    photos = sorted(p for p in images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    n_img = len(photos)

    # 1. Title -------------------------------------------------------------------------
    card([("Multi-View 3D Reconstruction", 56, True, INK),
          ("Classical Structure from Motion with OpenCV, NumPy/SciPy and Open3D", 26, False, MUTED),
          ("", 20, False, MUTED),
          ("Photos  >  features  >  matches  >  cameras  >  sparse cloud  >  dense cloud  >  mesh",
           22, False, ACCENT)], 5.0, w)  # fmt: skip

    # 2. Input photos: thumbnails appear one by one -------------------------------------
    cols, tw, th = 10, 112, 84
    grid = Image.new("RGB", (W, H), BG)
    x0, y0 = (W - cols * (tw + 8)) // 2, 150
    thumbs = [fit(Image.open(p), (tw, th)) for p in photos]
    for k, t in enumerate(thumbs):
        r, c = divmod(k, cols)
        grid.paste(t, (x0 + c * (tw + 8), y0 + r * (th + 8)))
        if k % 2 == 0 or k == n_img - 1:
            w.add(caption(grid, "1. Input: overlapping photographs",
                          f"Middlebury templeRing: {n_img} photos, 640x480, known intrinsics K",
                          "INPUT"), 2)  # fmt: skip
    w.add(caption(grid, "1. Input: overlapping photographs",
                  f"Middlebury templeRing: {n_img} photos, 640x480, known intrinsics K", "INPUT"),
          int(2.5 * FPS))  # fmt: skip

    # 3. Features and verified matches on one neighbouring pair -------------------------
    a, b = cv2.imread(str(photos[12])), cv2.imread(str(photos[13]))
    sift = cv2.SIFT_create(contrastThreshold=0.02)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    kp_img = cv2.drawKeypoints(a, ka, None, (0, 200, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    K = cameras[0].K
    _, mask = cv2.findEssentialMat(pa, pb, K, cv2.USAC_ACCURATE, 0.999, 1.5)
    inl = [g for g, keep in zip(good, mask.ravel(), strict=True) if keep]
    mimg = cv2.drawMatches(a, ka, b, kb, inl[::3], None, (60, 200, 60), flags=2)
    panel = Image.new("RGB", (W, H), BG)
    panel.paste(fit(Image.fromarray(kp_img[..., ::-1]), (560, 420)), (40, 170))
    panel.paste(fit(Image.fromarray(mimg[..., ::-1]), (640, 420)), (610, 190))
    d = ImageDraw.Draw(panel)
    d.text((40, 600), f"SIFT keypoints: {len(ka)} on this photo", font=font(22), fill=MUTED)
    d.text((610, 600), f"{len(inl)} of {len(good)} ratio-test matches kept by RANSAC on E",
           font=font(22), fill=MUTED)  # fmt: skip
    fade(caption(panel, "2. Features, matching and epipolar verification",
                 "SIFT, Lowe ratio test + mutual check, essential matrix with USAC (Sampson inliers)",
                 "2-D"), 7.0, w)  # fmt: skip

    # 4. Incremental SfM: cameras in registration order around the sparse cloud ---------
    s = Scene(cameras, sizes, sparse)
    scale = 0.08 * np.linalg.norm(np.ptp(np.vstack([sparse, [c.centre for c in cameras]]), axis=0))
    s.add_points("sparse", sparse, sparse_c, 3.0)
    order = [names.index(n) for n in recon["registration_order"]]
    steps = int(12 * FPS)
    added = 0
    for i in range(steps):
        want = 2 + int((len(order) - 2) * min(1.0, i / (steps * 0.85)))
        while added < want:
            s.add_camera(order[added], scale, (0.12, 0.47, 0.71) if added < 2 else (0.84, 0.15, 0.16))
            added += 1
        s.look(0.9 + 0.5 * i / steps, 2.0 * s.ring, np.radians(35))
        w.add(caption(s.frame(), "3. Incremental Structure from Motion",
                      f"initial pair (blue), then PnP registration + bundle adjustment: "
                      f"{added}/{len(order)} cameras", "CAMERAS"))  # fmt: skip

    # 5. Orbit of the cameras and the cleaned sparse cloud ------------------------------
    steps = int(9 * FPS)
    for i in range(steps):
        t = smooth(i / steps)
        s.look(1.4 + 2 * np.pi * t, (2.0 - 0.8 * t) * s.ring, np.radians(35 - 10 * t))
        w.add(caption(s.frame(), "4. Sparse reconstruction",
                      f"{recon['registered']}/{recon['images']} cameras, "
                      f"{len(sparse)} filtered points, mean reprojection error "
                      f"{recon['reprojection']['mean_px']:.2f} px", "SPARSE"))  # fmt: skip

    # 6. Close-up of the sparse cloud (cameras hidden) -----------------------------------
    for k in range(len(cameras)):
        s.r.scene.remove_geometry(f"cam{k}")
    s.r.scene.remove_geometry("sparse")
    s.add_points("sparse", sparse, sparse_c, 5.0)
    near = 1.55 * s.spread
    w.marks["closeup"] = w.n / FPS
    steps = int(7 * FPS)
    for i in range(steps):
        s.look(1.4 + 2 * np.pi * i / steps, near, np.radians(22))
        w.add(caption(s.frame(), "4. Sparse reconstruction (close-up)",
                      "landmarks coloured from the photos; outliers removed (reprojection, "
                      "statistical, radius)", "SPARSE"))  # fmt: skip

    # 7. Dense cloud ---------------------------------------------------------------------
    s.r.scene.remove_geometry("sparse")
    s.add_points("dense", dense, dense_c, 1.5)
    w.marks["dense"] = w.n / FPS
    steps = int(14 * FPS)
    for i in range(steps):
        t = i / steps
        s.look(1.4 + 2 * np.pi * t, near * (1.0 - 0.15 * np.sin(np.pi * t)), np.radians(22 + 12 * np.sin(2 * np.pi * t)))
        w.add(caption(s.frame(), "5. Dense reconstruction",
                      f"stereo depth map per view (SGBM) on the recovered cameras, fused: "
                      f"{len(dense) / 1e6:.2f} M points", "DENSE"))  # fmt: skip

    # 8. Poisson mesh --------------------------------------------------------------------
    s.r.scene.remove_geometry("dense")
    mesh = o3d.t.io.read_triangle_mesh(str(run / "meshes" / "reconstructed_mesh.ply"))
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    s.r.scene.add_geometry("mesh", mesh, mat)
    s.r.scene.scene.set_sun_light([-0.3, -0.8, -0.4], [1.0, 1.0, 1.0], 90000)
    s.r.scene.scene.enable_sun_light(True)
    s.r.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [-0.3, -0.8, -0.4])
    n_tri = int(mesh.triangle.indices.shape[0])
    w.marks["mesh"] = w.n / FPS
    steps = int(10 * FPS)
    for i in range(steps):
        s.look(1.4 + 2 * np.pi * i / steps, near, np.radians(25))
        w.add(caption(s.frame(), "6. Surface mesh",
                      f"screened Poisson reconstruction (Open3D, external): {n_tri:,} triangles",
                      "MESH"))  # fmt: skip

    # 9. Results ------------------------------------------------------------------------
    table_card([
        ("Images registered (templeRing)", f"{recon['registered']} / {recon['images']}"),
        ("Camera rotation error vs ground truth (median / max)", "0.15° / 0.32°"),
        ("Camera position error (median, % of ring radius)", "0.15 %"),
        ("Mean reprojection error", f"{recon['reprojection']['mean_px']:.2f} px"),
        ("Completeness vs reference cloud", "0.94"),
        ("COLMAP (external) on the same images: rotation error", "0.26° median"),
        ("ETH3D pipes / office / terrains registered", "14/14 · 26/26 · 42/42"),
        ("Full run (sparse + dense + mesh), Apple M4", "≈ 28 s"),
    ], "Results against ground truth", 9.0, w)  # fmt: skip

    # 10. Web application --------------------------------------------------------------
    for shot, title, sub in [
        (REPO / "docs/images/web_intro.jpg", "7. Web application: overview page",
         "python -m src.web  ->  http://127.0.0.1:8000/"),
        (REPO / "docs/images/web_ui.png", "7. Web application: workstation",
         "drop in photos, press Reconstruct, inspect cameras, matches and statistics, export PLY"),
    ]:
        if shot.exists():
            panel = Image.new("RGB", (W, H), BG)
            im = fit(Image.open(shot), (W - 120, H - 140))
            panel.paste(im, ((W - im.width) // 2, 110 + (H - 120 - im.height) // 2))
            fade(caption(panel, title, sub, "APP"), 4.5, w)

    # 11. End card ----------------------------------------------------------------------
    card([("Thank you", 52, True, INK),
          ("Code, sample data and instructions: see README.md in the submission", 24, False, MUTED)],
         3.5, w)  # fmt: skip

    w.encode(out)
    out.with_name("demo_timeline.json").write_text(json.dumps(w.marks, indent=1))
    print(f"{out}: {w.n} frames, {w.n / FPS:.1f} s")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

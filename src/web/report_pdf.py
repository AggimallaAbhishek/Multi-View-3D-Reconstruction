"""The downloadable report of a web run: an A4 PDF.

Built on request from what the run folder holds (``report.json`` is required; the rest —
``reconstruction.json``, ``registration.csv``, ``feature_statistics.csv``, figures,
thumbnails, the environment and the config — is used when present), so runs of earlier
servers get one too. The body gives the results, the figures and the accuracy; long
per-camera and per-photo tables, the environment and the full configuration go to the
appendices. Rendered with ReportLab (external).
"""

from __future__ import annotations

import csv
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    CondPageBreak,
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.run import read_json, run_stamp_of
from src.web.report import REPORT_NAME

# Figures of the run folder (written by src/outputs.py), in reading order.
FIGURES = (
    ("figures/filtered.png", "Filtered point cloud"),
    ("reconstruction/cameras.png", "Recovered camera positions"),
    ("reconstruction/reprojection_histogram.png", "Reprojection error of every observation"),
    ("matches/match_matrix.png", "Feature matches between every pair of photos"),
    ("verification/verified_matrix.png", "Matches kept after geometric verification"),
    ("pose/initial_pair.png", "The first camera pair"),
    ("triangulation/structure.png", "Points triangulated from the first pair"),
)
STAGE_LABELS = {
    "dataset": "Load and check photos",
    "features": "Detect features",
    "matches": "Match features",
    "verify": "Verify geometry",
    "pose": "First camera pair",
    "triangulate": "Triangulate first points",
    "cloud": "Two-view cloud",
    "reconstruct": "Register cameras, adjust",
    "filter": "Remove outliers",
    "outputs": "Write results",
}
DETECTORS = {"sift": "SIFT", "orb": "ORB", "disk": "DISK (external)"}
MATCHERS = {
    "bruteforce": "ratio test",
    "flann": "ratio test (FLANN)",
    "lightglue": "LightGlue (external)",
}
CALIBRATION_SOURCES = {"file": "camera.yaml", "exif": "EXIF", "fallback": "a guess"}

PDF_MEDIA_TYPE = "application/pdf"

# --- look -------------------------------------------------------------------------------

INK = colors.HexColor("#1b2328")
MUTED = colors.HexColor("#5d6a72")
LINE = colors.HexColor("#d5dadd")
ACCENT = colors.HexColor("#1f8a80")
TINT = colors.HexColor("#eef6f5")
ZEBRA = colors.HexColor("#f6f7f8")
WARN_BG = colors.HexColor("#fff4dc")
WARN = colors.HexColor("#7a5000")
BAD = colors.HexColor("#a8322a")

# DejaVu, shipped with matplotlib (external): unlike the PDF's built-in fonts it covers
# Greek and Cyrillic too, so photo names in those scripts print as named.
_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
SANS, BOLD, MONO = "DejaVuSans", "DejaVuSans-Bold", "DejaVuSansMono"
for _name in (SANS, BOLD, MONO):
    if _name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(_name, str(_FONT_DIR / f"{_name}.ttf")))
pdfmetrics.registerFontFamily(SANS, normal=SANS, bold=BOLD, italic=SANS, boldItalic=BOLD)

MARGIN = 18 * mm
WIDTH = A4[0] - 2 * MARGIN  # of the text column

_BASE = ParagraphStyle("base", fontName=SANS, fontSize=9.5, leading=13, textColor=INK)
S = {
    "eyebrow": ParagraphStyle(
        "eyebrow", _BASE, fontName=BOLD, fontSize=8, textColor=ACCENT, leading=10
    ),
    "title": ParagraphStyle("title", _BASE, fontName=BOLD, fontSize=24, leading=29),
    "meta": ParagraphStyle("meta", _BASE, textColor=MUTED),
    "h1": ParagraphStyle(
        "h1",
        _BASE,
        fontName=BOLD,
        fontSize=14,
        leading=18,
        spaceBefore=14,
        spaceAfter=6,
        textColor=INK,
    ),  # fmt: skip
    "h2": ParagraphStyle(
        "h2",
        _BASE,
        fontName=BOLD,
        fontSize=10.5,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
    ),  # fmt: skip
    "body": _BASE,
    "note": ParagraphStyle("note", _BASE, textColor=MUTED, fontSize=8.5, leading=11.5),
    "cell": ParagraphStyle("cell", _BASE, fontSize=8.5, leading=11),
    "cell_num": ParagraphStyle("cell_num", _BASE, fontSize=8.5, leading=11, alignment=TA_RIGHT),
    "cell_bad": ParagraphStyle("cell_bad", _BASE, fontSize=8.5, leading=11, textColor=BAD),
    "head": ParagraphStyle("head", _BASE, fontName=BOLD, fontSize=8, leading=10, textColor=MUTED),
    "head_num": ParagraphStyle(
        "head_num",
        _BASE,
        fontName=BOLD,
        fontSize=8,
        leading=10,
        textColor=MUTED,
        alignment=TA_RIGHT,
    ),  # fmt: skip
    "kpi_label": ParagraphStyle("kpi_label", _BASE, fontSize=8, leading=10, textColor=MUTED),
    "kpi": ParagraphStyle("kpi", _BASE, fontName=BOLD, fontSize=16, leading=20),
    "warn": ParagraphStyle("warn", _BASE, textColor=WARN, fontSize=9, leading=12),
    "caption": ParagraphStyle("caption", _BASE, fontSize=8.5, leading=11, textColor=MUTED),
    "code": ParagraphStyle("code", _BASE, fontName=MONO, fontSize=7, leading=8.6),
}


def report_filename(run_dir: str | Path) -> str:
    return f"{Path(run_dir).name}_report.pdf"


def render_report_pdf(run_dir: str | Path, *, compress: bool = True) -> bytes:
    """The PDF report of ``run_dir``; ``FileNotFoundError`` without ``report.json``.
    ``compress=False`` leaves the page streams readable (tests search them for text)."""
    run_dir = Path(run_dir)
    path = run_dir / REPORT_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    run = _Run(run_dir, read_json(path))

    body = [
        *_title(run),
        *_summary(run),
        *_warnings(run),
        *_inputs(run),
        *_pipeline(run),
        *_reconstruction(run),
        *_accuracy(run),
        *_figures(run),
    ]
    appendices = [
        part
        for part in (
            _registration(run),
            _per_camera(run),
            _features_per_photo(run),
            _environment(run),
            _configuration(run),
            _files(run),
        )
        if part
    ]
    story = body
    for letter, (title, flowables) in zip("ABCDEFGH", appendices, strict=False):
        story += [CondPageBreak(60 * mm), _h1(f"Appendix {letter}. {title}"), *flowables]

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN + 6 * mm,
        title=f"Reconstruction report: {run_dir.name}",
        author="Multi-view 3D reconstruction",
        subject="Structure-from-motion run report",
        pageCompression=1 if compress else 0,
    )
    doc.build(story, canvasmaker=_numbered_canvas(run_dir.name))
    return buffer.getvalue()


# --- the run's files --------------------------------------------------------------------


class _Run:
    def __init__(self, run_dir: Path, report: dict[str, Any]):
        self.dir = run_dir
        self.report = report
        self.rec = _optional_json(run_dir / "reconstruction" / "reconstruction.json") or {}
        self.env = _optional_json(run_dir / "environment.json")
        self.sections = 0

    def numbered(self, title: str) -> Paragraph:
        self.sections += 1
        return _h1(f"{self.sections}. {title}")


# --- body -------------------------------------------------------------------------------


def _title(run: _Run) -> list[Flowable]:
    stamp = run_stamp_of(run.dir.name)
    when = (
        datetime.strptime(stamp, "%Y%m%d-%H%M%S").strftime("%d %B %Y, %H:%M")
        if stamp
        else (run.env or {}).get("timestamp", "")
    )
    meta = f"Run {_mono(run.dir.name)}"
    if when:
        meta += f"  ·  {escape(when)}"
    return [
        Paragraph("MULTI-VIEW 3D RECONSTRUCTION", S["eyebrow"]),
        Spacer(0, 3),
        Paragraph("Reconstruction report", S["title"]),
        Paragraph(meta, S["meta"]),
        Spacer(0, 6),
        _Rule(WIDTH, ACCENT, 1.5),
        Spacer(0, 10),
    ]


def _summary(run: _Run) -> list[Flowable]:
    r = run.report
    images = r.get("images", {})
    points = r.get("points") or {}
    tiles = [
        ("Cameras registered", f"{images.get('registered', 0)} / {images.get('total', 0)}"),
        ("Points (filtered)", _int(points.get("filtered", points.get("sparse")))),
        ("Mean reprojection", _px((r.get("reprojection") or {}).get("mean_px"))),
        ("Total time", _seconds(r.get("total_s"))),
    ]
    gt = r.get("ground_truth") or {}
    if (gt.get("rotation_deg") or {}).get("median") is not None:
        tiles[3:3] = [("Median rotation", _deg(gt["rotation_deg"]["median"]))]
    cells = [
        [Paragraph(escape(k), S["kpi_label"]), Paragraph(escape(v), S["kpi"])] for k, v in tiles
    ]
    table = Table([cells], colWidths=[WIDTH / len(tiles)] * len(tiles))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), TINT),
                ("LINEBEFORE", (1, 0), (-1, -1), 0.75, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return [table, Spacer(0, 4)]


def _warnings(run: _Run) -> list[Flowable]:
    warnings = run.report.get("warnings") or []
    if not warnings:
        return []
    rows = [[Paragraph("<b>Warnings</b>", S["warn"])]] + [
        [Paragraph(f"•  {escape(w)}", S["warn"])] for w in warnings
    ]
    table = Table(rows, colWidths=[WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WARN_BG),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 7),
            ]
        )
    )
    return [Spacer(0, 8), table]


def _inputs(run: _Run) -> list[Flowable]:
    r = run.report
    images, features, cal = r.get("images", {}), r.get("features", {}), r.get("calibration", {})
    sources = ", ".join(CALIBRATION_SOURCES.get(s, s) for s in cal.get("sources", []))
    rows: list[tuple[str, Any]] = [
        ("Photos", _int(images.get("total"))),
        ("Feature detector", DETECTORS.get(features.get("detector"), features.get("detector"))),
        ("Matcher", MATCHERS.get(features.get("matcher"), features.get("matcher"))),
        ("Intrinsics from", sources or "—"),
    ]
    K = cal.get("K")
    if K:
        rows += [("Initial focal length", _focal(K)), ("Principal point", _principal(K))]
    refined = (run.rec.get("intrinsics") or {}).get("K")
    scale = cal.get("focal_scale")
    if refined:
        rows.append(("Refined focal length", _focal(refined)))
    elif scale is not None and K:
        rows.append(("Refined focal length", _focal([[v * scale for v in row] for row in K[:2]])))
    if scale is not None:
        rows.append(("Focal scale (bundle adjustment)", f"{scale:.4f}"))
    if run.rec.get("self_calibration"):
        rows.append(("Lens distortion", _compact(run.rec["self_calibration"])))
    if images.get("unregistered"):
        rows.append(("Not registered", ", ".join(images["unregistered"])))
    out = [run.numbered("Input and calibration"), _kv(rows)]
    thumbs = _thumbnails(run.dir)
    if thumbs:
        out += [_h2("Photos"), thumbs]
    return out


def _pipeline(run: _Run) -> list[Flowable]:
    stages = run.report.get("stages") or {}
    produced = {
        "features": _stage_features(stages.get("features")),
        "matches": _stage_matches(stages.get("matches")),
        "verify": _stage_verification(stages.get("verification")),
        "pose": " and ".join(stages["initial_pair"]) if stages.get("initial_pair") else "",
        "reconstruct": _stage_reconstruction(stages.get("reconstruction")),
        "filter": _stage_filter(stages.get("filter")),
    }
    timings = run.report.get("timings") or {}
    if not timings:
        return []
    total = sum(timings.values()) or 1.0
    rows = [
        [
            STAGE_LABELS.get(stage, stage),
            produced.get(stage, ""),
            _seconds(seconds),
            _Bar(seconds / total, 30 * mm),
        ]
        for stage, seconds in timings.items()
    ]
    return [
        run.numbered("Pipeline"),
        _table(
            ["Stage", "Result", "Time", "Share of time"],
            rows,
            [46 * mm, None, 20 * mm, 32 * mm],
            numeric=(2,),
        ),
    ]


def _reconstruction(run: _Run) -> list[Flowable]:
    r = run.report
    points, reproj = r.get("points"), r.get("reprojection")
    if points is None and reproj is None:
        return []
    rows: list[tuple[str, Any]] = []
    pair = (r.get("stages") or {}).get("initial_pair")
    if pair:
        rows.append(("Initial pair", " and ".join(pair)))
    if points:
        rows += [
            ("Sparse points", _int(points.get("sparse"))),
            ("Filtered points", _int(points.get("filtered"))),
        ]
    if "observations" in run.rec:
        rows.append(("Observations", _int(run.rec["observations"])))
    if "mean_track_length" in run.rec:
        rows.append(("Mean track length", f"{run.rec['mean_track_length']:.2f} views"))
    out: list[Flowable] = [run.numbered("Reconstruction"), _kv(rows)]
    if reproj:
        cells = [_int(reproj.get("n"))] + [
            _px(reproj.get(k)) for k in ("mean_px", "median_px", "p90_px", "max_px")
        ]
        out += [
            _h2("Reprojection error"),
            _table(
                ["Observations", "Mean", "Median", "90th percentile", "Max"],
                [cells],
                numeric=(0, 1, 2, 3, 4),
            ),
        ]
    steps = (run.rec.get("filtering") or {}).get("steps") or []
    if steps:
        rows_ = [
            [
                str(s.get("name", "")).replace("_", " "),
                _int(s.get("before")),
                _int(s.get("removed")),
            ]
            for s in steps
        ]
        out += [
            _h2("Outlier filtering"),
            _table(["Step", "Points before", "Removed"], rows_, numeric=(1, 2)),
        ]
    return out


def _accuracy(run: _Run) -> list[Flowable]:
    gt = run.report.get("ground_truth")
    if not gt:
        return []
    head = run.numbered("Accuracy against ground truth")
    if "error" in gt:
        return [head, Paragraph(escape(gt["error"]), S["body"])]
    rot, centre = gt.get("rotation_deg") or {}, gt.get("centre") or {}
    keys = ("median", "p90", "max")
    summary = _table(
        ["", "Median", "90th percentile", "Max"],
        [
            ["Rotation error", *(_deg(rot.get(k)) for k in keys)],
            ["Position error", *(_pct(centre.get(k)) for k in keys)],
        ],
        numeric=(1, 2, 3),
    )
    note = Paragraph(
        f"{_int(gt.get('cameras'))} recovered cameras compared with the uploaded ground truth "
        "after a similarity alignment (rotation, translation, scale). Position error is a share of "
        "the true cameras' spread. Every camera is listed in the appendix.",
        S["note"],
    )
    out: list[Flowable] = [head, note, Spacer(0, 6), summary]
    worst = sorted(gt.get("per_camera", []), key=lambda c: -(c.get("rotation_deg") or 0))[:5]
    if worst:
        out += [
            _h2("Largest rotation errors"),
            _table(
                ["Photo", "Rotation error", "Position error"],
                _camera_rows(worst),
                numeric=(1, 2),
            ),
        ]
    return out


def _figures(run: _Run) -> list[Flowable]:
    present = [(run.dir / p, c) for p, c in FIGURES if (run.dir / p).is_file()]
    if not present:
        return []
    out: list[Flowable] = [PageBreak(), run.numbered("Figures")]
    images = [(im, c) for p, c in present if (im := _image(p, WIDTH, 105 * mm)) is not None]
    for n, (image, caption) in enumerate(images, 1):
        out.append(
            KeepTogether(
                [
                    image,
                    Spacer(0, 3),
                    Paragraph(f"<b>Figure {n}.</b> {escape(caption)}", S["caption"]),
                    Spacer(0, 12),
                ]
            )
        )
    return out


# --- appendices: (title, flowables) or None ---------------------------------------------


def _registration(run: _Run) -> tuple[str, list[Flowable]] | None:
    rows = _csv(run.dir / "reconstruction" / "registration.csv")
    if not rows:
        return None
    ok = sum(r.get("success") == "True" for r in rows)
    data, failed = [], []
    for k, r in enumerate(rows):
        success = r.get("success") == "True"
        if not success:
            failed.append(k)
        data.append(
            [
                r.get("attempt", ""),
                r.get("image", ""),
                r.get("method", ""),
                "yes" if success else "no",
                f"{r.get('inliers', '')} / {r.get('correspondences', '')}",
                r.get("reason", ""),
            ]
        )
    note = Paragraph(
        f"{ok} of {len(rows)} registration attempts succeeded, in the order the cameras "
        "were added. Inliers are of the 2D–3D correspondences of each attempt.",
        S["note"],
    )
    table = _table(
        ["#", "Photo", "Method", "Registered", "Inliers", "Reason"],
        data,
        [10 * mm, 40 * mm, 16 * mm, 22 * mm, 24 * mm, None],
        numeric=(0, 4),
        bad_rows=failed,
    )
    return "Camera registration", [note, Spacer(0, 6), table]


def _per_camera(run: _Run) -> tuple[str, list[Flowable]] | None:
    gt = run.report.get("ground_truth") or {}
    cameras = gt.get("per_camera")
    if not cameras:
        return None
    return "Ground-truth error of every camera", [
        _table(["Photo", "Rotation error", "Position error"], _camera_rows(cameras), numeric=(1, 2))
    ]


def _camera_rows(cameras: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [c.get("image", ""), _deg(c.get("rotation_deg")), _pct(c.get("centre"))] for c in cameras
    ]


def _features_per_photo(run: _Run) -> tuple[str, list[Flowable]] | None:
    rows = _csv(run.dir / "features" / "feature_statistics.csv")
    if not rows:
        return None
    data = [
        [
            r.get("image", ""),
            _int(_number(r.get("n_features"))),
            _pct(_number(r.get("coverage")), 1),
            f"{r.get('time_ms', '')} ms",
        ]
        for r in rows
    ]
    return "Features per photo", [
        _table(["Photo", "Features", "Image coverage", "Time"], data, numeric=(1, 2, 3))
    ]


def _environment(run: _Run) -> tuple[str, list[Flowable]] | None:
    env = run.env
    if not env:
        return None
    rows = [
        ("Python", env.get("python")),
        ("Platform", env.get("platform")),
        ("Git commit", (env.get("git_commit") or "")[:12] or None),
    ] + list((env.get("packages") or {}).items())
    return "Environment", [_kv([(k, v) for k, v in rows if v])]


def _configuration(run: _Run) -> tuple[str, list[Flowable]] | None:
    path = run.dir / "config.yaml"
    if not path.is_file():
        return None
    note = Paragraph(f"Every setting the run used ({_mono('config.yaml')}).", S["note"])
    return "Configuration", [note, Spacer(0, 4), Preformatted(path.read_text(), S["code"])]


def _files(run: _Run) -> tuple[str, list[Flowable]]:
    files = sorted(
        rel
        for p in run.dir.rglob("*")
        if p.is_file() and not (rel := p.relative_to(run.dir).as_posix()).startswith("thumbnails/")
    )
    return "Run folder", [
        Paragraph(
            f"Files written to {_mono(run.dir.name)}:",
            S["note"],
        ),
        Spacer(0, 4),
        Preformatted("\n".join(files), S["code"]),
    ]


# --- building blocks --------------------------------------------------------------------


def _mono(text: str) -> str:
    """``text`` in the monospaced font, escaped, for a Paragraph."""
    return f"<font face='{MONO}'>{escape(text)}</font>"


def _h1(text: str) -> Paragraph:
    return Paragraph(escape(text), S["h1"])


def _h2(text: str) -> Paragraph:
    return Paragraph(escape(text), S["h2"])


def _cell(value: Any, style: str = "cell") -> Flowable:
    if isinstance(value, Flowable):
        return value
    return Paragraph(escape("" if value is None else str(value)), S[style])


def _table(
    headers: list[str],
    rows: list[list[Any]],
    widths: list[float | None] | None = None,
    numeric: tuple[int, ...] = (),
    bad_rows: tuple[int, ...] | list[int] = (),
) -> Table:
    """A data table: header repeated on every page, zebra rows, ``numeric`` columns
    right-aligned, ``bad_rows`` (indices into ``rows``) in red."""
    head = [_cell(h, "head_num" if k in numeric else "head") for k, h in enumerate(headers)]
    body = []
    for i, row in enumerate(rows):
        style = "cell_bad" if i in bad_rows else "cell"
        body.append(
            [
                _cell(v, "cell_num" if k in numeric and i not in bad_rows else style)
                for k, v in enumerate(row)
            ]
        )
    if widths is not None:
        fixed = sum(w for w in widths if w is not None)
        free = [w for w in widths if w is None]
        share = (WIDTH - fixed) / len(free) if free else 0
        widths = [share if w is None else w for w in widths]
    else:
        widths = [WIDTH / len(headers)] * len(headers)
    table = Table([head, *body], colWidths=widths, repeatRows=1)
    style = [
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    style += [("BACKGROUND", (0, r), (-1, r), ZEBRA) for r in range(2, len(rows) + 1, 2)]
    table.setStyle(TableStyle(style))
    return table


def _kv(rows: list[tuple[str, Any]]) -> Table:
    data = [
        [Paragraph(escape(k), S["note"]), Paragraph(escape(str(v)), S["body"])]
        for k, v in rows
        if v is not None
    ]
    table = Table(data, colWidths=[55 * mm, WIDTH - 55 * mm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def _thumbnails(run_dir: Path) -> Table | None:
    folder = run_dir / "thumbnails"
    paths = sorted(folder.glob("*.jpg")) if folder.is_dir() else []
    per_row, gap = 6, 2 * mm
    size = (WIDTH - gap * (per_row - 1)) / per_row
    images = [im for p in paths if (im := _image(p, size, size * 0.75)) is not None]
    if not images:
        return None
    rows = [images[k : k + per_row] for k in range(0, len(images), per_row)]
    rows[-1] += [""] * (per_row - len(rows[-1]))
    table = Table(rows, colWidths=[size + gap] * per_row)
    table.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), gap),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), gap),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    table.hAlign = "LEFT"
    return table


def _image(path: Path, max_w: float, max_h: float) -> Image | None:
    """``path`` scaled to fit ``max_w`` × ``max_h``, aspect kept; None if unreadable."""
    try:
        with PILImage.open(path) as im:
            w, h = im.size
    except (OSError, ValueError):
        return None
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


class _Rule(Flowable):
    def __init__(self, width: float, color: colors.Color, thickness: float):
        super().__init__()
        self.width, self.color, self.thickness = width, color, thickness
        self.height = thickness

    def draw(self) -> None:
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


class _Bar(Flowable):
    """A horizontal bar filled to ``fraction`` of ``width``, with the percentage after it."""

    def __init__(self, fraction: float, width: float):
        super().__init__()
        self.fraction, self.width, self.height = max(0.0, min(1.0, fraction)), width, 8

    def draw(self) -> None:
        bar = self.width - 11 * mm
        self.canv.setFillColor(ZEBRA)
        self.canv.rect(0, 1, bar, 5, stroke=0, fill=1)
        self.canv.setFillColor(ACCENT)
        self.canv.rect(0, 1, max(0.8, bar * self.fraction), 5, stroke=0, fill=1)
        self.canv.setFillColor(MUTED)
        self.canv.setFont(SANS, 7.5)
        self.canv.drawRightString(self.width, 1, f"{100 * self.fraction:.0f} %")


def _numbered_canvas(run_name: str) -> type[Canvas]:
    """A canvas whose footer reads 'run · Page n of N' (two passes: N is known at save)."""

    class NumberedCanvas(Canvas):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._pages: list[dict[str, Any]] = []

        def showPage(self) -> None:  # noqa: N802  (ReportLab's name)
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self) -> None:
            for state in self._pages:
                self.__dict__.update(state)
                self._footer(len(self._pages))
                super().showPage()
            super().save()

        def _footer(self, total: int) -> None:
            y = MARGIN - 2 * mm
            self.setStrokeColor(LINE)
            self.setLineWidth(0.5)
            self.line(MARGIN, y + 4 * mm, A4[0] - MARGIN, y + 4 * mm)
            self.setFont(SANS, 7.5)
            self.setFillColor(MUTED)
            self.drawString(MARGIN, y, f"Reconstruction report  ·  {run_name}")
            self.drawRightString(A4[0] - MARGIN, y, f"Page {self._pageNumber} of {total}")

    return NumberedCanvas


# --- reading and formatting -------------------------------------------------------------


def _optional_json(path: Path) -> dict[str, Any] | None:
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _stage_features(s: dict[str, Any] | None) -> str:
    if not s:
        return ""
    cached = " (cached)" if s.get("cached") else ""
    median = _int(s.get("per_image_median"))
    return f"{_int(s.get('total'))} features, median {median} per photo{cached}"


def _stage_matches(s: dict[str, Any] | None) -> str:
    if not s:
        return ""
    cached = " (cached)" if s.get("cached") else ""
    return f"{_int(s.get('matches'))} matches over {_int(s.get('pairs'))} pairs{cached}"


def _stage_verification(s: dict[str, Any] | None) -> str:
    if not s:
        return ""
    ratio = _pct(s.get("inlier_ratio"), 0)
    return f"{_int(s.get('pairs'))} pairs verified, {_int(s.get('inliers'))} inliers ({ratio})"


def _stage_reconstruction(s: dict[str, Any] | None) -> str:
    return f"{_int(s.get('registered'))} cameras, {_int(s.get('landmarks'))} points" if s else ""


def _stage_filter(s: dict[str, Any] | None) -> str:
    return f"kept {_int(s.get('kept'))} of {_int(s.get('total'))} points" if s else ""


def _number(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _int(value: float | None) -> str:
    return "—" if value is None else f"{round(value):,}"


def _px(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f} px"


def _deg(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}°"


def _pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f} %"


def _seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if round(value, 1) >= 60:  # rounded first: 59.96 s is "1 min 0 s", never "60.0 s"
        minutes, seconds = divmod(round(value), 60)
        return f"{minutes} min {seconds} s"
    return f"{value:.1f} s"


def _focal(K: list[list[float]]) -> str:
    fx, fy = K[0][0], K[1][1]
    return f"{fx:.2f} px" if abs(fx - fy) < 1e-9 else f"fx {fx:.2f} px, fy {fy:.2f} px"


def _principal(K: list[list[float]]) -> str:
    return f"({K[0][2]:.2f}, {K[1][2]:.2f}) px"


def _compact(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_compact(v)}" for k, v in value.items())
    if isinstance(value, list):
        return "[" + ", ".join(_compact(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)

"""Build the course submission: GroupX_3dcv_project/ and GroupX_3dcv_project.zip.

    source .venv/bin/activate
    python finalsubmission/build_submission.py              # everything
    python finalsubmission/build_submission.py --skip-demo  # reuse build/demo.mp4

Reads finalsubmission/group.json (group number, members, contributions) and
  1. copies the code (src, configs, scripts, tests, notebooks, docs, pyproject) into code/,
  2. copies two Middlebury sets into sample_data/,
  3. runs the pipeline on sample_data/templeRing and renders demo.mp4 from it,
  4. writes the report's author block and contribution table, compiles report.pdf
     (tectonic), with stills from the same run,
  5. writes README.md and requirements.txt from templates/,
  6. zips the folder and checks the size limit (200 MB).
Needs: the project venv, tectonic and ffmpeg on PATH, data/raw/{templeRing,dinoRing}.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BUILD = HERE / "build"

CODE_ITEMS = ["src", "configs", "scripts", "tests", "notebooks", "pyproject.toml"]
DOC_ITEMS = ["docs/images", "docs/architecture.md", "docs/methodology.md",
             "docs/experiments.md", "docs/ROADMAP.md", "data/sample/README.md"]  # fmt: skip
SAMPLE_SETS = ["templeRing", "dinoRing"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".ipynb_checkpoints",
                                "*.egg-info", ".pytest_cache", ".ruff_cache")  # fmt: skip
LIMIT_MB = 200


def run(cmd: list[str], **kw) -> None:
    print("  $", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kw)


def tex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ("<", r"\textless{}"), (">", r"\textgreater{}")]:  # fmt: skip
        s = s.replace(a, b)
    return s


def write_report_inputs(group: dict, report_dir: Path) -> None:
    members = group["members"]

    # Compact author block: three authors per row (then the rest), name, roll number and
    # email in small type, and the shared department / university once underneath.
    def cell(m: dict) -> str:
        return (
            "\\begin{tabular}[t]{@{}c@{}}{\\small %s}\\\\ {\\footnotesize Roll No. %s}\\\\ "
            "{\\footnotesize\\texttt{%s}}\\end{tabular}"
            % (tex_escape(m["name"]), tex_escape(m["id"]), tex_escape(m["email"]))
        )

    rows = [members[k : k + 3] for k in range(0, len(members), 3)]
    lines = ["\\hspace{2.5em}".join(cell(m) for m in row) for row in rows]
    (report_dir / "authors.tex").write_text(
        "\\author{%\n"
        + "\\\\[1.5mm]\n".join(lines)
        + "\\\\[1.5mm]\n{\\footnotesize\\textit{%s, %s}}\n}\n"
        % (tex_escape(group["department"]), tex_escape(group["university"])),
        encoding="utf-8",
    )

    body = "\n".join(
        f"{tex_escape(m['name'])} & {tex_escape(m['contribution'])} \\\\\n\\hline" for m in members
    )
    (report_dir / "contributions.tex").write_text(
        "\\vspace{3mm}\n\\noindent\\begin{minipage}{\\columnwidth}  % heading and table stay together\n"
        "\\section*{Author Contributions}\n"
        "\\noindent All members reviewed the results and the final report.\n\n"
        "\\begin{center}\\small\n"
        "\\begin{tabular}{|>{\\raggedright\\arraybackslash}p{0.27\\columnwidth}"
        "|>{\\raggedright\\arraybackslash}p{0.62\\columnwidth}|}\n\\hline\n"
        "\\textbf{Member} & \\textbf{Contribution} \\\\\n\\hline\n" + body + "\n\\end{tabular}\n"
        "\\end{center}\n\\end{minipage}\n",
        encoding="utf-8",
    )


def make_stills(video: Path, figures: Path) -> None:
    """Sparse / dense / mesh stills from the same viewpoint, cropped from demo.mp4."""
    # Segment start times follow make_demo.py: the close-up, dense and mesh orbits all
    # start at the same angle, so these three frames are directly comparable.
    times = {"still_sparse": None, "still_dense": None, "still_mesh": None}
    info = json.loads((BUILD / "demo_timeline.json").read_text())
    for name, key in [("still_sparse", "closeup"), ("still_dense", "dense"), ("still_mesh", "mesh")]:
        times[name] = info[key] + 0.05
    for name, t in times.items():
        run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video),
             "-frames:v", "1", "-vf", "crop=600:600:340:110", str(figures / f"{name}.png")])  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-demo", action="store_true", help="reuse build/renders.mp4 (source of the report stills)")
    args = ap.parse_args()

    group = json.loads((HERE / "group.json").read_text(encoding="utf-8"))
    if not 1 <= len(group["members"]) <= 5:
        sys.exit("group.json: 1-5 members")
    name = f"Group{group['group']}_3dcv_project"
    out = HERE / name
    print(f"Building {out}")
    if out.exists():
        shutil.rmtree(out)
    (out / "code").mkdir(parents=True)

    # 1. code -------------------------------------------------------------------------
    print("[1/6] code")
    for item in CODE_ITEMS:
        src = REPO / item
        if src.is_dir():
            shutil.copytree(src, out / "code" / item, ignore=IGNORE)
        else:
            shutil.copy2(src, out / "code" / item)
    for item in DOC_ITEMS:
        src, dst = REPO / item, out / "code" / item
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree(src, dst, ignore=IGNORE) if src.is_dir() else shutil.copy2(src, dst))
    shutil.copy2(REPO / "README.md", out / "code" / "README.md")
    for d in ["outputs", "data/raw"]:
        (out / "code" / d).mkdir(parents=True, exist_ok=True)
        (out / "code" / d / ".gitkeep").write_text("")

    # 2. sample data ------------------------------------------------------------------
    print("[2/6] sample_data")
    for s in SAMPLE_SETS:
        shutil.copytree(REPO / "data" / "raw" / s, out / "sample_data" / s, ignore=IGNORE)

    # 3. demo video: a recorded session in the web app ---------------------------------
    #    (build/renders.mp4, rendered offscreen from a CLI run, only supplies the report's
    #    sparse / dense / mesh stills)
    print("[3/6] demo.mp4")
    renders = BUILD / "renders.mp4"
    sample = out / "sample_data" / "templeRing"
    if not (args.skip_demo and renders.exists()):
        runs = BUILD / "runs"
        shutil.rmtree(runs, ignore_errors=True)
        run([sys.executable, "-m", "src.cli", "dense", str(sample), "--out", str(runs)], cwd=REPO)
        run_dir = next(runs.glob("*_dense_*"))
        run([sys.executable, str(HERE / "tools" / "make_demo.py"), str(run_dir), str(sample),
             str(renders)], cwd=REPO)  # fmt: skip
    # demo.mp4 is the group's own screen recording: put it at finalsubmission/demo.mp4.
    # (tools/record_webapp_demo.py can record one automatically if ever needed.)
    own = HERE / "demo.mp4"
    if own.exists():
        shutil.copy2(own, out / "demo.mp4")
    else:
        print("  WARNING: finalsubmission/demo.mp4 is missing, so the package has no demo video")

    # 4. report -----------------------------------------------------------------------
    print("[4/6] report.pdf")
    report = HERE / "report"
    figures = report / "figures"
    figures.mkdir(exist_ok=True)
    for f in ["templeRing_cameras.png", "e3b_focal_refinement.png"]:
        shutil.copy2(REPO / "docs" / "images" / f, figures / f)
    make_stills(renders, figures)
    write_report_inputs(group, report)
    run(["tectonic", "--keep-logs", "main.tex"], cwd=report)
    shutil.copy2(report / "main.pdf", out / "report.pdf")
    log = (report / "main.log").read_text(errors="replace")
    pages = re.search(r"Output written on .*?\((\d+) pages?", log)
    if pages:
        print(f"  report: {pages.group(1)} pages")

    # 5. README + requirements ----------------------------------------------------------
    print("[5/6] README.md, requirements.txt")
    table = "\n".join(f"| {m['name']} | {m['id']} | {m['email']} |" for m in group["members"])
    readme = (HERE / "templates" / "README.md").read_text(encoding="utf-8")
    readme = (readme.replace("{{GROUP}}", str(group["group"])).replace("{{COURSE}}", group["course"])
              .replace("{{MEMBERS_TABLE}}", table).replace("GroupX_3dcv_project", name))  # fmt: skip
    (out / "README.md").write_text(readme, encoding="utf-8")
    shutil.copy2(HERE / "templates" / "requirements.txt", out / "requirements.txt")

    # 6. zip --------------------------------------------------------------------------
    print("[6/6] zip")
    zpath = HERE / f"{name}.zip"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, Path(name) / f.relative_to(out))
    mb = zpath.stat().st_size / 2**20
    print(f"\n{zpath}  {mb:.1f} MB  ({'OK' if mb < LIMIT_MB else 'TOO LARGE'}, limit {LIMIT_MB} MB)")
    todo = [m for m in group["members"] for v in m.values() if "<" in str(v)]
    if todo or "<" in group["university"] or group["group"] == "X":
        print("NOTE: group.json still has placeholders (<...> or group X); fill them in and rebuild.")


if __name__ == "__main__":
    main()

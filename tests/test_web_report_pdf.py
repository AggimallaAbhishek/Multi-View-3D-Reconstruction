import re

import cv2
import numpy as np
import pytest

from src.run import write_json
from src.web.report_pdf import _seconds, render_report_pdf, report_filename
from tests.pdf_text import pdf_text

RUN_ID = "20261030-120000_web_reconstruct"


def page_count(pdf: bytes) -> int:
    return len(re.findall(rb"/Type /Page\b", pdf))


def full_report(**changes):
    report = {
        "images": {"total": 3, "registered": 2, "unregistered": ["c(1).png"]},
        "points": {"sparse": 1500, "filtered": 1234},
        "features": {"detector": "sift", "matcher": "bruteforce"},
        "calibration": {
            "sources": ["file"],
            "K": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            "focal_scale": 1.01,
        },
        "reprojection": {
            "n": 900,
            "mean_px": 0.3125,
            "median_px": 0.25,
            "p90_px": 0.6,
            "max_px": 1.9,
        },
        "stages": {
            "features": {"per_image_median": 1200.0, "total": 3600, "cached": False},
            "matches": {"pairs": 3, "matches": 2000, "cached": True},
            "verification": {"pairs": 2, "inliers": 1800, "inlier_ratio": 0.9},
            "initial_pair": ["a.png", "b.png"],
            "reconstruction": {"registered": 2, "landmarks": 1500},
            "filter": {"kept": 1234, "total": 1500},
        },
        "timings": {"dataset": 1.0, "features": 2.0, "matches": 3.0},
        "total_s": 6.0,
        "warnings": ["1 image(s) not registered: c(1).png <unusual & name>"],
    }
    return report | changes


GROUND_TRUTH = {
    "cameras": 2,
    "rotation_deg": {"median": 0.15, "p90": 0.3, "max": 0.32},
    "centre": {"median": 0.0015, "p90": 0.002, "max": 0.003},
    "per_camera": [
        {"image": "a.png", "rotation_deg": 0.1, "centre": 0.001},
        {"image": "b.png", "rotation_deg": 0.32, "centre": 0.003},
    ],
}


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / RUN_ID
    run.mkdir()
    write_json(run / "report.json", full_report())
    (run / "figures").mkdir()
    figure = np.full((120, 160, 3), 200, np.uint8)
    cv2.imwrite(str(run / "figures" / "filtered.png"), figure)
    return run


def render(run):
    return render_report_pdf(run, compress=False)


def test_report_is_a_pdf(run_dir):
    pdf = render_report_pdf(run_dir)

    assert pdf.startswith(b"%PDF-")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert page_count(pdf) >= 2  # the figures start on a page of their own


def test_report_states_the_headline_results(run_dir):
    text = pdf_text(render(run_dir))

    assert "Reconstruction report" in text
    assert RUN_ID in text
    assert "2 / 3" in text  # cameras registered
    assert "1,234" in text  # filtered points
    assert "0.31 px" in text  # mean reprojection error
    assert "a.png and b.png" in text  # initial pair
    assert "SIFT" in text
    assert "c(1).png <" in text and "unusual & name" in text  # printed as written


def test_pages_are_numbered(run_dir):
    pdf = render(run_dir)
    n = page_count(pdf)

    assert f"Page 1 of {n}" in pdf_text(pdf)
    assert f"Page {n} of {n}" in pdf_text(pdf)


def test_report_includes_the_figures_the_run_wrote(run_dir):
    text = pdf_text(render(run_dir))

    assert "Figure 1." in text and "Filtered point cloud" in text
    assert "Recovered camera positions" not in text  # not in the run folder


def test_report_includes_ground_truth_errors(run_dir):
    write_json(run_dir / "report.json", full_report(ground_truth=GROUND_TRUTH))

    text = pdf_text(render(run_dir))

    assert "Accuracy against ground truth" in text
    assert "0.32°" in text
    assert "0.30 %" in text  # the worst position error, as a share of the cameras' spread
    assert "Ground-truth error of every camera" in text  # the appendix


def test_report_without_ground_truth_has_no_such_section(run_dir):
    assert "ground truth" not in pdf_text(render(run_dir)).lower()


def test_report_includes_run_folder_details_when_present(run_dir):
    write_json(
        run_dir / "environment.json",
        {"python": "3.12.1", "git_commit": "abc1234def", "packages": {"opencv": "5.0.0"}},
    )
    (run_dir / "reconstruction").mkdir()
    write_json(
        run_dir / "reconstruction" / "reconstruction.json",
        {
            "observations": 900,
            "mean_track_length": 3.5,
            "intrinsics": {"K": [[505.0, 0, 320.0], [0, 505.0, 240.0], [0, 0, 1]]},
            "filtering": {"steps": [{"name": "statistical", "before": 1500, "removed": 266}]},
        },
    )
    (run_dir / "reconstruction" / "registration.csv").write_text(
        "attempt,image,method,success,reason,correspondences,inliers\n"
        "0,b.png,pnp,True,,352,349\n1,c.png,pnp,False,too few inliers,40,12\n"
    )
    (run_dir / "config.yaml").write_text("features:\n  detector: sift\n")

    text = pdf_text(render(run_dir))

    assert "opencv" in text and "abc1234" in text
    assert "3.50 views" in text  # mean track length
    assert "505.00 px" in text  # refined focal length
    assert "statistical" in text and "266" in text
    assert "1 of 2 registration attempts succeeded" in text
    assert "too few inliers" in text
    assert "detector: sift" in text


def test_long_tables_continue_over_pages(run_dir):
    many = [{"image": f"p{k:04d}.png", "rotation_deg": 0.1, "centre": 0.001} for k in range(400)]
    write_json(
        run_dir / "report.json", full_report(ground_truth=GROUND_TRUTH | {"per_camera": many})
    )

    pdf = render(run_dir)

    assert page_count(pdf) > 5
    assert "p0399.png" in pdf_text(pdf)


def test_report_of_a_run_without_a_reconstruction(tmp_path):
    run = tmp_path / RUN_ID
    run.mkdir()
    report = full_report(
        images={"total": 3, "registered": 0, "unregistered": ["a.png", "b.png", "c.png"]},
        points=None,
        reprojection=None,
        warnings=["no reconstruction: no image pair qualified as the initial pair"],
    )
    report["stages"] |= {"initial_pair": None, "reconstruction": None, "filter": None}
    write_json(run / "report.json", report)

    text = pdf_text(render(run))

    assert "0 / 3" in text
    assert "no image pair qualified" in text


def test_report_needs_report_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        render_report_pdf(tmp_path)


def test_file_name_is_the_run_name(tmp_path):
    assert report_filename(tmp_path / RUN_ID) == f"{RUN_ID}_report.pdf"


@pytest.mark.parametrize(
    ("seconds", "shown"),
    [(59.96, "1 min 0 s"), (119.6, "2 min 0 s"), (61.2, "1 min 1 s"), (4.04, "4.0 s")],
)
def test_times_round_without_a_sixtieth_second(seconds, shown):
    assert _seconds(seconds) == shown


def test_figures_are_numbered_without_gaps(run_dir):
    (run_dir / "reconstruction").mkdir()
    (run_dir / "reconstruction" / "cameras.png").write_bytes(b"not a png")  # skipped
    figure = np.full((120, 160, 3), 200, np.uint8)
    cv2.imwrite(str(run_dir / "reconstruction" / "reprojection_histogram.png"), figure)

    text = pdf_text(render(run_dir))

    assert "Figure 2." in text  # the histogram, after the unreadable cameras figure
    assert "Figure 3." not in text


def test_a_report_with_missing_fields_still_renders(run_dir):
    """A report.json of an older or partial run may lack fields: shown as blanks."""
    report = full_report(ground_truth={"cameras": 2, "per_camera": [{"image": "a.png"}]})
    report["stages"] = {"features": {"total": 10}, "matches": {}, "verification": {"pairs": 1}}
    write_json(run_dir / "report.json", report)

    text = pdf_text(render(run_dir))

    assert "Reconstruction report" in text


def test_names_in_any_script_are_printed(run_dir):
    write_json(
        run_dir / "report.json",
        full_report(images={"total": 3, "registered": 2, "unregistered": ["фото_1.jpg"]}),
    )

    assert "фото_1.jpg" in pdf_text(render(run_dir))

"""Per-stage outputs of a pipeline run: files in a run directory and printed summaries.

``save_stage_outputs`` writes what every completed stage produced (statistics, figures,
point clouds, cameras) into one run directory. The CLI and the web interface both call
it, so a run folder looks the same however it was made. Diagnostic images are drawn only
for the last stage.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from src.features.extractor import weak_images, write_feature_statistics
from src.features.matcher import (
    Matches,
    connectivity,
    isolated_images,
    match_matrix,
    resolve_pair_mode,
    save_matches,
    write_match_statistics,
)
from src.geometry.pose_estimation import write_pose_statistics
from src.geometry.reprojection import ReprojectionStats
from src.geometry.verification import (
    inlier_matches,
    unverified_images,
    verification_matrix,
    write_verification_statistics,
)
from src.pipeline import PipelineResult, Stage, stages_until
from src.pointcloud.export import save_cloud, write_ply
from src.reconstruction.incremental import FRAME as MULTI_VIEW_FRAME
from src.reconstruction.two_view import FRAME as TWO_VIEW_FRAME
from src.run import (
    atomic_savez,
    create_run_dir,
    read_json,
    write_csv,
    write_json,
    write_run_metadata,
)
from src.visualization.cameras import plot_cameras, plot_two_view
from src.visualization.epipolar import draw_epipolar_lines
from src.visualization.features import draw_keypoints
from src.visualization.matches import draw_matches, plot_match_matrix
from src.visualization.reprojection import plot_reprojection_histogram


def new_run_dir(result: PipelineResult, label: str, stamp: str | None = None) -> Path:
    """A new run folder under ``output.root`` (stamped now, or with ``stamp``) with the run
    metadata and ``dataset.json``, ready for ``save_stage_outputs``."""
    run_dir = create_run_dir(result.config.output.root, label, stamp)
    write_run_metadata(run_dir, result.config)
    result.dataset.save(run_dir / "dataset.json")
    return run_dir


def save_stage_outputs(
    result: PipelineResult, run_dir: Path, stage: Stage, draw: int = 0
) -> tuple[Path | None, int]:
    """Write the outputs of every stage up to and including ``stage`` into ``run_dir``.

    ``draw`` diagnostic images are saved for ``stage`` only. Returns the main file written
    (the last stage's, or the latest stage that wrote one) and an exit status: 1 if a stage
    found the run unusable (no initial pair), else 0.
    """
    written, status = None, 0
    for s in stages_until(stage):
        if s == "cloud" and stage != "cloud":
            continue  # the two-view cloud is `two-view`'s result; `reconstruct` replaces it
        path, stage_status = _SAVERS[s](result, run_dir, max(draw, 0) if s == stage else 0)
        written, status = path or written, max(status, stage_status)
    return written, status


def _spread(values: Sequence[float], fmt: str = "{:.0f}", unit: str = "") -> str:
    v = np.asarray(values, dtype=float)
    return ", ".join(
        f"{name} {fmt.format(x)}{unit}"
        for name, x in (("min", v.min()), ("median", np.median(v)), ("max", v.max()))
    )


def _inspection_order(matches: Sequence[Matches]) -> list[int]:
    """Indices of pairs, neighbouring images first: the pairs most worth looking at."""
    return sorted(range(len(matches)), key=lambda k: (matches[k].j - matches[k].i, matches[k].i))


def _pair_stem(name_i: str, name_j: str) -> str:
    return f"{Path(name_i).stem}_{Path(name_j).stem}"


def _save_features(r: PipelineResult, run_dir: Path, draw: int) -> tuple[Path, int]:
    out = run_dir / "features"
    csv_path = write_feature_statistics(r.feature_stats, out / "feature_statistics.csv")
    for im, f in list(zip(r.images, r.features, strict=True))[:draw]:
        cv2.imwrite(str(out / f"{Path(im.name).stem}_keypoints.png"), draw_keypoints(im.image, f))

    stats, cfg = r.feature_stats, r.config.features
    new = [s.time_ms for s in stats if not s.cached]
    print(f"Detector: {r.extractor.signature()}")
    print(
        f"Features per image: {_spread([s.n_features for s in stats])} "
        f"(cap {r.extractor.max_features})"
    )
    print(f"Grid coverage: {_spread([s.coverage for s in stats], '{:.2f}')}")
    print(
        f"Computed {len(new)} image(s)"
        + (f" at {np.mean(new):.0f} ms each" if new else "")
        + f", {len(stats) - len(new)} from cache"
    )
    for s in weak_images(stats, cfg):
        print(
            f"[WARNING] {s.image}: {s.n_features} features, coverage {s.coverage:.2f} "
            "- too little texture or features concentrated in one region"
        )
    return csv_path, 0


def _save_matches(r: PipelineResult, run_dir: Path, draw: int) -> tuple[Path, int]:
    out, names, stats, cfg = run_dir / "matches", r.names, r.match_stats, r.config.matching
    csv_path = write_match_statistics(stats, names, out / "match_statistics.csv")
    save_matches(r.matches, out / "matches.npz")
    plot_match_matrix(match_matrix(stats, len(names)), names, out / "match_matrix.png")
    for k in _inspection_order(r.matches)[:draw]:
        m = r.matches[k]
        a, b = r.images[m.i], r.images[m.j]
        image_a, image_b = a.image, b.image  # each access decodes from disk: do it once
        canvas = draw_matches(image_a, image_b, r.features[m.i], r.features[m.j], m)
        cv2.imwrite(str(out / f"{_pair_stem(a.name, b.name)}_matches.png"), canvas)

    final = np.array([s.final for s in stats])
    good = final >= cfg.min_matches
    print(f"Matching: ratio {cfg.ratio}, mutual {cfg.mutual}, {cfg.method}")
    print(
        f"Pairs: {len(stats)} ({resolve_pair_mode(len(names), cfg)}) in "
        f"{sum(s.time_ms for s in stats) / 1000:.1f} s, {sum(s.cached for s in stats)} from cache; "
        f"{good.sum()} with >= {cfg.min_matches} matches"
    )
    if good.any():
        print(f"Matches per connected pair: {_spread(final[good])}")
    best = connectivity(stats, len(names))
    for k in isolated_images(stats, len(names), cfg.min_matches):
        print(
            f"[WARNING] {names[k]}: best pair has only {best[k]} matches - "
            "it will probably not be registered"
        )
    return csv_path, 0


def _save_verification(r: PipelineResult, run_dir: Path, draw: int) -> tuple[Path, int]:
    out, names, stats, cfg = (
        run_dir / "verification", r.names, r.verification_stats, r.config.verification
    )  # fmt: skip
    csv_path = write_verification_statistics(stats, names, out / "verification_statistics.csv")
    verified = [inlier_matches(m, g) for m, g in zip(r.matches, r.geometries, strict=True)]
    save_matches(verified, out / "verified_matches.npz")
    inliers = verification_matrix(stats, len(names))
    plot_match_matrix(
        inliers, names, out / "verified_matrix.png",
        title="Verified inliers per image pair", label="geometric inliers",
    )  # fmt: skip
    for k in _inspection_order(r.matches)[:draw]:
        m, g = r.matches[k], r.geometries[k]
        a, b = r.images[m.i], r.images[m.j]
        stem = _pair_stem(a.name, b.name)
        fi, fj = r.features[m.i], r.features[m.j]
        image_a, image_b = a.image, b.image  # each access decodes from disk: do it once
        canvas = draw_matches(image_a, image_b, fi, fj, m, inlier_mask=g.inlier_mask)
        cv2.imwrite(str(out / f"{stem}_inliers.png"), canvas)
        if g.F is not None and g.n_inliers:
            x_i, x_j = m.points(fi, fj)
            canvas = draw_epipolar_lines(
                image_a, image_b, x_i[g.inlier_mask], x_j[g.inlier_mask], g.F
            )
            cv2.imwrite(str(out / f"{stem}_epipolar.png"), canvas)

    ok = [s for s in stats if s.success]
    models = ", ".join(sorted({s.model for s in stats}))
    print(
        f"Verification: {models} ({cfg.estimator}, threshold {cfg.ransac_threshold_px} px) "
        f"in {sum(s.time_ms for s in stats) / 1000:.1f} s"
    )
    print(f"Verified pairs: {len(ok)} of {len(stats)} (>= {cfg.min_inliers} inliers)")
    if ok:
        planar = sum(g.planar for g in r.geometries)
        print(
            f"Inlier ratio: {_spread([s.inlier_ratio for s in ok], '{:.2f}')}; "
            f"median residual {np.median([s.median_residual for s in ok]):.2f} px"
        )
        print(
            f"Pairs explained by a homography (H/F > {cfg.planar_ratio}): {planar} "
            "- planar scene or rotation only, weak for initialisation"
        )
    for k in unverified_images(stats, len(names)):
        print(f"[WARNING] {names[k]}: no verified pair - it cannot be registered")
    return csv_path, 0


def _save_poses(r: PipelineResult, run_dir: Path, draw: int) -> tuple[Path, int]:
    out, names, stats, cfg = run_dir / "pose", r.names, r.pose_stats, r.config
    csv_path = write_pose_statistics(stats, names, out / "relative_poses.csv")

    ok = [s for s in stats if s.success]
    reasons = Counter(s.reason for s in stats if not s.success)
    failed = ", ".join(f"{reason} {n}" for reason, n in reasons.items())
    print(
        f"Relative poses: {len(ok)} of {len(stats)} verified pairs"
        + (f" (failed: {failed})" if failed else "")
    )
    if ok:
        parallax = [s.median_parallax_deg for s in ok]
        print(f"Median parallax per pair: {_spread(parallax, '{:.1f}', ' deg')}")

    init = r.initial_pair
    if init is None:
        print(
            "[WARNING] no pair qualifies as the initial pair (needs a valid pose, "
            f"H/F <= {cfg.verification.planar_ratio} and parallax >= "
            f"{cfg.pose.init_min_parallax_deg} deg). Capture views with more baseline."
        )
        return csv_path, 1

    a, b, pose, g = names[init.i], names[init.j], init.pose, init.geometry
    plot_two_view(  # camera arrangement; the filtered points are in triangulation/structure.png
        pose.R, pose.t, np.empty((0, 3)), (a, b), out / "initial_pair.png",
        title=f"Initial pair {a} - {b}: rotation {pose.rotation_deg:.1f} deg, "
        f"parallax {pose.median_parallax_deg:.1f} deg",
    )  # fmt: skip
    write_json(out / "initial_pair.json", init.to_dict(names))
    print(
        f"Initial pair: {a} - {b}: {pose.n_valid} points, "
        f"parallax {pose.median_parallax_deg:.1f} deg, rotation {pose.rotation_deg:.1f} deg, "
        f"H/F {g.homography_ratio:.2f} ({g.model})"
    )
    return csv_path, 0


def _save_triangulation(r: PipelineResult, run_dir: Path, draw: int) -> tuple[Path | None, int]:
    if r.structure is None:  # no initial pair: _save_poses already reported it
        return None, 0
    out, s, cfg = run_dir / "triangulation", r.structure, r.config.triangulation
    tri = s.triangulation
    out.mkdir(parents=True, exist_ok=True)
    names = r.names
    summary = {
        "image_i": names[s.i],
        "image_j": names[s.j],
        **tri.summary(),
        "settings": r.config.to_dict()["triangulation"],
        "frame": TWO_VIEW_FRAME,
    }
    json_path = write_json(out / "triangulation.json", summary)
    atomic_savez(
        out / "points.npz",
        compressed=True,
        points=tri.points,
        valid=tri.valid,
        rejected_by=tri.rejected_by.astype(str),
        angles_deg=tri.angles_deg,
        max_errors_px=tri.max_errors,
    )
    plot_reprojection_histogram(
        tri.errors, cfg.max_reprojection_error_px, out / "reprojection_histogram.png",
        title=f"{names[s.i]} - {names[s.j]}",
    )  # fmt: skip
    pose = r.initial_pair.pose
    plot_two_view(
        pose.R, pose.t, s.points, (names[s.i], names[s.j]), out / "structure.png",
        title=f"Filtered structure: {len(s.points)} points",
    )  # fmt: skip

    counts = tri.counts
    rejected = ", ".join(
        f"{k} {v}" for k, v in counts.items() if k not in ("points", "valid") and v
    )
    print(
        f"Triangulated {counts['points']} correspondences of {names[s.i]} - {names[s.j]}: "
        f"{counts['valid']} valid" + (f" (rejected: {rejected})" if rejected else "")
    )
    rp = tri.reprojection
    print(
        f"Reprojection error of valid points: median {rp.median:.2f} px, "
        f"90th percentile {rp.p90:.2f} px, max {rp.max:.2f} px"
    )
    angles = summary["triangulation_angle_deg"]
    if angles["median"] is not None:
        print(
            f"Triangulation angle: median {angles['median']:.1f} deg, "
            f"min {angles['min']:.1f} deg (threshold {cfg.min_angle_deg} deg)"
        )
    return json_path, 0


def _save_cloud(r: PipelineResult, run_dir: Path, draw: int) -> tuple[Path | None, int]:
    """Coloured sparse cloud, cameras and renders of the initial structure (two-view)."""
    from src.visualization.viewer import RenderUnavailable, render_scene

    s = r.structure
    if s is None:
        return None, 0
    a, b = r.images[s.i], r.images[s.j]
    sizes = [(a.width, a.height), (b.width, b.height)]
    files = save_cloud(
        run_dir, s.points, r.colors, s.cameras, [a.name, b.name], sizes, TWO_VIEW_FRAME
    )
    print(f"Sparse cloud: {len(s.points)} coloured points -> {files.ply}")
    print(f"Cameras: {files.cameras}")
    try:
        figures = run_dir / "figures"
        for focus, name in (("cameras", "two_view.png"), ("points", "two_view_closeup.png")):
            render = render_scene(s.points, r.colors, s.cameras, sizes, figures / name, focus=focus)
            print(f"Render: {render}")
    except RenderUnavailable as exc:
        print(f"[WARNING] no offscreen rendering here ({exc}); view with: python -m src.cli view")
    print(f"View it: python -m src.cli view {files.ply} --cameras {files.cameras}")
    return files.ply, 0


def _save_reconstruction(r: PipelineResult, run_dir: Path, draw: int) -> tuple[Path | None, int]:
    """Multi-view cloud, cameras, registration log, summary and figures."""
    rec = r.reconstruction
    if rec is None:  # no initial pair: _save_poses already reported it
        return None, 0
    out, names, cfg = run_dir / "reconstruction", r.names, r.config
    out.mkdir(parents=True, exist_ok=True)
    registry, lm = rec.registry, rec.landmarks
    images, X = registry.images, lm.position
    cameras = [registry.camera(m) for m in images]
    errors = rec.reprojection_errors(r.features)
    stats = ReprojectionStats.from_errors(errors)
    unregistered = [names[m] for m in rec.unregistered(len(names))]
    track_lengths = lm.track_lengths()
    summary = {
        "images": len(names),
        "registered": len(images),
        "unregistered": unregistered,
        "registration_order": [names[m] for m in images],
        "landmarks": len(lm),
        "observations": len(errors),
        "mean_track_length": float(track_lengths.mean()) if len(lm) else None,
        "reprojection": stats.to_dict(),
        "intrinsics": {"initial_K": r.images[images[0]].K.tolist(), "K": cameras[0].K.tolist()},
        "settings": {k: cfg.to_dict()[k] for k in ("registration", "bundle_adjustment")},
        "frame": MULTI_VIEW_FRAME,
    }
    json_path = write_json(out / "reconstruction.json", summary)
    write_csv(
        out / "registration.csv",
        ["attempt", "image", "success", "reason", "correspondences", "inliers"],
        (
            [k, names[a.image], a.success, a.reason or "", a.n_correspondences, a.n_inliers]
            for k, a in enumerate(rec.attempts)
        ),
    )
    plot_reprojection_histogram(
        errors, cfg.triangulation.max_reprojection_error_px, out / "reprojection_histogram.png",
        title=f"{len(images)} cameras, {len(lm)} landmarks",
    )  # fmt: skip
    plot_cameras(
        [c.R for c in cameras], [c.t for c in cameras], X, [names[m] for m in images],
        out / "cameras.png", title=f"Reconstruction: {len(images)} of {len(names)} images",
    )  # fmt: skip

    sizes = [(r.images[m].width, r.images[m].height) for m in images]
    colors = lm.color / 255.0
    files = save_cloud(run_dir, X, colors, cameras, [names[m] for m in images], sizes,
                       MULTI_VIEW_FRAME)  # fmt: skip

    print(
        f"Registered {len(images)} of {len(names)} images "
        f"({sum(not a.success for a in rec.attempts)} failed attempts)"
    )
    print(
        f"Landmarks: {len(lm)}, observations {len(errors)} "
        f"(mean track length {summary['mean_track_length'] or 0:.1f})"
    )
    print(
        f"Reprojection error: mean {stats.mean:.2f} px, median {stats.median:.2f} px, "
        f"90th percentile {stats.p90:.2f} px, max {stats.max:.2f} px"
    )
    f0, f = (summary["intrinsics"][k][0][0] for k in ("initial_K", "K"))
    if f != f0:
        print(f"Focal length refined: {f:.1f} px (started at {f0:.1f} px, {f / f0 - 1:+.1%})")
    if unregistered:
        print(f"[WARNING] not registered: {', '.join(unregistered)}")
    print(f"Sparse cloud: {len(X)} coloured points -> {files.ply}")
    return json_path, 0


def _save_filtered(r: PipelineResult, run_dir: Path, draw: int) -> tuple[Path | None, int]:
    """filtered.ply, the filter counts (added to reconstruction.json) and a render."""
    from src.visualization.viewer import RenderUnavailable, render_scene

    if r.filtering is None:
        return None, 0
    rec, keep = r.reconstruction, r.filtering.keep
    lm, images = rec.landmarks, rec.registry.images
    X, colors = lm.position[keep], lm.color[keep] / 255.0
    ply = write_ply(run_dir / "point_clouds" / "filtered.ply", X, colors)
    json_path = run_dir / "reconstruction" / "reconstruction.json"
    summary = read_json(json_path)
    summary["filtering"] = {
        **r.filtering.summary(), "settings": r.config.to_dict()["filtering"]
    }  # fmt: skip
    write_json(json_path, summary)

    removed = ", ".join(f"{s.name} {s.removed}" for s in r.filtering.steps if s.removed)
    print(
        f"Filtered cloud: {len(X)} of {len(keep)} points kept"
        + (f" (removed: {removed})" if removed else "")
        + f" -> {ply}"
    )
    cameras = [rec.registry.camera(m) for m in images]
    sizes = [(r.images[m].width, r.images[m].height) for m in images]
    try:
        render = render_scene(
            X, colors, cameras, sizes, run_dir / "figures" / "filtered.png", focus="points"
        )
        print(f"Render: {render}")
    except RenderUnavailable as exc:
        print(f"[WARNING] no offscreen rendering here ({exc}); view with: python -m src.cli view")
    cameras_json = run_dir / "camera_poses" / "cameras.json"
    print(f"View it: python -m src.cli view {ply} --cameras {cameras_json}")
    return ply, 0


_SAVERS = {
    "features": _save_features,
    "matches": _save_matches,
    "verify": _save_verification,
    "pose": _save_poses,
    "triangulate": _save_triangulation,
    "cloud": _save_cloud,
    "reconstruct": _save_reconstruction,
    "filter": _save_filtered,
}

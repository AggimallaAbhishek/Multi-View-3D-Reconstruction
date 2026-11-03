// The workstation: upload photos, follow the job stage by stage, inspect the result. All
// logic about what is valid or what a run contains lives in the API (src/web/app.py);
// this only presents it. States (body[data-state]): empty, running, results, failed.

import { cameraLabels, count, small } from "./ui.js";
import { Viewer, parsePly } from "./viewer.js";

const $ = (id) => document.getElementById(id);

const STAGE_LABELS = {
  dataset: "Load and check photos",
  features: "Detect features",
  matches: "Match features",
  verify: "Verify geometry",
  pose: "First camera pair",
  triangulate: "Triangulate first points",
  cloud: "Two-view cloud",
  reconstruct: "Register cameras, adjust",
  filter: "Remove outliers",
  outputs: "Write results",
};

// What the page shows of a run folder, by path (written by src/outputs.py); only files the
// folder actually holds are shown (GET /api/jobs/{id}/files).
const FIGURES = {
  matches: [
    ["matches/match_matrix.png", "Feature matches between every pair of photos"],
    ["verification/verified_matrix.png", "Matches kept after geometric verification"],
    ["pose/initial_pair.png", "The first camera pair"],
  ],
  figures: [
    ["reconstruction/reprojection_histogram.png", "Reprojection error of every observation"],
    ["reconstruction/cameras.png", "Recovered camera positions"],
    ["figures/filtered.png", "Filtered point cloud"],
  ],
};

const DOWNLOADS = [
  ["point_clouds/filtered.ply", "Point cloud", "filtered.ply"],
  ["camera_poses/cameras.json", "Camera poses", "cameras.json"],
  ["report.json", "Report", "report.json"],
];

const CALIBRATION_SOURCES = { file: "camera.yaml", exif: "EXIF", fallback: "a guess" };
// report.features: which detector and matcher made the run (learned ones are external).
const DETECTORS = { sift: "SIFT", orb: "ORB", disk: "DISK" };
const MATCHERS = { bruteforce: "ratio test", flann: "ratio test (FLANN)", lightglue: "LightGlue" };

let config = null;      // GET /api/config
let photos = new Map(); // name -> File
const thumbUrls = new Map(); // File -> object URL, made once per file
let pollToken = 0;      // bumps to cancel a running poll loop
let viewer = null;
let viewerJob = null;   // {id, clouds: {filtered, sparse}, cameras}
let shownJob = null;    // the job on screen: what "Delete this run" deletes
let shownReport = null;
let gtSort = { key: "image", descending: false }; // ground-truth table order
let lightboxItems = [];  // [{href, caption}] of the gallery the lightbox was opened from
let lightboxIndex = 0;
let labels = null;       // camera labels of the viewer (ui.js cameraLabels)
let replaying = false;

// --- API --------------------------------------------------------------------------------

class ApiError extends Error {
  constructor(status, body) {
    super(detailText(body?.detail) || `request failed (${status})`);
    this.status = status;
    this.body = body;
  }
}

function detailText(detail) {
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join("; "); // FastAPI validation
  return detail;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, body);
  return body;
}

const jobUrl = (id, rest = "") => `/api/jobs/${encodeURIComponent(id)}${rest}`;
const fileUrl = (id, path) => jobUrl(id, `/files/${path}`);

// --- Formatting -------------------------------------------------------------------------

const seconds = (s) => (s < 10 ? s.toFixed(1) : Math.round(s).toString());
const percent = (fraction) => `${small(fraction * 100)}%`;
const megabytes = (bytes) => `${(bytes / 2 ** 20).toFixed(1)} MB`;
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const clock = (date = new Date()) => date.toTimeString().slice(0, 8);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.filter((c) => c !== null && c !== undefined && c !== false));
  return node;
}

/** A job's name: its start time (YYYYmmdd-HHMMSS, as run folders are named), else its id.
 * The dropdown adds the cameras registered (or the photo count) and the status of
 * unfinished or failed jobs, so runs of the same day can be told apart. */
function runLabel(job, detailed = false) {
  const m = (job.started ?? "").match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/);
  const when = m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]}` : `Run ${job.id}`;
  if (!detailed) return when;
  let label = when;
  if (job.registered != null && job.photos != null) label += ` · ${job.registered}/${job.photos} registered`;
  else if (job.photos != null) label += ` · ${plural(job.photos, "photo")}`;
  return job.status === "done" ? label : `${label} (${job.status})`;
}

/** The start of a job as a Date (from ``started``), or now. */
function startedAt(job) {
  const m = (job.started ?? "").match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/);
  return m ? new Date(+m[1], m[2] - 1, +m[3], +m[4], +m[5], +m[6]) : new Date();
}

// --- State ------------------------------------------------------------------------------

function show(state) {
  document.body.dataset.state = state;
}

function setStatus(title, text = "") {
  $("status-title").textContent = title;
  $("status-text").textContent = text;
}

// The run on screen survives a reload: it is kept in the address (#job=) and in this
// browser's storage until the user clears it or deletes the run. Storage can be missing
// (private windows, blocked site data): then only the address remembers it.
const OPEN_RUN_KEY = "mv3d.openRun";

/** Remember the run on screen: its job id (live this server session) and, once known,
 * its run folder, which also reopens it after the server restarts. */
function rememberOpenRun(id, folder = null) {
  try {
    if (id) localStorage.setItem(OPEN_RUN_KEY, JSON.stringify({ id, folder }));
    else localStorage.removeItem(OPEN_RUN_KEY);
  } catch { /* storage unavailable */ }
}

/** Ids to try, in order, for the run that was on screen before the reload. */
function rememberedOpenRun() {
  const ids = [];
  const match = location.hash.match(/^#job=(.+)$/);
  if (match) ids.push(decodeURIComponent(match[1]));
  try {
    const saved = JSON.parse(localStorage.getItem(OPEN_RUN_KEY) ?? "null");
    if (saved?.id) ids.push(saved.id);
    if (saved?.folder) ids.push(saved.folder);
  } catch { /* storage unavailable or old format */ }
  return [...new Set(ids)];
}

function openJobInUrl(id) {
  history.replaceState(null, "", id ? `#job=${encodeURIComponent(id)}` : location.pathname);
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(OPEN_RUN_KEY) ?? "null"); } catch { /* none */ }
  if (!id) rememberOpenRun(null);
  else if (saved?.id !== id) rememberOpenRun(id); // its folder is added once it has finished
}

async function openJob(id, { quiet = false } = {}) {
  pollToken++;
  openJobInUrl(id);
  try {
    const job = await api(jobUrl(id));
    showUploadMessage(null);
    if (job.status === "done") await showResults(job);
    else follow(job);
    return true;
  } catch (error) {
    openJobInUrl(null);
    clearRun();
    // a remembered run that was deleted elsewhere: start empty without an error
    if (!quiet) showUploadMessage(`Could not open run ${id}: ${error.message}`);
    return false;
  }
}

/** After a reload: the run that was on screen, by job id, else by its run folder (the
 * server forgets job ids when it restarts); empty if neither exists any more. */
async function reopenRememberedRun() {
  const ids = rememberedOpenRun();
  const folder = ids.at(-1);
  for (const id of ids) {
    if (await openJob(id, { quiet: true })) return;
    if (id !== folder) rememberOpenRun(folder, folder); // keep the folder for the next try
  }
  clearRun();
}

/** Nothing open: the empty workstation, ready for photos. */
function clearRun() {
  pollToken++;
  replaying = false;
  shownJob = shownReport = null;
  openJobInUrl(null);
  $("previous-runs").value = "";
  show("empty");
  setStatus("No run open", "Drop overlapping photos in the panel on the left, or open a previous run.");
  $("this-run").hidden = true;
  for (const id of ["summary-card", "export-card", "gt-card", "warnings", "viewer-tools", "layers"]) $(id).hidden = true;
  $("datastrip").replaceChildren();
  $("replay").disabled = true;
  setTabs({ poses: false, matches: false, figures: false });
  selectTab("viewer");
  renderPipeline(null);
  hideCameraLabel();
  viewer?.setCloud({ positions: new Float32Array(), colours: null });
  viewer?.setCameras([]);
}

async function refreshPreviousRuns(selected = "") {
  const select = $("previous-runs");
  let jobs = [];
  try { jobs = await api("/api/jobs"); } catch { /* keep the old list */ }
  select.replaceChildren(
    el("option", { value: "" }, jobs.length ? "Open a previous run…" : "No runs yet"),
    ...jobs.map((job) => el("option", { value: job.id }, runLabel(job, true))),
  );
  select.disabled = !jobs.length;
  select.value = jobs.some((j) => j.id === selected) ? selected : "";
}

// --- Console ----------------------------------------------------------------------------

/** One console line: ``kind`` is "ok" (a finished step), "bad" (a failure) or "" (news). */
function log(text, kind = "ok", when = new Date()) {
  const list = $("console");
  list.append(el("li", { class: `${kind} new` },
    el("time", {}, `[${clock(when)}]`),
    el("span", { class: "mark", "aria-hidden": "true" }), // drawn in CSS by kind
    el("span", { class: "text" }, text)));
  list.scrollTop = list.scrollHeight;
}

// --- Upload -----------------------------------------------------------------------------

function isImage(name) {
  const dot = name.lastIndexOf(".");
  return dot > 0 && config.extensions.includes(name.slice(dot).toLowerCase());
}

// files dropped or chosen with the photos under these names fill the optional fields
const OPTIONAL_FILES = { "camera.yaml": "camera", "gt_cameras.json": "ground-truth" };

function addPhotos(files) {
  for (const file of files) {
    const field = OPTIONAL_FILES[file.name];
    if (field) {
      setFile($(field), file);
      document.querySelector(".extras").open = true; // show where it went
    } else photos.set(file.name, file);
  }
  renderSelection();
}

function setFile(input, file) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
}

function clearPhotos() {
  photos = new Map();
  $("images").value = "";
  renderSelection();
}

function renderSelection() {
  const files = [...photos.values()];
  for (const [file, url] of thumbUrls) {
    if (!files.includes(file)) { URL.revokeObjectURL(url); thumbUrls.delete(file); }
  }
  const images = files.filter((f) => isImage(f.name));
  const others = files.filter((f) => !isImage(f.name));
  const bytes = images.reduce((sum, f) => sum + f.size, 0);

  $("selection").hidden = files.length === 0;
  $("selection-count").textContent =
    `${plural(images.length, "photo")} · ${megabytes(bytes)}` +
    (others.length ? ` · ${plural(others.length, "other file")} left out` : "");
  $("thumbs").replaceChildren(
    ...files.map((file) => {
      if (!thumbUrls.has(file)) thumbUrls.set(file, URL.createObjectURL(file));
      const img = el("img", { alt: "", loading: "lazy", src: thumbUrls.get(file) });
      img.addEventListener("error", () => img.removeAttribute("src"), { once: true }); // e.g. TIFF
      return el("li", { class: isImage(file.name) ? "" : "rejected", title: file.name }, img, el("span", {}, file.name));
    }),
  );

  let problem = null; // files that aren't photos are left out, not an error
  if (images.length && images.length < config.min_images) problem = `Add at least ${config.min_images} photos.`;
  else if (images.length > config.max_upload_images) problem = `At most ${config.max_upload_images} photos per run.`;
  else if (bytes > config.max_upload_mb * 2 ** 20) problem = `The photos add up to ${megabytes(bytes)}; the limit is ${config.max_upload_mb} MB.`;
  showUploadMessage(problem);

  const run = $("run");
  run.disabled = Boolean(problem) || images.length === 0;
  run.textContent = images.length ? `Reconstruct ${plural(images.length, "photo")}` : "Reconstruct";
}

function showUploadMessage(text, action = null, kind = "error") {
  const box = $("upload-message");
  box.className = `message ${kind}`;
  box.setAttribute("role", kind === "error" ? "alert" : "status");
  box.hidden = !text;
  box.replaceChildren(...(text ? [el("div", {}, text), action].filter(Boolean) : []));
}

async function submit() {
  const form = new FormData();
  for (const file of photos.values()) if (isImage(file.name)) form.append("images", file);
  if ($("camera").files[0]) form.append("camera", $("camera").files[0]);
  if ($("ground-truth").files[0]) form.append("ground_truth", $("ground-truth").files[0]);

  const run = $("run");
  run.disabled = true;
  run.textContent = "Uploading…";
  try {
    const { job_id } = await api("/api/jobs", { method: "POST", body: form });
    clearPhotos();
    $("camera").value = "";
    $("ground-truth").value = "";
    await refreshPreviousRuns(job_id);
    $("console").replaceChildren();
    log(`Uploaded the photos; reconstruction ${job_id} starts`, "");
    openJob(job_id);
  } catch (error) {
    renderSelection();
    const running = error.status === 409 ? error.body?.job_id : null;
    const action = running
      ? el("button", { type: "button", class: "btn btn-quiet btn-small", onclick: () => openJob(running) }, "Show the running reconstruction")
      : null;
    showUploadMessage(error.message, action);
  }
}

// --- This run (left) --------------------------------------------------------------------

function renderThisRun(job, report = null, files = []) {
  $("this-run").hidden = false;
  $("restart-run").hidden = job.status !== "failed";
  $("run-when").textContent = runLabel(job);
  $("run-status").textContent = { done: "done", failed: "failed", running: "running", queued: "queued" }[job.status] ?? job.status;
  const facts = [];
  if (job.photos != null) facts.push(["Photos", String(job.photos)]);
  if (report) {
    const sources = report.calibration.sources.map((s) => CALIBRATION_SOURCES[s] ?? s).join(", ");
    facts.push(["Intrinsics", sources]);
    if (report.features) facts.push(["Features", `${DETECTORS[report.features.detector] ?? report.features.detector} · ${MATCHERS[report.features.matcher] ?? report.features.matcher}`]);
  }
  $("run-facts").replaceChildren(...facts.map(([k, v]) => el("div", {}, el("dt", {}, k), el("dd", {}, v))));
  const thumbs = files.filter((f) => f.startsWith("thumbnails/")).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const shown = thumbs.slice(0, 8);
  const more = (job.photos ?? thumbs.length) - shown.length;
  const items = shown.map((path) => el("li", {}, el("img", { src: fileUrl(job.id, path), alt: path.slice(11, -4), loading: "lazy" })));
  if (more > 0 && shown.length) items.push(el("li", {}, el("span", { class: "more" }, `+${more} more`)));
  $("run-photos").replaceChildren(...items);
  $("run-photos").hidden = !items.length;
  showDeleteControls(job);
}

// --- Pipeline (right) -------------------------------------------------------------------

/** What a finished stage produced, from report.stages (null before the report exists). */
function stageDetail(stage, report) {
  if (!report?.stages) return "";
  const s = report.stages;
  switch (stage) {
    case "dataset": return `${plural(report.images.total, "photo")}`;
    case "features": return s.features && `${count(Math.round(s.features.per_image_median))} per photo`;
    case "matches": return s.matches && `${count(s.matches.matches)} matches`;
    case "verify": return s.verification && `${s.verification.pairs} pairs · ${Math.round(100 * s.verification.inlier_ratio)}% inliers`;
    case "pose": return s.initial_pair ? s.initial_pair.join(" + ") : "none found";
    case "reconstruct": return s.reconstruction && `${s.reconstruction.registered}/${report.images.total} cameras`;
    case "filter": return s.filter && `${count(s.filter.kept)} of ${count(s.filter.total)} kept`;
    default: return "";
  }
}

const cachedStage = (stage, report) =>
  (stage === "features" && report?.stages?.features?.cached) || (stage === "matches" && report?.stages?.matches?.cached);

function renderPipeline(job, report = null) {
  const stages = config?.stages ?? Object.keys(STAGE_LABELS);
  const done = new Set(job?.stages_done ?? []);
  $("pipeline").replaceChildren(
    ...stages.map((stage) => {
      let status = done.has(stage) ? "done" : "pending";
      if (job && stage === job.stage && job.status !== "done") status = job.status === "failed" ? "failed" : "running";
      const time = job?.timings?.[stage];
      const label = status === "running" ? "…" : time === undefined ? "" : cachedStage(stage, report) ? "cached" : `${seconds(time)} s`;
      return el("li", { "data-status": status },
        el("span", { class: "dot", "aria-hidden": "true" }),
        el("span", {},
          el("span", { class: "name" }, STAGE_LABELS[stage] ?? stage,
            el("span", { class: "visually-hidden" }, ` (${status})`)),
          el("span", { class: "detail" }, status === "failed" ? (job.error || "failed") : stageDetail(stage, report) || "")),
        label ? el("span", { class: "time" }, label) : el("span"));
    }),
  );
}

// --- Running ----------------------------------------------------------------------------

async function follow(job) {
  const token = ++pollToken;
  show("running");
  $("previous-runs").value = job.id;
  for (const id of ["summary-card", "export-card", "gt-card", "warnings", "viewer-tools", "layers"]) $(id).hidden = true;
  $("datastrip").replaceChildren();
  setTabs({ poses: false, matches: false, figures: false });
  selectTab("viewer");
  viewer?.setCloud({ positions: new Float32Array(), colours: null });
  viewer?.setCameras([]);
  renderThisRun(job);
  const logged = new Set(); // stages already logged as finished
  let started = null;       // the stage logged as started

  while (token === pollToken) {
    renderPipeline(job);
    for (const stage of job.stages_done) {
      if (logged.has(stage)) continue;
      logged.add(stage);
      const time = job.timings[stage];
      log(`${STAGE_LABELS[stage] ?? stage}${time !== undefined ? ` (${seconds(time)} s)` : ""}`);
    }
    if (job.status === "running" && job.stage && job.stage !== started && !logged.has(job.stage)) {
      started = job.stage;
      log(`${STAGE_LABELS[job.stage] ?? job.stage}…`, "");
    }
    setStatus("Reconstructing", job.stage ? `${STAGE_LABELS[job.stage] ?? job.stage}…` : "Waiting to start…");
    if (job.status === "done") {
      await refreshPreviousRuns(job.id);
      return showResults(job, { replay: true });
    }
    if (job.status === "failed") return showFailure(job);
    await sleep(config.poll_interval_ms);
    if (token !== pollToken) return;
    try {
      job = await api(jobUrl(job.id));
    } catch (error) {
      if (error.status === 404) { // jobs live in server memory: it was restarted mid-run
        show("failed");
        setStatus("Reconstruction lost", "The server was restarted while this reconstruction was running, so it stopped. Start it again.");
        log("The server was restarted; the reconstruction stopped", "bad");
        $("restart-run").hidden = false;
        return;
      }
      setStatus("Lost contact with the server", `${error.message}. Retrying…`);
    }
  }
}

async function showFailure(job) {
  show("failed");
  $("restart-run").hidden = false;
  renderPipeline(job);
  const stage = STAGE_LABELS[job.stage] ?? job.stage;
  setStatus(`Failed: ${stage}`, job.error || "unknown error");
  log(`${stage} failed: ${job.error || "unknown error"}`, "bad");
  refreshPreviousRuns(job.id);
  let files = [];
  try { files = await api(jobUrl(job.id, "/files")); } catch { /* nothing written */ }
  renderThisRun(job, null, files);
  renderFigures(job.id, files);
  const outputs = files.filter((f) => !f.startsWith("thumbnails/"));
  if (outputs.length) {
    $("this-run").querySelector(".partial-files")?.remove();
    $("run-facts").after(el("ul", { class: "partial-files", "aria-label": "Results written before the failure" },
      ...outputs.map((path) => el("li", {}, el("a", { href: fileUrl(job.id, path), target: "_blank", rel: "noopener" }, path)))));
  }
}

// --- Deleting a run ---------------------------------------------------------------------

/** "Delete this run" asks in the page (no browser dialog), then deletes ``shownJob``. */
function showDeleteControls(job) {
  shownJob = job;
  for (const control of document.querySelectorAll(".delete-run")) {
    control.querySelector(".delete-question").textContent = "Delete its results for good?";
    askToDelete(control, false, false);
  }
}

function askToDelete(control, asking, moveFocus = true) {
  control.querySelector('[data-delete="ask"]').hidden = asking;
  control.querySelector(".delete-confirm").hidden = !asking;
  if (moveFocus) control.querySelector(asking ? '[data-delete="cancel"]' : '[data-delete="ask"]').focus();
}

async function deleteShownRun(control) {
  const job = shownJob;
  const confirm = control.querySelector('[data-delete="confirm"]');
  confirm.disabled = true;
  try {
    await api(jobUrl(job.id), { method: "DELETE" });
  } catch (error) {
    control.querySelector(".delete-question").textContent = `Could not delete it: ${error.message}.`;
    return;
  } finally {
    confirm.disabled = false;
  }
  await refreshPreviousRuns();
  clearRun();
  $("previous-runs").focus(); // the card that had focus is gone
  log(`Deleted the run of ${runLabel(job)}`, "");
  showUploadMessage(`Deleted the run of ${runLabel(job)}.`, null, "notice");
}

// --- Results ----------------------------------------------------------------------------

async function showResults(job, { replay = false } = {}) {
  const [report, files] = await Promise.all([api(jobUrl(job.id, "/report")), api(jobUrl(job.id, "/files"))]);
  if (job.run_dir) rememberOpenRun(job.id, job.run_dir.split(/[\\/]/).pop()); // survives a restart
  shownReport = report;
  show("results");
  setStatus("");
  $("previous-runs").value = job.id;
  renderThisRun(job, report, files);
  renderPipeline(job, report);
  if (!replay) logFinishedRun(job, report); // a live run has logged its stages already
  logResults(report, replay ? new Date() : new Date(startedAt(job).getTime() + report.total_s * 1000));

  const warnings = $("warnings");
  warnings.hidden = !report.warnings.length;
  warnings.replaceChildren(...report.warnings.map((w) => el("li", {}, w)));

  renderDataStrip(report);
  renderSummary(report);
  renderGroundTruth(report.ground_truth);
  renderFigures(job.id, files);
  const downloads = DOWNLOADS.filter(([path]) => files.includes(path));
  $("export-card").hidden = !downloads.length;
  $("downloads").replaceChildren(
    ...downloads.map(([path, label, name], k) =>
      el("li", {}, el("a", { href: fileUrl(job.id, path), download: name, class: k === 0 ? "primary" : "" },
        label, el("small", {}, name)))),
  );
  await loadViewer(job.id, report);
  $("replay").disabled = !viewerJob?.cameras;
  if (replay) replayRun();
}

/** Console lines for a run opened from the list: its stages, at the times they finished. */
function logFinishedRun(job, report) {
  $("console").replaceChildren(); // log() scrolls; one layout per line is fine for ~10 lines
  let at = startedAt(job).getTime();
  for (const [stage, time] of Object.entries(report.timings)) {
    at += time * 1000;
    const detail = stageDetail(stage, report);
    log(`${STAGE_LABELS[stage] ?? stage}${detail ? `: ${detail}` : ""}`, "ok", new Date(at));
  }
}

function logResults(report, when) {
  const { images, points, reprojection, ground_truth: gt } = report;
  const parts = [`${images.registered}/${images.total} cameras`];
  if (points) parts.push(`${count(points.filtered)} points`);
  if (reprojection) parts.push(`reprojection ${small(reprojection.median_px)} px`);
  if (gt && !gt.error) parts.push(`rotation error ${small(gt.rotation_deg.median)}° against ground truth`);
  log(`Done in ${seconds(report.total_s)} s: ${parts.join(", ")}`, "ok", when);
}

function cell(label, value, note = null, warn = false) {
  return el("div", {}, el("dt", {}, label), el("dd", { class: warn ? "warn" : "" }, value, note ? " " : null, note ? el("small", {}, note) : null));
}

// Focal length after bundle adjustment (which refines it), with where it started from.
function focal(calibration) {
  const source = calibration.sources.map((s) => CALIBRATION_SOURCES[s] ?? s).join(", ");
  const start = calibration.K[0][0];
  const final = start * (calibration.focal_scale ?? 1);
  const note = Math.round(final) === Math.round(start)  // as displayed: whole pixels
    ? `from ${source}`
    : `refined from ${Math.round(start)} px (${source})`;
  return { value: `${Math.round(final)} px`, note, fallback: calibration.sources.includes("fallback") };
}

function renderDataStrip(report) {
  const { images, points, reprojection, calibration, ground_truth: gt } = report;
  const f = focal(calibration);
  $("datastrip").replaceChildren(
    cell("Cameras", `${images.registered}/${images.total}`, null, images.registered < images.total),
    points ? cell("Points", count(points.filtered), `of ${count(points.sparse)}`) : cell("Points", "none", null, true),
    reprojection ? cell("Reprojection", `${small(reprojection.median_px)} px`, "median") : cell("Reprojection", "–"),
    gt && !gt.error
      ? cell("Rotation error", `${small(gt.rotation_deg.median)}°`, "median")
      : cell("Focal length", f.value, null, f.fallback),
    cell("Time", `${seconds(report.total_s)} s`),
  );
}

function renderSummary(report) {
  const { images, points, reprojection, calibration, stages, ground_truth: gt, features } = report;
  const f = focal(calibration);
  const rows = [
    ["Registered cameras", `${images.registered} / ${images.total}`, images.unregistered.length ? `missing ${images.unregistered.join(", ")}` : null, images.registered < images.total],
    points && ["Points (sparse)", count(points.sparse)],
    points && ["After the outlier filter", count(points.filtered)],
    stages?.verification && ["Inlier matches", count(stages.verification.inliers), `${stages.verification.pairs} verified pairs`],
    reprojection && ["Reprojection error", `${small(reprojection.median_px)} px`, `median · 90th percentile ${small(reprojection.p90_px)} px`],
    ["Focal length", f.value, f.note, f.fallback],
    features && ["Features", DETECTORS[features.detector] ?? features.detector, `${MATCHERS[features.matcher] ?? features.matcher}${features.detector === "disk" || features.matcher === "lightglue" ? " · learned, external" : ""}`],
    gt && !gt.error && ["Rotation error", `${small(gt.rotation_deg.median)}°`, `median · max ${small(gt.rotation_deg.max)}°`],
    gt && !gt.error && ["Position error", percent(gt.centre.median), `median · max ${percent(gt.centre.max)}`],
  ].filter(Boolean);
  $("summary-card").hidden = false;
  $("summary").replaceChildren(...rows.map(([label, value, note, warn]) => cell(label, value, note, warn)));
}

// --- Ground truth -----------------------------------------------------------------------

function renderGroundTruth(gt) {
  const usable = gt && !gt.error;
  setTabs({ poses: Boolean(gt) });
  $("gt-card").hidden = !usable;
  if (!gt) return;
  if (gt.error) {
    $("gt-summary").replaceChildren(el("div", {}, el("dt", {}, "Could not score against ground truth"), el("dd", {}, gt.error)));
    $("gt-rows").replaceChildren();
    return;
  }
  $("gt-summary").replaceChildren(
    cell("Cameras scored", String(gt.cameras)),
    cell("Rotation error", `${small(gt.rotation_deg.median)}° median · ${small(gt.rotation_deg.max)}° max`),
    cell("Position error", `${percent(gt.centre.median)} median · ${percent(gt.centre.max)} max`),
  );
  $("gt-mini").replaceChildren(
    cell("Rotation error", `${small(gt.rotation_deg.median)}°`, "median"),
    cell("Position error", percent(gt.centre.median), "median"),
  );
  const n = gt.cameras;
  $("gt-ring").innerHTML = Array.from({ length: n }, (_, k) => {
    const a = (k / n) * 2 * Math.PI;
    return `<circle class="ring-camera" cx="${(60 + 52 * Math.cos(a)).toFixed(1)}" cy="${(40 + 30 * Math.sin(a)).toFixed(1)}" r="1.8"/>`;
  }).join("") + '<circle class="ring-object" cx="60" cy="40" r="7"/>';
  renderGroundTruthRows();
}

// An error with a bar scaled to the worst camera's, so outliers stand out in a long list.
function errorCell(value, max, text) {
  const share = max > 0 ? value / max : 0;
  return el("td", {},
    el("span", { class: "bar", style: `--share: ${share.toFixed(3)}`, "aria-hidden": "true" }),
    el("span", { class: "bar-value" }, text));
}

function renderGroundTruthRows() {
  const gt = shownReport?.ground_truth;
  if (!gt || gt.error) return;
  const { key, descending } = gtSort;
  const byName = (a, b) => a.image.localeCompare(b.image, undefined, { numeric: true });
  const rows = [...gt.per_camera].sort((a, b) =>
    (key === "image" ? byName(a, b) : a[key] - b[key] || byName(a, b)) * (descending ? -1 : 1));
  const maxRot = Math.max(...gt.per_camera.map((c) => c.rotation_deg));
  const maxCentre = Math.max(...gt.per_camera.map((c) => c.centre));
  const picked = labels?.pinned >= 0 ? viewerJob?.cameras?.[labels.pinned]?.name : null;
  $("gt-rows").replaceChildren(
    ...rows.map((c) =>
      el("tr", { class: [c.rotation_deg === maxRot || c.centre === maxCentre ? "worst" : "", c.image === picked ? "picked" : ""].join(" ").trim() },
        el("td", {}, c.image,
          c.rotation_deg === maxRot || c.centre === maxCentre ? el("span", { class: "visually-hidden" }, " (worst camera)") : null,
          c.image === picked ? el("span", { class: "visually-hidden" }, " (selected in the 3-D view)") : null),
        errorCell(c.rotation_deg, maxRot, `${small(c.rotation_deg)}°`),
        errorCell(c.centre, maxCentre, percent(c.centre)))),
  );
  for (const th of document.querySelectorAll(".gt-table th")) {
    const sorted = th.querySelector("[data-sort]").dataset.sort === key;
    th.setAttribute("aria-sort", sorted ? (descending ? "descending" : "ascending") : "none");
  }
}

// Photos in name order; errors worst first on the first click.
function sortGroundTruth(key) {
  gtSort = gtSort.key === key
    ? { key, descending: !gtSort.descending }
    : { key, descending: key !== "image" };
  renderGroundTruthRows();
}

// --- Tabs -------------------------------------------------------------------------------

/** Show or hide tabs: ``{poses: true, ...}`` (the 3-D view is always there). */
function setTabs(visible) {
  for (const [name, on] of Object.entries(visible)) {
    $(`tab-${name}`).hidden = !on;
    if (!on && $(`tab-${name}`).getAttribute("aria-selected") === "true") selectTab("viewer");
  }
}

function selectTab(name, focus = false) {
  for (const tab of document.querySelectorAll('[role="tab"]')) {
    const on = tab.dataset.tab === name;
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
    if (on && focus) tab.focus();
  }
  for (const panel of document.querySelectorAll("[data-panel]")) panel.hidden = panel.dataset.panel !== name;
}

/** Arrow keys move between the visible tabs (WAI-ARIA tabs pattern). */
function onTabKey(event) {
  const tabs = [...document.querySelectorAll('[role="tab"]')].filter((t) => !t.hidden);
  const at = tabs.indexOf(document.activeElement);
  if (at < 0) return;
  const next = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: tabs.length - 1 }[event.key];
  if (next === undefined) return;
  event.preventDefault();
  selectTab(tabs[(next + tabs.length) % tabs.length].dataset.tab, true);
}

// --- Figures and the lightbox -----------------------------------------------------------

function renderFigures(id, files) {
  for (const [tab, figures] of Object.entries(FIGURES)) {
    const items = figures.filter(([path]) => files.includes(path)).map(([path, caption]) => ({ href: fileUrl(id, path), caption }));
    setTabs({ [tab]: items.length > 0 });
    $(`gallery-${tab}`).replaceChildren(
      ...items.map(({ href, caption }, index) => {
        const link = el("a", { href, target: "_blank", rel: "noopener" }, el("img", { src: href, alt: caption, loading: "lazy" }));
        link.addEventListener("click", (e) => {
          if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return; // a new tab as asked
          e.preventDefault();
          openLightbox(items, index);
        });
        return el("figure", {}, link, el("figcaption", {}, caption));
      }),
    );
  }
}

function openLightbox(items, index) {
  lightboxItems = items;
  showLightboxItem(index);
  const dialog = $("lightbox");
  if (!dialog.open) dialog.showModal();
}

function showLightboxItem(index) {
  const n = lightboxItems.length;
  lightboxIndex = (index + n) % n;
  const { href, caption } = lightboxItems[lightboxIndex];
  $("lightbox-image").src = href;
  $("lightbox-image").alt = caption;
  $("lightbox-caption").textContent = caption;
  $("lightbox-count").textContent = n > 1 ? `${lightboxIndex + 1} / ${n}` : "";
  $("lightbox-open").href = href;
  for (const b of document.querySelectorAll("#lightbox [data-step]")) b.hidden = n < 2;
}

// --- Viewer -----------------------------------------------------------------------------

async function fetchCloud(id, name) {
  const response = await fetch(fileUrl(id, `point_clouds/${name}.ply`));
  if (!response.ok) throw new Error(`${name}.ply is missing`);
  return parsePly(await response.arrayBuffer());
}

async function loadViewer(id, report) {
  try {
    viewer ??= createViewer();
  } catch (error) { // no WebGL: the numbers, tables and figures still work
    setStatus("3-D view unavailable", `This browser could not start WebGL (${error.message}). The summary, tables and figures below are complete.`);
    return;
  }
  viewerJob = { id, clouds: {}, cameras: null };
  setPressed("cloud", "filtered");
  hideCameraLabel();
  if (!report.points) {
    viewer.setCloud({ positions: new Float32Array(), colours: null });
    viewer.setCameras([]);
    setStatus("No point cloud", "The reconstruction did not get past the first camera pair.");
    return;
  }
  setStatus("Loading the point cloud…");
  try {
    const [cloud, cameras] = await Promise.all([
      fetchCloud(id, "filtered"),
      api(fileUrl(id, "camera_poses/cameras.json")),
    ]);
    if (viewerJob.id !== id) return; // another run was opened meanwhile
    viewerJob.clouds.filtered = cloud;
    viewerJob.cameras = cameras.cameras;
    viewer.setCloud(cloud);
    viewer.setCameras(cameras.cameras);
    viewer.reveal(1, Infinity);
    viewer.fit("overview");
    $("layer-points-count").textContent = count(cloud.count);
    $("layer-cameras-count").textContent = String(cameras.cameras.length);
    $("viewer-tools").hidden = $("layers").hidden = false;
    setStatus("");
  } catch (error) {
    setStatus("Could not show the point cloud", error.message);
  }
}

function createViewer() {
  const v = new Viewer($("viewer"));
  v.setPointSize(Number($("point-size").value));
  labels = cameraLabels(v, $("cam-label"), (k) => {
    const camera = viewerJob?.cameras?.[k];
    if (!camera) return null;
    const gt = shownReport?.ground_truth?.per_camera?.find((c) => c.image === camera.name);
    return [el("b", {}, `Camera ${k + 1}`), ` · ${camera.name}`, gt ? ` · ${small(gt.rotation_deg)}°` : ""];
  }, { onPin: renderGroundTruthRows, announce: (text) => { $("viewer-announce").textContent = text; } });
  return v;
}

function hideCameraLabel() {
  labels?.unpin();
}

/** The run again at presentation speed: cameras appear in registration order and the
 * cloud grows, while the view turns. Instant under reduced motion. */
async function replayRun() {
  if (!viewer || !viewerJob?.cameras || replaying) return;
  replaying = true;
  const id = viewerJob.id;
  const n = viewerJob.cameras.length;
  const button = $("replay");
  button.setAttribute("aria-disabled", "true"); // stays focusable while it plays
  hideCameraLabel();
  if (!reducedMotion()) {
    setSpin(true);
    for (let k = 0; k <= n; k++) {
      if (!replaying || viewerJob?.id !== id) break;
      viewer.reveal(k / n, k);
      await sleep(Math.max(40, 3600 / n));
    }
  }
  if (viewerJob?.id === id) viewer.reveal(1, Infinity);
  replaying = false;
  button.removeAttribute("aria-disabled");
}

function setSpin(on) {
  const spinning = viewer?.spin(on) ?? false;
  $("spin").setAttribute("aria-pressed", String(spinning));
}

/** Press the button of a segmented control (``data-<key>`` buttons) whose value is ``value``. */
function setPressed(key, value) {
  for (const button of document.querySelectorAll(`[data-${key}]`)) {
    button.setAttribute("aria-pressed", String(button.dataset[key] === value));
  }
}

async function switchCloud(name) {
  if (!viewerJob) return;
  const { id, clouds } = viewerJob;
  setPressed("cloud", name);
  try {
    clouds[name] ??= await fetchCloud(id, name);
    if (viewerJob.id !== id) return;
    viewer.setCloud(clouds[name]);
    $("layer-points-count").textContent = count(clouds[name].count);
  } catch (error) {
    setStatus("Could not load that cloud", error.message);
  }
}

// --- Wiring -----------------------------------------------------------------------------

function wire() {
  const dropzone = $("dropzone");
  $("images").addEventListener("change", (e) => addPhotos(e.target.files));
  for (const type of ["dragenter", "dragover"]) {
    dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.add("dragging"); });
  }
  for (const type of ["dragleave", "drop"]) {
    dropzone.addEventListener(type, () => dropzone.classList.remove("dragging"));
  }
  dropzone.addEventListener("drop", (e) => { e.preventDefault(); addPhotos(e.dataTransfer.files); });
  $("clear").addEventListener("click", clearPhotos);
  $("run").addEventListener("click", submit);
  for (const control of document.querySelectorAll(".delete-run")) {
    control.querySelector('[data-delete="ask"]').addEventListener("click", () => askToDelete(control, true));
    control.querySelector('[data-delete="cancel"]').addEventListener("click", () => askToDelete(control, false));
    control.querySelector('[data-delete="confirm"]').addEventListener("click", () => deleteShownRun(control));
  }
  $("previous-runs").addEventListener("change", (e) => (e.target.value ? openJob(e.target.value) : clearRun()));
  $("replay").addEventListener("click", replayRun);
  $("restart-run").addEventListener("click", () => {
    clearRun();
    $("images").focus(); // straight to choosing photos again
  });

  for (const tab of document.querySelectorAll('[role="tab"]')) tab.addEventListener("click", () => selectTab(tab.dataset.tab));
  document.querySelector(".tabs").addEventListener("keydown", onTabKey);

  for (const button of document.querySelectorAll("[data-cloud]")) button.addEventListener("click", () => switchCloud(button.dataset.cloud));
  for (const button of document.querySelectorAll("[data-fit]")) button.addEventListener("click", () => viewer?.fit(button.dataset.fit));
  for (const button of document.querySelectorAll("[data-tool]")) {
    button.addEventListener("click", () => { viewer?.setTool(button.dataset.tool); setPressed("tool", button.dataset.tool); });
  }
  for (const button of document.querySelectorAll("[data-projection]")) {
    button.addEventListener("click", () => { viewer?.setProjection(button.dataset.projection); setPressed("projection", button.dataset.projection); });
  }
  $("spin").addEventListener("click", () => setSpin($("spin").getAttribute("aria-pressed") !== "true"));
  for (const box of document.querySelectorAll("[data-layer]")) box.addEventListener("change", () => viewer?.setLayer(box.dataset.layer, box.checked));
  $("point-size").addEventListener("input", (e) => viewer?.setPointSize(Number(e.target.value)));

  for (const button of document.querySelectorAll("[data-sort]")) button.addEventListener("click", () => sortGroundTruth(button.dataset.sort));
  const lightbox = $("lightbox");
  for (const button of lightbox.querySelectorAll("[data-step]")) {
    button.addEventListener("click", () => showLightboxItem(lightboxIndex + Number(button.dataset.step)));
  }
  $("lightbox-close").addEventListener("click", () => lightbox.close());
  lightbox.addEventListener("click", (e) => { // the backdrop, not the dialog's own padding
    const r = lightbox.getBoundingClientRect();
    if (e.target === lightbox && (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)) lightbox.close();
  });
  lightbox.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      showLightboxItem(lightboxIndex + (e.key === "ArrowRight" ? 1 : -1));
    }
  });
}

async function init() {
  wire();
  config = await api("/api/config");
  // the optional files too, so they can be chosen with the photos (see OPTIONAL_FILES)
  const optional = Object.keys(OPTIONAL_FILES).map((name) => name.slice(name.lastIndexOf(".")));
  $("images").accept = ["image/*", ...config.extensions, ...optional].join(",");
  $("dropzone-hint").textContent =
    `At least ${config.min_images} photos (${config.extensions.map((e) => e.slice(1).toUpperCase()).join(", ")}). ` +
    "Walk round the subject so neighbouring photos overlap by about two thirds.";
  renderPipeline(null);
  await refreshPreviousRuns();
  // Empty on a first visit; after a reload, the run that was on screen (until the user
  // clears it or deletes it). Nothing is run here: a running job is only followed.
  if (rememberedOpenRun().length) await reopenRememberedRun();
  else clearRun();
}

init().catch((error) => showUploadMessage(`The page could not start: ${error.message}`));

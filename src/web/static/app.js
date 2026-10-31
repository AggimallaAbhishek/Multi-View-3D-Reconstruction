// The page: upload photos, follow the job, show the results. All logic about what is
// valid or what a run contains lives in the API (src/web/app.py); this only presents it.

import { Viewer, parsePly } from "./viewer.js";

const $ = (id) => document.getElementById(id);

const STAGE_LABELS = {
  dataset: "Load and check photos",
  features: "Detect features",
  matches: "Match features between photos",
  verify: "Verify matches geometrically",
  pose: "Estimate the first camera pair",
  triangulate: "Triangulate the first points",
  cloud: "Build the two-view cloud",
  reconstruct: "Register remaining photos, bundle adjustment",
  filter: "Remove outlier points",
  outputs: "Write results and figures",
};

// What the page shows of a run folder, by path (written by src/outputs.py); only files the
// folder actually holds are shown (GET /api/jobs/{id}/files).
const DIAGNOSTICS = [
  ["matches/match_matrix.png", "Feature matches between every pair of photos"],
  ["verification/verified_matrix.png", "Matches kept after geometric verification"],
  ["pose/initial_pair.png", "The first camera pair"],
  ["reconstruction/reprojection_histogram.png", "Reprojection error of every observation"],
  ["reconstruction/cameras.png", "Recovered camera positions"],
  ["figures/filtered.png", "Filtered point cloud"],
];

const DOWNLOADS = [
  ["point_clouds/filtered.ply", "Point cloud", "filtered.ply"],
  ["camera_poses/cameras.json", "Camera poses", "cameras.json"],
  ["report.json", "Report", "report.json"],
];

const CALIBRATION_SOURCES = { file: "camera.yaml", exif: "EXIF", fallback: "a guess" };

let config = null;      // GET /api/config
let photos = new Map(); // name -> File
const thumbUrls = new Map(); // File -> object URL, made once per file
let pollToken = 0;      // bumps to cancel a running poll loop
let viewer = null;
let viewerJob = null;   // {id, clouds: {filtered, sparse}}

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

const count = (n) => n.toLocaleString("en-GB").replace(/,/g, " "); // thin-space groups
const seconds = (s) => (s < 10 ? s.toFixed(1) : Math.round(s).toString());
const small = (x) => (x < 0.1 ? x.toFixed(3) : x < 10 ? x.toFixed(2) : x.toFixed(1));
const percent = (fraction) => `${small(fraction * 100)}%`;
const megabytes = (bytes) => `${(bytes / 2 ** 20).toFixed(1)} MB`;
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.filter((c) => c !== null && c !== undefined));
  return node;
}

/** A job's name: the start time from its run folder (<YYYYmmdd-HHMMSS>_web_reconstruct[_n]),
 * else its id. The dropdown adds the photo count, the cameras registered and the status of
 * unfinished or failed jobs, so runs of the same day can be told apart. */
function runLabel(job, detailed = false) {
  const name = job.run_dir ? job.run_dir.split("/").pop() : job.id;
  const m = name.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/);
  const when = m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]}` : `Run ${name}`;
  if (!detailed) return when;
  const parts = [when];
  if (job.photos != null) parts.push(plural(job.photos, "photo"));
  if (job.registered != null) parts.push(`${job.registered}/${job.photos} registered`);
  const label = parts.join(" · ");
  return job.status === "done" ? label : `${label} (${job.status})`;
}

// --- State ------------------------------------------------------------------------------

function show(state) {
  document.body.dataset.state = state;
  window.scrollTo({ top: 0 });
}

function openJobInUrl(id) {
  history.replaceState(null, "", id ? `#job=${encodeURIComponent(id)}` : location.pathname);
}

async function openJob(id) {
  pollToken++;
  openJobInUrl(id);
  try {
    const job = await api(jobUrl(id));
    if (job.status === "done") await showResults(job);
    else follow(job);
  } catch (error) {
    openJobInUrl(null);
    show("upload");
    showUploadMessage(`Could not open run ${id}: ${error.message}`);
  }
}

function startOver() {
  pollToken++;
  openJobInUrl(null);
  $("previous-runs").value = "";
  show("upload");
}

async function refreshPreviousRuns(selected = "") {
  const select = $("previous-runs");
  let jobs = [];
  try { jobs = await api("/api/jobs"); } catch { /* keep the old list */ }
  select.replaceChildren(
    el("option", { value: "" }, jobs.length ? "Choose a run" : "No runs yet"),
    ...jobs.map((job) => el("option", { value: job.id }, runLabel(job, true))),
  );
  select.disabled = !jobs.length;
  select.value = jobs.some((j) => j.id === selected) ? selected : "";
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
    if (field) setFile($(field), file);
    else photos.set(file.name, file);
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

function showUploadMessage(text, action = null) {
  const box = $("upload-message");
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
    openJob(job_id);
  } catch (error) {
    renderSelection();
    const running = error.status === 409 ? error.body?.job_id : null;
    const action = running
      ? el("button", { type: "button", class: "quiet", onclick: () => openJob(running) }, "Show the running reconstruction")
      : null;
    showUploadMessage(error.message, action);
  }
}

// --- Running ----------------------------------------------------------------------------

function renderStages(job) {
  const done = new Set(job.stages_done);
  $("stages").replaceChildren(
    ...config.stages.map((stage) => {
      let status = done.has(stage) ? "done" : "pending";
      if (stage === job.stage && job.status !== "done") status = job.status === "failed" ? "failed" : "running";
      const mark =
        status === "done" ? "✓" :
        status === "failed" ? "✗" :
        status === "running" ? el("span", { class: "spinner", "aria-hidden": "true" }) : "·";
      const time = job.timings[stage] !== undefined ? `${seconds(job.timings[stage])} s` : "";
      return el("li", { "data-status": status },
        el("span", { class: "mark" }, mark),
        el("span", {}, STAGE_LABELS[stage] ?? stage),
        el("span", { class: "time" }, time));
    }),
  );
}

async function follow(job) {
  const token = ++pollToken;
  show("running");
  $("previous-runs").value = job.id;
  $("running-id").textContent = runLabel(job);
  $("failure").hidden = true;
  $("partial").hidden = true;
  $("restart-failed").hidden = true;
  $("running-title").textContent = "Reconstructing";

  while (token === pollToken) {
    renderStages(job);
    if (job.status === "done") {
      await refreshPreviousRuns(job.id);
      return showResults(job);
    }
    if (job.status === "failed") return showFailure(job);
    await new Promise((resolve) => setTimeout(resolve, config.poll_interval_ms));
    if (token !== pollToken) return;
    try {
      job = await api(jobUrl(job.id));
      $("failure").hidden = true; // back in contact after a blip
    } catch (error) {
      $("failure").hidden = false;
      if (error.status === 404) { // jobs live in server memory: it was restarted mid-run
        $("running-title").textContent = "Reconstruction lost";
        $("failure").textContent = "The server was restarted while this reconstruction was running, so it stopped. Start it again.";
        $("restart-failed").hidden = false;
        return;
      }
      $("failure").textContent = `Lost contact with the server: ${error.message}. Retrying…`;
    }
  }
}

function showFailure(job) {
  $("running-title").textContent = "Reconstruction failed";
  $("failure").hidden = false;
  const stage = STAGE_LABELS[job.stage] ?? job.stage;
  $("failure").replaceChildren(el("strong", {}, `${stage}: `), job.error || "unknown error");
  $("restart-failed").hidden = false;
  refreshPreviousRuns(job.id);
  api(jobUrl(job.id, "/files")).then((files) => {
    $("partial").hidden = !files.length;
    renderGallery($("partial-gallery"), job.id, files);
    $("partial-files").replaceChildren(
      ...files.map((path) => el("li", {}, el("a", { href: fileUrl(job.id, path), target: "_blank", rel: "noopener" }, path))),
    );
  }).catch(() => {});
}

// --- Results ----------------------------------------------------------------------------

async function showResults(job) {
  const [report, files] = await Promise.all([api(jobUrl(job.id, "/report")), api(jobUrl(job.id, "/files"))]);
  show("results");
  $("previous-runs").value = job.id;
  $("results-id").textContent = runLabel(job);

  const warnings = $("warnings");
  warnings.hidden = !report.warnings.length;
  warnings.replaceChildren(...report.warnings.map((w) => el("li", {}, w)));

  renderDataStrip(report);
  renderGroundTruth(report.ground_truth);
  renderGallery($("gallery"), job.id, files);
  $("downloads").replaceChildren(
    ...DOWNLOADS.filter(([path]) => files.includes(path)).map(([path, label, name]) =>
      el("li", {}, el("a", { href: fileUrl(job.id, path), download: name }, label, el("small", {}, name)))),
  );
  await loadViewer(job.id, report);
}

function cell(label, value, note = null, warn = false) {
  return el("div", {}, el("dt", {}, label), el("dd", { class: warn ? "warn" : "" }, value, note ? " " : null, note ? el("small", {}, note) : null));
}

function renderDataStrip(report) {
  const { images, points, reprojection, calibration } = report;
  const fallback = calibration.sources.includes("fallback");
  $("datastrip").replaceChildren(
    el("dl", { class: "cells" },
      cell("Photos registered", `${images.registered}/${images.total}`, null, images.registered < images.total),
      points
        ? cell("Points", count(points.filtered), `of ${count(points.sparse)}`)
        : cell("Points", "none", null, true),
      reprojection
        ? cell("Reprojection error", `${small(reprojection.median_px)} px`, `median · p90 ${small(reprojection.p90_px)}`)
        : cell("Reprojection error", "–"),
      cell("Focal length from", calibration.sources.map((s) => CALIBRATION_SOURCES[s] ?? s).join(", "), null, fallback),
      cell("Time", `${seconds(report.total_s)} s`)),
  );
}

function renderGroundTruth(gt) {
  const panel = $("gt-panel");
  panel.hidden = !gt;
  if (!gt) return;
  if (gt.error) {
    $("gt-summary").replaceChildren(el("p", { class: "message error" }, `Could not score against ground truth: ${gt.error}`));
    $("gt-rows").replaceChildren();
    return;
  }
  $("gt-summary").replaceChildren(
    el("dl", { class: "cells" },
      cell("Cameras scored", String(gt.cameras)),
      cell("Rotation error", `${small(gt.rotation_deg.median)}° median · ${small(gt.rotation_deg.max)}° max`),
      cell("Position error", `${percent(gt.centre.median)} median · ${percent(gt.centre.max)} max`)),
  );
  $("gt-rows").replaceChildren(
    ...[...gt.per_camera].sort((a, b) => a.image.localeCompare(b.image, undefined, { numeric: true })).map((c) =>
      el("tr", {}, el("td", {}, c.image), el("td", {}, `${small(c.rotation_deg)}°`), el("td", {}, percent(c.centre)))),
  );
}

/** The diagnostic figures among the run's ``files``. */
function renderGallery(container, id, files) {
  container.replaceChildren(
    ...DIAGNOSTICS.filter(([path]) => files.includes(path)).map(([path, caption]) =>
      el("figure", {},
        el("a", { href: fileUrl(id, path), target: "_blank", rel: "noopener" },
          el("img", { src: fileUrl(id, path), alt: caption, loading: "lazy" })),
        el("figcaption", {}, caption))),
  );
}

// --- Viewer -----------------------------------------------------------------------------

async function fetchCloud(id, name) {
  const response = await fetch(fileUrl(id, `point_clouds/${name}.ply`));
  if (!response.ok) throw new Error(`${name}.ply is missing`);
  return parsePly(await response.arrayBuffer());
}

async function loadViewer(id, report) {
  const status = $("viewer-status");
  viewer ??= new Viewer($("viewer"));
  viewerJob = { id, clouds: {} };
  setCloudButtons("filtered");
  document.querySelector(".viewer-controls").hidden = !report.points;
  if (!report.points) {
    viewer.setCloud({ positions: new Float32Array(), colours: null });
    status.textContent = "No point cloud: the reconstruction did not get past the first camera pair.";
    return;
  }
  status.textContent = "Loading point cloud…";
  try {
    const [cloud, cameras] = await Promise.all([
      fetchCloud(id, "filtered"),
      api(fileUrl(id, "camera_poses/cameras.json")),
    ]);
    if (viewerJob.id !== id) return; // another run was opened meanwhile
    viewerJob.clouds.filtered = cloud;
    viewer.setCloud(cloud);
    viewer.setCameras(cameras.cameras);
    viewer.fit();
    status.textContent = "";
  } catch (error) {
    status.textContent = `Could not show the point cloud: ${error.message}`;
  }
}

function setCloudButtons(active) {
  for (const button of document.querySelectorAll("[data-cloud]")) {
    button.setAttribute("aria-pressed", String(button.dataset.cloud === active));
  }
}

async function switchCloud(name) {
  if (!viewerJob) return;
  const { id, clouds } = viewerJob;
  setCloudButtons(name);
  try {
    clouds[name] ??= await fetchCloud(id, name);
    if (viewerJob.id === id) viewer.setCloud(clouds[name]);
  } catch (error) {
    $("viewer-status").textContent = error.message;
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
  $("restart").addEventListener("click", startOver);
  $("restart-failed").addEventListener("click", startOver);
  $("previous-runs").addEventListener("change", (e) => (e.target.value ? openJob(e.target.value) : startOver()));
  for (const button of document.querySelectorAll("[data-cloud]")) {
    button.addEventListener("click", () => switchCloud(button.dataset.cloud));
  }
  $("point-size").addEventListener("input", (e) => viewer?.setPointSize(Number(e.target.value)));
  $("fit").addEventListener("click", () => viewer?.fit());
}

async function init() {
  wire();
  config = await api("/api/config");
  // the optional files too, so they can be chosen with the photos (see OPTIONAL_FILES)
  const optional = Object.keys(OPTIONAL_FILES).map((name) => name.slice(name.lastIndexOf(".")));
  $("images").accept = ["image/*", ...config.extensions, ...optional].join(",");
  $("dropzone-hint").textContent =
    `At least ${config.min_images} photos (${config.extensions.map((e) => e.slice(1).toUpperCase()).join(", ")}). ` +
    "Walk around the subject so neighbouring photos overlap by about two thirds.";
  await refreshPreviousRuns();
  openJobFromUrl();
  // a #job= link pasted into an open tab, or edited by hand
  window.addEventListener("hashchange", openJobFromUrl);
}

function openJobFromUrl() {
  const match = location.hash.match(/^#job=(.+)$/);
  if (match) openJob(decodeURIComponent(match[1]));
  else startOver();
}

init().catch((error) => showUploadMessage(`The page could not start: ${error.message}`));

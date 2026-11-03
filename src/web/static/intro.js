// The intro page: the committed showcase (showcase/, made by scripts/make_showcase.py from
// a real run) shown live in the hero, its numbers filled in, and the pinned "how it was
// made" story. The page reads fully without JavaScript; this adds the live parts.

import { cameraLabels, count, small } from "./ui.js";
import { Viewer, parsePly } from "./viewer.js";

const $ = (id) => document.getElementById(id);
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

document.documentElement.classList.add("js");

/** Every element with ``data-n="<key>"`` gets ``values[key]``. */
function fill(values) {
  for (const node of document.querySelectorAll("[data-n]")) {
    const value = values[node.dataset.n];
    if (value !== undefined && value !== null) node.textContent = value;
  }
}

async function loadShowcase() {
  const [showcase, temple] = await Promise.all([
    fetch("showcase/showcase.json").then((r) => r.json()),
    fetch("showcase/temple.json").then((r) => r.json()),
  ]);
  const { report, frames } = showcase;
  const frame = Object.fromEntries(frames.map((f) => [f.file.replace(".jpg", ""), f]));
  const gt = report.ground_truth && !report.ground_truth.error ? report.ground_truth : null;
  fill({
    photos: String(report.images.total),
    cameras: `${report.images.registered} / ${report.images.total}`,
    "cameras-short": String(report.images.registered),
    points: count(report.points.filtered),
    rotation: gt ? `${small(gt.rotation_deg.median)}°` : null,
    reprojection: `${small(report.reprojection.median_px)} px`,
    keypoints: count(frame.features.keypoints),
    verified: count(frame.matches.verified),
    matches: count(frame.matches.matches),
    "temple-views": String(temple.views),
    "temple-keypoints": count(temple.keypoints),
    "temple-verified": count(temple.verified),
    "temple-matches": count(temple.matches),
  });
  renderPhotos(showcase.photos ?? []);
  renderFragment(report);
  return { showcase, frame, gt, temple };
}

function renderPhotos(photos) {
  $("photo-row").replaceChildren(...photos.map(({ file, name }) => {
    const item = document.createElement("li");
    const figure = document.createElement("figure");
    const img = Object.assign(document.createElement("img"), { src: `showcase/${file}`, alt: `templeRing photo ${name}`, loading: "lazy", width: 480, height: 360 });
    const caption = Object.assign(document.createElement("figcaption"), { textContent: name.replace(/\.[^.]+$/, "") });
    figure.append(img, caption);
    item.append(figure);
    return item;
  }));
}

/** The workstation's pipeline column for the showcase run, with its real times. */
function renderFragment(report) {
  const t = report.timings;
  const s = report.stages ?? {};
  const time = (x, cached) =>
    cached ? "cached" : Number.isFinite(x) ? `${x < 10 ? x.toFixed(1) : Math.round(x)} s` : "";
  const sum = (...keys) => keys.reduce((total, key) => total + (t[key] ?? 0), 0);
  const rows = [
    ["Load and check photos", `${report.images.total} photos`, time(t.dataset)],
    ["Detect features", s.features ? `${count(Math.round(s.features.per_image_median))} per photo` : "", time(t.features, s.features?.cached)],
    ["Match features", s.matches ? `${count(s.matches.matches)} matches` : "", time(t.matches, s.matches?.cached)],
    ["Verify geometry", s.verification ? `${s.verification.pairs} pairs · ${Math.round(100 * s.verification.inlier_ratio)}% inliers` : "", time(t.verify)],
    ["Register cameras, adjust", s.reconstruction ? `${s.reconstruction.registered}/${report.images.total} cameras` : "", time(sum("pose", "triangulate", "cloud", "reconstruct"))],
    ["Remove outliers", s.filter ? `${count(s.filter.kept)} of ${count(s.filter.total)} kept` : "", time(t.filter)],
  ];
  $("fragment-pipeline").replaceChildren(...rows.map(([name, detail, when]) => {
    const li = document.createElement("li");
    li.innerHTML = '<span class="dot" aria-hidden="true"></span><span><b></b><small></small></span><span class="time"></span>';
    li.querySelector("b").textContent = name;
    li.querySelector("small").textContent = detail;
    li.querySelector(".time").textContent = when;
    return li;
  }));
}

// --- Hero: the reconstruction, live -----------------------------------------------------

async function startHero({ showcase, gt }) {
  const container = $("hero-viewer");
  const viewer = new Viewer(container);
  const [cloud, cameras] = await Promise.all([
    fetch("showcase/cloud.ply").then((r) => r.arrayBuffer()).then(parsePly),
    fetch("showcase/cameras.json").then((r) => r.json()),
  ]);
  viewer.setPointSize(2.5);
  viewer.setCloud(cloud);
  viewer.setCameras(cameras.cameras);
  viewer.setLayer("grid", true);
  viewer.fit("overview");
  $("hero-caption").textContent = `${showcase.name} · ${count(cloud.count)} points · drag to orbit`;

  // Hold the orbit while the visitor drives it, and while the hero is off screen.
  // Paused by the button, by the visitor moving the view (pointer or keys), off screen, or
  // under reduced motion (WCAG 2.2.2: anything moving for more than 5 s can be paused).
  let visible = true;
  let held = false;
  const pause = $("pause");
  const spin = () => {
    const turning = viewer.spin(visible && !held && !reducedMotion);
    pause.setAttribute("aria-pressed", String(held));
    pause.textContent = held ? "Resume rotation" : "Pause rotation";
    return turning;
  };
  const hold = () => { held = true; spin(); };
  viewer.controls.addEventListener("start", hold);
  container.addEventListener("keydown", (e) => { if (e.key.startsWith("Arrow")) hold(); });
  pause.hidden = reducedMotion;
  pause.addEventListener("click", () => { held = !held; spin(); });
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; spin(); }).observe(container);
  spin();

  // Camera labels: hover shows one, a click keeps it.
  const errors = Object.fromEntries((gt?.per_camera ?? []).map((c) => [c.image, c.rotation_deg]));
  const announcer = $("hero-announce");
  cameraLabels(viewer, $("cam-label"), (k) => {
    const name = cameras.cameras[k].name;
    const bold = Object.assign(document.createElement("b"), { textContent: `Camera ${k + 1}` });
    return [bold, ` · ${name}${errors[name] !== undefined ? ` · ${small(errors[name])}° from the truth` : ""}`];
  }, {
    announce: (text) => {
      if (text) { held = true; spin(); } // a camera chosen by keyboard holds the view still
      announcer.textContent = text;
    },
  });
}

// --- How it was made: the pinned stage follows the step in view ---------------------------

function startStory({ frame, temple }) {
  const steps = [...document.querySelectorAll(".step")];
  const layers = [$("stage-a"), $("stage-b")];
  const caption = $("stage-caption");
  const describe = {
    "temple-photos": () => [`The temple seen from ${temple.views} positions`, `The temple seen from ${temple.views} positions`],
    "temple-features": () => [`The temple with its ${count(temple.keypoints)} SIFT keypoints`, `${count(temple.keypoints)} SIFT keypoints on one view`],
    "temple-matches": () => [`${count(temple.verified)} verified matches between two views of the temple`, `${count(temple.verified)} of ${count(temple.matches)} matches verified (150 drawn)`],
    cameras: () => [`The ${frame.cameras.cameras} recovered cameras round the benchmark temple`, `Benchmark temple: ${frame.cameras.cameras} recovered cameras`],
    cloud: () => [`The filtered point cloud of ${count(frame.cloud.points)} points`, `Benchmark temple: ${count(frame.cloud.points)} points after filtering`],
  };
  let current = "temple-photos";
  let front = 0; // the layer on show
  const select = (name) => {
    if (name === current) return;
    current = name;
    for (const step of steps) step.classList.toggle("is-active", step.dataset.frame === name);
    const [alt, text] = describe[name]();
    const next = layers[1 - front];
    next.src = `showcase/${name}.jpg`;
    next.alt = alt;
    const show = () => {
      next.classList.add("is-shown");
      layers[front].classList.remove("is-shown");
      layers[front].alt = "";
      front = 1 - front;
      caption.textContent = text;
    };
    if (next.complete) show(); else next.addEventListener("load", show, { once: true });
  };
  for (const name of Object.keys(describe)) new Image().src = `showcase/${name}.jpg`; // preload
  caption.textContent = describe[current]()[1];

  // One source of truth: the step under the middle of the viewport, and how far through
  // it the reader is (the progress line), recomputed once per frame on scroll.
  const update = () => {
    const middle = innerHeight / 2;
    let active = steps[0];
    for (const step of steps) {
      const r = step.getBoundingClientRect();
      if (r.top <= middle) active = step;
      step.style.setProperty("--progress", Math.min(1, Math.max(0, (middle - r.top) / r.height)).toFixed(3));
    }
    select(active.dataset.frame);
  };
  let queued = false;
  addEventListener("scroll", () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; update(); });
  }, { passive: true });
  addEventListener("resize", update);
  update();
}

// --- Entrances --------------------------------------------------------------------------

function startEntrances() {
  const parts = document.querySelectorAll(".arch .flow li, .stack-row > div, .results-table tbody tr, .credits, .cta h2, .cta p, .cta-actions, .footer .tile");
  if (reducedMotion) return;
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("is-in");
        observer.unobserve(entry.target);
      }
    },
    { rootMargin: "0px 0px -8% 0px" },
  );
  parts.forEach((part, k) => {
    part.classList.add("reveal");
    part.style.transitionDelay = `${(k % 4) * 60}ms`;
    observer.observe(part);
  });
}

startEntrances();
loadShowcase()
  .then((data) => {
    startStory(data);
    return startHero(data);
  })
  .catch((error) => {
    const box = $("hero-error");
    box.textContent = `The live 3-D view could not load (${error.message}). The rest of the page works; reload to try again.`;
    box.hidden = false;
  });

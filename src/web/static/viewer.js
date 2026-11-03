// Point cloud and camera viewer (three.js, rendered on demand).
//
// The reconstruction frame follows OpenCV: x right, y down, z forward from the first
// camera. three.js is y-up, so everything sits in one group rotated to put the scene's up
// direction on +y (see sceneUp: the normal of a camera ring, else the cameras' mean image
// up, R^T [0, -1, 0]). Before cameras are known: 180° about x.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const ACCENT = 0x3cc2b5;
const FRUSTUM_SCALE = 0.05; // frustum depth as a fraction of the cloud's extent ...
const FRUSTUM_RING_SCALE = 0.1; // ... or of the cameras' median distance to it, if larger
const GRID_COLOURS = [0x2c3d46, 0x16222a];

// --- PLY ------------------------------------------------------------------------------

const PLY_TYPES = {
  char: ["getInt8", 1], int8: ["getInt8", 1],
  uchar: ["getUint8", 1], uint8: ["getUint8", 1],
  short: ["getInt16", 2], int16: ["getInt16", 2],
  ushort: ["getUint16", 2], uint16: ["getUint16", 2],
  int: ["getInt32", 4], int32: ["getInt32", 4],
  uint: ["getUint32", 4], uint32: ["getUint32", 4],
  float: ["getFloat32", 4], float32: ["getFloat32", 4],
  double: ["getFloat64", 8], float64: ["getFloat64", 8],
};

/** Vertex positions and colours (0..1, or null) of a PLY file.
 *
 * Written here rather than three's PLYLoader, which assumes 0..255 colours: Open3D
 * writes float colours in 0..1. Reads ascii and binary files whose first element is
 * the vertex list. */
export function parsePly(buffer) {
  const head = new TextDecoder("latin1").decode(new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 8192)));
  const end = head.indexOf("end_header");
  if (!head.startsWith("ply") || end < 0) throw new Error("not a PLY file");
  const bodyStart = head.indexOf("\n", end) + 1;
  let format = "", count = 0, inVertex = false;
  const props = [];
  for (const line of head.slice(0, end).split(/\r?\n/)) {
    const words = line.trim().split(/\s+/);
    if (words[0] === "format") format = words[1];
    else if (words[0] === "element") { inVertex = words[1] === "vertex"; if (inVertex) count = Number(words[2]); }
    else if (words[0] === "property" && inVertex) {
      if (words[1] === "list") throw new Error("PLY vertex lists are not supported");
      props.push({ type: words[1], name: words[2] });
    }
  }
  const index = Object.fromEntries(props.map((p, k) => [p.name, k]));
  if (!["x", "y", "z"].every((n) => n in index)) throw new Error("PLY has no x, y, z");
  const hasColour = ["red", "green", "blue"].every((n) => n in index);
  const colourScale = hasColour && PLY_TYPES[props[index.red].type][1] === 1 ? 1 / 255 : 1;

  const positions = new Float32Array(count * 3);
  const colours = hasColour ? new Float32Array(count * 3) : null;
  const { x, y, z, red, green, blue } = index;
  const store = (k, values) => { // no per-point allocation: clouds can have millions
    positions[3 * k] = values[x]; positions[3 * k + 1] = values[y]; positions[3 * k + 2] = values[z];
    if (colours) {
      colours[3 * k] = values[red] * colourScale;
      colours[3 * k + 1] = values[green] * colourScale;
      colours[3 * k + 2] = values[blue] * colourScale;
    }
  };

  if (format === "ascii") {
    const lines = new TextDecoder().decode(new Uint8Array(buffer, bodyStart)).split(/\r?\n/);
    for (let k = 0; k < count; k++) store(k, lines[k].trim().split(/\s+/).map(Number));
  } else if (format === "binary_little_endian" || format === "binary_big_endian") {
    const little = format === "binary_little_endian";
    const view = new DataView(buffer, bodyStart);
    const readers = props.map((p) => {
      if (!(p.type in PLY_TYPES)) throw new Error(`PLY type ${p.type} is not supported`);
      return PLY_TYPES[p.type];
    });
    const values = new Array(props.length);
    let offset = 0;
    for (let k = 0; k < count; k++) {
      for (let j = 0; j < readers.length; j++) {
        const [getter, size] = readers[j];
        values[j] = view[getter](offset, little);
        offset += size;
      }
      store(k, values);
    }
  } else {
    throw new Error(`PLY format ${format} is not supported`);
  }
  return { positions, colours, count };
}

/** The ``q`` quantile of ``values`` (sorted in place). */
const quantile = (values, q) => values.sort((a, b) => a - b)[Math.min(values.length - 1, Math.floor(q * values.length))];

/** Centre (per-axis median) and radius (90th percentile distance) of the bulk of the
 * points, so a few stray points far out don't shrink the view. Samples large clouds. */
function robustSphere(positions) {
  const n = positions.length / 3;
  if (!n) return null;
  const step = Math.max(1, Math.floor(n / 20000));
  const xs = [], ys = [], zs = [];
  for (let i = 0; i < n; i += step) {
    xs.push(positions[3 * i]); ys.push(positions[3 * i + 1]); zs.push(positions[3 * i + 2]);
  }
  const centre = new THREE.Vector3(quantile([...xs], 0.5), quantile([...ys], 0.5), quantile([...zs], 0.5));
  const distances = xs.map((x, k) => Math.hypot(x - centre.x, ys[k] - centre.y, zs[k] - centre.z));
  return { centre, radius: Math.max(quantile(distances, 0.9), 1e-6) };
}

/** The scene's up direction in the reconstruction frame, and whether the cameras form a
 * ring round the object: ``{ up, ring }``, from the cameras (centres, and sums of their
 * image-up and optical-axis vectors).
 *
 * Cameras on a ring round the object (their centres enclose a loop): the ring's normal,
 * pointing away from where the cameras look (they usually look down at the object); the
 * image-up vectors are no guide there (rolled cameras, e.g. Middlebury templeRing).
 * Otherwise (an arc or a line of shots): the mean image-up.
 *
 * The loop test does not depend on the cameras' order (cameras.json lists them in
 * registration order): the centres are sorted by angle in their best-fit plane first. */
function sceneUp(centres, upSum, forwardSum) {
  const n = centres.length;
  const mean = centres.reduce((acc, c) => acc.add(c), new THREE.Vector3()).divideScalar(Math.max(n, 1));
  const offsets = centres.map((c) => c.clone().sub(mean));
  const spread = offsets.reduce((acc, a) => acc + a.lengthSq(), 0) / Math.max(n, 1);
  const plane = n >= 4 && principalPlane(offsets);
  if (plane) {
    const angle = (a) => Math.atan2(a.dot(plane[1]), a.dot(plane[0]));
    const ring = [...offsets].sort((a, b) => angle(a) - angle(b));
    const normal = new THREE.Vector3(); // Newell: twice the loop's vector area
    ring.forEach((a, k) => normal.add(a.clone().cross(ring[(k + 1) % n])));
    if (normal.length() / 2 > 0.5 * Math.PI * spread) { // ~1 for a ring, ~0.2 for a 40° arc
      normal.normalize();
      return { up: normal.dot(forwardSum) > 0 ? normal.negate() : normal, ring: true };
    }
  }
  const up = upSum.length() > 0.3 * n ? upSum.clone().normalize() : new THREE.Vector3(0, -1, 0);
  return { up, ring: false };
}

/** The two main axes of a set of offsets from their mean (unit vectors), or null when
 * they span less than a plane. */
function principalPlane(offsets) {
  const e = Array(9).fill(0); // covariance, row-major
  for (const a of offsets) {
    const v = [a.x, a.y, a.z];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) e[3 * i + j] += v[i] * v[j];
  }
  const first = dominantAxis(e);
  if (!first) return null;
  const f = [first.x, first.y, first.z];
  const lambda = f.reduce((acc, fi, i) => acc + fi * f.reduce((s, fj, j) => s + e[3 * i + j] * fj, 0), 0);
  const second = dominantAxis(e.map((x, k) => x - lambda * f[Math.floor(k / 3)] * f[k % 3]), lambda);
  return second && [first, second.addScaledVector(first, -second.dot(first)).normalize()];
}

/** Unit eigenvector of the largest eigenvalue of a symmetric 3×3 matrix (row-major), by
 * power iteration; null if that eigenvalue is ~0 (relative to ``scale``). */
function dominantAxis(e, scale = Math.max(...e.map(Math.abs))) {
  const m = new THREE.Matrix3().set(...e);
  const columns = [0, 1, 2].map((i) => new THREE.Vector3(e[i], e[3 + i], e[6 + i]));
  const v = columns.reduce((a, b) => (b.lengthSq() > a.lengthSq() ? b : a)); // in the range of m
  if (!(v.length() > 1e-6 * scale)) return null;
  for (let k = 0; k < 100; k++) v.applyMatrix3(m).normalize();
  return v;
}

const srgbToLinear = (c) => (c < 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

// --- Viewer ---------------------------------------------------------------------------

/** A round sprite (white disc, or a soft glow) for point materials. */
function spriteTexture(glow) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const g = canvas.getContext("2d");
  if (glow) {
    const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, "rgba(255,255,255,1)");
    grad.addColorStop(0.25, "rgba(255,255,255,0.9)");
    grad.addColorStop(0.5, "rgba(255,255,255,0.25)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, 64, 64);
  } else {
    g.fillStyle = "#fff";
    g.beginPath();
    g.arc(32, 32, 30, 0, 2 * Math.PI);
    g.fill();
  }
  return new THREE.CanvasTexture(canvas);
}

const motionQuery = matchMedia("(prefers-reduced-motion: reduce)");
const reducedMotion = () => motionQuery.matches;

export class Viewer {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // 3x screens: no gain, 2.25x cost
    this.renderer.setClearColor(0x000000, 0); // the container's background shows through
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.perspective = new THREE.PerspectiveCamera(50, 1, 0.001, 1000);
    this.camera = this.perspective;
    this.orthoHalf = 1; // half the orthographic view's height, set when switching
    this.group = new THREE.Group();
    this.group.rotation.x = Math.PI; // OpenCV (y down) -> three.js (y up) until setCameras
    this.scene.add(this.group);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.addEventListener("change", () => this.render());
    this.controls.listenToKeyEvents(container); // arrow keys pan when the viewer has focus

    this.pointSize = 2;
    this.points = null;
    this.frustums = null;
    this.markers = null; // glowing camera centres
    this.path = null; // camera centres in photo-name order
    this.grid = null;
    this.layers = { points: true, cameras: true, path: false, grid: true };
    this.cameraForward = null; // mean optical axis (group frame), or null: no common one
    this.cameraCentres = [];
    this.spinning = false;
    this.textures = { dot: spriteTexture(false), glow: spriteTexture(true) };

    new ResizeObserver(() => this.resize()).observe(container);
    // asking for reduced motion mid-session stops a turning view at once
    motionQuery.addEventListener("change", () => { if (reducedMotion()) this.spin(false); });
    this.resize();
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.perspective.aspect = w / h;
    this.perspective.updateProjectionMatrix();
    if (this.camera !== this.perspective) this.frameOrthographic();
    this.render();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  #dispose(object) {
    if (!object) return;
    object.parent?.remove(object);
    object.geometry.dispose();
    object.material.dispose();
  }

  /** Replace the point cloud (from ``parsePly``). */
  setCloud({ positions, colours }) {
    this.#dispose(this.points);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (colours) {
      // PLY colours are sRGB; three.js wants linear vertex colours (converted in place:
      // the parsed arrays belong to this cloud)
      for (let i = 0; i < colours.length; i++) colours[i] = srgbToLinear(colours[i]);
      geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
    }
    geometry.computeBoundingBox();
    this.sphere = robustSphere(positions);
    const material = new THREE.PointsMaterial({
      size: this.pointSize * window.devicePixelRatio,
      sizeAttenuation: false,
      vertexColors: Boolean(colours),
      color: colours ? 0xffffff : 0xc9d3d6,
      map: this.textures.dot, // round points
      alphaTest: 0.45,
    });
    this.points = new THREE.Points(geometry, material);
    this.points.visible = this.layers.points;
    this.group.add(this.points);
    this.render();
  }

  /** Camera frustums from ``cameras.json`` entries (in registration order), with a glow at
   * each centre, the capture path and a ground grid; sized to the current cloud. */
  setCameras(cameras) {
    for (const old of [this.frustums, this.markers, this.path]) this.#dispose(old);
    const box = this.points ? this.points.geometry.boundingBox : null;
    const extent = box ? box.getSize(new THREE.Vector3()).length() : 1;
    const middle = box ? box.getCenter(new THREE.Vector3()) : new THREE.Vector3();
    const vertices = [];
    const forward = new THREE.Vector3();
    const up = new THREE.Vector3();
    const rotations = cameras.map((cam) => new THREE.Matrix3().set(...cam.R.flat()).transpose()); // R^T
    this.cameraCentres = cameras.map((cam, k) => new THREE.Vector3(...cam.t).applyMatrix3(rotations[k]).negate()); // C = -R^T t
    const median = this.cameraCentres.length ? quantile(this.cameraCentres.map((c) => c.distanceTo(middle)), 0.5) : 0;
    // cameras on a wide ring would be specks at the cloud's scale
    const depth = Math.max(FRUSTUM_SCALE * extent, FRUSTUM_RING_SCALE * median);
    cameras.forEach((cam, k) => {
      const Rt = rotations[k];
      const centre = this.cameraCentres[k];
      const Kinv = new THREE.Matrix3().set(...cam.K.flat()).invert();
      const corners = [[0, 0], [cam.width, 0], [cam.width, cam.height], [0, cam.height]].map(([u, v]) => {
        const ray = new THREE.Vector3(u, v, 1).applyMatrix3(Kinv); // z = 1
        return ray.multiplyScalar(depth).applyMatrix3(Rt).add(centre);
      });
      for (let i = 0; i < 4; i++) vertices.push(centre, corners[i], corners[i], corners[(i + 1) % 4]);
      forward.add(new THREE.Vector3(0, 0, 1).applyMatrix3(Rt)); // optical axis in the world
      up.add(new THREE.Vector3(0, -1, 0).applyMatrix3(Rt)); // image up in the world
    });
    this.frustums = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(vertices),
      new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.85 }),
    );
    this.markers = new THREE.Points(
      new THREE.BufferGeometry().setFromPoints(this.cameraCentres),
      new THREE.PointsMaterial({
        color: ACCENT, size: 14 * window.devicePixelRatio, sizeAttenuation: false, map: this.textures.glow,
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      }),
    );
    const order = cameras.map((c, k) => [c.name ?? "", k])
      .sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true }));
    this.path = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(order.map(([, k]) => this.cameraCentres[k])),
      new THREE.LineDashedMaterial({ color: 0x8fa3a9, dashSize: 0.04 * median, gapSize: 0.03 * median, transparent: true, opacity: 0.7 }),
    );
    this.path.computeLineDistances();
    this.group.add(this.frustums, this.markers, this.path);
    this.frustums.visible = this.markers.visible = this.layers.cameras;
    this.path.visible = this.layers.path;

    const scene = sceneUp(this.cameraCentres, up, forward);
    this.group.quaternion.setFromUnitVectors(scene.up, new THREE.Vector3(0, 1, 0));
    // no common viewing direction for a ring, even when the cameras all look down steeply
    // (their sum need not cancel: 0.28 per camera on Middlebury dinoRing), nor when they cancel
    const common = !scene.ring && forward.length() > 0.2 * cameras.length;
    this.cameraForward = common ? forward.normalize() : null;
    this.#buildGrid();
    this.render();
  }

  /** A ground grid under the cloud, wide enough for the cameras. */
  #buildGrid() {
    this.#dispose(this.grid);
    this.grid = null;
    if (!this.points || !this.cameraCentres.length) return;
    this.group.updateMatrixWorld();
    const box = new THREE.Box3().setFromObject(this.points);
    if (box.isEmpty()) return;
    const centre = box.getCenter(new THREE.Vector3());
    const reach = Math.max(...this.cameraCentres.map((c) =>
      c.clone().applyMatrix4(this.group.matrixWorld).setY(centre.y).distanceTo(centre)));
    this.grid = new THREE.GridHelper(2.6 * Math.max(reach, box.getSize(new THREE.Vector3()).length()), 26, ...GRID_COLOURS);
    this.grid.position.set(centre.x, box.min.y, centre.z);
    this.grid.visible = this.layers.grid;
    this.scene.add(this.grid);
  }

  setPointSize(px) {
    this.pointSize = px;
    if (this.points) {
      this.points.material.size = px * window.devicePixelRatio;
      this.render();
    }
  }

  /** Show or hide a layer: "points", "cameras", "path" or "grid". */
  setLayer(name, visible) {
    this.layers[name] = visible;
    const objects = { points: [this.points], cameras: [this.frustums, this.markers], path: [this.path], grid: [this.grid] }[name];
    for (const object of objects ?? []) if (object) object.visible = visible;
    this.render();
  }

  /** Draw only a share of the points and the first ``cameraCount`` cameras (registration
   * order): the replay of a finished run. ``reveal(1, Infinity)`` draws everything. */
  reveal(pointShare, cameraCount) {
    if (this.points) {
      const n = this.points.geometry.attributes.position.count;
      this.points.geometry.setDrawRange(0, Math.round(n * Math.min(1, pointShare)));
    }
    const shown = Math.min(cameraCount, this.cameraCentres.length);
    this.frustums?.geometry.setDrawRange(0, shown * 16); // 8 segments per frustum
    this.markers?.geometry.setDrawRange(0, shown);
    // the path runs in photo order, not registration order: only once it is complete
    if (this.path) this.path.visible = this.layers.path && shown >= this.cameraCentres.length;
    this.render();
  }

  /** What a left-drag does: "orbit" or "pan". */
  setTool(tool) {
    this.controls.mouseButtons.LEFT = tool === "pan" ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
  }

  frameOrthographic() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    const half = this.orthoHalf;
    Object.assign(this.camera, { left: (-half * w) / h, right: (half * w) / h, top: half, bottom: -half });
    this.camera.updateProjectionMatrix();
  }

  /** "perspective" or "orthographic", keeping the view. */
  setProjection(kind) {
    const ortho = kind === "orthographic";
    if (ortho === (this.camera !== this.perspective)) return;
    const from = this.camera;
    if (ortho) {
      const distance = from.position.distanceTo(this.controls.target);
      this.orthoHalf = distance * Math.tan(THREE.MathUtils.degToRad(this.perspective.fov / 2));
      this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, from.near, from.far * 2);
      this.camera.position.copy(from.position);
      this.camera.quaternion.copy(from.quaternion);
      this.frameOrthographic();
    } else {
      // the orthographic zoom becomes distance, so the view keeps its size
      const half = this.orthoHalf / from.zoom;
      const distance = half / Math.tan(THREE.MathUtils.degToRad(this.perspective.fov / 2));
      const back = from.position.clone().sub(this.controls.target).normalize();
      this.perspective.position.copy(this.controls.target).addScaledVector(back, distance);
      this.perspective.quaternion.copy(from.quaternion);
      this.camera = this.perspective;
    }
    this.controls.object = this.camera;
    this.controls.update();
    this.render();
  }

  /** Turn slowly round the object (never under reduced motion). */
  spin(on) {
    const wasSpinning = this.spinning;
    this.spinning = on && !reducedMotion();
    this.controls.autoRotate = this.spinning;
    this.controls.autoRotateSpeed = 0.8;
    const loop = () => {
      if (!this.spinning) return;
      this.controls.update();
      this.render();
      requestAnimationFrame(loop);
    };
    if (this.spinning && !wasSpinning) loop(); // one loop, however often this is called
    return this.spinning;
  }

  /** Screen positions (px in the container) of cameras ``0..count-1``; null when behind
   * the view. */
  #screenPositions(count = this.cameraCentres.length) {
    const { clientWidth: w, clientHeight: h } = this.container;
    this.group.updateMatrixWorld();
    this.camera.updateMatrixWorld();
    const toView = new THREE.Matrix4().multiplyMatrices(this.camera.matrixWorldInverse, this.group.matrixWorld);
    const p = new THREE.Vector3();
    return this.cameraCentres.slice(0, count).map((c) => {
      p.copy(c).applyMatrix4(toView);
      if (p.z >= 0) return null; // behind the view
      p.applyMatrix4(this.camera.projectionMatrix);
      return [((p.x + 1) / 2) * w, ((1 - p.y) / 2) * h];
    });
  }

  /** Screen position (px in the container) of camera ``k``, or null when behind the view. */
  screenOf(k) {
    return this.#screenPositions(k + 1)[k];
  }

  /** Index of the drawn camera nearest (x, y) within ``radius`` px, or -1. */
  pick(x, y, radius = 16) {
    if (!this.markers?.visible) return -1;
    const drawn = Math.min(this.markers.geometry.drawRange.count, this.cameraCentres.length);
    let best = -1;
    let bestDistance = radius;
    this.#screenPositions(drawn).forEach((p, k) => {
      const d = p ? Math.hypot(p[0] - x, p[1] - y) : Infinity;
      if (d < bestDistance) [best, bestDistance] = [k, d];
    });
    return best;
  }

  /** Frame the cloud ("object"), the cloud and every camera ("cameras"), or the cloud with
   * the nearer cameras round it ("overview"), looking the way the photos did. */
  fit(target = "object") {
    this.group.updateMatrixWorld();
    const sphere = this.points && this.sphere;
    if (!sphere) return;
    const reaches = this.cameraCentres.map((c) => c.distanceTo(sphere.centre));
    const radius = Math.max(sphere.radius, ...(target === "cameras" ? reaches : []));
    const centre = sphere.centre.clone().applyMatrix4(this.group.matrixWorld);
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(this.perspective.fov / 2));
    // a little further back, so the cameras in front of the cloud stay in view
    let distance = (1.3 * radius) / tanHalf;
    if (target === "overview" && reaches.length) distance = Math.max(distance, 1.15 * Math.min(...reaches));
    const direction = this.cameraForward
      ? this.cameraForward.clone().transformDirection(this.group.matrixWorld)
      : new THREE.Vector3(0, 0, -1);
    if (Math.hypot(direction.x, direction.z) < 1e-3) direction.set(0, 0, -1); // looking straight down
    direction.setY(0).normalize().add(new THREE.Vector3(0, this.cameraForward ? -0.35 : -0.8, 0)).normalize(); // from above
    this.camera.position.copy(centre).addScaledVector(direction, -distance);
    this.camera.lookAt(centre);
    this.perspective.near = distance / 1000;
    this.perspective.far = distance * 10;
    this.perspective.updateProjectionMatrix();
    if (this.camera !== this.perspective) { // orthographic: the same view's size, no zoom
      this.orthoHalf = distance * tanHalf;
      this.camera.zoom = 1;
      Object.assign(this.camera, { near: this.perspective.near, far: this.perspective.far * 2 });
      this.frameOrthographic();
    }
    this.controls.target.copy(centre);
    this.controls.update();
    this.render();
  }
}

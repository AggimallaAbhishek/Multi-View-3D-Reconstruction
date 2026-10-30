// Point cloud and camera viewer (three.js, rendered on demand).
//
// The reconstruction frame follows OpenCV: x right, y down, z forward from the first
// camera. three.js is y-up, so everything sits in one group rotated to put the cameras'
// mean up direction (R^T [0, -1, 0]) on +y: upright photos give an upright cloud, even
// when the first camera was tilted. Before cameras are known: 180° about x.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const FRUSTUM_COLOUR = 0x3cc2b5;
const FRUSTUM_SCALE = 0.05; // frustum depth as a fraction of the cloud's extent

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
  const quantile = (values, q) => values.sort((a, b) => a - b)[Math.min(values.length - 1, Math.floor(q * values.length))];
  const centre = new THREE.Vector3(quantile([...xs], 0.5), quantile([...ys], 0.5), quantile([...zs], 0.5));
  const distances = xs.map((x, k) => Math.hypot(x - centre.x, ys[k] - centre.y, zs[k] - centre.z));
  return { centre, radius: Math.max(quantile(distances, 0.9), 1e-6) };
}

/** The scene's up direction in the reconstruction frame, from the cameras (centres, and
 * sums of their image-up and optical-axis vectors).
 *
 * Cameras on a ring round the object (their centres enclose a loop): the ring's normal,
 * pointing away from where the cameras look (they usually look down at the object); the
 * image-up vectors are no guide there (rolled cameras, e.g. Middlebury templeRing).
 * Otherwise (an arc or a line of shots): the mean image-up. */
function sceneUp(centres, upSum, forwardSum) {
  const n = centres.length;
  const mean = centres.reduce((acc, c) => acc.add(c), new THREE.Vector3()).divideScalar(Math.max(n, 1));
  const normal = new THREE.Vector3(); // Newell: twice the loop's vector area
  let spread = 0;
  centres.forEach((c, k) => {
    const a = c.clone().sub(mean);
    normal.add(a.clone().cross(centres[(k + 1) % n].clone().sub(mean)));
    spread += a.lengthSq() / n;
  });
  const enclosesLoop = n >= 4 && normal.length() / 2 > 0.5 * Math.PI * spread; // ~1 for a ring, ~0.2 for a 40° arc
  if (enclosesLoop) {
    normal.normalize();
    return normal.dot(forwardSum) > 0 ? normal.negate() : normal;
  }
  return upSum.length() > 0.3 * n ? upSum.clone().normalize() : new THREE.Vector3(0, -1, 0);
}

const srgbToLinear =(c) => (c < 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

// --- Viewer ---------------------------------------------------------------------------

export class Viewer {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(0x0c1013);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.001, 1000);
    this.group = new THREE.Group();
    this.group.rotation.x = Math.PI; // OpenCV (y down) -> three.js (y up) until setCameras
    this.scene.add(this.group);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.addEventListener("change", () => this.render());
    this.controls.listenToKeyEvents(container); // arrow keys pan when the viewer has focus

    this.pointSize = 2;
    this.points = null;
    this.frustums = null;
    this.cameraForward = null; // mean optical axis (group frame), or null: no common one
    this.cameraCentres = [];

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  /** Replace the point cloud (from ``parsePly``). */
  setCloud({ positions, colours }) {
    if (this.points) {
      this.group.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (colours) {
      // PLY colours are sRGB; three.js wants linear vertex colours
      const linear = colours.map(srgbToLinear);
      geometry.setAttribute("color", new THREE.BufferAttribute(linear, 3));
    }
    geometry.computeBoundingBox();
    const material = new THREE.PointsMaterial({
      size: this.pointSize * window.devicePixelRatio,
      sizeAttenuation: false,
      vertexColors: Boolean(colours),
      color: colours ? 0xffffff : 0xc9d3d6,
    });
    this.points = new THREE.Points(geometry, material);
    this.group.add(this.points);
    this.render();
  }

  /** Camera frustums from ``cameras.json`` entries, sized to the current cloud. */
  setCameras(cameras) {
    if (this.frustums) {
      this.group.remove(this.frustums);
      this.frustums.geometry.dispose();
      this.frustums.material.dispose();
    }
    const box = this.points ? this.points.geometry.boundingBox : null;
    const extent = box ? box.getSize(new THREE.Vector3()).length() : 1;
    const depth = FRUSTUM_SCALE * extent;
    const vertices = [];
    const forward = new THREE.Vector3();
    const up = new THREE.Vector3();
    this.cameraCentres = [];
    for (const cam of cameras) {
      const R = new THREE.Matrix3().set(...cam.R.flat());
      const Rt = R.clone().transpose();
      const t = new THREE.Vector3(...cam.t);
      const centre = t.clone().applyMatrix3(Rt).negate(); // C = -R^T t
      const Kinv = new THREE.Matrix3().set(...cam.K.flat()).invert();
      const corners = [[0, 0], [cam.width, 0], [cam.width, cam.height], [0, cam.height]].map(([u, v]) => {
        const ray = new THREE.Vector3(u, v, 1).applyMatrix3(Kinv); // z = 1
        return ray.multiplyScalar(depth).applyMatrix3(Rt).add(centre);
      });
      for (let k = 0; k < 4; k++) {
        vertices.push(centre, corners[k], corners[k], corners[(k + 1) % 4]);
      }
      forward.add(new THREE.Vector3(0, 0, 1).applyMatrix3(Rt)); // optical axis in the world
      up.add(new THREE.Vector3(0, -1, 0).applyMatrix3(Rt)); // image up in the world
      this.cameraCentres.push(centre);
    }
    const geometry = new THREE.BufferGeometry().setFromPoints(vertices);
    this.frustums = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: FRUSTUM_COLOUR }));
    this.group.add(this.frustums);
    this.group.quaternion.setFromUnitVectors(sceneUp(this.cameraCentres, up, forward), new THREE.Vector3(0, 1, 0));
    // cameras all round the object cancel out: no common viewing direction
    this.cameraForward = forward.length() > 0.2 * cameras.length ? forward.normalize() : null;
    this.render();
  }

  setPointSize(px) {
    this.pointSize = px;
    if (this.points) {
      this.points.material.size = px * window.devicePixelRatio;
      this.render();
    }
  }

  /** Frame the cloud and cameras, looking the way the photos did. */
  fit() {
    this.group.updateMatrixWorld();
    const sphere = this.points && robustSphere(this.points.geometry.attributes.position.array);
    if (!sphere) return;
    // cameras round the object (no common viewing direction): frame their ring too
    const ring = this.cameraForward ? 0 : Math.max(0, ...this.cameraCentres.map((c) => c.distanceTo(sphere.centre)));
    const radius = Math.max(sphere.radius, ring);
    const centre = sphere.centre.clone().applyMatrix4(this.group.matrixWorld);
    // a little further back, so the cameras in front of the cloud stay in view
    const distance = (1.3 * radius) / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const direction = this.cameraForward
      ? this.cameraForward.clone().transformDirection(this.group.matrixWorld)
      : new THREE.Vector3(0, 0, -1);
    if (Math.hypot(direction.x, direction.z) < 1e-3) direction.set(0, 0, -1); // looking straight down
    direction.setY(0).normalize().add(new THREE.Vector3(0, this.cameraForward ? -0.35 : -0.8, 0)).normalize(); // from above
    this.camera.position.copy(centre).addScaledVector(direction, -distance);
    this.camera.near = distance / 1000;
    this.camera.far = distance * 10;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(centre);
    this.controls.update();
    this.render();
  }
}

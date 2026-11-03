// The intro model's formation: the cameras appear one by one round their ring while the
// cloud rises from the ground, as if the cameras were building it. Pure array maths (no
// three.js), so the viewer only draws a growing share of data put in this order.

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalise = (v) => {
  const n = Math.hypot(...v);
  return n > 1e-12 ? v.map((x) => x / n) : null;
};

/** A camera's centre ``C = -Rᵀ t`` (``cameras.json``: world -> camera ``x_c = R X + t``). */
export function cameraCentre({ R, t }) {
  return [0, 1, 2].map((i) => -(R[0][i] * t[0] + R[1][i] * t[1] + R[2][i] * t[2]));
}

/** The scene's up direction: ``up`` when measured, else from the cloud's mean to the
 * cameras' mean (a dome of cameras sits above its object), else null. */
export function upOf({ positions }, cameras, up = null) {
  if (up) return normalise(up);
  const n = positions.length / 3;
  if (!n || cameras.length < 3) return null;
  const mean = [0, 0, 0];
  for (let i = 0; i < positions.length; i++) mean[i % 3] += positions[i] / n;
  const centres = cameras.map(cameraCentre);
  const above = [0, 1, 2].map((i) => centres.reduce((s, c) => s + c[i], 0) / centres.length - mean[i]);
  return normalise(above);
}

/** The cloud's points from the lowest to the highest along ``up`` (copies), and their
 * heights in that order. */
export function sortByHeight({ positions, colours }, up) {
  const n = positions.length / 3;
  const height = new Float64Array(n);
  for (let k = 0; k < n; k++) height[k] = positions[3 * k] * up[0] + positions[3 * k + 1] * up[1] + positions[3 * k + 2] * up[2];
  const order = Array.from({ length: n }, (_, k) => k).sort((a, b) => height[a] - height[b]);
  const sorted = { positions: new Float32Array(3 * n), colours: colours ? new Float32Array(3 * n) : null, heights: new Float64Array(n) };
  order.forEach((k, i) => {
    for (let a = 0; a < 3; a++) {
      sorted.positions[3 * i + a] = positions[3 * k + a];
      if (colours) sorted.colours[3 * i + a] = colours[3 * k + a];
    }
    sorted.heights[i] = height[k];
  });
  return sorted;
}

/** The cameras in order of their angle round ``up`` (a copy): revealed in this order they
 * appear one after another round the ring. */
export function aroundRing(cameras, up) {
  const centres = cameras.map(cameraCentre);
  const middle = [0, 1, 2].map((i) => centres.reduce((s, c) => s + c[i], 0) / Math.max(1, centres.length));
  // two directions across ``up``: from the axis least aligned with it
  const axis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].reduce((best, a) => (Math.abs(dot(a, up)) < Math.abs(dot(best, up)) ? a : best));
  const across = normalise(cross(up, axis));
  const along = cross(up, across);
  const angle = centres.map((c) => {
    const d = c.map((x, i) => x - middle[i]);
    return Math.atan2(dot(d, along), dot(d, across));
  });
  return cameras.map((cam, k) => [cam, angle[k]]).sort((a, b) => a[1] - b[1]).map(([cam]) => cam);
}

/** How many of the height-sorted points lie below a level that rises at an even speed
 * from the bottom (``progress`` 0) to the top (1); the ends are the 0.5 % and 99.5 %
 * heights, so a few stray points don't stretch the rise. */
export function pointsBelow(heights, progress) {
  const n = heights.length;
  if (progress >= 1) return n;
  if (progress <= 0 || !n) return 0;
  const low = heights[Math.floor(0.005 * (n - 1))];
  const high = heights[Math.ceil(0.995 * (n - 1))];
  const level = low + progress * (high - low);
  let lo = 0;
  let hi = n; // first index above the level
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (heights[mid] <= level) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Smooth start and end: the formation eases in and settles. */
export const ease = (p) => p * p * (3 - 2 * p);

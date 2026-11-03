// Small helpers shared by the intro page (intro.js) and the workstation (app.js).

/** 12 345 with thin-space groups. */
export const count = (n) => n.toLocaleString("en-GB").replace(/,/g, " ");

/** A small measurement with sensible precision: 0.053, 0.15, 12.3. */
export const small = (x) => (x < 0.1 ? x.toFixed(3) : x < 10 ? x.toFixed(2) : x.toFixed(1));

/** Hover and click labels for the cameras of ``viewer`` in ``label``: hovering shows one,
 * a click keeps it up (a drag does not count as a click). ``describe(k)`` returns the
 * label's content (nodes or strings) for camera ``k``, or null for none. Returns
 * ``{show, pin, unpin}``: ``show()`` re-places the label, e.g. after the view moved. */
export function cameraLabels(viewer, label, describe, { onPin = () => {} } = {}) {
  const container = viewer.container;
  const canvas = viewer.renderer.domElement;
  let pinned = -1;
  const show = (k = pinned) => {
    const content = k >= 0 ? describe(k) : null;
    const at = content && viewer.screenOf(k);
    if (!at) { label.hidden = true; return; }
    label.replaceChildren(...content);
    label.classList.toggle("flip", at[0] > container.clientWidth - 340);
    label.style.left = `${at[0]}px`;
    label.style.top = `${at[1]}px`;
    label.hidden = false;
  };
  canvas.addEventListener("pointermove", (e) => {
    if (e.buttons) return; // dragging
    const k = viewer.pick(e.offsetX, e.offsetY);
    canvas.style.cursor = k >= 0 ? "pointer" : "";
    show(k >= 0 ? k : pinned);
  });
  canvas.addEventListener("pointerleave", () => show(pinned));
  let down = null;
  canvas.addEventListener("pointerdown", (e) => { down = [e.offsetX, e.offsetY]; });
  canvas.addEventListener("click", (e) => {
    if (down && Math.hypot(e.offsetX - down[0], e.offsetY - down[1]) > 4) return;
    pinned = viewer.pick(e.offsetX, e.offsetY);
    show(pinned);
    onPin(pinned);
  });
  viewer.controls.addEventListener("change", () => show(pinned));
  return {
    show,
    get pinned() { return pinned; },
    unpin() { pinned = -1; label.hidden = true; onPin(-1); },
  };
}

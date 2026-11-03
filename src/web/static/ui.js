// Small helpers shared by the intro page (intro.js) and the workstation (app.js).

/** 12 345 with narrow no-break spaces between digit groups (never split across lines). */
export const count = (n) => n.toLocaleString("en-GB").replace(/,/g, "\u202f");

/** A small measurement with sensible precision: 0.053, 0.15, 12.3. */
export const small = (x) => (x < 0.1 ? x.toFixed(3) : x < 10 ? x.toFixed(2) : x.toFixed(1));

/** Hover and click labels for the cameras of ``viewer`` in ``label``: hovering shows one,
 * a click keeps it up (a drag does not count as a click). ``describe(k)`` returns the
 * label's content (nodes or strings) for camera ``k``, or null for none. Returns
 * ``{show, pin, unpin}``: ``show()`` re-places the label, e.g. after the view moved.
 * ``announce(text)`` (optional) reads a camera chosen by keyboard to screen readers. */
export function cameraLabels(viewer, label, describe, { onPin = () => {}, announce = null } = {}) {
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
  // Keyboard: "]" and "[" step through the cameras, Escape lets go.
  container.addEventListener("keydown", (e) => {
    const n = viewer.cameraCentres.length;
    if (!n) return;
    if (e.key === "]" || e.key === "[") {
      e.preventDefault();
      pinned = pinned < 0 ? 0 : (pinned + (e.key === "]" ? 1 : -1) + n) % n;
    } else if (e.key === "Escape" && pinned >= 0) {
      pinned = -1;
    } else return;
    show(pinned);
    onPin(pinned);
    announce?.(pinned >= 0 ? label.textContent : "");
  });
  return {
    show,
    get pinned() { return pinned; },
    unpin() { pinned = -1; label.hidden = true; onPin(-1); },
  };
}

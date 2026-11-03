// Point colours: PLY files hold sRGB, three.js wants linear vertex colours.

export const srgbToLinear = (c) => (c < 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

/** Linear copies of sRGB colours. A copy, never in place: callers keep their parsed
 * clouds and show them again (converting the same array twice darkened it each time). */
export function linearColours(colours) {
  return Float32Array.from(colours, srgbToLinear);
}

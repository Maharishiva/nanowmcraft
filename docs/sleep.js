// sleep detector for the player
// rgb: one 63x63 frame, RGB bytes in row-major order (length 63 * 63 * 3), as the player draws it
// returns true while the frame shows the player asleep
//
// Craftax draws sleep as a filter on the map (the top 49 rows, 9x7 tiles of 7x7 px): every pixel turns grey at half
// brightness with a small blue tint, so R = G and B = R + 8 exactly, and the player tile shows the sleeping sprite
// in the same grey. Night only darkens and tints the map blue (G and B well above R) and leaves the player in colour,
// so it does not pass. Two tests, both must hold:
//   1. at least 80% of the map pixels fit the filter (|G - R| <= 1 and |B - R - 8| <= 2)
//   2. the centre tile (the player, rows 21-27, columns 28-34) fits it too: mean |G - R| + |B - R - 8| <= 10
// Real frames: asleep 1.00 / 0.00, awake at most 0.68 of the map fits and the player tile is at 24 or more.
// Frames the world model makes: asleep at least 0.94 / at most 2.6, awake at most 0.28 / 22 or more.
const MAP = 49 * 63, NEED = Math.ceil(0.8 * MAP);

export function asleep(rgb) {
  let fit = 0;
  for (let p = 0, i = 0; p < MAP; p++, i += 3) {
    const r = rgb[i], dg = rgb[i + 1] - r, db = rgb[i + 2] - r - 8;
    if (dg >= -1 && dg <= 1 && db >= -2 && db <= 2) fit++;
    else if (p + 1 - fit > MAP - NEED) return false;   // too many misses already: fewer than NEED can fit
  }
  let dev = 0;
  for (let y = 21; y < 28; y++) {
    for (let x = 28; x < 35; x++) {
      const i = (y * 63 + x) * 3, r = rgb[i];
      dev += Math.abs(rgb[i + 1] - r) + Math.abs(rgb[i + 2] - r - 8);
    }
  }
  return dev <= 10 * 49;
}

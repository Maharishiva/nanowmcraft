// Reads a 63x63 frame the way the tutorial needs: inventory counts from the strip under the map, and what is on the map
// tiles around the player; and whether an imagined first frame is a fresh start (fresh). Nearest template of
// Craftax-Classic's own textures (hud_tpl.js, made by make_hud.py); the frames are generated and a little blurry, so
// every read comes with a margin to the second best template.
import { TPL } from './hud_tpl.js';

const bytes = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const TILES = bytes(TPL.tiles), PLAYER = bytes(TPL.player), WOOD = bytes(TPL.wood), DIAMOND = bytes(TPL.diamond);
export const NAMES = TPL.names;                     // grass, sand, water, stone, tree, path, table, ..., cow, zombie, skeleton
const WALK = new Set(['grass', 'sand', 'path']);   // what the player walks on
export const ME = [3, 4];                           // the player's tile: row, column of the 7x9 map
export const STEP = { 1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0] };   // action: row, column it moves

function tile(rgb, x0, y0) {  // the 7x7 pixels at x0, y0 as 147 floats
  const p = new Float32Array(147);
  for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) for (let c = 0; c < 3; c++)
    p[(y * 7 + x) * 3 + c] = rgb[((y0 + y) * 63 + x0 + x) * 3 + c];
  return p;
}

function nearest(p, tpl, n, scale = false) {  // index of the nearest of n templates, its mean squared error, the margin
  // to the second; scale: each template brightened or dimmed to fit first (dusk, blur)
  let best = -1, d0 = Infinity, d1 = Infinity;
  for (let k = 0; k < n; k++) {
    let s = 1;
    if (scale) {
      let pt = 0, tt = 0;
      for (let i = 0; i < 147; i++) { pt += p[i] * tpl[k * 147 + i]; tt += tpl[k * 147 + i] ** 2; }
      s = Math.min(1.25, Math.max(0.6, pt / Math.max(tt, 1)));
    }
    let d = 0;
    for (let i = 0; i < 147; i++) d += (p[i] - s * tpl[k * 147 + i]) ** 2;
    d /= 147;
    if (d < d0) [d1, d0, best] = [d0, d, k];
    else if (d < d1) d1 = d;
  }
  return { k: best, d: d0, margin: d1 - d0 };
}

export function inventory(rgb) {  // { wood, diamond }: each { n: the count, d, margin }
  const slot = (x, y, tpl) => { const r = nearest(tile(rgb, 7 * x, 49 + 7 * y), tpl, 10); return { n: r.k, d: r.d, margin: r.margin }; };
  return { wood: slot(5, 0, WOOD), diamond: slot(0, 1, DIAMOND) };
}

const SLOTS = bytes(TPL.slots);
export const SLOT_NAMES = TPL.slot_names;          // health, food, drink, energy, sapling, wood, ..., iron_sword

export function slots(rgb) {  // the counts of all 16 slots of the strip, in SLOT_NAMES' order (nearest template, as
  // inventory() reads wood and diamonds)
  return TPL.slot_xy.map(([x, y], i) => nearest(tile(rgb, 7 * x, 49 + 7 * y), SLOTS.subarray(i * 1470, (i + 1) * 1470), 10).k);
}

export function fresh(rgb) {  // a frame a world can start from: the status bar full (9/9/9/9) and nothing held
  return slots(rgb).every((n, i) => n === (i < 4 ? 9 : 0));
}

export function tiles(rgb) {  // the 63 map tiles, row by row, as names
  const out = [];
  for (let r = 0; r < 7; r++) for (let c = 0; c < 9; c++) out.push(NAMES[nearest(tile(rgb, 7 * c, 7 * r), TILES, NAMES.length, true).k]);
  return out;
}

export function facing(rgb) {  // the action the player sprite faces (1 left, 2 right, 3 up, 4 down), from its opaque pixels
  const p = tile(rgb, 7 * ME[1], 7 * ME[0]);
  let best = 3, bd = Infinity;
  for (let k = 0; k < 4; k++) {
    let pt = 0, tt = 0, d = 0, n = 0;
    for (let i = 0; i < 49; i++) if (PLAYER[(k * 49 + i) * 4 + 3]) for (let c = 0; c < 3; c++) {
      pt += p[i * 3 + c] * PLAYER[(k * 49 + i) * 4 + c];
      tt += PLAYER[(k * 49 + i) * 4 + c] ** 2;
    }
    const s = pt / Math.max(tt, 1);
    for (let i = 0; i < 49; i++) if (PLAYER[(k * 49 + i) * 4 + 3]) for (let c = 0; c < 3; c++, n++)
      d += (p[i * 3 + c] - s * PLAYER[(k * 49 + i) * 4 + c]) ** 2;
    if (d / n < bd) [bd, best] = [d / n, k + 1];
  }
  return best;
}

export function route(names, face, want = 'tree') {  // the way to the nearest `want` tile on the map: { at: [row, col] of
  // it, next: the action to press now (into it if it is next to the player: that turns the player to face it), facing:
  // the player faces it now }, or null if there is none in view. Walks on grass, sand and path only, breadth first.
  const at = (r, c) => names[r * 9 + c], ok = (r, c) => r >= 0 && r < 7 && c >= 0 && c < 9;
  const seen = new Map([[ME.join(), 0]]), queue = [[...ME, 0]];
  const order = [face, 2, 3, 1, 4].filter((a, i, l) => a && l.indexOf(a) === i);   // the faced side first
  while (queue.length) {
    const [r, c, first] = queue.shift();
    for (const a of order) {
      const [dr, dc] = STEP[a], R = r + dr, C = c + dc;
      if (!ok(R, C) || seen.has(R + ',' + C)) continue;
      seen.set(R + ',' + C, 1);
      if (at(R, C) === want) return { at: [R, C], next: first || a, facing: !first && a === face };
      if (WALK.has(at(R, C))) queue.push([R, C, first || a]);
    }
  }
  return null;
}

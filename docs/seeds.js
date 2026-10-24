// Checks a new play.wgr before the site plays it, with the player's own runtime and noise (play.js): every world starts
// from a first frame imagined from noise at slot 0, so the model decides what a start looks like. For seeds from..from+n-1
// it imagines that first frame and reads it (hud.js): fresh (9/9/9/9, nothing held; play.js takes the next seed until one
// is), brightness, grass, mobs, asleep (sleep.js should never fire on a start), a tree straight up or right 1-3 tiles
// away over grass. Then it plays the walkthrough's path (walk to the tree, chop until 2 wood, place a table) in the
// best candidates for the tutorial world (TUT_SEED in play.js): fewest moves to the table first.
//   seeds.html?model=play.wgr&n=1500&from=0&cands=40
// The result is window.result and the page text (JSON); thumbnails of the candidates show under it.
import { create } from './wgr.js';
import { fresh, tiles, slots, facing, route, inventory } from './hud.js';
import { asleep } from './sleep.js';
import { noise, PAST, NOISE_BUF, RNOISE } from './seeded.js';

const q = new URLSearchParams(location.search);
const MODEL = q.get('model') ?? 'play.wgr', N = +(q.get('n') ?? 1500), FROM = +(q.get('from') ?? 0);
const CANDS = +(q.get('cands') ?? 40), REROLL = 20, L = 31 * 512;
const SEED_ABC = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
const code = s => [15, 10, 5, 0].map(b => SEED_ABC[(s >> b) & 31]).join('');
const out = document.getElementById('out'), grid = document.getElementById('grid');
const say = s => { out.textContent = s; };

function world(sess, seed) {  // one world as play.js runs it: call 0 imagines the first frame after a noop, then a move per call
  const z = new Float32Array(L), tau = new Float32Array(31), act = new Int32Array(32);
  let fill = 0, calls = 0;
  return async a => {
    act[31] = calls ? a : 0;
    const eps = noise(seed, calls, PAST, L), s = Math.min(fill, 31), pos = new Float32Array(32), prev = new Float32Array(31);
    pos[s] = 1;
    if (s) prev[s - 1] = 1;
    const o = await sess.run({ past: z.map((v, i) => tau[i >> 9] * v + (1 - tau[i >> 9]) * eps[i]), tau, act,
      noise: noise(seed, calls, NOISE_BUF, 512), rnoise: noise(seed, calls, RNOISE, 512), pos, prev,
      kmask: new Float32Array(32).fill(-1e30, s, 31) });
    if (fill < 31) { z.set(o.z, fill * 512); tau[fill] = 0.9; act[fill++] = calls ? a : 0; }
    else { z.copyWithin(0, 512); z.set(o.z, 30 * 512); tau.copyWithin(0, 1); tau[30] = 0.9; act.copyWithin(0, 1); }
    calls++;
    return { rgb: o.rgb, reward: o.reward[0], done: o.done[0] };
  };
}

function look(rgb) {  // what a first frame shows
  const names = tiles(rgb);
  let bright = 0;
  for (let i = 0; i < 49 * 63 * 3; i++) bright += rgb[i];
  let tree = null;
  for (const [dr, dc, dir] of [[-1, 0, 'up'], [0, 1, 'right']]) {
    for (let d = 1; d <= 3 && !tree; d++) {
      const n = names[(3 + dr * d) * 9 + 4 + dc * d];
      if (n === 'tree') tree = { dir, d };
      else if (n !== 'grass') break;
    }
  }
  return { fresh: fresh(rgb), slots: slots(rgb).join(''), bright: Math.round(bright / (49 * 63 * 3)), asleep: asleep(rgb),
    grass: +(names.filter(n => n === 'grass').length / 63).toFixed(2),
    mobs: names.filter((n, i) => i !== 31 && (n === 'zombie' || n === 'skeleton')).length, tree, face: facing(rgb) };
}

async function walkthrough(sess, seed) {  // the tutorial's path: moves until the first and second wood and the table
  const step = world(sess, seed);
  let o = await step(0), face = facing(o.rgb), wood = 0, last = -1, table = null, deaths = 0;
  const got = [], acts = [];
  for (let n = 0; n < 40 && table === null; n++) {
    const way = route(tiles(o.rgb), face);
    const a = wood >= 2 ? 8 : way?.facing ? 5 : way ? way.next : 3;
    o = await step(a);
    acts.push(a);
    if (a >= 1 && a <= 4) face = a;
    const w = inventory(o.rgb).wood;
    if (w.margin > 600 || w.n === last) wood = Math.max(wood, w.n);
    last = w.n;
    if (a === 5 && o.reward > 0.5) wood = Math.max(wood, 1);
    if (got.length < wood) got.push(n + 1);
    if (a === 8 && (o.reward > 0.5 || route(tiles(o.rgb), face, 'table')?.facing)) table = n + 1;
    deaths += o.done > 0.5;
  }
  return { wood1: got[0] ?? null, wood2: got[1] ?? null, table, deaths, acts: acts.join(',') };
}

function thumb(rgb, label) {
  const c = document.createElement('canvas'), f = document.createElement('figure');
  [c.width, c.height] = [63, 63];
  const img = c.getContext('2d').createImageData(63, 63);
  for (let i = 0; i < 63 * 63; i++) img.data.set([rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], 255], 4 * i);
  c.getContext('2d').putImageData(img, 0, 0);
  f.append(c, Object.assign(document.createElement('figcaption'), { textContent: label }));
  grid.append(f);
}

async function main() {
  const device = await (await navigator.gpu.requestAdapter()).requestDevice();
  const bytes = new Uint8Array(await (await fetch(MODEL, { cache: 'no-store' })).arrayBuffer());
  const sess = await create(device, bytes);
  if (!sess.meta.inputs.pos) throw new Error(`${MODEL} has no slot inputs: export it with export_web.py`);
  const rows = [], firsts = new Map(), t0 = performance.now();
  for (let seed = FROM; seed < FROM + N; seed++) {
    const o = await world(sess, seed)(0);
    rows.push({ seed, ...look(o.rgb) });
    firsts.set(seed, o.rgb);
    if (seed % 50 === 0) say(`first frames: ${seed - FROM} of ${N}`);
  }
  const ms = (performance.now() - t0) / N, isFresh = new Set(rows.filter(r => r.fresh).map(r => r.seed));
  // play.js: a random seed, then the next one until a first frame is fresh (at most REROLL tries)
  const tries = rows.map(r => { let k = 0; while (k < REROLL - 1 && r.seed + k < FROM + N && !isFresh.has(r.seed + k)) k++; return k + 1; });
  const cand = rows.filter(r => r.fresh && r.tree?.dir === 'up' && r.tree.d >= 2 && !r.mobs && r.bright > 60)
    .sort((a, b) => a.tree.d - b.tree.d || b.grass - a.grass).slice(0, CANDS);
  for (const [i, r] of cand.entries()) {
    say(`walkthrough: candidate ${i + 1} of ${cand.length}`);
    Object.assign(r, await walkthrough(sess, r.seed));
  }
  const ok = cand.filter(r => r.table && !r.deaths).sort((a, b) => a.table - b.table || a.wood2 - b.wood2);
  for (const r of ok) thumb(firsts.get(r.seed), `${r.seed} ${code(r.seed)}: table at ${r.table}`);
  const share = f => +(rows.filter(f).length / rows.length).toFixed(3);
  window.result = {
    model: MODEL, seeds: [FROM, FROM + N - 1], ms_per_frame: +ms.toFixed(1),
    fresh: share(r => r.fresh), bright: share(r => r.bright > 60), asleep: share(r => r.asleep), mobs: share(r => r.mobs > 0),
    // what a world that play.js accepts can still start with: dark, asleep, a zombie or skeleton in view
    fresh_dark: share(r => r.fresh && r.bright <= 60), fresh_asleep: share(r => r.fresh && r.asleep),
    fresh_mobs: share(r => r.fresh && r.mobs > 0),
    tries_mean: +(tries.reduce((x, y) => x + y, 0) / tries.length).toFixed(2), tries_max: Math.max(...tries),
    tutorial: ok.slice(0, 10).map(({ seed, tree, grass, bright, table, wood1, wood2, acts }) =>
      ({ seed, code: code(seed), tree, grass, bright, wood1, wood2, table, acts })),
    not_fresh: rows.filter(r => !r.fresh).slice(0, 40).map(r => `${r.seed}:${r.slots}`),
  };
  say(JSON.stringify(window.result, null, 1));
  document.title = 'done';
}

main().catch(e => { say(`error: ${e.message ?? e}`); document.title = 'error'; });

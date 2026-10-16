// Play inside the tinywmcraft world model of Craftax-Classic, in the browser on WebGPU. Mounts into #play.
// Icons are Craftax's own sprites (github.com/MichaelTMatthews/Craftax, MIT, (c) 2024 Michael Matthews).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';

const GIFENC = 'https://cdn.jsdelivr.net/npm/gifenc@1.0.3/dist/gifenc.esm.js';
const q = new URLSearchParams(location.search);
const MODEL = q.get('model') ?? 'https://huggingface.co/maharishiva/tinywmcraft/resolve/main/web/play.onnx';
const STARTS = q.get('starts') ?? 'starts.bin';
const DONE_THRESHOLD = 0.5;                   // done probability that ends the episode
const CLIP = 300, MAX_SCALE = 7;              // frames kept for the clip, largest pixel scale
const L = 31 * 512, RGB = 63 * 63 * 3, H = 88; // one 31-frame latent window, one frame, canvas height
const BG = '#0a0a0a', FG = '#f0f0ee', ORANGE = '#ff6a13', KEY = '#1c1c1e', EDGE = '#3a3a3e', CAP = '#aaaaa8';
// index = action; e.code, so the keys work on any keyboard layout
const CODES = ['KeyQ', 'KeyA', 'KeyD', 'KeyW', 'KeyS', 'Space', 'Tab', 'KeyR', 'KeyT', 'KeyF', 'KeyP',
  'Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5', 'Digit6'];
const CAPS = ['Q', 'A', 'D', 'W', 'S', 'SPACE', 'TAB', 'R', 'T', 'F', 'P', '1', '2', '3', '4', '5', '6'];
const ARROWS = { ArrowLeft: 1, ArrowRight: 2, ArrowUp: 3, ArrowDown: 4 };
const NAMES = ['noop', 'left', 'right', 'up', 'down', 'do', 'sleep', 'place stone', 'place table', 'place furnace',
  'place plant', 'make wood pickaxe', 'make stone pickaxe', 'make iron pickaxe', 'make wood sword',
  'make stone sword', 'make iron sword'];
// the keyboard of play.gif under the frame: action, x, y, width of each key; -1 shows any other key pressed
const BOARD = [[3, 10, 0, 9], [1, 0, 10, 9], [4, 10, 10, 9], [2, 20, 10, 9], [5, 31, 10, 23], [-1, 31, 0, 15]];
const ICONS = [8, 7, 9, 10, 6, 0, 11, 12, 13, 14, 15, 16];
const FONT = {  // 3x5 pixel font of docs/make_play_gif.py
  A: '010101111101101', B: '110101110101110', C: '011100100100011', D: '110101101101110', E: '111100110100111',
  F: '111100110100100', P: '110101110100100', Q: '010101101110011', R: '110101110101101', S: '011100010001110',
  T: '111010010010010', W: '101101111111101', 1: '010110010010111', 2: '111001111100111', 3: '111001111001111',
  4: '101101111001001', 5: '111100111001111', 6: '111100111101111', '+': '000010111010000',
};

const root = document.getElementById('play');
root.innerHTML = `<div class="screen"><canvas width="63" height="${H}"></canvas><div class="msg"></div>
<div class="bar"><span class="st"></span><button data-a="reset">new world</button><button data-a="save">save clip</button></div></div>
<div class="acts">${ICONS.map(a => `<button data-a="${a}" title="${NAMES[a]}">` +
  `<img src="icons/${NAMES[a].replaceAll(' ', '_')}.png" alt="${NAMES[a]}"><small>${CAPS[a].toLowerCase()}</small></button>`).join('')}</div>`;
const canvas = root.querySelector('canvas'), msg = root.querySelector('.msg'), st = root.querySelector('.st');
const acts = root.querySelector('.acts'), ctx = canvas.getContext('2d', { willReadFrequently: true });

let session, starts, z, tau, act, frames = [], busy = false, dead = false, visible = false;

const status = (s, hot) => { st.textContent = s; st.className = hot ? 'st hot' : 'st'; };

function fit() {  // largest integer scale that fits next to (or above) the icons and in the viewport
  const side = getComputedStyle(root).flexDirection === 'row' ? acts.offsetWidth + 16 : 0;
  const s = Math.min(MAX_SCALE, Math.floor((root.clientWidth - side) / 63), Math.floor((innerHeight - 80) / H));
  canvas.parentNode.style.width = canvas.style.width = 63 * Math.max(1, s) + 'px';
}

function text(s, x, y, color) {
  ctx.fillStyle = color;
  for (const ch of s) {
    for (let i = 0; i < 15; i++) if (FONT[ch][i] === '1') ctx.fillRect(x + i % 3, y + (i / 3 | 0), 1, 1);
    x += 4;
  }
}

function board(a) {  // the keys under the frame, the pressed one in orange
  ctx.fillStyle = BG;
  ctx.fillRect(0, 63, 63, H - 63);
  const other = a !== undefined && !BOARD.some(b => b[0] === a);
  for (const [k, x, y, w] of BOARD) {
    const on = k < 0 ? other : k === a, cap = k < 0 ? (other ? CAPS[a] : '') : CAPS[k];
    ctx.fillStyle = on ? ORANGE : EDGE;
    ctx.fillRect(x + 4, y + 66, w, 9);
    ctx.fillStyle = on ? ORANGE : KEY;
    ctx.fillRect(x + 5, y + 67, w - 2, 7);
    text(cap, x + 4 + ((w - 4 * cap.length + 1) >> 1), y + 68, on ? BG : CAP);
  }
}

function draw(rgb, a, plus) {  // the frame, the +1 of a reward, the keys; recorded for the clip
  const img = ctx.createImageData(63, 63);
  for (let i = 0; i < 63 * 63; i++) img.data.set([rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], 255], 4 * i);
  ctx.putImageData(img, 0, 0);
  if (plus) text('+1', 55, 1, FG);
  board(a);
  frames.push(ctx.getImageData(0, 0, 63, H));
  if (frames.length > CLIP) frames.shift();
}

const gauss = n => {  // standard normal, Box-Muller
  const x = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * Math.log(1 - Math.random())), t = 2 * Math.PI * Math.random();
    x[i] = r * Math.cos(t);
    x[i + 1] = r * Math.sin(t);
  }
  return x;
};

function reset() {  // a random real start: 31 clean latents with their incoming actions, and the last real frame
  const k = Math.floor(Math.random() * starts.n);
  z = starts.lat.slice(k * L, (k + 1) * L);
  tau = new Float32Array(31).fill(1);
  act = new Int32Array(32);
  act.set(starts.acts.subarray(k * 31, k * 31 + 31));
  frames = [];
  dead = false;
  msg.textContent = '';
  draw(starts.rgb.subarray(k * RGB, (k + 1) * RGB));
  if (session) status('ready: W A S D to move, SPACE to hit', true);
}

function feeds() {  // the past fed corrupted to its signal levels tau, fresh noise every call
  const eps = gauss(L), t = (d, x, dims) => new ort.Tensor(d, x, dims);
  return {
    past: t('float32', z.map((v, i) => tau[i >> 9] * v + (1 - tau[i >> 9]) * eps[i]), [1, 31, 4, 4, 32]),
    tau: t('float32', tau, [1, 31]), act: t('int32', act, [1, 32]),
    noise: t('float32', gauss(512), [1, 4, 4, 32]), rnoise: t('float32', gauss(512), [1, 4, 4, 32]),
  };
}

async function step(a) {
  busy = true;
  act[31] = a;
  const t0 = performance.now(), o = await session.run(feeds());
  z.copyWithin(0, 512);
  z.set(o.z.data, 30 * 512);
  tau.copyWithin(0, 1);
  tau[30] = 0.9;
  act.copyWithin(0, 1);
  dead = o.done.data[0] > DONE_THRESHOLD;
  draw(o.rgb.data, a, o.reward.data[0] > 0.5);
  status(`${NAMES[a]}, frame ${frames.length - 1}, ${Math.round(performance.now() - t0)} ms on your GPU`);
  if (dead) msg.innerHTML = '<b>dead</b><button class="hot" data-a="save">save clip</button>' +
    '<button data-a="reset">new world</button>';
  busy = false;
}

async function save() {  // the last CLIP frames since the reset as a GIF: 4x nearest-neighbour, about 6 fps
  const { GIFEncoder, quantize, applyPalette } = await import(GIFENC);
  const G = 4, W = 63 * G, N = W * H * G, gif = GIFEncoder(), big = new Uint8Array(N);
  frames.forEach((f, i) => {
    const palette = quantize(f.data, 256), idx = applyPalette(f.data, palette);
    for (let p = 0; p < N; p++) big[p] = idx[(p / W / G | 0) * 63 + ((p % W) / G | 0)];
    gif.writeFrame(big, W, H * G, { palette, delay: i === frames.length - 1 ? 2000 : 167 });
  });
  gif.finish();
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([gif.bytes()], { type: 'image/gif' }));
  link.download = 'tinywmcraft.gif';
  link.click();
}

const fail = e => { msg.textContent = String(e.message ?? e); status('error'); };

function press(a) {  // an action index, 'reset' or 'save'
  if (a === 'save') return frames.length && save();
  if (!session) return;
  if (a === 'reset') return reset();
  if (!busy && !dead) step(a).catch(fail);
}

async function load(url, what) {  // fetch with a progress line
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = (+r.headers.get('content-length') / 1e6).toFixed(0), parts = [];
  let got = 0;
  for (const reader = r.body.getReader(); ;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    status(`loading the ${what}: ${(got / 1e6).toFixed(0)} / ${total} MB`);
  }
  return new Uint8Array(await new Blob(parts).arrayBuffer());
}

async function main() {
  const b = await load(STARTS, 'start frames'), n = new DataView(b.buffer).getUint32(0, true);
  starts = { n, lat: new Float32Array(b.buffer, 4, n * L), acts: b.subarray(4 + 4 * n * L, 4 + 4 * n * L + 31 * n) };
  starts.rgb = b.subarray(4 + 4 * n * L + 31 * n);
  reset();
  const model = await load(MODEL, 'world model');
  status('compiling the world model for your GPU');
  const s = await ort.InferenceSession.create(model, { executionProviders: ['webgpu'], logSeverityLevel: 3 });
  await s.run(feeds());   // warm-up: compiles the shaders before the first keypress
  session = s;
  status('ready: W A S D to move, SPACE to hit', true);
}

root.onmousedown = e => e.preventDefault();   // buttons never take focus, so space stays DO
root.onclick = e => {
  const b = e.target.closest('button');
  if (b) return press(/^\d+$/.test(b.dataset.a) ? +b.dataset.a : b.dataset.a);
  if (e.target !== canvas) return;
  const s = canvas.clientWidth / 63, x = e.offsetX / s - 4, y = e.offsetY / s - 66;   // a click on a drawn key
  const k = BOARD.find(([a, kx, ky, w]) => a >= 0 && x >= kx && x < kx + w && y >= ky && y < ky + 9);
  if (k) press(k[0]);
};
addEventListener('keydown', e => {
  const a = e.code in ARROWS ? ARROWS[e.code] : CODES.indexOf(e.code);
  if (!visible || e.ctrlKey || e.metaKey || e.altKey || (a < 0 && e.code !== 'Backspace')) return;
  e.preventDefault();
  press(a < 0 ? 'reset' : a);
});
new IntersectionObserver(([e]) => { visible = e.intersectionRatio >= 0.3; }, { threshold: [0, 0.3] }).observe(root);
addEventListener('resize', fit);
fit();
if (navigator.gpu) main().catch(fail);
else status('needs WebGPU: Chrome, Edge, or Safari 26+');

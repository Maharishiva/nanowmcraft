// Play inside the tinywmcraft world model of Craftax-Classic, in the browser on WebGPU. Mounts into #play.
// Icons are Craftax's own sprites (github.com/MichaelTMatthews/Craftax, MIT, (c) 2024 Michael Matthews).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';

const MEDIABUNNY = 'https://cdn.jsdelivr.net/npm/mediabunny@1.61.5/dist/bundles/mediabunny.min.mjs';
const LINK = ['MAHARISHIVA.GITHUB.IO/', 'TINYWMCRAFT'];   // on top of every saved clip
const SITE = 'https://maharishiva.github.io/tinywmcraft/';
const q = new URLSearchParams(location.search);
const MODEL = q.get('model') ??   // a pinned revision, so the copy kept in the browser's cache stays valid
  'https://huggingface.co/maharishiva/tinywmcraft/resolve/8346c17f18c91ad577f9b7f1227fb7c614d4a432/web/play.onnx';
const STARTS = q.get('starts') ?? 'starts.bin';
const DONE_THRESHOLD = 0.5;                   // done probability that ends the episode
const CLIP = 300, MAX_SCALE = 7;              // frames kept for the clip, largest pixel scale
const L = 31 * 512, RGB = 63 * 63 * 3;        // one 31-frame latent window, one frame
const TOP = 7, H = TOP + 88, MAP = 49;        // canvas: the score strip, the frame, the keys; the map part of the frame
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
  G: '011100101101011', H: '101101111101101', I: '111010010010111', M: '101111111101101', N: '110101101101101',
  O: '010101101101010', U: '101101101101111', V: '101101101101010', Y: '101101010010010', '.': '000000000000010',
  '/': '001001010100100', 0: '111101101101111', 7: '111001001001001', 8: '111101111101111', 9: '111101111001111',
  ' ': '000000000000000',
};
// the clip can go to the system share sheet only where the browser shares files
const FILES = !!navigator.canShare?.({ files: [new File([''], 'x.mp4', { type: 'video/mp4' })] });

const root = document.getElementById('play');
root.innerHTML = `<div class="screen"><div class="view"><canvas width="63" height="${H}"></canvas><div class="msg"></div></div>
<div class="bar"><span class="st"></span><button data-a="save">save clip</button><button data-a="share">share</button>
<button data-a="reset">new world</button></div></div>
<div class="acts">${ICONS.map(a => `<button data-a="${a}" title="${NAMES[a]}">` +
  `<img src="icons/${NAMES[a].replaceAll(' ', '_')}.png" alt="${NAMES[a]}"><img class="k" src="${label(CAPS[a])}" alt=""></button>`).join('')}</div>`;
const canvas = root.querySelector('canvas'), msg = root.querySelector('.msg'), st = root.querySelector('.st');
const acts = root.querySelector('.acts'), ctx = canvas.getContext('2d', { willReadFrequently: true });

let session, starts, z, tau, act, mux, made, frames = [], tick = 0, score = 0, episode = 0;
let best = +localStorage.getItem('tinywmcraft.best') || 0, busy = false, dead = false, saving = false;

const status = (s, hot) => { st.textContent = s; st.className = hot ? 'st hot' : 'st'; };

function fit() {  // the largest integer scale that fits; the action keys span the map's height beside it, or its width under it
  const row = getComputedStyle(root).flexDirection === 'row', W = root.clientWidth;
  const s = Math.max(1, Math.min(MAX_SCALE, Math.floor((W - (row ? 16 : 0)) / (row ? 99.5 : 63)), Math.floor((innerHeight - 80) / H)));
  const b = row ? (MAP - 3) * s / 4 : 58 * s / 6;   // one game pixel between keys, as under the frame
  root.querySelector('.screen').style.width = canvas.style.width = 63 * s + 'px';
  Object.assign(acts.style, { gridTemplateColumns: `repeat(${row ? 3 : 6}, ${b}px)`, gridAutoRows: b + 'px', gap: s + 'px',
    marginTop: row ? TOP * s + 'px' : '' });
  root.style.setProperty('--px', s + 'px');   // one game pixel: the keys' border, and the icons drawn at the game's tile size
  root.style.setProperty('--label', 5 * Math.max(1, s >> 1) + 'px');
  msg.style.padding = `${TOP * s}px 16px ${25 * s}px`;   // what shows over the game sits on the frame
}

function text(s, x, y, color, c = ctx, u = 1) {  // the 3x5 pixel font, u screen pixels per font pixel; returns the end x
  c.fillStyle = color;
  for (const ch of s) {
    for (let i = 0; i < 15; i++) if (FONT[ch][i] === '1') c.fillRect(x + u * (i % 3), y + u * (i / 3 | 0), u, u);
    x += 4 * u;
  }
  return x;
}

function label(s) {  // a key name in the pixel font, as an image for the buttons
  const c = document.createElement('canvas');
  [c.width, c.height] = [4 * s.length - 1, 5];
  text(s, 0, 0, CAP, c.getContext('2d'));
  return c.toDataURL();
}

function head(hot) {  // the score and the best score
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, 63, TOP);
  text(String(score), text('SCORE', 2, 1, CAP) + 2, 1, hot ? ORANGE : FG);
  const right = `HI ${best}`;
  text(right, 62 - 4 * right.length, 1, CAP);
}

function board(a, r = 0) {  // the keys under the frame, the pressed one in orange; a square filled by the reward
  const y0 = TOP + 66;
  ctx.fillStyle = BG;
  ctx.fillRect(0, TOP + 63, 63, 25);
  const other = a !== undefined && !BOARD.some(b => b[0] === a);
  for (const [k, x, y, w] of BOARD) {
    const on = k < 0 ? other : k === a, cap = k < 0 ? (other ? CAPS[a] : '') : CAPS[k];
    ctx.fillStyle = on ? ORANGE : EDGE;
    ctx.fillRect(x + 4, y + y0, w, 9);
    ctx.fillStyle = on ? ORANGE : KEY;
    ctx.fillRect(x + 5, y + y0 + 1, w - 2, 7);
    text(cap, x + 4 + ((w - 4 * cap.length + 1) >> 1), y + y0 + 2, on ? BG : CAP);
  }
  const fill = Math.round(7 * Math.min(Math.max(r, 0), 1));
  ctx.fillStyle = EDGE;
  ctx.fillRect(53, y0, 9, 9);
  ctx.fillStyle = KEY;
  ctx.fillRect(54, y0 + 1, 7, 7);
  ctx.fillStyle = r > 0.5 ? ORANGE : CAP;
  ctx.fillRect(54, y0 + 8 - fill, 7, fill);
}

function draw(rgb, a, r = 0) {  // the score, the frame with a +1 for a predicted reward over 0.5, the keys; kept for the clip
  const img = ctx.createImageData(63, 63);
  for (let i = 0; i < 63 * 63; i++) img.data.set([rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], 255], 4 * i);
  ctx.putImageData(img, 0, TOP);
  if (r > 0.5) text('+1', 55, TOP + 1, FG);
  head(r > 0.5);
  board(a, r);
  tick++;
  frames.push(ctx.getImageData(0, 0, 63, H));
  if (frames.length > CLIP) frames.shift();
}

function wasted() {  // the death screen: the frame washed red, DEAD and the score; it replaces the last frame, so it ends the clip
  const img = ctx.getImageData(0, TOP, 63, 63), d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const y = 0.3 * d[i] + 0.59 * d[i + 1] + 0.11 * d[i + 2];
    [d[i], d[i + 1], d[i + 2]] = [70 + 0.7 * y, 0.18 * y, 0.18 * y];
  }
  ctx.putImageData(img, 0, TOP);
  ctx.fillStyle = 'rgba(10, 10, 10, 0.6)';
  ctx.fillRect(0, TOP + 15, 63, 31);
  text('DEAD', 9, TOP + 18, FG, ctx, 3);
  const n = String(score), w = 4 * 'SCORE '.length + 8 * n.length - 2;
  text(n, text('SCORE ', (63 - w) >> 1, TOP + 39, CAP), TOP + 36, ORANGE, ctx, 2);
  frames[frames.length - 1] = ctx.getImageData(0, 0, 63, H);
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
  episode++;
  const k = Math.floor(Math.random() * starts.n);
  z = starts.lat.slice(k * L, (k + 1) * L);
  tau = new Float32Array(31).fill(1);
  act = new Int32Array(32);
  act.set(starts.acts.subarray(k * 31, k * 31 + 31));
  frames = [];
  score = 0;
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

async function step(a) {  // results of a step started before a reset are dropped
  busy = true;
  const ep = episode;
  act[31] = a;
  try {
    const t0 = performance.now(), o = await session.run(feeds());
    if (ep !== episode) return;
    z.copyWithin(0, 512);
    z.set(o.z.data, 30 * 512);
    tau.copyWithin(0, 1);
    tau[30] = 0.9;
    act.copyWithin(0, 1);
    dead = o.done.data[0] > DONE_THRESHOLD;
    const r = o.reward.data[0];
    score += r > 0.5;
    if (score > best) localStorage.setItem('tinywmcraft.best', best = score);
    draw(o.rgb.data, a, r);
    status(`${NAMES[a]}, ${Math.round(performance.now() - t0)} ms on your GPU`, r > 0.5);
    if (!dead) return;
    wasted();
    msg.innerHTML = '<div class="end"><button data-a="save">save clip</button><button class="hot" data-a="share">share</button>' +
      '<button data-a="reset">new world</button></div>';
    clip().catch(() => {});   // encoded now, so share opens at once
  } finally {
    busy = false;
  }
}

async function encode() {  // the last CLIP frames as an MP4 under the link: 8x nearest-neighbour, 6 fps, last frame 2 s
  const fs = frames.slice(), d = new Date(), p2 = n => String(n).padStart(2, '0');   // play goes on while it encodes
  const name = `tinywmcraft-score-${score}-${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}-` +
    `${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}.mp4`;
  const { Output, Mp4OutputFormat, BufferTarget, CanvasSource, QUALITY_HIGH } = await (mux ??= import(MEDIABUNNY));
  const G = 8, LINE = 5, U = 3, out = document.createElement('canvas'), one = document.createElement('canvas');
  [out.width, out.height, one.width, one.height] = [63 * G, (LINE + H) * G, 63, H];
  const c = out.getContext('2d'), y = (LINE * G - 5 * U) >> 1, x = (out.width - U * (4 * LINK.join('').length - 1)) >> 1;
  c.imageSmoothingEnabled = false;
  c.fillStyle = BG;
  c.fillRect(0, 0, out.width, LINE * G);
  text(LINK[1], text(LINK[0], x, y, CAP, c, U), y, ORANGE, c, U);
  const output = new Output({ format: new Mp4OutputFormat(), target: new BufferTarget() });
  const source = new CanvasSource(out, { codec: 'avc', bitrate: QUALITY_HIGH });
  output.addVideoTrack(source, { frameRate: 6 });
  await output.start();
  for (const [i, f] of fs.entries()) {
    one.getContext('2d').putImageData(f, 0, 0);
    c.drawImage(one, 0, LINE * G, out.width, H * G);
    await source.add(i / 6, i === fs.length - 1 ? 2 : 1 / 6);
  }
  await output.finalize();
  return new File([output.target.buffer], name, { type: 'video/mp4' });
}

const clip = () => (made?.[0] === tick ? made : (made = [tick, encode()]))[1];   // encoded once per frame shown

async function save() {
  status('saving the clip');
  const f = await clip(), link = document.createElement('a');
  link.href = URL.createObjectURL(f);
  link.download = f.name;
  link.click();
  status('clip saved');
}

async function share() {  // the clip to the system share sheet; without file sharing, a post on X and the clip saved to attach
  const text = `I scored ${score} inside a world model trained in 10 minutes for $0.1 of compute. ` +
    'Every frame is generated by a neural network on my GPU. Can you beat it?';
  if (!FILES) {
    open(`https://x.com/intent/post?text=${encodeURIComponent(text)}&url=${encodeURIComponent(SITE)}`, '_blank');
    await save();
    return status('clip saved: attach it to the post', true);
  }
  status('preparing the clip');
  const files = [await clip()];
  try {
    await navigator.share({ files, text: `${text} ${SITE}` });
    status('');
  } catch (e) {
    if (e.name === 'NotAllowedError') status('the clip is ready: press share again', true);   // the click expired while encoding
    else if (e.name === 'AbortError') status('');
    else throw e;
  }
}

const fail = e => { msg.textContent = String(e.message ?? e); status('error'); };

function press(a) {  // an action index, 'reset', 'save' or 'share'
  if (a === 'save' || a === 'share') {
    if (!frames.length || saving) return;
    saving = true;
    return (a === 'save' ? save() : share()).catch(fail).finally(() => { saving = false; });
  }
  if (!session) return;
  if (a === 'reset') return reset();
  if (busy || dead) return;
  for (const b of acts.children) b.classList.toggle('on', +b.dataset.a === a);
  step(a).catch(fail);
}

function loading(what, frac, note) {  // the screen over the game until it plays: what is loading, a progress bar
  msg.innerHTML = `<b>${what}</b><div class="prog${frac < 0 ? ' wait' : ''}"><i style="width:${frac < 0 ? 100 : 100 * frac}%"></i></div>` +
    `<span>${note}</span>`;
}

async function load(url, what, keep) {  // fetch with a progress bar; keep: from and into the browser's cache
  const box = keep && await self.caches?.open('tinywmcraft').catch(() => null), hit = await box?.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  loading(`loading the ${what}`, 0, 'connecting');
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +r.headers.get('content-length'), parts = [];
  let got = 0;
  for (const reader = r.body.getReader(); ;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    loading(`loading the ${what}`, got / total, `${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB, then it runs on your GPU`);
  }
  const bytes = new Uint8Array(await new Blob(parts).arrayBuffer());
  if (box) {   // the next visit loads it from here; older versions go
    for (const k of await box.keys()) if (k.url !== url) await box.delete(k);
    await box.put(url, new Response(bytes)).catch(() => {});
  }
  return bytes;
}

async function main() {
  const b = await load(STARTS, 'start frames'), n = new DataView(b.buffer).getUint32(0, true);
  starts = { n, lat: new Float32Array(b.buffer, 4, n * L), acts: b.subarray(4 + 4 * n * L, 4 + 4 * n * L + 31 * n) };
  starts.rgb = b.subarray(4 + 4 * n * L + 31 * n);
  reset();
  const model = await load(MODEL, 'world model', true);
  loading('compiling it for your GPU', -1, 'no server, no queue: every frame is made on your machine');
  const s = await ort.InferenceSession.create(model, { executionProviders: ['webgpu'], logSeverityLevel: 3 });
  await s.run(feeds());   // warm-up: compiles the shaders before the first keypress
  session = s;
  msg.textContent = '';
  status('ready: W A S D to move, SPACE to hit', true);
  mux = import(MEDIABUNNY);   // the MP4 encoder, loaded in the background so saving a clip is quick
}

root.onmousedown = e => e.preventDefault();   // buttons never take focus, so space stays DO
root.onclick = e => {
  const b = e.target.closest('button');
  if (b) return press(/^\d+$/.test(b.dataset.a) ? +b.dataset.a : b.dataset.a);
  if (e.target !== canvas) return;
  const s = canvas.clientWidth / 63, x = e.offsetX / s - 4, y = e.offsetY / s - TOP - 66;   // a click on a drawn key
  const k = BOARD.find(([a, kx, ky, w]) => a >= 0 && x >= kx && x < kx + w && y >= ky && y < ky + 9);
  if (k) press(k[0]);
};
addEventListener('keydown', e => {
  const a = e.code in ARROWS ? ARROWS[e.code] : CODES.indexOf(e.code);
  const r = root.getBoundingClientRect(), visible = r.bottom > innerHeight / 3 && r.top < innerHeight * 2 / 3;
  if (!visible || e.ctrlKey || e.metaKey || e.altKey || (a < 0 && e.code !== 'Backspace')) return;
  e.preventDefault();
  press(a < 0 ? 'reset' : a);
});
addEventListener('resize', fit);
fit();
if (navigator.gpu) main().catch(fail);
else msg.innerHTML = '<b>needs WebGPU</b><span>Chrome, Edge, or Safari 26+</span>';

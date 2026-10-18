// Seeded sampler noise, so a seed always gives the same world: noise 'tw-noise-1'.
// Every noise number is a pure function of (seed, call, buffer, index). call: the model calls since the world began,
// 0 first: call 0 imagines the first frame, step t is call t.
// - Threefry-2x32, 20 rounds (Salmon et al. 2011, Random123; the generator under jax.random): key (seed, 0x74776d31),
//   counter (call, buffer << 16 | pair) -> two u32. Buffers: 0 the past's corruption (31 x 512), 1 noise, 2 rnoise.
// - Box-Muller turns the two u32 of pair p into the numbers 2p and 2p + 1: u1 = (x0 + 0.5) / 2^32, angle 2 pi x1 / 2^32,
//   r = sqrt(-2 ln u1), (r cos, r sin), rounded to float32. ln, sin and cos are written out with + - * / on doubles
//   (series below) and Math.sqrt, all correctly rounded IEEE 754 in every JS engine and in numpy, so every browser
//   makes the identical float32 numbers (Math.log, Math.sin, Math.cos differ in the last bit between engines).
export const PAST = 0, NOISE_BUF = 1, RNOISE = 2;
const KEY1 = 0x74776d31;   // 'twm1'
const ROT = [13, 15, 26, 6, 17, 29, 16, 24];

function threefry(k0, k1, c0, c1) {  // Threefry-2x32-20 of counter (c0, c1) under key (k0, k1): two u32
  const ks = [k0 >>> 0, k1 >>> 0, (0x1bd11bda ^ k0 ^ k1) >>> 0];
  let x0 = (c0 + ks[0]) >>> 0, x1 = (c1 + ks[1]) >>> 0;
  for (let r = 0; r < 20; r++) {
    const s = ROT[r & 7];
    x0 = (x0 + x1) >>> 0;
    x1 = ((x1 << s) | (x1 >>> (32 - s))) >>> 0;
    x1 = (x1 ^ x0) >>> 0;
    if ((r & 3) === 3) {
      const j = (r + 1) >> 2;
      x0 = (x0 + ks[j % 3]) >>> 0;
      x1 = (x1 + ks[(j + 1) % 3] + j) >>> 0;
    }
  }
  return [x0, x1];
}

const INV2 = [1];                                   // 2^-e, exact
for (let e = 1; e <= 64; e++) INV2.push(INV2[e - 1] * 0.5);
const LN2 = 0.6931471805599453, HALF_PI = 1.5707963267948966, SQRT_HALF = 0.7071067811865476;
const LOGC = [], SINC = [1], COSC = [1];            // 1/(2j+1); (-1)^j/(2j+1)!; (-1)^j/(2j)!
for (let j = 0; j <= 12; j++) LOGC.push(1 / (2 * j + 1));
for (let j = 1; j <= 11; j++) SINC.push(-SINC[j - 1] / ((2 * j) * (2 * j + 1)));
for (let j = 1; j <= 11; j++) COSC.push(-COSC[j - 1] / ((2 * j - 1) * (2 * j)));

function lnu(x0) {  // ln((x0 + 0.5) / 2^32): n = 2 x0 + 1 = f 2^e, f in [sqrt(1/2), sqrt(2)), ln f by the atanh series
  let e = 33 - Math.clz32(x0);                      // the bit length of n
  let f = (2 * x0 + 1) * INV2[e];
  if (f < SQRT_HALF) { f = f * 2; e = e - 1; }
  const s = (f - 1) / (f + 1), s2 = s * s;
  let p = LOGC[12];
  for (let j = 11; j >= 0; j--) p = p * s2 + LOGC[j];
  return (e - 33) * LN2 + 2 * s * p;
}

function pair(x0, x1, out, i) {  // Box-Muller: out[i], out[i + 1]
  const r = Math.sqrt(-2 * lnu(x0));
  const a = (x1 & 0x3fffffff) * INV2[30] * HALF_PI, a2 = a * a, q = x1 >>> 30;
  let ps = SINC[11], pc = COSC[11];
  for (let j = 10; j >= 0; j--) { ps = ps * a2 + SINC[j]; pc = pc * a2 + COSC[j]; }
  const s = a * ps, c = pc;                         // sin a, cos a for a in [0, pi/2); the angle is q pi/2 + a
  const [cs, sn] = q === 0 ? [c, s] : q === 1 ? [-s, c] : q === 2 ? [-c, -s] : [s, -c];
  out[i] = r * cs;
  if (i + 1 < out.length) out[i + 1] = r * sn;
}

export function noise(seed, call, buf, n) {  // n standard normals (float32) for this seed, model call and buffer
  const out = new Float32Array(n);
  for (let p = 0; 2 * p < n; p++) {
    const [x0, x1] = threefry(seed, KEY1, call, (buf << 16 | p) >>> 0);
    pair(x0, x1, out, 2 * p);
  }
  return out;
}

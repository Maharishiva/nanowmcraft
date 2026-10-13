"""Tokenizer + flow-matching world model for 63x63 pixel environments (Craftax-Classic): a hybrid tokenizer (63x63 RGB
<-> 4x4x32 standardized latents: 7x7 tiles, six transformer blocks over the 9x9 tile grid, four conv blocks at 4x4) and a
diffusion-forcing transformer that predicts the next latent in x-space by flow matching, conditioned on the action, with
symlog two-hot reward and termination heads read from its per-frame summary; Muon on the kernels of both, Pallas kernels
for the attention, norms and activations on GPU. Ten minutes on one A100, compilation included: no persistent compilation
cache is used, so every run compiles fresh.

    python train.py --data data/train.npz --out runs/base/model.npz --seed 0
"""
import argparse
import hashlib
import json
import os
import time
from collections import namedtuple
from dataclasses import dataclass, asdict, fields
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
import jax
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
import jax.numpy as jnp
import numpy as np
import optax
import flax.nnx as nnx

import kernels

H = W = 63
TILE, GRID, GLAT, TOK = 7, 9, 4, 32
ENC_HID, DEC_HID, NB_HI, NB_LO = 128, 128, 6, 4
D, DEPTH, HEADS, THEADS = 256, 6, 8, 8
A_N, LMAX, WIN, STEPS, CTX_SIGNAL = 17, 40, 32, 8, 0.9
RBINS, RLIM = 255, 20.0
CDT, PDT = jnp.bfloat16, jnp.float32
MP = dict(dtype=CDT, param_dtype=PDT)
NRM = dict(dtype=PDT, param_dtype=PDT)
ZEROS = jax.nn.initializers.zeros

@dataclass(frozen=True)
class Config:
    seed: int = 0
    tok_steps: int = 9_000
    tok_batch: int = 256
    wm_steps: int = 18_000
    wm_batch: int = 64
    rollout: int = WIN
    lr: float = 9e-4
    tok_lr: float = 9e-4
    decay: float = 0.2
    hardmine: float = 0.5
    startsamp: float = 0.5
    reward_weight: float = 0.1
    done_weight: float = 0.1
    data_budget: int = 500_000

ModelState = namedtuple("ModelState", "enc dec df win", defaults=(WIN,))

def _to01(u8): return u8.astype(jnp.float32) / 255.0

def _tau_embed(tau, dim):
    ang = tau[:, None] * jnp.exp(jnp.linspace(0.0, jnp.log(1000.0), dim // 2))[None, :]
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)

def _chain(blocks, h):
    for b in blocks:
        h = b(h)
    return h

def _symlog(x): return jnp.sign(x) * jnp.log1p(jnp.abs(x))

def _symexp(x): return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)

def _two_hot(value):
    pos = (jnp.clip(value, -RLIM, RLIM) + RLIM) / (2.0 * RLIM) * (RBINS - 1)
    lo = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0, RBINS - 2)
    w = (pos - lo)[..., None]
    return (1.0 - w) * jax.nn.one_hot(lo, RBINS) + w * jax.nn.one_hot(lo + 1, RBINS)

class ResBlock(nnx.Module):
    def __init__(self, ch, rngs):
        self.n1 = nnx.RMSNorm(ch, rngs=rngs, **NRM)
        self.c1 = nnx.Conv(ch, ch, (3, 3), padding="SAME", rngs=rngs, **MP)
        self.n2 = nnx.RMSNorm(ch, rngs=rngs, **NRM)
        self.c2 = nnx.Conv(ch, ch, (3, 3), padding="SAME", rngs=rngs, **MP)

    def __call__(self, h):
        return h + self.c2(nnx.gelu(self.n2(self.c1(nnx.gelu(self.n1(h))))))

class TBlock(nnx.Module):
    """Pre-norm transformer block over the 81 tile tokens of one frame (Pallas attention and gelu on GPU)."""
    def __init__(self, ch, rngs):
        self.n1 = nnx.RMSNorm(ch, rngs=rngs, **NRM)
        self.qkv = nnx.Linear(ch, 3 * ch, rngs=rngs, **MP)
        self.o = nnx.Linear(ch, ch, rngs=rngs, **MP)
        self.n2 = nnx.RMSNorm(ch, rngs=rngs, **NRM)
        self.m1 = nnx.Linear(ch, 4 * ch, rngs=rngs, **MP)
        self.m2 = nnx.Linear(4 * ch, ch, rngs=rngs, **MP)
        self.pos = nnx.Param(0.02 * jax.random.normal(rngs.params(), (1, GRID, GRID, ch)))

    def __call__(self, h):
        B = h.shape[0]
        h = h + self.pos.value.astype(h.dtype)
        t = self.qkv(self.n1(h)).reshape((B, 1, GRID * GRID, -1))
        h = h + self.o(kernels.attention(t, jnp.zeros((t.shape[-1] // 3,), t.dtype), THEADS, False)[:, 0].reshape(h.shape))
        return h + self.m2(kernels.gelu(self.m1(self.n2(h))))

def _tiles(x):
    """Frames (N, H, W, 3) -> (N, GRID, GRID, TILE * TILE * 3): every 7x7 tile as one vector, so the tile stem and the
    tile readout are matmuls (a 7x7 stride-7 conv and its transpose, which cuDNN runs as slow generic kernels)."""
    N = x.shape[0]
    return x.reshape((N, GRID, TILE, GRID, TILE, 3)).transpose((0, 1, 3, 2, 4, 5)).reshape((N, GRID, GRID, TILE * TILE * 3))

def _untile(t):
    N = t.shape[0]
    return t.reshape((N, GRID, GRID, TILE, TILE, 3)).transpose((0, 1, 3, 2, 4, 5)).reshape((N, H, W, 3))

class TileOut(nnx.Module):
    """The 7x7 stride-7 transpose conv as a matmul over the tile layout; the kernel keeps the conv's (TILE, TILE, in, 3)
    shape and per-channel bias, so its init and Muon's flattening are the conv's."""
    def __init__(self, features, rngs):
        self.kernel = nnx.Param(jax.nn.initializers.lecun_normal()(rngs.params(), (TILE, TILE, features, 3), PDT))
        self.bias = nnx.Param(jnp.zeros((3,), PDT))

    def __call__(self, h):
        w = self.kernel[...].astype(CDT).transpose((2, 0, 1, 3)).reshape((h.shape[-1], TILE * TILE * 3))
        return h.astype(CDT) @ w + jnp.tile(self.bias[...].astype(CDT), TILE * TILE)

class Encoder(nnx.Module):
    """Frames (N, H, W, 3) in [0, 1] -> latents (N, GLAT, GLAT, TOK): 7x7 tiles, 6 transformer blocks at 9x9, 4 conv
    blocks at 4x4."""
    def __init__(self, rngs):
        self.stem = nnx.Linear(TILE * TILE * 3, 64, rngs=rngs, **MP)
        self.proj = nnx.Conv(64, ENC_HID, (1, 1), rngs=rngs, **MP)
        self.hi = nnx.List([TBlock(ENC_HID, rngs) for _ in range(NB_HI)])
        self.down = nnx.Conv(ENC_HID, 128, (3, 3), strides=(2, 2), padding="VALID", rngs=rngs, **MP)
        self.lo = nnx.List([ResBlock(128, rngs) for _ in range(NB_LO)])
        self.out = nnx.Conv(128, TOK, (1, 1), rngs=rngs, **MP)

    def __call__(self, x):
        h = _chain(self.hi, self.proj(nnx.gelu(self.stem(_tiles(x).astype(CDT)))))
        return self.out(_chain(self.lo, nnx.gelu(self.down(h)))).astype(jnp.float32)

class Decoder(nnx.Module):
    def __init__(self, rngs):
        self.inp = nnx.Conv(TOK, 128, (1, 1), rngs=rngs, **MP)
        self.lo = nnx.List([ResBlock(128, rngs) for _ in range(NB_LO)])
        self.up = nnx.ConvTranspose(128, DEC_HID, (3, 3), strides=(2, 2), padding="VALID", rngs=rngs, **MP)
        self.hi = nnx.List([TBlock(DEC_HID, rngs) for _ in range(NB_HI)])
        self.out = TileOut(DEC_HID, rngs)

    def __call__(self, z):
        h = _chain(self.lo, nnx.gelu(self.inp(z.astype(CDT))))
        return jax.nn.sigmoid(_untile(self.out(_chain(self.hi, nnx.gelu(self.up(h)))))).astype(jnp.float32)

class Tok(nnx.Module):
    def __init__(self, rngs):
        self.enc, self.dec = Encoder(rngs), Decoder(rngs)

class Block(nnx.Module):
    """adaLN-zero: spatial attention over a frame's tokens, causal temporal attention over frames, MLP (bf16);
    the attention, modulated norms, gated residuals and gelu are Pallas kernels on GPU (kernels.py)."""
    def __init__(self, rngs, d):
        self.mod = nnx.Linear(d, 9 * d, kernel_init=ZEROS, bias_init=ZEROS, rngs=rngs, **MP)
        self.sqkv = nnx.Linear(d, 3 * d, use_bias=False, rngs=rngs, **MP)
        self.sq_bias = nnx.Param(jnp.zeros((d,), PDT))
        self.so = nnx.Linear(d, d, rngs=rngs, **MP)
        self.tqkv = nnx.Linear(d, 3 * d, use_bias=False, rngs=rngs, **MP)
        self.tq_bias = nnx.Param(jnp.zeros((d,), PDT))
        self.to = nnx.Linear(d, d, rngs=rngs, **MP)
        self.m1 = nnx.Linear(d, 2 * d, rngs=rngs, **MP)
        self.m2 = nnx.Linear(2 * d, d, rngs=rngs, **MP)

    def __call__(self, h, cond):
        norm, gate, attend = kernels.norm_mod, kernels.gated_add, kernels.attention
        ss, sh, sg, ts, th, tg, ms, mh, mg = jnp.split(self.mod(cond)[:, :, None, :], 9, axis=-1)
        h = gate(h, sg, self.so(attend(self.sqkv(norm(h, ss, sh)), self.sq_bias[...], HEADS, False)))
        h = gate(h, tg, self.to(attend(self.tqkv(norm(h, ts, th)), self.tq_bias[...], HEADS, True)))
        return gate(h, mg, self.m2(kernels.gelu(self.m1(norm(h, ms, mh)))))

class DF(nnx.Module):
    """Predicts clean latents from (B, L, GLAT, GLAT, TOK) windows at per-frame signal levels tau (B, L) under the
    incoming actions a (B, L); zmean/zstd standardize the tokenizer's latents. The action embedding is read by a one-hot
    matmul in f32 at full precision: the gather's gradient was a scatter-add of every token into 17 rows, whose atomics
    cost a fifth of the step and which accumulated in bf16; the matmul's gradient is exact. The reward head reads each
    frame's summary (its mean token), the termination head that plus the outgoing action."""
    def __init__(self, rngs, num_actions=A_N, max_length=LMAX):
        self.zmean = nnx.Variable(jnp.zeros((TOK,), PDT))
        self.zstd = nnx.Variable(jnp.ones((TOK,), PDT))
        self.inp = nnx.Linear(TOK, D, rngs=rngs, **MP)
        self.spos = nnx.Param(0.02 * jax.random.normal(rngs.params(), (1, 1, GLAT * GLAT, D)))
        self.tpos = nnx.Param(0.02 * jax.random.normal(rngs.params(), (1, max_length, 1, D)))
        self.aemb = nnx.Embed(num_actions, D, rngs=rngs, **MP)
        self.taulin = nnx.Linear(D, D, rngs=rngs, **MP)
        self.blocks = nnx.List([Block(rngs, D) for _ in range(DEPTH)])
        self.outn = nnx.RMSNorm(D, use_scale=False, rngs=rngs, **NRM)
        self.out = nnx.Linear(D, TOK, kernel_init=ZEROS, rngs=rngs, **MP)
        self.rew1 = nnx.Linear(D, D, rngs=rngs, **MP)
        self.rew2 = nnx.Linear(D, RBINS, kernel_init=ZEROS, bias_init=ZEROS, rngs=rngs, **MP)
        self.done1 = nnx.Linear(2 * D, D, rngs=rngs, **MP)
        self.done2 = nnx.Linear(D, 1, kernel_init=ZEROS, bias_init=ZEROS, rngs=rngs, **MP)

    def __call__(self, x, tau, a):
        return self.forward(x, tau, a)[0]

    def action_embed(self, a):
        return jnp.matmul(jax.nn.one_hot(a, self.aemb.num_embeddings, dtype=PDT), self.aemb.embedding[...], precision=jax.lax.Precision.HIGHEST).astype(CDT)

    def reward_logits(self, summary):
        return self.rew2(nnx.gelu(self.rew1(summary.astype(CDT)))).astype(jnp.float32)

    def done_logits(self, summary, next_actions):
        h = jnp.concatenate([summary.astype(CDT), self.action_embed(next_actions)], -1)
        return self.done2(nnx.gelu(self.done1(h)))[..., 0].astype(jnp.float32)

    def forward(self, x, tau, a):
        """Clean-latent prediction (B, L, GLAT, GLAT, TOK) and the per-frame summary (B, L, D)."""
        B, L = x.shape[:2]
        ae = self.action_embed(a)
        h = self.inp(x.reshape((B, L, GLAT * GLAT, TOK)).astype(CDT))
        h = h + self.spos.value.astype(h.dtype) + self.tpos.value[:, :L].astype(h.dtype) + ae[:, :, None, :]
        cond = self.taulin(_tau_embed(tau.reshape((-1,)), D).astype(CDT)).reshape((B, L, D)) + ae
        for blk in self.blocks:
            h = blk(h, cond)
        h = self.outn(h)
        return self.out(h).reshape((B, L, GLAT, GLAT, TOK)).astype(jnp.float32), h.mean(2)

@nnx.jit
def _encode(enc, zmean, zstd, obs_u8):
    z = enc(_to01(obs_u8).reshape((-1, H, W, 3)))
    return ((z - zmean) / zstd).reshape(obs_u8.shape[:-3] + z.shape[-3:])

@nnx.jit
def _decode(dec, zmean, zstd, latent):
    z = (latent * zstd + zmean).reshape((-1,) + latent.shape[-3:])
    return jnp.clip(dec(z) * 255.0, 0, 255).astype(jnp.uint8).reshape(latent.shape[:-3] + (H, W, 3))

def encode(model, obs_u8):
    """uint8 frames (..., H, W, 3) -> standardized latents (..., GLAT, GLAT, TOK); jitted, the tokenizer's Pallas
    kernels would otherwise recompile on every eager call."""
    return _encode(model.enc, model.df.zmean.value, model.df.zstd.value, jnp.asarray(obs_u8))

def decode(model, latent):
    """Standardized latents (..., GLAT, GLAT, TOK) -> uint8 frames (..., H, W, 3)."""
    return _decode(model.dec, model.df.zmean.value, model.df.zstd.value, latent)

def _flow_one(apply, seq, taus, aseq, s, rng, n_steps=STEPS):
    """Denoise slot s in n_steps Euler steps; the past (taus > 0.5) is fed corrupted to the signal levels in taus."""
    B = seq.shape[0]
    kc, kz = jax.random.split(rng)
    t5 = taus[..., None, None, None]
    seq = jnp.where(t5 > 0.5, t5 * seq + (1.0 - t5) * jax.random.normal(kc, seq.shape), seq)
    dt = 1.0 / n_steps

    def step(z, tau_s):
        x1 = apply(seq.at[:, s].set(z), taus.at[:, s].set(jnp.full((B,), tau_s)), aseq)[:, s]
        return z + dt * (x1 - z) / jnp.maximum(1.0 - tau_s, 1e-4), None
    return jax.lax.scan(step, jax.random.normal(kz, (B,) + seq.shape[-3:]), jnp.arange(n_steps) * dt)[0]

def _rollout_latents(apply, ctx, ctx_a, acts, rng, steps=STEPS, win=WIN):
    """Free run from context latents ctx (B, n0, ...) with incoming actions ctx_a (B, n0): one latent per action in
    acts (T, B) -> (T, B, ...); the window slides once full; real context frames are fed at signal 1.0, generated
    ones at CTX_SIGNAL."""
    B, n0 = ctx.shape[:2]
    seq = jnp.zeros((B, win) + ctx.shape[-3:]).at[:, :n0].set(ctx)
    aseq = jnp.zeros((B, win), jnp.int32).at[:, :n0].set(ctx_a)

    def step(carry, inputs):
        seq, aseq, n, r = carry
        action, key = inputs
        slide = lambda sa: tuple(jnp.concatenate([v[:, 1:], v[:, -1:]], 1) for v in sa)
        seq, aseq = jax.lax.cond(n == win, slide, lambda sa: sa, (seq, aseq))
        r = jnp.where(n == win, jnp.maximum(r - 1, 0), r)
        n = jnp.minimum(n, win - 1)
        aseq = aseq.at[:, n].set(action)
        taus = jnp.broadcast_to(jnp.where(jnp.arange(win) < n, jnp.where(jnp.arange(win) < r, 1.0, CTX_SIGNAL), 0.0), (B, win))
        z = _flow_one(apply, seq, taus, aseq, n, key, steps)
        return (seq.at[:, n].set(z), aseq, n + 1, r), z
    return jax.lax.scan(step, (seq, aseq, jnp.asarray(n0), jnp.asarray(n0)), (acts, jax.random.split(rng, acts.shape[0])))[1]

def _read_heads(df, latents, actions, key):
    """Rewards (B, L) for arriving at each frame and the probability (B, L) that the transition into it ended the
    episode, read from a clean latent sequence (L <= WIN, padded to WIN for the kernels) fed at the context signal level."""
    L = latents.shape[1]
    latents, actions = jnp.pad(latents, ((0, 0), (0, WIN - L)) + ((0, 0),) * 3), jnp.pad(actions, ((0, 0), (0, WIN - L)))
    x = CTX_SIGNAL * latents + (1.0 - CTX_SIGNAL) * jax.random.normal(key, latents.shape)
    s = df.forward(x, jnp.full(latents.shape[:2], CTX_SIGNAL), actions)[1][:, :L]
    rewards = _symexp(jnp.sum(jax.nn.softmax(df.reward_logits(s), -1) * jnp.linspace(-RLIM, RLIM, RBINS), -1))
    p = jax.nn.sigmoid(df.done_logits(s[:, :-1], actions[:, 1:L]))
    return rewards, jnp.concatenate([jnp.zeros_like(p[:, :1]), p], 1)

def imagine(model, obs_u8, ctx_actions, actions, rng, steps=STEPS):
    """Real context frames (B, n0, H, W, 3), their incoming actions (B, n0), actions (B, T) -> frames (B, T, H, W, 3),
    the reward for arriving at each (B, T) and the probability that the transition into it ended the episode (B, T)."""
    ctx = encode(model, jnp.asarray(obs_u8))
    ca, fa = jnp.asarray(ctx_actions, jnp.int32), jnp.asarray(actions, jnp.int32)
    gdef, state = nnx.split(model.df)

    @jax.jit
    def run(state, ctx, ca, fa, key):
        kr, kz = jax.random.split(key)
        df = nnx.merge(gdef, state)
        zt = _rollout_latents(df, ctx, ca, fa.T, kr, steps, model.win)
        heads = _read_heads(df, jnp.concatenate([ctx, zt.transpose(1, 0, 2, 3, 4)], 1), jnp.concatenate([ca, fa], 1), kz)
        return zt, jax.tree.map(lambda v: v[:, ctx.shape[1]:], heads)
    zt, (rewards, dones) = run(state, ctx, ca, fa, rng)
    return decode(model, zt).transpose(1, 0, 2, 3, 4), rewards, dones

def predict_heads(model, obs_u8, actions, rng):
    """Teacher-forced rewards and termination probabilities (B, L) for real frames (B, L, H, W, 3), actions (B, L)."""
    gdef, state = nnx.split(model.df)
    read = jax.jit(lambda state, z, a, key: _read_heads(nnx.merge(gdef, state), z, a, key))
    return read(state, encode(model, jnp.asarray(obs_u8)), jnp.asarray(actions, jnp.int32), rng)

def _orthogonalize(g, steps=5, coeffs=(3.4445, -4.775, 2.0315)):
    """Newton-Schulz orthogonalization of a stack of matrices (n, r, c)."""
    a, b, c = coeffs
    x = g / (jnp.linalg.norm(g, axis=(-2, -1), keepdims=True) + 1e-7)
    tall = x.shape[-2] > x.shape[-1]
    x = jnp.swapaxes(x, -2, -1) if tall else x
    for _ in range(steps):
        aa = x @ jnp.swapaxes(x, -2, -1)
        x = a * x + (b * aa + c * (aa @ aa)) @ x
    return jnp.swapaxes(x, -2, -1) if tall else x

def _muon(sched, weight_decay, beta=0.95, rms=0.2):
    """Muon with Nesterov momentum: conv kernels flattened to (taps * in, out), same-shape kernels orthogonalized as
    one stack; update RMS matched to rms."""
    def update(grads, state, params):
        mu, count = state
        mu = jax.tree.map(lambda m, g: beta * m + g, mu, grads)
        leaves, tdef = jax.tree.flatten(jax.tree.map(lambda m, g: g + beta * m, mu, grads))
        flat = [l.reshape((-1, l.shape[-1])) for l in leaves]
        groups = {shape: [i for i, l in enumerate(flat) if l.shape == shape] for shape in {l.shape for l in flat}}
        out = list(flat)
        for idx in groups.values():
            for j, i in enumerate(idx):
                out[i] = _orthogonalize(jnp.stack([flat[k] for k in idx]))[j]
        new = [-sched(count) * (rms * max(o.shape) ** 0.5 * o.reshape(p.shape) + weight_decay * p) for o, p in zip(out, jax.tree.leaves(params))]
        return jax.tree.unflatten(tdef, new), (mu, count + 1)
    return optax.GradientTransformation(lambda params: (jax.tree.map(jnp.zeros_like, params), jnp.zeros((), jnp.int32)), update)

def _optimizer(lr, steps, decay, warm, weight_decay):
    """Muon on the matrix and conv kernels, AdamW on embeddings, biases and norms; warmup, constant, linear decay."""
    dec = int(steps * decay)
    sched = optax.join_schedules([optax.linear_schedule(0.0, lr, warm), optax.constant_schedule(lr),
                                  optax.linear_schedule(lr, 0.0, max(dec, 1))], [warm, steps - dec])
    label = lambda p, x: "muon" if x.ndim in (2, 4) and "aemb" not in jax.tree_util.keystr(p) else "adam"
    labels = lambda params: jax.tree_util.tree_map_with_path(label, params)
    opts = {"muon": _muon(sched, weight_decay), "adam": optax.adamw(sched, weight_decay=weight_decay)}
    return optax.chain(optax.clip_by_global_norm(1.0), optax.partition(opts, labels))

def _train_tokenizer(tok, frames, cfg):
    """Batches of thirds: clean frames, half the tiles masked to grey, Gaussian noise up to 0.2; clean targets."""
    warm = min(300, max(1, cfg.tok_steps // 10))
    opt = nnx.Optimizer(tok, _optimizer(cfg.tok_lr, cfg.tok_steps, cfg.decay, warm, 0.0), wrt=nnx.Param)
    n = cfg.tok_batch // 3

    @nnx.jit
    def run(tok, opt, frames, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def loop(carry, key):
            tok, opt = carry
            kr, km, kn, ke = jax.random.split(key, 4)
            x = _to01(frames[jax.random.randint(kr, (cfg.tok_batch,), 0, frames.shape[0])])
            xc, xm, xd = x[:n], x[n:2 * n], x[2 * n:]
            m = (jax.random.uniform(km, (n, GRID, GRID, 1)) > 0.5).astype(jnp.float32)
            m = jnp.repeat(jnp.repeat(m, TILE, 1), TILE, 2)
            sig = jax.random.uniform(kn, (xd.shape[0], 1, 1, 1)) * 0.2
            inp = jnp.concatenate([xc, xm * m + 0.5 * (1 - m), jnp.clip(xd + sig * jax.random.normal(ke, xd.shape), 0, 1)], 0)
            loss, grads = nnx.value_and_grad(lambda tok: jnp.mean((tok.dec(tok.enc(inp)) - x) ** 2))(tok)
            opt.update(tok, grads)
            return (tok, opt), loss
        return loop((tok, opt), keys)[1]
    return run(tok, opt, frames, jax.random.split(jax.random.PRNGKey(cfg.seed + 1), cfg.tok_steps))

def _encode_all(enc, frames):
    gdef, state = nnx.split(enc)
    return jax.lax.map(lambda f: nnx.merge(gdef, state)(_to01(f)), frames)

def _layout(actions, dones, length):
    """Incoming action and reset flag per frame, and the (t, e) of every episode start that fits a full window."""
    T, E = actions.shape[0] + 1, actions.shape[1]
    incoming = jnp.concatenate([jnp.zeros((1, E), jnp.int32), actions], 0)
    resets = jnp.concatenate([jnp.ones((1, E), jnp.float32), dones], 0)
    rt, re = np.nonzero(np.asarray(resets[:T - length + 1]) > 0.5)
    return incoming, resets, jnp.asarray(rt), jnp.asarray(re)

def _train_wm(df, Z, A, Dd, R, cfg):
    """Windows drawn uniformly, from the recent-loss table (hardmine) or at episode starts (startsamp); per-frame
    signal levels; the loss weight ramps with the signal. The heads train on the same forward pass, their gradient
    reaches the trunk."""
    warm = min(100, max(1, cfg.wm_steps // 10))
    opt = nnx.Optimizer(df, _optimizer(cfg.lr, cfg.wm_steps, cfg.decay, warm, 0.01), wrt=nnx.Param)
    T, E = Z.shape[:2]
    L, B = cfg.rollout, cfg.wm_batch
    A, Dd, rst_t, rst_e = _layout(A, Dd, L)
    R = jnp.concatenate([jnp.zeros((1, E), jnp.float32), R], 0)
    ltab, offz = jnp.full((T - L + 1, E), 10.0), jnp.arange(L)

    @nnx.jit
    def run(df, opt, ltab, Z, A, Dd, R, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def loop(carry, key):
            df, opt, ltab = carry
            ke, kt, ktau, keps, kp, kq, krj, krm = jax.random.split(key, 8)
            e = jax.random.randint(ke, (B,), 0, E)
            t = jax.random.randint(kt, (B,), 0, T - L + 1)
            cdf = jnp.cumsum(ltab.reshape(-1))
            jf = jnp.minimum(jnp.searchsorted(cdf, jax.random.uniform(kp, (B,)) * cdf[-1], method="scan_unrolled"), cdf.shape[0] - 1)
            hm = jax.random.bernoulli(kq, cfg.hardmine, (B,))
            t, e = jnp.where(hm, jf // E, t), jnp.where(hm, jf % E, e)
            jr = jax.random.randint(krj, (B,), 0, len(rst_t))
            sm = jax.random.bernoulli(krm, cfg.startsamp, (B,))
            t, e = jnp.where(sm, rst_t[jr], t), jnp.where(sm, rst_e[jr], e)
            zi = t[:, None] + offz
            ei = jnp.broadcast_to(e[:, None], zi.shape)
            z, a, dwin, r = Z[zi, ei].astype(jnp.float32), A[zi, ei], Dd[zi, ei], R[zi, ei]
            a = a.at[:, 0].set(jnp.where(dwin[:, 0] > 0.5, 0, a[:, 0]))
            r = r.at[:, 0].set(jnp.where(dwin[:, 0] > 0.5, 0.0, r[:, 0]))
            valid = jnp.cumprod(1.0 - dwin.at[:, 0].set(0.0), axis=1)
            tau = jax.random.uniform(ktau, (B, L))
            t5 = tau[:, :, None, None, None]
            x_tau = (1.0 - t5) * jax.random.normal(keps, z.shape) + t5 * z

            def loss_fn(df):
                x1, s = df.forward(x_tau, tau, a)
                ramp = 0.9 * tau + 0.1
                err = ramp * jnp.mean((x1 - z) ** 2, axis=(2, 3, 4))
                ce = -jnp.sum(_two_hot(_symlog(r)) * jax.nn.log_softmax(df.reward_logits(s), -1), -1)
                dl = optax.sigmoid_binary_cross_entropy(df.done_logits(s[:, :-1], a[:, 1:]), dwin[:, 1:])
                loss = (jnp.sum(err * valid) + cfg.reward_weight * jnp.sum(ramp * ce * valid)) / (jnp.sum(valid) + 1e-6)
                loss = loss + cfg.done_weight * jnp.sum(ramp[:, :-1] * dl * valid[:, :-1]) / (jnp.sum(valid[:, :-1]) + 1e-6)
                return loss, jnp.sum(err * valid, 1) / (jnp.sum(valid, 1) + 1e-6)
            (loss, perwin), grads = nnx.value_and_grad(loss_fn, has_aux=True)(df)
            opt.update(df, grads)
            ltab = ltab.at[t, e].set(0.5 * ltab[t, e] + 0.5 * perwin, unique_indices=True)
            return (df, opt, ltab), loss
        return loop((df, opt, ltab), keys)[1]
    return run(df, opt, ltab, Z, A, Dd, R, jax.random.split(jax.random.PRNGKey(cfg.seed + 2), cfg.wm_steps))

def train(data_npz, out, cfg):
    t0 = time.time()
    data = np.load(data_npz)
    num_actions = json.loads(str(data["meta"])).get("num_actions", A_N)
    A, Dd, R = jnp.asarray(data["actions"], jnp.int32), jnp.asarray(data["dones"], jnp.float32), jnp.asarray(data["rewards"], jnp.float32)
    if A.size > cfg.data_budget:
        raise ValueError(f"the training data budget is {cfg.data_budget} transitions")
    frames = jnp.asarray(data["frames"])
    with open(data_npz, "rb") as fh:
        data_sha = hashlib.file_digest(fh, "sha256").hexdigest()
    print(f"[data] frames {frames.shape} in {time.time()-t0:.1f}s", flush=True)
    tok = Tok(nnx.Rngs(cfg.seed))
    t1 = time.time()
    tl = _train_tokenizer(tok, frames.reshape((-1, H, W, 3)), cfg)
    tl.block_until_ready()
    print(f"[tok] {cfg.tok_steps} steps {time.time()-t1:.1f}s  loss {float(tl[0]):.4f}->{float(tl[-1]):.5f}", flush=True)
    df = DF(nnx.Rngs(cfg.seed + 100), num_actions, max(LMAX, cfg.rollout))
    t2 = time.time()
    Z = _encode_all(tok.enc, frames)
    flat = Z.reshape((-1, Z.shape[-1]))
    df.zmean[...], df.zstd[...] = flat.mean(0), flat.std(0)
    Z = (Z - df.zmean.value) / df.zstd.value
    print(f"[cache] latents {Z.shape} {time.time()-t2:.1f}s  raw std {float(flat.std()):.3f}", flush=True)
    del frames, flat
    t3 = time.time()
    wl = _train_wm(df, Z, A, Dd, R, cfg)
    wl.block_until_ready()
    q = [0, len(wl) // 4, len(wl) // 2, 3 * len(wl) // 4, -1]
    print(f"[wm] {cfg.wm_steps} steps {time.time()-t3:.1f}s  loss " + "  ".join(f"{float(wl[i]):.4f}" for i in q), flush=True)
    model = ModelState(enc=tok.enc, dec=tok.dec, df=df, win=cfg.rollout)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.parent / "losses.npz", tok=np.asarray(tl), wm=np.asarray(wl))
    meta = {"cfg": asdict(cfg) | {"num_actions": num_actions}, "train_s": round(time.time() - t0, 1),
            "tokenizer_train_s": round(t2 - t1, 1), "data_sha256": data_sha,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    arrs = {f"{k}.{i}": np.asarray(jax.device_get(leaf)) for k in ("enc", "dec", "df")
            for i, leaf in enumerate(jax.tree.leaves(nnx.split(getattr(model, k))[1]))}
    np.savez(out, meta=json.dumps(meta), **arrs)
    print(f"[done] total {time.time()-t0:.1f}s -> {out}", flush=True)
    return model

def load_meta(path): return json.loads(str(np.load(path)["meta"]))

def load_model(path):
    z = np.load(path)
    cfg = json.loads(str(z["meta"]))["cfg"]
    mods = {"enc": Encoder(nnx.Rngs(0)), "dec": Decoder(nnx.Rngs(0)), "df": DF(nnx.Rngs(0), cfg["num_actions"], max(LMAX, cfg["rollout"]))}
    out = {}
    for k, mod in mods.items():
        gdef, ref = nnx.split(mod)
        leaves = [jnp.asarray(z[f"{k}.{i}"]) for i in range(len(jax.tree.leaves(ref)))]
        out[k] = nnx.merge(gdef, jax.tree.unflatten(jax.tree.structure(ref), leaves))
    return ModelState(out["enc"], out["dec"], out["df"], cfg["rollout"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/train.npz"))
    p.add_argument("--out", type=Path, default=Path("runs/base/model.npz"))
    for f in fields(Config):
        p.add_argument(f"--{f.name.replace('_', '-')}", type=f.type, default=f.default)
    a = p.parse_args()
    train(a.data, a.out, Config(**{f.name: getattr(a, f.name) for f in fields(Config)}))

if __name__ == "__main__":
    main()

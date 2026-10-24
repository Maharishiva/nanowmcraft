"""train.py: iteration 3 of the tinywmcraft world model. A tokenizer maps 63x63 Craftax frames to 4x4x32 latents: each
7x7 tile is a token, transformer blocks mix the 9x9 tiles, a stride-2 conv and residual convs reach 4x4. A transformer
with spatial attention in frames, causal attention across them and adaLN-zero on the action and each frame's noise
level (diffusion forcing) predicts the next latent by flow matching, with reward and episode-end heads. Also: the
tokenizer learns from masked and noised frames too; a shortcut loss (told the step size, one Euler step learns to land
where two half steps land); the heads' gradient into the trunk is cut to a tenth in the second half; half the windows
start a game; Muon; Pallas kernels (kernels.py). Trains in ten minutes on one A100.

    python train.py --data data/train.npz --out runs/base/model.npz --seed 0
"""
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")  # cuBLAS matmuls compile much faster
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import kernels

jax.config.update("jax_default_matmul_precision", "tensorfloat32")
H = W = 63
TILE, GRID, LAT, TOK = 7, 9, 4, 32  # 7x7 pixel tiles on a 9x9 grid; latents are 4x4 tokens of 32 channels
CH, TBLOCKS, RBLOCKS = 128, 6, 4  # tokenizer width, transformer blocks at 9x9, residual blocks at 4x4
D, DEPTH, HEADS, WIN = 256, 6, 8, 32  # transformer width, blocks, heads, frames per window
ACTIONS, STEPS, CTX_SIGNAL, BUDGET = 17, 8, 0.9, 500_000
RBINS, RLIM = 255, 20.0  # reward head: two-hot over symlog(reward) in [-RLIM, RLIM]
Linear, Conv = partial(nnx.Linear, dtype=jnp.bfloat16), partial(nnx.Conv, dtype=jnp.bfloat16)
Norm, zeros = partial(nnx.RMSNorm, dtype=jnp.float32), nnx.initializers.zeros

@dataclass(frozen=True)
class Config:
    seed: int = 0
    tok_steps: int = 9000
    tok_batch: int = 256
    wm_steps: int = 19000
    wm_batch: int = 64
    lr: float = 9e-4

def noised(z, tau, key): return tau * z + (1 - tau) * jax.random.normal(key, z.shape)

def scaled_grad(x, s): return jax.lax.stop_gradient(x) + s * (x - jax.lax.stop_gradient(x))

class ResBlock(nnx.Module):
    def __init__(self, rngs):
        self.n1, self.c1 = Norm(CH, rngs=rngs), Conv(CH, CH, (3, 3), rngs=rngs)
        self.n2, self.c2 = Norm(CH, rngs=rngs), Conv(CH, CH, (3, 3), rngs=rngs)

    def __call__(self, h):
        return h + self.c2(nnx.gelu(self.n2(self.c1(nnx.gelu(self.n1(h))))))

class TileBlock(nnx.Module):
    """Pre-norm transformer block over the 9x9 tile tokens of a frame."""
    def __init__(self, rngs):
        self.n1, self.qkv, self.o = Norm(CH, rngs=rngs), Linear(CH, 3 * CH, rngs=rngs), Linear(CH, CH, rngs=rngs)
        self.n2 = Norm(CH, rngs=rngs)
        self.mlp = nnx.Sequential(Linear(CH, 4 * CH, rngs=rngs), nnx.gelu, Linear(4 * CH, CH, rngs=rngs))
        self.pos = nnx.Param(0.02 * jax.random.normal(rngs.params(), (GRID, GRID, CH)))

    def __call__(self, h):
        h = h + self.pos[...].astype(h.dtype)
        qkv = self.qkv(self.n1(h)).reshape(-1, 1, GRID * GRID, 3 * CH)
        h = h + self.o(kernels.attention(qkv, HEADS, False).reshape(h.shape))
        return h + self.mlp(self.n2(h))

class Tokenizer(nnx.Module):
    """uint8 frames (..., 63, 63, 3) <-> standardized latents (..., 4, 4, TOK); the decoder mirrors the encoder and
    maps each 9x9 token back to its tile."""
    def __init__(self, rngs):
        tiles, res = lambda: [TileBlock(rngs) for _ in range(TBLOCKS)], lambda: [ResBlock(rngs) for _ in range(RBLOCKS)]
        down = Conv(CH, CH, (3, 3), strides=2, padding="VALID", rngs=rngs)
        up = nnx.ConvTranspose(CH, CH, (3, 3), strides=2, padding="VALID", dtype=jnp.bfloat16, rngs=rngs)
        self.enc = nnx.Sequential(Linear(3 * TILE**2, 64, rngs=rngs), nnx.gelu, Linear(64, CH, rngs=rngs), *tiles(),
                                  down, nnx.gelu, *res(), Linear(CH, TOK, rngs=rngs))
        self.dec = nnx.Sequential(Linear(TOK, CH, rngs=rngs), nnx.gelu, *res(), up, nnx.gelu, *tiles())
        self.tile = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (TILE, TILE, CH, 3)))  # a 7x7 conv
        self.bias = nnx.Param(jnp.zeros(3))
        self.mean, self.std = nnx.Variable(jnp.zeros(TOK)), nnx.Variable(jnp.ones(TOK))

    def encode(self, frames):
        x = frames.reshape(-1, GRID, TILE, GRID, TILE, 3).swapaxes(2, 3).reshape(-1, GRID, GRID, 3 * TILE**2)
        z = (self.enc(x / 255.0).astype(jnp.float32) - self.mean[...]) / self.std[...]
        return z.reshape(frames.shape[:-3] + z.shape[1:])

    def decode(self, z):
        h = self.dec((z * self.std[...] + self.mean[...]).reshape(-1, LAT, LAT, TOK))
        x = h @ self.tile[...].astype(h.dtype).transpose(2, 0, 1, 3).reshape(CH, -1) + jnp.tile(self.bias[...], TILE**2)
        x = jax.nn.sigmoid(x.astype(jnp.float32))
        return x.reshape(-1, GRID, GRID, TILE, TILE, 3).swapaxes(2, 3).reshape(z.shape[:-3] + (H, W, 3))

class Attention(nnx.Module):
    """Attention over the tokens of each frame of x (B, L, N, D), or causally over its frames; also returns the keys
    and values. Given those of a whole window as a cache, x is the one frame at slot n."""
    def __init__(self, rngs):
        self.qkv, self.out = Linear(D, 3 * D, rngs=rngs), Linear(D, D, rngs=rngs)

    def __call__(self, x, temporal=False, cache=None, n=0):
        qkv = self.qkv(x)
        if cache is None:
            return self.out(kernels.attention(qkv, HEADS, temporal)), qkv[..., D:]
        kv = jax.lax.dynamic_update_slice_in_dim(cache, qkv[..., D:], n, axis=1)
        q, (k, v) = qkv[:, 0, :, :D].reshape(x.shape[0], -1, HEADS, D // HEADS), jnp.split(
            kv.reshape(kv.shape[:3] + (2 * HEADS, -1)), 2, axis=3)
        s = jnp.einsum("bnhd,blnhd->bnhl", q, k, preferred_element_type=jnp.float32) * (D // HEADS) ** -0.5
        p = jax.nn.softmax(jnp.where(jnp.arange(WIN) <= n, s, -1e30), -1).astype(v.dtype)
        return self.out(jnp.einsum("bnhl,blnhd->bnhd", p, v).reshape(x.shape)), kv

class Block(nnx.Module):
    """adaLN-zero: spatial attention within each frame, causal attention across frames, MLP."""
    def __init__(self, rngs):
        self.mod = Linear(D, 9 * D, kernel_init=zeros, rngs=rngs)
        self.space, self.time = Attention(rngs), Attention(rngs)
        self.mlp = nnx.Sequential(Linear(D, 2 * D, rngs=rngs), nnx.gelu, Linear(2 * D, D, rngs=rngs))

    def __call__(self, h, cond, cache=None, n=0):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = jnp.split(self.mod(cond)[:, :, None], 9, axis=-1)
        h, x = kernels.residual_norm(h, jnp.zeros_like(g1), h, s1, b1)  # the first norm, with nothing to add
        h, x = kernels.residual_norm(h, g1, self.space(x)[0], s2, b2)
        y, kv = self.time(x, True, cache, n)
        h, x = kernels.residual_norm(h, g2, y, s3, b3)
        return h + g3 * self.mlp(x), kv

class WorldModel(nnx.Module):
    """Clean latents (B, L, 4, 4, TOK) from latents noised to levels tau (B, L) under the action into each frame
    (B, L), from window slot n on, told the Euler step size delta; also each frame's mean token and the temporal keys
    and values of every block."""
    def __init__(self, rngs):
        self.inp, self.level = Linear(TOK, D, rngs=rngs), Linear(D, D, rngs=rngs)
        self.spos = nnx.Param(0.02 * jax.random.normal(rngs.params(), (LAT * LAT, D)))
        self.tpos = nnx.Param(0.02 * jax.random.normal(rngs.params(), (WIN, 1, D)))
        self.act, self.step = nnx.Embed(ACTIONS, D, rngs=rngs), nnx.Param(jnp.zeros(D))
        self.blocks = nnx.List([Block(rngs) for _ in range(DEPTH)])
        self.out = Linear(D, TOK, kernel_init=zeros, rngs=rngs)
        self.reward = nnx.Sequential(Linear(D, D, rngs=rngs), nnx.gelu, Linear(D, RBINS, kernel_init=zeros, rngs=rngs))
        self.done = nnx.Sequential(Linear(2 * D, D, rngs=rngs), nnx.gelu, Linear(D, 1, kernel_init=zeros, rngs=rngs))

    def embed(self, a):  # a one-hot matmul: the gather's gradient would be a slow scatter-add
        return (jax.nn.one_hot(a, ACTIONS) @ self.act.embedding[...]).astype(jnp.bfloat16)

    def __call__(self, x, tau, a, cache=None, n=0, delta=0.0):
        B, L = a.shape
        freqs = tau[..., None] * jnp.exp(jnp.linspace(0.0, jnp.log(1000.0), D // 2))
        cond = self.level(jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)], -1)) + self.embed(a)
        cond = (cond + STEPS * delta * self.step[...]).astype(jnp.bfloat16)
        pos = jax.lax.dynamic_slice_in_dim(self.tpos[...], n, L) + self.spos[...]
        h, kvs = self.inp(x.reshape(B, L, LAT * LAT, TOK)) + pos.astype(jnp.bfloat16) + self.embed(a)[:, :, None], []
        for block, c in zip(self.blocks, cache or [None] * DEPTH):
            h, kv = block(h, cond, c, n)
            kvs.append(kv)
        h = h * jax.lax.rsqrt(jnp.mean(jnp.square(h.astype(jnp.float32)), -1, keepdims=True) + 1e-6)
        return self.out(h).reshape(x.shape).astype(jnp.float32), h.mean(2), kvs

    def heads(self, s, a, grad=1.0):
        """Reward logits from each frame's mean token s, episode-end logits of each next transition."""
        done = self.done(jnp.concatenate([s[:, :-1], scaled_grad(self.embed(a[:, 1:]), grad)], -1))[..., 0]
        return self.reward(s).astype(jnp.float32), done.astype(jnp.float32)

class Model(nnx.Module):
    def __init__(self, rngs):
        self.tok, self.wm = Tokenizer(rngs), WorldModel(rngs)

def sample(wm, z, a, n, real, key):
    """The latent at slot n of the window z (B, WIN, ...): the past (the first `real` frames real and clean, the rest
    generated, noised to CTX_SIGNAL) runs once to cache its keys and values, then STEPS Euler steps carry slot n
    alone from noise to a clean latent."""
    kc, kx = jax.random.split(key)
    tau = jnp.broadcast_to(jnp.where(jnp.arange(WIN) < real, 1.0, CTX_SIGNAL), a.shape)
    cache = wm(noised(z, tau[..., None, None, None], kc), tau, a)[2]
    an = jax.lax.dynamic_slice_in_dim(a, n, 1, axis=1)

    def euler(x, t):
        x1 = wm(x[:, None], jnp.full(an.shape, t), an, cache, n, 1 / STEPS)[0][:, 0]
        return x + (x1 - x) / (STEPS * (1 - t)), None
    return jax.lax.scan(euler, jax.random.normal(kx, z[:, 0].shape), jnp.arange(STEPS) / STEPS)[0]

@jax.jit
def start(model, frames, actions):
    """Real frames uint8 (B, n0, 63, 63, 3) and the action into each (B, n0) -> rollout state."""
    n = min(frames.shape[1], WIN)
    z = jnp.zeros((frames.shape[0], WIN, LAT, LAT, TOK)).at[:, :n].set(model.tok.encode(frames[:, -n:]))
    return z, jnp.zeros(z.shape[:2], jnp.int32).at[:, :n].set(actions[:, -n:]), jnp.int32(n), jnp.int32(n)

@jax.jit
def step(model, state, action, key):
    """The frame (B, 63, 63, 3) uint8 that follows the actions (B,); the window slides once full."""
    z, a, n, real = state
    z, a = (jnp.where(n == WIN, jnp.roll(v, -1, axis=1), v) for v in (z, a))
    real = jnp.where(n == WIN, jnp.maximum(real - 1, 0), real)
    n = jnp.minimum(n, WIN - 1)
    a = a.at[:, n].set(action)
    x = sample(model.wm, z, a, n, real, key)
    return jnp.round(model.tok.decode(x) * 255).astype(jnp.uint8), (z.at[:, n].set(x), a, n + 1, real)

@jax.jit
def heads(model, frames, actions, key):
    """Teacher-forced reward and episode-end probability (B, L) of the step into each frame (L <= WIN, padded to WIN
    for the kernels)."""
    L, pad = frames.shape[1], ((0, 0), (0, WIN - frames.shape[1]))
    x, a = noised(jnp.pad(model.tok.encode(frames), pad + ((0, 0),) * 3), CTX_SIGNAL, key), jnp.pad(actions, pad)
    reward, done = model.wm.heads(model.wm(x, jnp.full(a.shape, CTX_SIGNAL), a)[1][:, :L], a[:, :L])
    reward = jax.nn.softmax(reward) @ jnp.linspace(-RLIM, RLIM, RBINS)
    return jnp.sign(reward) * jnp.expm1(jnp.abs(reward)), jnp.pad(jax.nn.sigmoid(done), ((0, 0), (1, 0)))

def tok_loss(tok, frames, key, step, batch):
    """A third of the batch clean, a third with half the tiles grey, a third with Gaussian noise up to 0.2; clean
    targets."""
    ki, km, ks, kn = jax.random.split(key, 4)
    x, n = frames[jax.random.randint(ki, (batch,), 0, frames.shape[0])] / 255.0, batch // 3
    keep = jnp.repeat(jnp.repeat(jax.random.uniform(km, (n, GRID, GRID, 1)) > 0.5, TILE, 1), TILE, 2)
    noise = 0.2 * jax.random.uniform(ks, (batch - 2 * n, 1, 1, 1)) * jax.random.normal(kn, x[2 * n:].shape)
    inp = jnp.concatenate([x[:n], jnp.where(keep, x[n:2 * n], 0.5), jnp.clip(x[2 * n:] + noise, 0, 1)])
    return jnp.mean((tok.decode(tok.encode(inp * 255.0)) - x) ** 2)

def shortcut_loss(wm, z, a, valid, key):
    """On 8 windows at a random slot after a past at CTX_SIGNAL: one Euler step told its size 1/STEPS learns to land
    where two half steps of the plain model land."""
    z, a, valid = z[:8], a[:8], valid[:8]
    ks, kt, kn = jax.random.split(key, 3)
    n, tau = jax.random.randint(ks, (), 0, WIN), jax.random.randint(kt, (8, 1), 0, STEPS) / STEPS
    eps, t5, h = jax.random.normal(kn, z.shape), tau[..., None, None, None], 1 / (2 * STEPS)
    cache = jax.lax.stop_gradient(wm(CTX_SIGNAL * z + (1 - CTX_SIGNAL) * eps, jnp.full(a.shape, CTX_SIGNAL), a)[2])
    x, an = t5 * z[:, n][:, None] + (1 - t5) * eps[:, n][:, None], a[:, n][:, None]
    flow = lambda x, t: (jax.lax.stop_gradient(wm(x, t, an, cache, n)[0]) - x) / (1 - t[..., None, None, None])
    v1 = flow(x, tau)
    target = x + (1 - t5) * (v1 + flow(x + h * v1, tau + h)) / 2
    err = (0.1 + 0.9 * tau[:, 0]) * jnp.mean((wm(x, tau, an, cache, n, 1 / STEPS)[0] - target) ** 2, (1, 2, 3, 4))
    return jnp.sum(err * valid[:, n]) / (jnp.sum(valid[:, n]) + 1e-6)

def wm_loss(wm, data, key, step, batch, steps):
    """Flow matching on windows at independent per-frame signal levels, weighted up with the signal, plus the heads on
    the same pass; half the windows start a game; frames after an episode end are masked out; a truncation is no
    death. Every 4th step after the first quarter adds the shortcut loss; the heads' gradient into the trunk is cut
    to a tenth in the second half."""
    Z, A, R, Dn, Tr, starts = data
    kt, ks, kn, kd, kp, kq, kc = jax.random.split(key, 7)
    rand = jax.random.randint(kt, (batch, 2), 0, jnp.array([Z.shape[0] - WIN + 1, Z.shape[1]]))
    first = starts[jax.random.randint(kp, (batch,), 0, len(starts))]
    t, e = jnp.where(jax.random.bernoulli(kq, 0.5, (batch, 1)), first, rand).T
    i = t[:, None] + jnp.arange(WIN)
    z, a, r, d, tr = Z[i, e[:, None]], A[i, e[:, None]], R[i, e[:, None]], Dn[i, e[:, None]], Tr[i, e[:, None]]
    a, r = (v.at[:, 0].set(jnp.where(d[:, 0] > 0, 0, v[:, 0])) for v in (a, r))
    valid = jnp.cumprod(1 - d.at[:, 0].set(0), axis=1)
    tau = jax.random.uniform(ks, (batch, WIN))
    pred, s, _ = wm(noised(z, tau[..., None, None, None], kn), tau, a, delta=jax.random.bernoulli(kd) / STEPS)
    grad = jnp.where(step < steps // 2, 1.0, 0.1)
    reward, done = wm.heads(scaled_grad(s, grad), a, grad)
    pos = (jnp.clip(jnp.sign(r) * jnp.log1p(jnp.abs(r)), -RLIM, RLIM) + RLIM) / (2 * RLIM) * (RBINS - 1)
    lo = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0, RBINS - 2)
    hot = lambda i, w: w[..., None] * jax.nn.one_hot(i, RBINS)
    two_hot = hot(lo, lo + 1 - pos) + hot(lo + 1, pos - lo)
    rew = -jnp.sum(jax.nn.log_softmax(reward) * two_hot, -1)
    flow = jnp.mean((pred - z) ** 2, axis=(2, 3, 4))
    end = optax.sigmoid_binary_cross_entropy(done, d[:, 1:] * (1 - tr[:, 1:]))
    w = valid * (0.9 * tau + 0.1)
    loss = (jnp.sum(w * (flow + 0.1 * rew)) + 0.1 * jnp.sum(w[:, :-1] * (1 - tr[:, 1:]) * end)) / valid.sum()
    shortcut = lambda: shortcut_loss(wm, z, a, valid, kc)
    return loss + 0.25 * jax.lax.cond((step >= steps // 4) & (step % 4 == 0), shortcut, lambda: jnp.zeros(()))

def polar(g):
    """Five Polar Express steps in bf16: the nearest semi-orthogonal matrix of each of a stack (n, rows, columns)."""
    x = (g / (1.02 * jnp.linalg.norm(g, axis=(-2, -1), keepdims=True) + 1e-7)).astype(jnp.bfloat16)
    x = x.mT if x.shape[-2] > x.shape[-1] else x
    for a, b, c in ((8.15655, -22.48329, 15.87877), (4.04293, -2.80892, 0.500018), (3.89167, -2.77248, 0.506065),
                    (3.28575, -2.36813, 0.464490), (2.34654, -1.70978, 0.423236)):
        aa = x @ x.mT
        x = a * x + (b * aa + c * aa @ aa) @ x
    return (x.mT if g.shape[-2] > g.shape[-1] else x).astype(jnp.float32)

def muon(lr, wd, beta=0.95):
    """Muon: Nesterov momentum, orthogonalized (conv kernels flattened, same-shape matrices as one stack), RMS 0.2."""
    def update(grads, state, params):
        mu, count = jax.tree.map(lambda m, g: beta * m + g, state[0], grads), state[1]
        flat, tdef = jax.tree.flatten(jax.tree.map(lambda m, g: (g + beta * m).reshape(-1, g.shape[-1]), mu, grads))
        out = list(flat)
        for shape in {f.shape for f in flat}:
            idx = [i for i, f in enumerate(flat) if f.shape == shape]
            for i, o in zip(idx, polar(jnp.stack([flat[i] for i in idx]))):
                out[i] = 0.2 * max(shape) ** 0.5 * o
        new = [-lr(count) * (o.reshape(p.shape) + wd * p) for o, p in zip(out, jax.tree.leaves(params))]
        return jax.tree.unflatten(tdef, new), (mu, count + 1)
    return optax.GradientTransformation(lambda p: (jax.tree.map(jnp.zeros_like, p), jnp.zeros((), jnp.int32)), update)

def optimizer(steps, warm, lr, wd):
    """Muon on the matrices, AdamW on the action embedding, biases and norms; warmup, then linear decay over the last
    fifth."""
    sched = optax.join_schedules([optax.linear_schedule(0.0, lr, warm), optax.constant_schedule(lr),
                                  optax.linear_schedule(lr, 0.0, steps // 5)], [warm, steps - steps // 5])
    label = lambda path, p: "muon" if p.ndim > 1 and "act" not in str(path) else "adam"
    return optax.chain(optax.clip_by_global_norm(1.0), optax.partition(
        {"muon": muon(sched, wd), "adam": optax.adamw(sched, weight_decay=wd)},
        lambda params: jax.tree_util.tree_map_with_path(label, params)))

def fit(model, loss_fn, data, steps, cfg, opt):
    """All optimizer steps on loss_fn(model, data, key, step) as one compiled scan; returns the losses."""
    @nnx.scan(in_axes=(nnx.Carry, None, 0, 0), out_axes=(nnx.Carry, 0))
    def one(carry, data, key, i):
        model, opt = carry
        loss, grads = nnx.value_and_grad(lambda m: loss_fn(m, data, key, i))(model)
        opt.update(model, grads)
        return (model, opt), loss
    keys = jax.random.split(jax.random.key(cfg.seed), steps)
    return nnx.jit(one)((model, nnx.Optimizer(model, opt, wrt=nnx.Param)), data, keys, jnp.arange(steps))[1]

def train(data_path, out, cfg):
    t0, data = time.time(), np.load(data_path)
    assert data["actions"].size <= BUDGET, f"the data budget is {BUDGET} transitions"
    frames = jnp.asarray(data["frames"])
    def incoming(k, v): return jnp.pad(jnp.asarray(data[k], float), ((1, 0), (0, 0)), constant_values=v)
    A, R, Dn = incoming("actions", 0).astype(jnp.int32), incoming("rewards", 0), incoming("dones", 1)
    Tr = incoming("truncated", 0) if "truncated" in data.files else 0 * R  # an end, but no death
    Dn = jnp.maximum(Dn, Tr)
    first = data["starts"] if "starts" in data.files else Dn  # frames that begin a game, else every piece start
    starts = jnp.asarray(np.argwhere(np.asarray(first[:Dn.shape[0] - WIN + 1]) > 0))  # their (t, e)
    model = Model(nnx.Rngs(cfg.seed))
    print(f"[data] frames {frames.shape}, {time.time() - t0:.0f} s", flush=True)
    tok, opt = lambda m, f, k, i: tok_loss(m, f, k, i, cfg.tok_batch), optimizer(cfg.tok_steps, 300, cfg.lr, 0)
    loss = fit(model.tok, tok, frames.reshape(-1, H, W, 3), cfg.tok_steps, cfg, opt)
    print(f"[tokenizer] loss {loss[0]:.4f} -> {loss[-1]:.5f}, {time.time() - t0:.0f} s", flush=True)
    Z = jax.jit(lambda tok, f: jax.lax.map(tok.encode, f))(model.tok, frames)
    model.tok.mean[...], model.tok.std[...] = Z.mean((0, 1, 2, 3)), Z.std((0, 1, 2, 3))
    Z = (Z - model.tok.mean[...]) / model.tok.std[...]
    wm = lambda m, d, k, i: wm_loss(m, d, k, i, cfg.wm_batch, cfg.wm_steps)
    loss = fit(model.wm, wm, (Z, A, R, Dn, Tr, starts), cfg.wm_steps, cfg, optimizer(cfg.wm_steps, 100, cfg.lr, 0.01))
    print(f"[world model] loss {loss[0]:.4f} -> {loss[-1]:.4f}, {time.time() - t0:.0f} s", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, *jax.tree.leaves(model), meta=json.dumps(dict(cfg=asdict(cfg), train_s=round(time.time() - t0, 1))))

def load_meta(path): return json.loads(str(np.load(path)["meta"]))

def load_model(path):
    z, model = np.load(path), Model(nnx.Rngs(0))
    leaves = [jnp.asarray(z[f"arr_{i}"]) for i in range(len(z.files) - 1)]
    return jax.tree.unflatten(jax.tree.structure(model), leaves)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train.npz")
    p.add_argument("--out", default="runs/base/model.npz")
    for f in fields(Config):
        p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = vars(p.parse_args())
    train(args.pop("data"), args.pop("out"), Config(**args))

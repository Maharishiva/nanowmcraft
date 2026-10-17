"""train_tiny.py: the tinywmcraft world model in one file. A conv tokenizer maps 63x63 Craftax frames to
4x4x32 latents; a transformer with spatial attention in frames, causal attention across them and adaLN-zero on
the action and each frame's noise level (diffusion forcing) predicts the next latent by flow matching, with
reward and episode-end heads. Trains in ten minutes on one A100.

    python train_tiny.py --data data/train.npz --out runs/tiny/model.npz --seed 0
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
from optax.contrib import MuonDimensionNumbers

H = W = 63
TILE, GRID, LAT, TOK = 7, 9, 4, 32  # 7x7 pixel tiles on a 9x9 grid; latents are 4x4 tokens of 32 channels
CH, RES = 128, 3  # tokenizer width, residual blocks per resolution
D, DEPTH, HEADS, WIN = 256, 6, 8, 32  # transformer width, blocks, heads, frames per window
ACTIONS, STEPS, CTX_SIGNAL, BUDGET = 17, 8, 0.9, 500_000
RSTEP, RBINS = 0.1, 41  # reward classes -2.0, -1.9, ..., 2.0
Linear, Conv = partial(nnx.Linear, dtype=jnp.bfloat16), partial(nnx.Conv, dtype=jnp.bfloat16)

@dataclass(frozen=True)
class Config:
    seed: int = 0
    tok_steps: int = 5000
    tok_batch: int = 256
    wm_steps: int = 11000
    wm_batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01

def rms(x): return x * jax.lax.rsqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), -1, keepdims=True) + 1e-6)

def noised(z, tau, key): return tau * z + (1 - tau) * jax.random.normal(key, z.shape)

class ResBlock(nnx.Module):
    def __init__(self, rngs):
        self.c1, self.c2 = Conv(CH, CH, (3, 3), rngs=rngs), Conv(CH, CH, (3, 3), rngs=rngs)

    def __call__(self, h):
        return h + self.c2(nnx.gelu(rms(self.c1(nnx.gelu(rms(h))))))

class Tokenizer(nnx.Module):
    """uint8 frames (..., 63, 63, 3) <-> standardized latents (..., 4, 4, TOK): each 7x7 tile is one vector
    of a 9x9 grid, residual convs, a stride-2 conv to 4x4, more residual convs; the decoder mirrors it."""
    def __init__(self, rngs):
        lin, res = partial(Linear, rngs=rngs), lambda: [ResBlock(rngs) for _ in range(RES)]
        down = Conv(CH, CH, (3, 3), strides=2, padding="VALID", rngs=rngs)
        up = nnx.ConvTranspose(CH, CH, (3, 3), strides=2, padding="VALID", dtype=jnp.bfloat16, rngs=rngs)
        self.enc = nnx.Sequential(lin(3 * TILE**2, CH), *res(), down, *res(), lin(CH, TOK))
        self.dec = nnx.Sequential(lin(TOK, CH), *res(), up, *res(), lin(CH, 3 * TILE**2), jax.nn.sigmoid)
        self.mean, self.std = nnx.Variable(jnp.zeros(TOK)), nnx.Variable(jnp.ones(TOK))

    def encode(self, frames):
        x = frames.reshape(-1, GRID, TILE, GRID, TILE, 3).swapaxes(2, 3).reshape(-1, GRID, GRID, 3 * TILE**2)
        z = (self.enc(x / 255.0) - self.mean[...]) / self.std[...]
        return z.reshape(frames.shape[:-3] + z.shape[1:])

    def decode(self, z):
        x = self.dec((z * self.std[...] + self.mean[...]).reshape(-1, LAT, LAT, TOK)).astype(jnp.float32)
        return x.reshape(-1, GRID, GRID, TILE, TILE, 3).swapaxes(2, 3).reshape(z.shape[:-3] + (H, W, 3))

class Attention(nnx.Module):
    """Attention along axis 2 of x (B, S, T, D); also returns the keys and values it used. Given the keys and
    values of a whole window as a cache, x is the one frame at slot n."""
    def __init__(self, rngs):
        self.qkv, self.out = Linear(D, 3 * D, rngs=rngs), Linear(D, D, rngs=rngs)

    def __call__(self, x, causal=False, cache=None, n=0):
        q, k, v = jnp.split(self.qkv(x).reshape(x.shape[:3] + (3 * HEADS, D // HEADS)), 3, axis=3)
        if cache is not None:
            k, v = (jax.lax.dynamic_update_slice_in_dim(c, new, n, axis=2) for c, new in zip(cache, (k, v)))
        mask = None if cache is None else (jnp.arange(WIN) <= n)[None, None, None]
        o = jax.nn.dot_product_attention(*(t.reshape((-1,) + t.shape[2:]) for t in (q, k, v)), mask=mask,
                                         is_causal=causal and cache is None)
        return self.out(o.reshape(x.shape)), (k, v)

class Block(nnx.Module):
    """adaLN-zero: spatial attention within each frame, causal attention across frames, MLP."""
    def __init__(self, rngs):
        self.mod = Linear(D, 9 * D, kernel_init=nnx.initializers.zeros, rngs=rngs)
        self.space, self.time = Attention(rngs), Attention(rngs)
        self.mlp = nnx.Sequential(Linear(D, 2 * D, rngs=rngs), nnx.gelu, Linear(2 * D, D, rngs=rngs))

    def __call__(self, h, cond, cache=None, n=0):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = jnp.split(self.mod(nnx.silu(cond))[:, :, None], 9, axis=-1)
        h = h + g1 * self.space(rms(h) * (1 + s1) + b1)[0]
        y, kv = self.time((rms(h) * (1 + s2) + b2).swapaxes(1, 2), True, cache, n)
        h = h + g2 * y.swapaxes(1, 2)
        return h + g3 * self.mlp(rms(h) * (1 + s3) + b3), kv

class WorldModel(nnx.Module):
    """Clean latents (B, L, 4, 4, TOK) from latents noised to levels tau (B, L) under the action into each
    frame (B, L), from window slot n on; also each frame's mean token and the temporal keys and values."""
    def __init__(self, rngs):
        self.inp, self.level = Linear(TOK, D, rngs=rngs), Linear(D, D, rngs=rngs)
        self.pos = nnx.Param(0.02 * jax.random.normal(rngs.params(), (WIN, LAT * LAT, D)))
        self.act = nnx.Embed(ACTIONS, D, rngs=rngs)
        self.blocks = nnx.List([Block(rngs) for _ in range(DEPTH)])
        self.out = Linear(D, TOK, kernel_init=nnx.initializers.zeros, rngs=rngs)
        self.reward = nnx.Sequential(Linear(D, D, rngs=rngs), nnx.gelu, Linear(D, RBINS, rngs=rngs))
        self.done = nnx.Sequential(Linear(2 * D, D, rngs=rngs), nnx.gelu, Linear(D, 1, rngs=rngs))

    def __call__(self, x, tau, a, cache=None, n=0):
        B, L = a.shape
        freqs = tau[..., None] * jnp.exp(jnp.linspace(0.0, jnp.log(1000.0), D // 2))
        cond = self.level(jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)], -1)) + self.act(a)
        pos = jax.lax.dynamic_slice_in_dim(self.pos[...], n, L)
        h, kvs = self.inp(x.reshape(B, L, LAT * LAT, TOK)) + pos + self.act(a)[:, :, None], []
        for block, c in zip(self.blocks, cache or [None] * DEPTH):
            h, kv = block(h, cond, c, n)
            kvs.append(kv)
        h = rms(h)
        return self.out(h).reshape(x.shape).astype(jnp.float32), h.mean(2), kvs

    def heads(self, s, a):
        """Reward-class logits from each frame's mean token s, episode-end logits of each next transition."""
        done = self.done(jnp.concatenate([s[:, :-1], self.act(a[:, 1:])], -1))[..., 0]
        return self.reward(s).astype(jnp.float32), done.astype(jnp.float32)

class Model(nnx.Module):
    def __init__(self, rngs):
        self.tok, self.wm = Tokenizer(rngs), WorldModel(rngs)

def sample(wm, z, a, n, key):
    """The latent at slot n of the window z (B, WIN, ...): the past is noised to CTX_SIGNAL and its keys and
    values cached, then STEPS Euler steps carry slot n alone from noise to a clean latent."""
    kc, kx = jax.random.split(key)
    cache = wm(noised(z, CTX_SIGNAL, kc), jnp.full(a.shape, CTX_SIGNAL), a)[2]
    an = jax.lax.dynamic_slice_in_dim(a, n, 1, axis=1)

    def euler(x, t):
        x1 = wm(x[:, None], jnp.full(an.shape, t), an, cache, n)[0][:, 0]
        return x + (x1 - x) / (STEPS * (1 - t)), None
    return jax.lax.scan(euler, jax.random.normal(kx, z[:, 0].shape), jnp.arange(STEPS) / STEPS)[0]

@jax.jit
def start(model, frames, actions):
    """Real frames uint8 (B, n0, 63, 63, 3) and the action into each (B, n0) -> rollout state."""
    n = min(frames.shape[1], WIN)
    z = jnp.zeros((frames.shape[0], WIN, LAT, LAT, TOK)).at[:, :n].set(model.tok.encode(frames[:, -n:]))
    return z, jnp.zeros(z.shape[:2], jnp.int32).at[:, :n].set(actions[:, -n:]), jnp.int32(n)

@jax.jit
def step(model, state, action, key):
    """The frame (B, 63, 63, 3) uint8 that follows the actions (B,); the window slides once full."""
    z, a, n = state
    z, a = (jnp.where(n == WIN, jnp.roll(v, -1, axis=1), v) for v in (z, a))
    n = jnp.minimum(n, WIN - 1)
    a = a.at[:, n].set(action)
    x = sample(model.wm, z, a, n, key)
    return jnp.round(model.tok.decode(x) * 255).astype(jnp.uint8), (z.at[:, n].set(x), a, n + 1)

@jax.jit
def heads(model, frames, actions, key):
    """Teacher-forced reward and episode-end probability (B, L) of the step into each frame."""
    x = noised(model.tok.encode(frames), CTX_SIGNAL, key)
    reward, done = model.wm.heads(model.wm(x, jnp.full(actions.shape, CTX_SIGNAL), actions)[1], actions)
    reward = jax.nn.softmax(reward) @ (RSTEP * (jnp.arange(RBINS) - RBINS // 2))
    return reward, jnp.pad(jax.nn.sigmoid(done), ((0, 0), (1, 0)))

def tok_loss(tok, frames, key, batch):
    x = frames[jax.random.randint(key, (batch,), 0, frames.shape[0])]
    return jnp.mean((tok.decode(tok.encode(x)) - x / 255.0) ** 2)

def wm_loss(wm, data, key, batch):
    """Flow matching on random windows at independent per-frame signal levels, weighted up with the signal,
    plus the heads on the same pass; frames after an episode end are masked out; a truncation is no death."""
    Z, A, R, Dn, Tr = data
    kt, ke, ks, kn = jax.random.split(key, 4)
    t = jax.random.randint(kt, (batch, 1), 0, Z.shape[0] - WIN + 1) + jnp.arange(WIN)
    e = jax.random.randint(ke, (batch, 1), 0, Z.shape[1])
    z, a, r, d, tr = Z[t, e], A[t, e], R[t, e], Dn[t, e], Tr[t, e]
    a, r = (v.at[:, 0].set(jnp.where(d[:, 0] > 0, 0, v[:, 0])) for v in (a, r))
    valid = jnp.cumprod(1 - d.at[:, 0].set(0), axis=1)
    tau = jax.random.uniform(ks, (batch, WIN))
    pred, s, _ = wm(noised(z, tau[..., None, None, None], kn), tau, a)
    reward, done = wm.heads(s, a)
    label = jnp.clip(jnp.round(r / RSTEP).astype(jnp.int32) + RBINS // 2, 0, RBINS - 1)
    flow = jnp.mean((pred - z) ** 2, axis=(2, 3, 4))
    rew = optax.softmax_cross_entropy_with_integer_labels(reward, label)
    end = optax.sigmoid_binary_cross_entropy(done, d[:, 1:] * (1 - tr[:, 1:]))
    w = valid * (0.9 * tau + 0.1)
    return (jnp.sum(w * (flow + 0.1 * rew)) + 0.1 * jnp.sum(w[:, :-1] * (1 - tr[:, 1:]) * end)) / valid.sum()

def optimizer(cfg, steps):
    """Muon on every Linear and Conv kernel, AdamW on the rest; linear warmup, cosine decay."""
    def dims(path, p):
        return MuonDimensionNumbers(tuple(range(p.ndim - 1)), -1) if "kernel" in str(path) else None
    sched = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, min(200, steps // 10 + 1), steps)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.contrib.muon(
        sched, weight_decay=cfg.weight_decay, adam_weight_decay=cfg.weight_decay, consistent_rms=0.2,
        muon_weight_dimension_numbers=lambda params: jax.tree_util.tree_map_with_path(dims, params)))

def fit(model, loss_fn, data, steps, batch, cfg):
    """All optimizer steps on loss_fn(model, data, key, batch) as one compiled scan; returns the losses."""
    @nnx.scan(in_axes=(nnx.Carry, None, 0), out_axes=(nnx.Carry, 0))
    def one(carry, data, key):
        model, opt = carry
        loss, grads = nnx.value_and_grad(lambda m: loss_fn(m, data, key, batch))(model)
        opt.update(model, grads)
        return (model, opt), loss
    opt = nnx.Optimizer(model, optimizer(cfg, steps), wrt=nnx.Param)
    return nnx.jit(one)((model, opt), data, jax.random.split(jax.random.key(cfg.seed), steps))[1]

def train(data_path, out, cfg):
    t0, data = time.time(), np.load(data_path)
    assert data["actions"].size <= BUDGET, f"the data budget is {BUDGET} transitions"
    frames = jnp.asarray(data["frames"])
    def incoming(k, v): return jnp.pad(jnp.asarray(data[k], float), ((1, 0), (0, 0)), constant_values=v)
    A, R, Dn = incoming("actions", 0).astype(jnp.int32), incoming("rewards", 0), incoming("dones", 1)
    Tr = incoming("truncated", 0) if "truncated" in data.files else 0 * R  # an end, but no death
    model = Model(nnx.Rngs(cfg.seed))
    print(f"[data] frames {frames.shape}, {time.time() - t0:.0f} s", flush=True)
    loss = fit(model.tok, tok_loss, frames.reshape(-1, H, W, 3), cfg.tok_steps, cfg.tok_batch, cfg)
    print(f"[tokenizer] loss {loss[0]:.4f} -> {loss[-1]:.5f}, {time.time() - t0:.0f} s", flush=True)
    Z = jax.jit(lambda tok, f: jax.lax.map(tok.encode, f))(model.tok, frames)
    model.tok.mean[...], model.tok.std[...] = Z.mean((0, 1, 2, 3)), Z.std((0, 1, 2, 3))
    Z = (Z - model.tok.mean[...]) / model.tok.std[...]
    loss = fit(model.wm, wm_loss, (Z, A, R, jnp.maximum(Dn, Tr), Tr), cfg.wm_steps, cfg.wm_batch, cfg)
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
    p.add_argument("--out", default="runs/tiny/model.npz")
    for f in fields(Config):
        p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = vars(p.parse_args())
    train(args.pop("data"), args.pop("out"), Config(**args))

"""Factorial causal dynamics: temporal codec, shortcut bootstrap, resampled history and temporal loss."""
import argparse, hashlib, json, os, time
from collections import namedtuple
from dataclasses import dataclass, asdict, fields
from pathlib import Path
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
import jax
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
import jax.numpy as jnp, numpy as np, optax, flax.nnx as nnx, kernels
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
    wm_steps: int = 16_000
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
    temporal_codec: int = 1
    history_dropout: float = 0.25
    shortcut_weight: float = 0.25
    shortcut_every: int = 4
    shortcut_warm: float = 0.25
    resample_frames: int = 2
    resample_every: int = 8
    resample_warm: float = 0.5
    resample_signal: float = 0.5
    temporal_weight: float = 0.25
    head_trunk: float = 0.1
    head_warm: float = 0.5
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
def _grad_scale(x, strength): return jax.lax.stop_gradient(x) + strength * (x - jax.lax.stop_gradient(x))
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
    N = x.shape[0]
    return x.reshape((N, GRID, TILE, GRID, TILE, 3)).transpose((0, 1, 3, 2, 4, 5)).reshape((N, GRID, GRID, TILE * TILE * 3))
def _untile(t):
    N = t.shape[0]
    return t.reshape((N, GRID, GRID, TILE, TILE, 3)).transpose((0, 1, 3, 2, 4, 5)).reshape((N, H, W, 3))
class TileOut(nnx.Module):
    def __init__(self, features, rngs):
        self.kernel = nnx.Param(jax.nn.initializers.lecun_normal()(rngs.params(), (TILE, TILE, features, 3), PDT))
        self.bias = nnx.Param(jnp.zeros((3,), PDT))
    def __call__(self, h):
        w = self.kernel[...].astype(CDT).transpose((2, 0, 1, 3)).reshape((h.shape[-1], TILE * TILE * 3))
        return h.astype(CDT) @ w + jnp.tile(self.bias[...].astype(CDT), TILE * TILE)
class Encoder(nnx.Module):
    def __init__(self, rngs, temporal=1):
        self.temporal = bool(temporal)
        self.stem = nnx.Linear((1 + self.temporal) * TILE * TILE * 3, 64, rngs=rngs, **MP)
        self.proj = nnx.Conv(64, ENC_HID, (1, 1), rngs=rngs, **MP)
        self.hi = nnx.List([TBlock(ENC_HID, rngs) for _ in range(NB_HI)])
        self.down = nnx.Conv(ENC_HID, 128, (3, 3), strides=(2, 2), padding="VALID", rngs=rngs, **MP)
        self.lo = nnx.List([ResBlock(128, rngs) for _ in range(NB_LO)])
        self.out = nnx.Conv(128, TOK, (1, 1), rngs=rngs, **MP)
    def __call__(self, x):
        tiles = jnp.concatenate([_tiles(x[..., :3]), _tiles(x[..., 3:])], -1) if self.temporal else _tiles(x)
        h = _chain(self.hi, self.proj(nnx.gelu(self.stem(tiles.astype(CDT)))))
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
    def __init__(self, rngs, temporal=1):
        self.enc, self.dec = Encoder(rngs, temporal), Decoder(rngs)
class Block(nnx.Module):
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
    def __call__(self, h, cond, kv=None, slot=None, cache=False):
        norm, gate, attend = kernels.norm_mod, kernels.gated_add, kernels.attention
        ss, sh, sg, ts, th, tg, ms, mh, mg = jnp.split(self.mod(cond)[:, :, None, :], 9, axis=-1)
        y = self.so(attend(self.sqkv(norm(h, ss, sh)), self.sq_bias[...], HEADS, False))
        h, z = kernels.residual_norm(h, sg, y, ts, th)
        qkv = self.tqkv(z)
        attn = attend(qkv, self.tq_bias[...], HEADS, True) if kv is None else kernels.attention_one(qkv[:, 0], self.tq_bias[...], kv, slot, HEADS)[:, None]
        y = self.to(attn)
        h, z = kernels.residual_norm(h, tg, y, ms, mh)
        h = gate(h, mg, self.m2(kernels.gelu(self.m1(z))))
        return (h, qkv[..., h.shape[-1]:]) if cache else h
class DF(nnx.Module):
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
        self.step_embed = nnx.Param(jnp.zeros((D,), PDT))
    def __call__(self, x, tau, a):
        return self.forward(x, tau, a)[0]
    def action_embed(self, a):
        return jnp.matmul(jax.nn.one_hot(a, self.aemb.num_embeddings, dtype=PDT), self.aemb.embedding[...], precision=jax.lax.Precision.HIGHEST).astype(CDT)
    def reward_logits(self, summary):
        return self.rew2(nnx.gelu(self.rew1(summary.astype(CDT)))).astype(jnp.float32)
    def done_logits(self, summary, next_actions, head_scale=1.0):
        action = self.action_embed(next_actions)
        h = jnp.concatenate([summary.astype(CDT), _grad_scale(action, head_scale)], -1)
        return self.done2(nnx.gelu(self.done1(h)))[..., 0].astype(jnp.float32)
    def forward(self, x, tau, a, caches=None, slot=None, cache=False, delta=0.0):
        B, L = x.shape[:2]
        ae = self.action_embed(a)
        h = self.inp(x.reshape((B, L, GLAT * GLAT, TOK)).astype(CDT))
        pos = self.tpos.value[:, :L] if slot is None else self.tpos.value[:, slot][:, None]
        h = h + self.spos.value.astype(h.dtype) + pos.astype(h.dtype) + ae[:, :, None, :]
        cond = self.taulin(_tau_embed(tau.reshape((-1,)), D).astype(CDT)).reshape((B, L, D)) + ae + 8 * delta * self.step_embed[...].astype(CDT)
        stored = []
        for i, blk in enumerate(self.blocks):
            if cache:
                h, kv = blk(h, cond, cache=True)
                stored.append(kv)
            else: h = blk(h, cond, None if caches is None else caches[i], slot)
        if cache: return tuple(stored)
        h = self.outn(h)
        return self.out(h).reshape((B, L, GLAT, GLAT, TOK)).astype(jnp.float32), h.mean(2)
@nnx.jit
def _encode(enc, zmean, zstd, obs_u8):
    if enc.temporal:
        past = jnp.take(obs_u8, jnp.maximum(jnp.arange(obs_u8.shape[-4]) - 1, 0), axis=-4) if obs_u8.ndim >= 5 else obs_u8
        obs_u8 = jnp.concatenate([past, obs_u8], -1)
    z = enc(_to01(obs_u8).reshape((-1, H, W, obs_u8.shape[-1])))
    return ((z - zmean) / zstd).reshape(obs_u8.shape[:-3] + z.shape[-3:])
@nnx.jit
def _decode(dec, zmean, zstd, latent):
    z = (latent * zstd + zmean).reshape((-1,) + latent.shape[-3:])
    return jnp.clip(dec(z) * 255.0, 0, 255).astype(jnp.uint8).reshape(latent.shape[:-3] + (H, W, 3))
def encode(model, obs_u8):
    return _encode(model.enc, model.df.zmean.value, model.df.zstd.value, jnp.asarray(obs_u8))
def decode(model, latent):
    return _decode(model.dec, model.df.zmean.value, model.df.zstd.value, latent)
def _flow_one(apply, seq, taus, aseq, s, rng, n_steps=STEPS):
    B = seq.shape[0]
    kc, kz = jax.random.split(rng)
    t5 = taus[..., None, None, None]
    seq = jnp.where(t5 > 0.5, t5 * seq + (1.0 - t5) * jax.random.normal(kc, seq.shape), seq)
    dt = 1.0 / n_steps
    caches = apply.forward(seq, taus, aseq, cache=True)
    action = aseq[:, s][:, None]
    def step(z, tau_s):
        x1 = apply.forward(z[:, None], jnp.full((B, 1), tau_s), action, caches=caches, slot=s, delta=dt)[0][:, 0]
        return z + dt * (x1 - z) / jnp.maximum(1.0 - tau_s, 1e-4), None
    return jax.lax.scan(step, jax.random.normal(kz, (B,) + seq.shape[-3:]), jnp.arange(n_steps) * dt)[0]
def _rollout_latents(apply, ctx, ctx_a, acts, rng, steps=STEPS, win=WIN):
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
    L = latents.shape[1]
    latents, actions = jnp.pad(latents, ((0, 0), (0, WIN - L)) + ((0, 0),) * 3), jnp.pad(actions, ((0, 0), (0, WIN - L)))
    x = CTX_SIGNAL * latents + (1.0 - CTX_SIGNAL) * jax.random.normal(key, latents.shape)
    s = df.forward(x, jnp.full(latents.shape[:2], CTX_SIGNAL), actions)[1][:, :L]
    rewards = _symexp(jnp.sum(jax.nn.softmax(df.reward_logits(s), -1) * jnp.linspace(-RLIM, RLIM, RBINS), -1))
    p = jax.nn.sigmoid(df.done_logits(s[:, :-1], actions[:, 1:L]))
    return rewards, jnp.concatenate([jnp.zeros_like(p[:, :1]), p], 1)
def imagine(model, obs_u8, ctx_actions, actions, rng, steps=STEPS):
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
    gdef, state = nnx.split(model.df)
    read = jax.jit(lambda state, z, a, key: _read_heads(nnx.merge(gdef, state), z, a, key))
    return read(state, encode(model, jnp.asarray(obs_u8)), jnp.asarray(actions, jnp.int32), rng)
_orthogonalize = kernels.polar_express
def _muon(sched, weight_decay, beta=0.95, rms=0.2):
    "NorMuon: normalize per-row/column variance while preserving each update's norm."
    def init(params):
        def variance(p):
            r, c = p.size // p.shape[-1], p.shape[-1]
            return jnp.zeros((r, 1) if r >= c else (1, c), PDT)
        return jax.tree.map(jnp.zeros_like, params), jax.tree.map(variance, params), jnp.zeros((), jnp.int32)
    def update(grads, state, params):
        mu, nu, count = state
        mu = jax.tree.map(lambda m, g: beta * m + g, mu, grads)
        leaves, tdef = jax.tree.flatten(jax.tree.map(lambda m, g: g + beta * m, mu, grads))
        flat = [l.reshape((-1, l.shape[-1])) for l in leaves]
        groups = {shape: [i for i, l in enumerate(flat) if l.shape == shape] for shape in {l.shape for l in flat}}
        out, moments = list(flat), jax.tree.leaves(nu)
        for idx in groups.values():
            orth = _orthogonalize(jnp.stack([flat[k] for k in idx]))
            for j, i in enumerate(idx):
                o = orth[j]
                axis = -1 if o.shape[-2] >= o.shape[-1] else -2
                v = 0.95 * moments[i] + 0.05 * jnp.mean(o * o, axis, keepdims=True)
                scaled = o * jax.lax.rsqrt(v + 1e-10)
                out[i] = scaled * jnp.sqrt(jnp.sum(o * o) / (jnp.sum(scaled * scaled) + 1e-10))
                moments[i] = v
        new = [-sched(count) * (rms * max(o.shape) ** 0.5 * o.reshape(p.shape) + weight_decay * p) for o, p in zip(out, jax.tree.leaves(params))]
        return jax.tree.unflatten(tdef, new), (mu, jax.tree.unflatten(tdef, moments), count + 1)
    return optax.GradientTransformation(init, update)
def _optimizer(lr, steps, decay, warm, weight_decay, head_split=False):
    if head_split: return optax.partition({k: _optimizer(lr, steps, decay, warm, weight_decay) for k in ("body", "heads")}, lambda params: jax.tree_util.tree_map_with_path(lambda p, x: "heads" if any(n in jax.tree_util.keystr(p) for n in ("rew1", "rew2", "done1", "done2")) else "body", params))
    dec = int(steps * decay)
    sched = optax.join_schedules([optax.linear_schedule(0.0, lr, warm), optax.constant_schedule(lr),
                                  optax.linear_schedule(lr, 0.0, max(dec, 1))], [warm, steps - dec])
    label = lambda p, x: "muon" if x.ndim in (2, 4) and "aemb" not in jax.tree_util.keystr(p) else "adam"
    labels = lambda params: jax.tree_util.tree_map_with_path(label, params)
    opts = {"muon": _muon(sched, weight_decay), "adam": optax.adamw(sched, weight_decay=weight_decay)}
    return optax.chain(optax.clip_by_global_norm(1.0), optax.partition(opts, labels))
def _train_tokenizer(tok, frames, cfg, envs=1):
    warm = min(300, max(1, cfg.tok_steps // 10))
    opt = nnx.Optimizer(tok, _optimizer(cfg.tok_lr, cfg.tok_steps, cfg.decay, warm, 0.0), wrt=nnx.Param)
    n = cfg.tok_batch // 3
    @nnx.jit
    def run(tok, opt, frames, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def loop(carry, key):
            tok, opt = carry
            kr, km, kn, ke = jax.random.split(key, 4)
            ids = jax.random.randint(kr, (cfg.tok_batch,), 0, frames.shape[0])
            x = _to01(frames[ids])
            xc, xm, xd = x[:n], x[n:2 * n], x[2 * n:]
            m = (jax.random.uniform(km, (n, GRID, GRID, 1)) > 0.5).astype(jnp.float32)
            m = jnp.repeat(jnp.repeat(m, TILE, 1), TILE, 2)
            sig = jax.random.uniform(kn, (xd.shape[0], 1, 1, 1)) * 0.2
            inp = jnp.concatenate([xc, xm * m + 0.5 * (1 - m), jnp.clip(xd + sig * jax.random.normal(ke, xd.shape), 0, 1)], 0)
            if tok.enc.temporal:
                past = _to01(frames[jnp.where(ids >= envs, ids - envs, ids)])
                absent = jax.random.bernoulli(jax.random.fold_in(km, 1), cfg.history_dropout, (cfg.tok_batch, 1, 1, 1))
                inp = jnp.concatenate([jnp.where(absent, inp, past), inp], -1)
            loss, grads = nnx.value_and_grad(lambda tok: jnp.mean((tok.dec(tok.enc(inp)) - x) ** 2))(tok)
            opt.update(tok, grads)
            return (tok, opt), loss
        return loop((tok, opt), keys)[1]
    return run(tok, opt, frames, jax.random.split(jax.random.PRNGKey(cfg.seed + 1), cfg.tok_steps))
def _encode_all(enc, frames):
    gdef, state = nnx.split(enc)
    def one(t):
        f = jnp.concatenate([frames[jnp.maximum(t - 1, 0)], frames[t]], -1) if enc.temporal else frames[t]
        return nnx.merge(gdef, state)(_to01(f))
    return jax.lax.map(one, jnp.arange(frames.shape[0]))
def _layout(actions, dones, length):
    T, E = actions.shape[0] + 1, actions.shape[1]
    incoming = jnp.concatenate([jnp.zeros((1, E), jnp.int32), actions], 0)
    resets = jnp.concatenate([jnp.ones((1, E), jnp.float32), dones], 0)
    rt, re = np.nonzero(np.asarray(resets[:T - length + 1]) > 0.5)
    return incoming, resets, jnp.asarray(rt), jnp.asarray(re)
def _shortcut_loss(df, z, a, valid, key, history):
    z, a, valid = z[:8], a[:8], valid[:8]
    B, L = z.shape[:2]
    ks, kt, kn = jax.random.split(key, 3)
    slot = jax.random.randint(ks, (), 0, L)
    tau = jax.random.randint(kt, (B, 1), 0, STEPS) / STEPS
    eps = jax.random.normal(kn, z.shape)
    levels = jnp.broadcast_to(jnp.where(jnp.arange(L) < slot, CTX_SIGNAL, 0.0), (B, L))
    seq = CTX_SIGNAL * history[:8] + (1.0 - CTX_SIGNAL) * eps
    caches = jax.tree.map(jax.lax.stop_gradient, df.forward(seq, levels, a, cache=True))
    t5 = tau[..., None, None, None]
    query = t5 * z[:, slot][:, None] + (1.0 - t5) * eps[:, slot][:, None]
    action = a[:, slot][:, None]
    fine = lambda x, t: jax.lax.stop_gradient(df.forward(x, t, action, caches=caches, slot=slot)[0])
    v1 = (fine(query, tau) - query) / (1.0 - t5)
    middle = query + v1 / (2 * STEPS)
    v2 = (fine(middle, tau + 1 / (2 * STEPS)) - middle) / (1.0 - t5 - 1 / (2 * STEPS))
    target = jax.lax.stop_gradient(query + (1.0 - t5) * (v1 + v2) / 2)
    pred = df.forward(query, tau, action, caches=caches, slot=slot, delta=1 / STEPS)[0]
    error = (0.1 + .9 * tau[:, 0]) * jnp.mean((pred - target) ** 2, (1, 2, 3, 4))
    return jnp.sum(error * valid[:, slot]) / (jnp.sum(valid[:, slot]) + 1e-6)
def _resample_context(df, z, a, key, cfg):
    B, L = z.shape[:2]
    ks, kn = jax.random.split(key)
    start = jax.random.randint(ks, (B,), 0, L - cfg.resample_frames + 1)
    slots, batch = jnp.arange(L)[None], jnp.arange(B)
    def one(seq, item):
        offset, key = item
        kc, kt = jax.random.split(key)
        slot = start + offset
        tau = jnp.where(slots < slot[:, None], CTX_SIGNAL, 0.0)
        xt = CTX_SIGNAL * seq + (1.0 - CTX_SIGNAL) * jax.random.normal(kc, seq.shape)
        query = cfg.resample_signal * z[batch, slot] + (1.0 - cfg.resample_signal) * jax.random.normal(kt, z[:, 0].shape)
        xt, tau = xt.at[batch, slot].set(query), tau.at[batch, slot].set(cfg.resample_signal)
        return seq.at[batch, slot].set(df(xt, tau, a)[batch, slot]), None
    result = jax.lax.scan(one, z, (jnp.arange(cfg.resample_frames), jax.random.split(kn, cfg.resample_frames)))[0]
    return jax.lax.stop_gradient(result), (slots >= start[:, None]) & (slots < start[:, None] + cfg.resample_frames)
def _train_wm(df, Z, A, Dd, R, cfg):
    warm = min(100, max(1, cfg.wm_steps // 10))
    opt = nnx.Optimizer(df, _optimizer(cfg.lr, cfg.wm_steps, cfg.decay, warm, 0.01, True), wrt=nnx.Param)
    T, E = Z.shape[:2]
    L, B = cfg.rollout, cfg.wm_batch
    A, Dd, rst_t, rst_e = _layout(A, Dd, L)
    R = jnp.concatenate([jnp.zeros((1, E), jnp.float32), R], 0)
    ltab, offz = jnp.full((T - L + 1, E), 10.0), jnp.arange(L)
    @nnx.jit
    def run(df, opt, ltab, Z, A, Dd, R, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def loop(carry, item):
            key, step = item
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
            history = z
            if cfg.resample_frames:
                active = (step >= cfg.wm_steps * cfg.resample_warm) & (step % cfg.resample_every == 0)
                history, mask = jax.lax.cond(active, lambda: _resample_context(df, z, a, jax.random.fold_in(keps, 1), cfg), lambda: (z, jnp.zeros((B, L), jnp.bool_)))
                tau = jnp.where(mask, CTX_SIGNAL, tau)
            t5 = tau[:, :, None, None, None]
            x_tau = (1.0 - t5) * jax.random.normal(keps, z.shape) + t5 * history
            def loss_fn(df):
                delta = jax.random.bernoulli(jax.random.fold_in(key, 90), .5) / STEPS if cfg.shortcut_weight else 0.0
                x1, s = df.forward(x_tau, tau, a, delta=delta)
                ramp = 0.9 * tau + 0.1
                err = ramp * jnp.mean((x1 - z) ** 2, axis=(2, 3, 4))
                strength = jnp.where(step < cfg.wm_steps * cfg.head_warm, 1.0, cfg.head_trunk)
                s = _grad_scale(s, strength)
                ce = -jnp.sum(_two_hot(_symlog(r)) * jax.nn.log_softmax(df.reward_logits(s), -1), -1)
                dl = optax.sigmoid_binary_cross_entropy(df.done_logits(s[:, :-1], a[:, 1:], head_scale=strength), dwin[:, 1:])
                loss = (jnp.sum(err * valid) + cfg.reward_weight * jnp.sum(ramp * ce * valid)) / (jnp.sum(valid) + 1e-6)
                loss = loss + cfg.done_weight * jnp.sum(ramp[:, :-1] * dl * valid[:, :-1]) / (jnp.sum(valid[:, :-1]) + 1e-6)
                if cfg.shortcut_weight:
                    active = (step >= cfg.wm_steps * cfg.shortcut_warm) & (step % cfg.shortcut_every == 0)
                    loss += cfg.shortcut_weight * jax.lax.cond(active, lambda: _shortcut_loss(df, z, a, valid, jax.random.fold_in(key, 91), history), lambda: jnp.array(0.0))
                if cfg.temporal_weight:
                    drift = (x1[:, 1:] - x1[:, :-1]) - (z[:, 1:] - z[:, :-1])
                    motion = jnp.minimum(tau[:, 1:], tau[:, :-1]) * jnp.mean(drift ** 2, (2, 3, 4))
                    loss += cfg.temporal_weight * jnp.sum(motion * valid[:, 1:]) / (jnp.sum(valid[:, 1:]) + 1e-6)
                return loss, jnp.sum(err * valid, 1) / (jnp.sum(valid, 1) + 1e-6)
            (loss, perwin), grads = nnx.value_and_grad(loss_fn, has_aux=True)(df)
            opt.update(df, grads)
            ltab = ltab.at[t, e].set(0.5 * ltab[t, e] + 0.5 * perwin, unique_indices=False)
            return (df, opt, ltab), loss
        return loop((df, opt, ltab), (keys, jnp.arange(len(keys))))[1]
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
    tok = Tok(nnx.Rngs(cfg.seed), cfg.temporal_codec)
    t1 = time.time()
    tl = _train_tokenizer(tok, frames.reshape((-1, H, W, 3)), cfg, frames.shape[1])
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
    mods = {"enc": Encoder(nnx.Rngs(0), cfg.get('temporal_codec', 0)), "dec": Decoder(nnx.Rngs(0)), "df": DF(nnx.Rngs(0), cfg["num_actions"], max(LMAX, cfg["rollout"]))}
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

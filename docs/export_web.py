"""Export the world model to one ONNX graph for in-browser play (onnxruntime-web on WebGPU) and pack start states.

    python docs/export_web.py --model runs/record1/model.npz

Writes docs/play.onnx (gitignored; fp32) and docs/starts.bin. One call imagines the
last frame of a full 32-frame window: a pass over the 31 past frames caches every block's temporal keys and values and
the last past frame's summary; 8 Euler steps as in train._flow_one then run only the new frame's 16 tokens against that
cache, exact since spatial attention is per frame and temporal attention causal (6.8 GFLOP per call by XLA's count,
instead of 42 for eight full-window passes); one more new-frame pass at the context signal reads the reward, and the
decoder renders the frame. Needs jax2onnx, onnx and onnxruntime on top of the repo's dependencies.

play.onnx   in   past f32 [1,31,4,4,32] (tau*z + (1-tau)*eps where tau > 0.5, else z), tau f32 [1,31],
                 act i32 [1,32] (incoming action of each past frame, then the new action), noise, rnoise f32 [1,4,4,32]
            out  z f32 [1,4,4,32], rgb u8 [63,63,3], reward f32 [1], done f32 [1]
starts.bin  uint32 N, f32 clean latents [N,31,4,4,32], u8 incoming actions [N,31], u8 last frame [N,63,63,3]
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kernels
import train

train.CDT, train.MP = jnp.float32, dict(dtype=jnp.float32, param_dtype=jnp.float32)  # fp32 on CPU, as play_wm.py
jax.config.update("jax_default_matmul_precision", "highest")
PAST, G = train.WIN - 1, train.GLAT * train.GLAT


def gelu(x):
    """jax.nn.gelu (tanh), what kernels.gelu and nnx.gelu compute off GPU, written out as plain ops."""
    return x * (0.5 * (1.0 + jnp.tanh(np.float32(np.sqrt(2 / np.pi)) * (x + 0.044715 * (x * x * x)))))


def norm(h, scale, shift):
    """kernels.norm_mod off GPU."""
    return h * jax.lax.rsqrt(jnp.mean(h * h, -1, keepdims=True) + kernels.EPS) * (1.0 + scale) + shift


def heads(x, n=train.HEADS):
    return x.reshape(x.shape[:-1] + (n, x.shape[-1] // n))


def attend(q, k, v, causal=False):
    """kernels.attention off GPU, on split heads: q (..., Lq, heads, hd) with its bias added, k and v (..., Lk, heads,
    hd)."""
    s = jnp.einsum("...qhd,...khd->...hqk", q * q.shape[-1] ** -0.5, k)
    if causal:
        s = jnp.where(np.tril(np.ones(s.shape[-2:], bool)), s, kernels.MASKED)
    return jnp.einsum("...hqk,...khd->...qhd", jax.nn.softmax(s, -1), v)


def decode(dec, z):
    """train.Decoder on one latent (1, GLAT, GLAT, TOK), with gelu and attention as above -> (H, W, 3) in [0, 1]."""
    h = gelu(dec.inp(z))
    for r in dec.lo:
        h = h + r.c2(gelu(r.n2(r.c1(gelu(r.n1(h))))))
    h = gelu(dec.up(h))
    for t in dec.hi:                                  # TBlock: its attention's query bias is zero
        h = h + t.pos[...]
        qkv = t.qkv(t.n1(h)).reshape((1, -1, 3 * h.shape[-1]))
        q, k, v = (heads(x, train.THEADS) for x in jnp.split(qkv, 3, -1))
        h = h + t.o(attend(q, k, v).reshape(h.shape))
        h = h + t.m2(gelu(t.m1(t.n2(h))))
    return jax.nn.sigmoid(train._untile(dec.out(h)))[0]


def block(b, h, cond, cache=None):
    """train.Block on h (1, L, G, D). Without a cache, causal temporal attention over the L frames; with the past's
    temporal (k, v) (1, G, PAST, HEADS, D / HEADS), h is the new frame and attends to the past plus itself. Returns
    h and this block's temporal (k, v)."""
    ss, sh, sg, ts, th, tg, ms, mh, mg = jnp.split(b.mod(cond)[:, :, None, :], 9, -1)
    q, k, v = jnp.split(b.sqkv(norm(h, ss, sh)), 3, -1)
    h = h + sg * b.so(attend(heads(q + b.sq_bias[...]), heads(k), heads(v)).reshape(h.shape))
    q, k, v = jnp.split(b.tqkv(norm(h, ts, th)), 3, -1)
    q, k, v = (heads(x).swapaxes(1, 2) for x in (q + b.tq_bias[...], k, v))    # (1, G, L, HEADS, D / HEADS)
    kc, vc = (k, v) if cache is None else (jnp.concatenate([cache[0], k], 2), jnp.concatenate([cache[1], v], 2))
    h = h + tg * b.to(attend(q, kc, vc, cache is None).swapaxes(1, 2).reshape(h.shape))
    return h + mg * b.m2(gelu(b.m1(norm(h, ms, mh)))), (k, v)


def make_step(model):
    """(past, tau, act, noise, rnoise) -> (z, rgb, reward, done), the contract above, in plain jnp."""
    df = model.df
    aemb = lambda a: jnp.take(df.aemb.embedding[...], a, axis=0)   # a gather, equal to DF.action_embed's matmul

    def embed(x, tau, a, slots):
        """train.DF.forward up to the blocks for frames x (1, L, GLAT, GLAT, TOK) at window slots `slots`."""
        ae = aemb(a)
        h = df.inp(x.reshape(x.shape[:2] + (G, train.TOK))) + df.spos[...] + df.tpos[...][:, slots] + ae[:, :, None]
        return h, df.taulin(train._tau_embed(tau.reshape(-1), train.D)).reshape(ae.shape) + ae

    def step(past, tau, act, noise, rnoise):
        h, cond = embed(past, tau, act[:, :PAST], slice(0, PAST))
        cache = []
        for b in df.blocks:
            h, kv = block(b, h, cond)
            cache.append(kv)
        s_past = df.outn(h).mean(2)[:, -1]

        def new_frame(z, level):
            h, cond = embed(z[:, None], jnp.full((1, 1), level), act[:, PAST:], slice(PAST, PAST + 1))
            for b, kv in zip(df.blocks, cache):
                h = block(b, h, cond, kv)[0]
            h = df.outn(h)
            return df.out(h).reshape(z.shape), h.mean(2)[:, 0]

        z, dt = noise, 1.0 / train.STEPS
        for i in range(train.STEPS):                 # train._flow_one's update
            z = z + dt * (new_frame(z, i * dt)[0] - z) / max(1.0 - i * dt, 1e-4)
        s_new = new_frame(train.CTX_SIGNAL * z + (1.0 - train.CTX_SIGNAL) * rnoise, train.CTX_SIGNAL)[1]
        bins = np.linspace(-train.RLIM, train.RLIM, train.RBINS, dtype=np.float32)
        reward = train._symexp(jnp.sum(jax.nn.softmax(df.rew2(gelu(df.rew1(s_new))), -1) * bins, -1))
        done = jax.nn.sigmoid(df.done2(gelu(df.done1(jnp.concatenate([s_past, aemb(act[:, PAST])], -1))))[:, 0])
        img = decode(model.dec, z * df.zstd[...] + df.zmean[...])
        return z, jnp.clip(img * 255.0, 0, 255).astype(jnp.uint8), reward, done
    return step


def light(t):
    """craftax_classic.game_logic.calculate_light_level at episode step t (day length 300)."""
    return 1.0 - np.abs(np.cos(np.pi * ((t / 300.0) % 1 + 0.3))) ** 3


def terrain(frame):
    """How many of trees, water, stone and sand cover at least two of the 9x7 map tiles (by tile mean colour), minus
    two if lava is in view."""
    tiles = frame[:49].reshape(7, 7, 9, 7, 3).astype(np.float32)
    (r, g, b), std = tiles.mean((1, 3)).transpose(2, 0, 1), tiles.std((1, 3)).mean(-1)
    kinds = [(g > r + 30) & (g < 120) & (std > 30), (b > g + 40) & (b > r + 100),
             (np.abs(r - b) < 12) & (r > 100) & (r < 150) & (std > 8), (r > 170) & (g > 150) & (b < 150)]
    return sum(int(k.sum() >= 2) for k in kinds) - 2 * int(((r > 180) & (r > g + 60) & (b < 60)).any())


def pick_starts(data, n, seed=0):
    """(env, last frame) of n windows of PAST frames from different held-out episodes with no reset inside: daylight
    over the window and 20 more steps, full health throughout (the status bar's health tile equals a fresh episode's),
    alive 30 steps on, no sleeping; the most varied terrain first."""
    F, A, D = data["frames"], data["actions"].astype(np.int64), data["dones"]
    T, E = A.shape
    clock = np.zeros((T + 1, E))
    for t in range(T):
        clock[t + 1] = np.where(D[t], 0, clock[t] + 1)
    healthy = (F[:, :, 49:56, :7] == F[0, 0, 49:56, :7]).all((2, 3, 4))      # streams start with a fresh episode
    rng, best = np.random.default_rng(seed), []
    for e in range(E):
        ok = [t for t in range(PAST, T + 1) if not D[t - PAST:min(t + 30, T), e].any()
              and light(clock[t - PAST + 1:t + 1, e]).min() >= 0.8 and light(clock[t, e] + 20) >= 0.8
              and healthy[t - PAST + 1:t + 1, e].all() and not (A[max(t - 2 * PAST, 0):t, e] == 6).any()]
        if ok:
            scores = [terrain(F[t, e]) + rng.uniform() for t in ok]
            best.append((max(scores), e, ok[int(np.argmax(scores))]))
    return [(e, t) for _, e, t in sorted(best, reverse=True)[:n]]


def pack_starts(model, data, path, n):
    """starts.bin, little-endian: uint32 N, f32 clean standardized latents [N, PAST, GLAT, GLAT, TOK], u8 incoming
    actions [N, PAST], u8 last frame [N, H, W, 3]."""
    picks = pick_starts(data, n)
    frames = np.stack([data["frames"][t - PAST + 1:t + 1, e] for e, t in picks])
    acts = np.stack([data["actions"][t - PAST:t, e] for e, t in picks]).astype(np.uint8)
    lat = np.asarray(train.encode(model, frames), "<f4")
    Path(path).write_bytes(np.uint32(len(picks)).astype("<u4").tobytes() + lat.tobytes() + acts.tobytes()
                           + np.ascontiguousarray(frames[:, -1]).tobytes())
    return picks


SPECS = dict(past=((1, PAST, train.GLAT, train.GLAT, train.TOK), jnp.float32), tau=((1, PAST), jnp.float32),
             act=((1, train.WIN), jnp.int32), noise=((1, train.GLAT, train.GLAT, train.TOK), jnp.float32),
             rnoise=((1, train.GLAT, train.GLAT, train.TOK), jnp.float32))


def export(model, path):
    """jax2onnx at opset 18 and onnxruntime's basic constant folding. Weights stay fp32: stored fp16 with a Cast each,
    the file halves, but onnxruntime-web's asyncify build (Safari, which has no JSPI) leaks the cast weights on every
    call, about 40 MB a step, and iPhones kill the tab within a few moves."""
    import onnx
    import onnxruntime as ort
    from jax2onnx import to_onnx
    m = to_onnx(make_step(model), [jax.ShapeDtypeStruct(*s) for s in SPECS.values()], model_name="tinywmcraft",
                opset=18, input_names=list(SPECS), output_names=["z", "rgb", "reward", "done"])
    with tempfile.TemporaryDirectory() as tmp:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath, so.log_severity_level = f"{tmp}/folded.onnx", 3
        ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
        m = onnx.load(so.optimized_model_filepath)
    del m.opset_import[:]
    m.opset_import.append(onnx.helper.make_opsetid("", 18))     # onnxruntime's save declares every domain it knows
    m.ir_version = 8
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, path)


def check(model, path, data, pick):
    """One call of play.onnx (onnxruntime, CPU) against the JAX function, from a start state."""
    import onnxruntime as ort
    (e, t), rng = pick, np.random.default_rng(0)
    inp = dict(past=np.asarray(train.encode(model, data["frames"][t - PAST + 1:t + 1, e]))[None],
               tau=np.ones((1, PAST), np.float32), act=np.r_[data["actions"][t - PAST:t, e], 2].astype(np.int32)[None],
               noise=rng.standard_normal(SPECS["noise"][0], np.float32),
               rnoise=rng.standard_normal(SPECS["rnoise"][0], np.float32))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out, ref = sess.run(None, inp), jax.jit(make_step(model))(*inp.values())
    print("onnx vs jax max abs diff:", "  ".join(f"{k} {np.abs(np.float64(a) - np.float64(b)).max():.2e}"
                                                for k, a, b in zip(["z", "rgb", "reward", "done"], out, ref)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("runs/record1/model.npz"))
    p.add_argument("--data", type=Path, default=Path("data/eval.npz"), help="held-out episodes for the start states")
    p.add_argument("--starts", type=int, default=24)
    p.add_argument("--copy", type=Path, help="also copy play.onnx here")
    a = p.parse_args()
    docs, model, data = Path(__file__).resolve().parent, train.load_model(a.model), dict(np.load(a.data))
    picks = pack_starts(model, data, docs / "starts.bin", a.starts)
    print(f"starts.bin: {len(picks)} windows (env, last step) {picks}, "
          f"{(docs / 'starts.bin').stat().st_size / 1e6:.2f} MB")
    export(model, docs / "play.onnx")
    print(f"play.onnx: {(docs / 'play.onnx').stat().st_size / 1e6:.1f} MB")
    if a.copy:
        a.copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(docs / "play.onnx", a.copy)
    check(model, docs / "play.onnx", data, picks[0])


if __name__ == "__main__":
    main()

"""Export the world model to one ONNX graph for in-browser play (convert_web.py turns it into play.wgr).

    python docs/export_web.py --model runs/base/model.npz [--train path/to/train.py]
    python docs/convert_web.py

Writes docs/play.onnx (gitignored; weights stored fp16, computed in fp32). One call imagines the frame at window slot
s (0..31) after the frames in slots 0..s-1, as train.sample does while the window fills and, at s = 31, once it is
full (the player imagines every world's first frame from noise at slot 0, then fills the window): a pass over the 31
past slots caches every block's temporal keys and values and every past frame's summary; 8 Euler steps as in
train.sample then run only the new frame's 16 tokens against that cache, its temporal attention masked to slots below
s and itself (kmask), exact since spatial attention is per frame and temporal attention causal (about 7 GFLOP per
call, instead of 42 for eight full-window passes); one more new-frame pass at the context signal reads the reward,
the summary of slot s - 1 (prev) reads the done head, and the tokenizer's decoder renders the frame. Past slots from
s on are ignored, whatever they hold. The graph is train.py's model written out in plain operations (no Pallas, no
bf16); the check compares it with the training code's own sampler run in fp32. --train picks the module that defines
the model (default: the repo's train.py; its kernels.py is the one next to it). Needs jax2onnx, onnx and onnxruntime
on top of the repo's dependencies.

play.onnx   in   past f32 [1,31,4,4,32] (tau*z + (1-tau)*eps), tau f32 [1,31],
                 act i32 [1,32] (incoming action of each past slot, then the new action), noise, rnoise f32 [1,4,4,32],
                 pos f32 [1,32] (one-hot of s), prev f32 [1,31] (one-hot of s - 1, zeros at s = 0),
                 kmask f32 [1,32] (0 for past slots below s and for index 31, the new frame itself; -1e30 elsewhere)
            out  z f32 [1,4,4,32], rgb u8 [63,63,3], reward f32 [1], done f32 [1] (meaningless at s = 0)
"""
import argparse
import importlib.util
import shutil
import sys
import tempfile
from functools import partial
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from clips import streams

def _load_train():
    """The module given by --train (read before the full argument parse: everything below depends on it)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--train", type=Path, default=Path(__file__).resolve().parents[1] / "train.py")
    path = p.parse_known_args()[0].train.resolve()
    sys.path.insert(0, str(path.parent))                     # its own kernels.py
    spec = importlib.util.spec_from_file_location("train", path)
    mod = sys.modules["train"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load_train()
MASKED = -1e30


class _F32:
    """train's jax.numpy with bfloat16 read as float32, for fp32() below."""
    bfloat16 = jnp.float32

    def __getattr__(self, name):
        return getattr(jnp, name)


def fp32(model):
    """The training code in fp32 on CPU: train.py makes its layers bf16 for the GPU and casts to bf16 in places; here
    every layer computes in fp32 and the casts are to fp32, the precision of the browser."""
    train.jnp = _F32()
    for _, m in nnx.iter_graph(model):
        if isinstance(m, nnx.Module) and getattr(m, "dtype", None) is not None:
            m.dtype = jnp.float32


jax.config.update("jax_default_matmul_precision", "highest")
PAST, G, TOK = train.WIN - 1, train.LAT * train.LAT, train.TOK


def gelu(x):
    """jax.nn.gelu (tanh), what nnx.gelu computes, written out as plain ops."""
    return x * (0.5 * (1.0 + jnp.tanh(np.float32(np.sqrt(2 / np.pi)) * (x + 0.044715 * (x * x * x)))))


def rms(x, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps)


def mlp(seq, x):
    """nnx.Sequential(Linear, gelu, Linear) with gelu as above."""
    return seq.layers[2](gelu(seq.layers[0](x)))


def heads(x, n=train.HEADS):
    return x.reshape(x.shape[:-1] + (n, x.shape[-1] // n))


def attend(q, k, v, causal=False, kmask=None):
    """kernels.attention off GPU, on split heads: q (..., Lq, heads, hd), k and v (..., Lk, heads, hd); kmask (1, Lk),
    added to the scores of every query."""
    s = jnp.einsum("...qhd,...khd->...hqk", q * q.shape[-1] ** -0.5, k)
    if causal:
        s = jnp.where(np.tril(np.ones(s.shape[-2:], bool)), s, MASKED)
    if kmask is not None:
        s = s + kmask.reshape((1,) * (s.ndim - 1) + kmask.shape[-1:])
    return jnp.einsum("...hqk,...khd->...qhd", jax.nn.softmax(s, -1), v)


def tile_block(t, h):
    """train.TileBlock on h (1, 9, 9, CH)."""
    h = h + t.pos[...]
    qkv = t.qkv(t.n1.scale[...] * rms(h, t.n1.epsilon)).reshape((1, -1, 3 * h.shape[-1]))
    q, k, v = (heads(x) for x in jnp.split(qkv, 3, -1))
    h = h + t.o(attend(q, k, v).reshape(h.shape))
    return h + mlp(t.mlp, t.n2.scale[...] * rms(h, t.n2.epsilon))


def res_block(r, h):
    """train.ResBlock."""
    h1 = r.c1(gelu(r.n1.scale[...] * rms(h, r.n1.epsilon)))
    return h + r.c2(gelu(r.n2.scale[...] * rms(h1, r.n2.epsilon)))


def decode(tok, z):
    """train.Tokenizer.decode of one latent (1, LAT, LAT, TOK) -> (H, W, 3) in [0, 1]."""
    h = z * tok.std[...] + tok.mean[...]
    for layer in tok.dec.layers:
        if isinstance(layer, train.ResBlock):
            h = res_block(layer, h)
        elif isinstance(layer, train.TileBlock):
            h = tile_block(layer, h)
        else:
            h = layer(h) if isinstance(layer, nnx.Module) else gelu(h)
    CH, T, n = h.shape[-1], train.TILE, train.GRID
    x = jnp.einsum("abcd,de->abce", h, tok.tile[...].transpose(2, 0, 1, 3).reshape(CH, -1)) + jnp.tile(tok.bias[...], T * T)
    return jax.nn.sigmoid(x).reshape(1, n, n, T, T, 3).swapaxes(2, 3).reshape(train.H, train.W, 3)


def block(b, h, cond, cache=None, kmask=None):
    """train.Block on h (1, L, G, D). Without a cache, causal temporal attention over the L frames; with the past's
    temporal (k, v) (1, G, PAST, HEADS, D / HEADS), h is the new frame and attends to the past slots kmask keeps plus
    itself. Returns h and this block's temporal (k, v)."""
    s1, b1, g1, s2, b2, g2, s3, b3, g3 = jnp.split(b.mod(cond)[:, :, None], 9, -1)
    x = rms(h) * (1 + s1) + b1
    q, k, v = jnp.split(b.space.qkv(x), 3, -1)
    h = h + g1 * b.space.out(attend(heads(q), heads(k), heads(v)).reshape(h.shape))
    x = rms(h) * (1 + s2) + b2
    q, k, v = (heads(t).swapaxes(1, 2) for t in jnp.split(b.time.qkv(x), 3, -1))    # (1, G, L, HEADS, D / HEADS)
    kc, vc = (k, v) if cache is None else (jnp.concatenate([cache[0], k], 2), jnp.concatenate([cache[1], v], 2))
    h = h + g2 * b.time.out(attend(q, kc, vc, cache is None, kmask).swapaxes(1, 2).reshape(h.shape))
    x = rms(h) * (1 + s3) + b3
    return h + g3 * mlp(b.mlp, x), (k, v)


def make_step(model):
    """(past, tau, act, noise, rnoise, pos, prev, kmask) -> (z, rgb, reward, done), the contract above, in plain jnp."""
    wm = model.wm
    aemb = lambda a: jnp.take(wm.act.embedding[...], a, axis=0)   # a gather, equal to WorldModel.embed's matmul
    freqs = np.exp(np.linspace(0.0, np.log(1000.0), train.D // 2)).astype(np.float32)

    def embed(x, tau, a, tpos, delta=0.0):
        """train.WorldModel.__call__ up to the blocks for frames x (1, L, LAT, LAT, TOK) at levels tau (1, L) with
        temporal position embeddings tpos (L, 1, D); delta is the Euler step size the model is told about."""
        ae, f = aemb(a), tau[..., None] * freqs
        cond = wm.level(jnp.concatenate([jnp.sin(f), jnp.cos(f)], -1)) + ae
        if delta:
            cond = cond + train.STEPS * delta * wm.step[...]
        return wm.inp(x.reshape(x.shape[:2] + (G, TOK))) + tpos + wm.spos[...] + ae[:, :, None], cond

    def step(past, tau, act, noise, rnoise, pos, prev, kmask):
        h, cond = embed(past, tau, act[:, :PAST], wm.tpos[...][:PAST])
        cache = []
        for b in wm.blocks:
            h, kv = block(b, h, cond)
            cache.append(kv)
        # the summary of slot s - 1 and slot s's position embedding, as one-hot contractions in the form of the dense
        # layers (convert_web.py's 'abcd,de->abce')
        s_past = jnp.einsum("abcd,de->abce", prev[:, None, None], rms(h).mean(2)[0])[:, 0, 0]
        tpos_new = jnp.einsum("abcd,de->abce", pos[:, None, None], wm.tpos[...][:, 0])[0]

        def new_frame(z, level, delta=0.0):
            h, cond = embed(z[:, None], jnp.full((1, 1), level), act[:, PAST:], tpos_new, delta)
            for b, kv in zip(wm.blocks, cache):
                h = block(b, h, cond, kv, kmask)[0]
            h = rms(h)
            return wm.out(h).reshape(z.shape), h.mean(2)[:, 0]

        z, dt = noise, 1.0 / train.STEPS
        for i in range(train.STEPS):                 # train.sample's update
            z = z + dt * (new_frame(z, i * dt, dt)[0] - z) / (1.0 - i * dt)
        s_new = new_frame(train.CTX_SIGNAL * z + (1.0 - train.CTX_SIGNAL) * rnoise, train.CTX_SIGNAL)[1]
        bins = np.linspace(-train.RLIM, train.RLIM, train.RBINS, dtype=np.float32)
        r = jnp.sum(jax.nn.softmax(mlp(wm.reward, s_new), -1) * bins, -1)
        reward = jnp.sign(r) * (jnp.exp(jnp.abs(r)) - 1.0)
        done = jax.nn.sigmoid(mlp(wm.done, jnp.concatenate([s_past, aemb(act[:, PAST])], -1))[:, 0])
        img = decode(model.tok, z)
        return z, jnp.clip(img * 255.0 + 0.5, 0, 255).astype(jnp.uint8), reward, done   # rounded, as train.step
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


def any_window(data):
    """(env, last frame) of the first window of PAST frames, and the frame before them, inside one episode: the check's
    window when pick_starts finds none (clips that start mid-episode, at night)."""
    D = data["dones"]
    return next((e, t) for e in range(D.shape[1]) for t in range(PAST, D.shape[0]) if not D[t - PAST:t, e].any())


def latents(model, data, picks):
    """Standardized latents (N, PAST, LAT, LAT, TOK) of the PAST frames up to each (env, t)."""
    frames = np.stack([data["frames"][t - PAST + 1:t + 1, e] for e, t in picks])
    return np.asarray(model.tok.encode(jnp.asarray(frames)))


LATENT = (1, train.LAT, train.LAT, TOK)
SPECS = dict(past=((1, PAST) + LATENT[1:], jnp.float32), tau=((1, PAST), jnp.float32),
             act=((1, train.WIN), jnp.int32), noise=(LATENT, jnp.float32),
             rnoise=(LATENT, jnp.float32), pos=((1, train.WIN), jnp.float32),
             prev=((1, PAST), jnp.float32), kmask=((1, train.WIN), jnp.float32))


def slot_inputs(s):
    """pos, prev and kmask for the new frame at window slot s (0..31)."""
    pos, prev = np.eye(train.WIN, dtype=np.float32)[s][None], np.zeros((1, PAST), np.float32)
    if s:
        prev[0, s - 1] = 1.0
    kmask = np.where((np.arange(train.WIN) < s) | (np.arange(train.WIN) == PAST), 0.0, MASKED)
    return dict(pos=pos, prev=prev, kmask=kmask.astype(np.float32)[None])


def half_weights(model, min_size=1024):
    """Round every matrix-sized weight to fp16 (in place), so that the fp16 file computes exactly what the JAX
    reference with these weights computes."""
    half = lambda w: w.astype(jnp.float16).astype(w.dtype) if w.size >= min_size else w
    for mod in (model.wm, model.tok):
        nnx.update(mod, jax.tree.map(half, nnx.state(mod)))


def gpu_only(g):
    """Edit the graph so that every node runs on onnxruntime-web's WebGPU EP, which graph capture needs: Sign becomes
    Where(x < 0, -1, 1), the action is sliced as float (its Slice takes floats) and the embedding Gathers take the int32
    slices as indices, without Casts to int64."""
    import onnx
    from onnx import helper
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def from_act(t):
        while t in prod and prod[t].op_type in ("Slice", "Squeeze"):
            t = prod[t].input[0]
        return t in ("act", "act_f")

    g.initializer.extend(onnx.numpy_helper.from_array(np.array(v, np.float32), k)
                         for k, v in (("sg_zero", 0.0), ("sg_one", 1.0), ("sg_minus_one", -1.0)))
    nodes = [helper.make_node("Cast", ["act"], ["act_f"], to=onnx.TensorProto.FLOAT, name="act_to_float")]
    for n in g.node:
        if n.op_type == "Sign":
            nodes += [helper.make_node("Less", [n.input[0], "sg_zero"], [n.name + "_lt"], name=n.name + "_less"),
                      helper.make_node("Where", [n.name + "_lt", "sg_minus_one", "sg_one"], list(n.output), name=n.name + "_where")]
        elif n.op_type == "Slice" and n.input[0] == "act":
            out = n.output[0]
            n.input[0], n.output[0] = "act_f", out + "_f"
            nodes += [n, helper.make_node("Cast", [out + "_f"], [out], to=onnx.TensorProto.INT32, name=n.name + "_i32")]
        elif n.op_type == "Cast" and n.attribute[0].i == onnx.TensorProto.INT64 and from_act(n.input[0]) \
                and all(c.op_type == "Gather" and c.input[1] == n.output[0] for c in cons[n.output[0]]):
            for c in cons[n.output[0]]:
                c.input[1] = n.input[0]
        else:
            nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)


def export(model, path, min_size=1024):
    """jax2onnx at opset 18, onnxruntime's basic constant folding, gpu_only, then fp16 storage with a Cast for each
    large initializer (onnxruntime folds the Casts back to fp32 weights at load)."""
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
    gpu_only(m.graph)
    nodes = []
    for init in m.graph.initializer:
        w, name = onnx.numpy_helper.to_array(init), init.name
        if w.dtype == np.float32 and w.size >= min_size:
            assert np.array_equal(w.astype(np.float16).astype(np.float32), w), name
            init.CopyFrom(onnx.numpy_helper.from_array(w.astype(np.float16), name + "_f16"))
            nodes.append(onnx.helper.make_node("Cast", [name + "_f16"], [name], to=onnx.TensorProto.FLOAT))
    nodes += m.graph.node
    del m.graph.node[:], m.opset_import[:]
    m.graph.node.extend(nodes)
    m.opset_import.append(onnx.helper.make_opsetid("", 18))     # onnxruntime's save declares every domain it knows
    m.ir_version = 8
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, path)


def reference(model, past, tau, act, noise, s):
    """The new latent at window slot s as the training code computes it (train.sample: the window holds the frames of
    slots 0..s-1, one pass caches their temporal keys and values, then the Euler steps run train.WorldModel on slot s
    against that cache with the step size), from the same corrupted past and noise."""
    wm, dt = model.wm, 1.0 / train.STEPS
    seq = jnp.zeros((1, train.WIN) + past.shape[2:]).at[:, :s].set(past[:, :s])
    taus = jnp.zeros((1, train.WIN)).at[:, :s].set(tau[:, :s])
    aseq = jnp.zeros((1, train.WIN), jnp.int32).at[:, :s].set(act[:, :s]).at[:, s].set(act[:, PAST])
    cache = wm(seq, taus, aseq)[2]
    z = noise
    for i in range(train.STEPS):
        x1 = wm(z[:, None], jnp.full((1, 1), i * dt), aseq[:, s:s + 1], cache, s, dt)[0][:, 0]
        z = z + (x1 - z) / (train.STEPS * (1.0 - i * dt))
    return z


def check(model, path, data, pick, slots=(0, 1, 7, 30, 31)):
    """At several window slots: one call of play.onnx (onnxruntime, CPU) against the exported JAX function, and the
    JAX function's frame against the training code's own sampler, from a start state. The past slots from s on hold
    noise and must not matter."""
    import onnxruntime as ort
    (e, t), rng = pick, np.random.default_rng(0)
    real, acts = latents(model, data, [pick]), np.asarray(data["actions"][t - PAST:t, e], np.int32)
    sess, step = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]), jax.jit(make_step(model))
    ref = jax.jit(partial(reference, model), static_argnums=4)
    diff = lambda a, b: np.abs(np.float64(a) - np.float64(b)).max()
    for s in slots:
        past = np.where(np.arange(PAST)[None, :, None, None, None] < s, real, rng.standard_normal(real.shape))
        inp = dict(past=past.astype(np.float32), tau=(np.arange(PAST) < s).astype(np.float32)[None],
                   act=np.r_[np.where(np.arange(PAST) < s, acts, rng.integers(0, train.ACTIONS, PAST)), 2].astype(np.int32)[None],
                   noise=rng.standard_normal(SPECS["noise"][0], np.float32),
                   rnoise=rng.standard_normal(SPECS["rnoise"][0], np.float32), **slot_inputs(s))
        out, jx = sess.run(None, inp), step(*inp.values())
        z = ref(inp["past"], inp["tau"], inp["act"], inp["noise"], s)
        rgb = np.round(np.asarray(model.tok.decode(z))[0] * 255).astype(np.uint8)   # as train.step
        print(f"slot {s:2d}  onnx vs jax: " + "  ".join(f"{k} {diff(a, b):.1e}" for k, a, b in zip(["z", "rgb", "reward", "done"], out, jx))
              + f"   jax vs train's sampler: z {diff(jx[0], z):.1e}  rgb {diff(jx[1], rgb):.0f} / 255, "
              f"{np.mean(np.asarray(jx[1]) != np.asarray(rgb)):.2%} of values differ")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("runs/base/model.npz"))
    p.add_argument("--train", type=Path, help="the module that defines the model (read before this parse)")
    p.add_argument("--data", type=Path, default=Path("data/eval.npz"), help="held-out play for the check (clips.py)")
    p.add_argument("--copy", type=Path, help="also copy play.onnx here")
    a = p.parse_args()
    docs, model, data = Path(__file__).resolve().parent, train.load_model(a.model), streams(a.data)
    pick = (pick_starts(data, 1) or [any_window(data)])[0]
    fp32(model)
    half_weights(model)
    export(model, docs / "play.onnx")
    print(f"play.onnx: {(docs / 'play.onnx').stat().st_size / 1e6:.1f} MB")
    if a.copy:
        a.copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(docs / "play.onnx", a.copy)
    check(model, docs / "play.onnx", data, pick)


if __name__ == "__main__":
    main()

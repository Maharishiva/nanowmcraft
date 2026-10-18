"""Export the world model to one ONNX graph for in-browser play (convert_web.py turns it into play.wgr).

    python docs/export_web.py --model runs/base/model.npz [--train path/to/train.py]
    python docs/convert_web.py

Writes docs/play.onnx (gitignored; weights stored fp16, computed in fp32). One call imagines the frame at window slot
s (0..31) after the frames in slots 0..s-1, as train._advance does while the window fills and, at s = 31, once it is
full (the player imagines every world's first frame from noise at slot 0, then fills the window): a pass over the 31
past slots caches every block's temporal keys and values and every past frame's summary; 8 Euler steps as in
train._flow_one then run only the new frame's 16 tokens against that cache, its temporal attention masked to slots
below s and itself (kmask), exact since spatial attention is per frame and temporal attention causal (about 7 GFLOP
per call, instead of 42 for eight full-window passes); one more new-frame pass at the context signal reads the reward,
the summary of slot s - 1 (prev) reads the done head, and the decoder renders the frame. Past slots from s on are
ignored, whatever they hold. A model with a step embedding (train.DF.step_embed) gets it on the Euler passes, as in
train._flow_one; a two-frame tokenizer (train.Encoder.temporal) encodes each frame of the check with its true
predecessor. --train picks the module that defines the model (default: the repo's train.py; its kernels.py is the one
next to it). Needs jax2onnx, onnx and onnxruntime on top of the repo's dependencies.

play.onnx   in   past f32 [1,31,4,4,32] (tau*z + (1-tau)*eps where tau > 0.5, else z), tau f32 [1,31],
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
import kernels  # noqa: E402  (the train module's)

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


def attend(q, k, v, causal=False, kmask=None):
    """kernels.attention off GPU, on split heads: q (..., Lq, heads, hd) with its bias added, k and v (..., Lk, heads,
    hd); kmask (1, Lk), added to the scores of every query."""
    s = jnp.einsum("...qhd,...khd->...hqk", q * q.shape[-1] ** -0.5, k)
    if causal:
        s = jnp.where(np.tril(np.ones(s.shape[-2:], bool)), s, kernels.MASKED)
    if kmask is not None:
        s = s + kmask.reshape((1,) * (s.ndim - 1) + kmask.shape[-1:])
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


def block(b, h, cond, cache=None, kmask=None):
    """train.Block on h (1, L, G, D). Without a cache, causal temporal attention over the L frames; with the past's
    temporal (k, v) (1, G, PAST, HEADS, D / HEADS), h is the new frame and attends to the past slots kmask keeps plus
    itself. Returns h and this block's temporal (k, v)."""
    ss, sh, sg, ts, th, tg, ms, mh, mg = jnp.split(b.mod(cond)[:, :, None, :], 9, -1)
    q, k, v = jnp.split(b.sqkv(norm(h, ss, sh)), 3, -1)
    h = h + sg * b.so(attend(heads(q + b.sq_bias[...]), heads(k), heads(v)).reshape(h.shape))
    q, k, v = jnp.split(b.tqkv(norm(h, ts, th)), 3, -1)
    q, k, v = (heads(x).swapaxes(1, 2) for x in (q + b.tq_bias[...], k, v))    # (1, G, L, HEADS, D / HEADS)
    kc, vc = (k, v) if cache is None else (jnp.concatenate([cache[0], k], 2), jnp.concatenate([cache[1], v], 2))
    h = h + tg * b.to(attend(q, kc, vc, cache is None, kmask).swapaxes(1, 2).reshape(h.shape))
    return h + mg * b.m2(gelu(b.m1(norm(h, ms, mh)))), (k, v)


def make_step(model):
    """(past, tau, act, noise, rnoise, pos, prev, kmask) -> (z, rgb, reward, done), the contract above, in plain jnp."""
    df = model.df
    aemb = lambda a: jnp.take(df.aemb.embedding[...], a, axis=0)   # a gather, equal to DF.action_embed's matmul
    step_embed = getattr(df, "step_embed", None)

    def embed(x, tau, a, tpos, delta=0.0):
        """train.DF.forward up to the blocks for frames x (1, L, GLAT, GLAT, TOK) with temporal position embeddings
        tpos (1, L, 1, D); delta is the Euler step size the model is told about (DF.forward's 8 * delta * step_embed,
        zero for the past)."""
        ae = aemb(a)
        h = df.inp(x.reshape(x.shape[:2] + (G, train.TOK))) + df.spos[...] + tpos + ae[:, :, None]
        cond = df.taulin(train._tau_embed(tau.reshape(-1), train.D)).reshape(ae.shape) + ae
        if step_embed is not None and delta:
            cond = cond + 8.0 * delta * step_embed[...]
        return h, cond

    def step(past, tau, act, noise, rnoise, pos, prev, kmask):
        h, cond = embed(past, tau, act[:, :PAST], df.tpos[...][:, :PAST])
        cache = []
        for b in df.blocks:
            h, kv = block(b, h, cond)
            cache.append(kv)
        # the summary of slot s - 1 and slot s's position embedding, as one-hot contractions in the form of the dense
        # layers (convert_web.py's 'abcd,de->abce')
        s_past = jnp.einsum("abcd,de->abce", prev[:, None, None], df.outn(h).mean(2)[0])[:, 0, 0]
        tpos_new = jnp.einsum("abcd,de->abce", pos[:, None, None], df.tpos[...][0, :train.WIN, 0]).swapaxes(1, 2)

        def new_frame(z, level, delta=0.0):
            h, cond = embed(z[:, None], jnp.full((1, 1), level), act[:, PAST:], tpos_new, delta)
            for b, kv in zip(df.blocks, cache):
                h = block(b, h, cond, kv, kmask)[0]
            h = df.outn(h)
            return df.out(h).reshape(z.shape), h.mean(2)[:, 0]

        z, dt = noise, 1.0 / train.STEPS
        for i in range(train.STEPS):                 # train._flow_one's update
            z = z + dt * (new_frame(z, i * dt, dt)[0] - z) / max(1.0 - i * dt, 1e-4)
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


def latents(model, data, picks):
    """Standardized latents (N, PAST, GLAT, GLAT, TOK) of the PAST frames up to each (env, t): encoded with one more
    frame in front and that one dropped, so a two-frame tokenizer sees every frame's true predecessor (pick_starts
    keeps the frame before the window in the same episode)."""
    frames = np.stack([data["frames"][t - PAST:t + 1, e] for e, t in picks])
    return np.asarray(train.encode(model, frames))[:, 1:]


SPECS = dict(past=((1, PAST, train.GLAT, train.GLAT, train.TOK), jnp.float32), tau=((1, PAST), jnp.float32),
             act=((1, train.WIN), jnp.int32), noise=((1, train.GLAT, train.GLAT, train.TOK), jnp.float32),
             rnoise=((1, train.GLAT, train.GLAT, train.TOK), jnp.float32), pos=((1, train.WIN), jnp.float32),
             prev=((1, PAST), jnp.float32), kmask=((1, train.WIN), jnp.float32))


def slot_inputs(s):
    """pos, prev and kmask for the new frame at window slot s (0..31)."""
    pos, prev = np.eye(train.WIN, dtype=np.float32)[s][None], np.zeros((1, PAST), np.float32)
    if s:
        prev[0, s - 1] = 1.0
    kmask = np.where((np.arange(train.WIN) < s) | (np.arange(train.WIN) == PAST), 0.0, kernels.MASKED)
    return dict(pos=pos, prev=prev, kmask=kmask.astype(np.float32)[None])


def half_weights(model, min_size=1024):
    """Round every matrix-sized weight to fp16 (in place), so that the fp16 file computes exactly what the JAX
    reference with these weights computes."""
    half = lambda w: w.astype(jnp.float16).astype(w.dtype) if w.size >= min_size else w
    for mod in (model.df, model.dec):
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
    """The new latent at window slot s as the training code computes it (train._advance: the window holds the frames
    of slots 0..s-1, train.DF.forward with its temporal cache and step size), from the same corrupted past and noise."""
    df, dt = model.df, 1.0 / train.STEPS
    seq = jnp.zeros((1, train.WIN) + past.shape[2:]).at[:, :s].set(past[:, :s])
    taus = jnp.zeros((1, train.WIN)).at[:, :s].set(tau[:, :s])
    aseq = jnp.zeros((1, train.WIN), jnp.int32).at[:, :s].set(act[:, :s]).at[:, s].set(act[:, PAST])
    caches = df.forward(seq, taus, aseq, cache=True)
    x1 = lambda z, level: df.forward(z[:, None], jnp.full((1, 1), level), aseq[:, s:s + 1], caches=caches, slot=s,
                                     delta=dt)[0][:, 0]
    z = noise
    for i in range(train.STEPS):
        z = z + dt * (x1(z, i * dt) - z) / max(1.0 - i * dt, 1e-4)
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
                   act=np.r_[np.where(np.arange(PAST) < s, acts, rng.integers(0, train.A_N, PAST)), 2].astype(np.int32)[None],
                   noise=rng.standard_normal(SPECS["noise"][0], np.float32),
                   rnoise=rng.standard_normal(SPECS["rnoise"][0], np.float32), **slot_inputs(s))
        out, jx = sess.run(None, inp), step(*inp.values())
        z = ref(inp["past"], inp["tau"], inp["act"], inp["noise"], s)
        rgb = train.decode(model, z)[0]              # its own jit, outside the ones above
        print(f"slot {s:2d}  onnx vs jax: " + "  ".join(f"{k} {diff(a, b):.1e}" for k, a, b in zip(["z", "rgb", "reward", "done"], out, jx))
              + f"   jax vs train's sampler: z {diff(jx[0], z):.1e}  rgb {diff(jx[1], rgb):.0f} / 255, "
              f"{np.mean(np.asarray(jx[1]) != np.asarray(rgb)):.2%} of values differ")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("runs/base/model.npz"))
    p.add_argument("--train", type=Path, help="the module that defines the model (read before this parse)")
    p.add_argument("--data", type=Path, default=Path("data/eval.npz"), help="held-out episodes for the check")
    p.add_argument("--copy", type=Path, help="also copy play.onnx here")
    a = p.parse_args()
    docs, model, data = Path(__file__).resolve().parent, train.load_model(a.model), dict(np.load(a.data))
    pick = pick_starts(data, 1)[0]
    half_weights(model)
    export(model, docs / "play.onnx")
    print(f"play.onnx: {(docs / 'play.onnx').stat().st_size / 1e6:.1f} MB")
    if a.copy:
        a.copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(docs / "play.onnx", a.copy)
    check(model, docs / "play.onnx", data, pick)


if __name__ == "__main__":
    main()

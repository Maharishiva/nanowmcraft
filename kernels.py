"""Pallas (Triton) kernels for train.py on NVIDIA GPUs, plain XLA elsewhere. Both read the activations in place, so a
world-model step needs no transposes and no reductions over the token axis (in XLA these took about 40% of the step).

attention: softmax attention from qkv (B, L, N, 3d), over the N tokens of each frame or causally over the L frames of
    each token position; one program per (frame or token position, head). The forward keeps the log-sum-exp, the
    backward recomputes the probabilities. Token counts that are not a power of two use masked rows (81 -> 128).
residual_norm: a gated residual add and the next adaLN-modulated RMS norm in one pass, one program per frame; the
    backward sums the per-frame gate, scale and shift gradients inside the program.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


def _params(elements): return plgpu.CompilerParams(num_warps=2 if elements <= 4096 else 4, num_stages=1)


def _tile(ids, col, w, width, N, T, temporal):
    """Index and row mask of the (T, w) tile at columns col:col+w of a program's rows (row width `width`)."""
    if temporal:  # the L frames of token position (b, n) in a (B, L, N * width) view
        b, n = ids
        return (b, slice(None), pl.ds(pl.multiple_of(n * width + col, w), w)), None
    rows = jax.lax.broadcasted_iota(jnp.int32, (T, w), 0) < N if T > N else None
    return (pl.ds(ids[0] * N, T), pl.ds(pl.multiple_of(col, w), w)), rows  # the N tokens of frame f: (B * L * N, width)


def _load(ref, at):
    return ref[at[0]] if at[1] is None else plgpu.load(ref.at[at[0]], mask=at[1], other=0.0)


def _store(ref, at, x):
    if at[1] is None:
        ref[at[0]] = x.astype(ref.dtype)
    else:
        plgpu.store(ref.at[at[0]], x.astype(ref.dtype), mask=at[1])


def _setup(d, heads, N, T, temporal):
    ids = (pl.program_id(0), pl.program_id(1)) if temporal else (pl.program_id(0),)
    h, hd = pl.program_id(len(ids)), d // heads
    at = partial(_tile, ids=ids, N=N, T=T, temporal=temporal)
    qkv = [at(col=j * d + h * hd, w=hd, width=3 * d) for j in range(3)]
    o, lse = at(col=h * hd, w=hd, width=d), at(col=h, w=1, width=heads)
    row, col = jax.lax.broadcasted_iota(jnp.int32, (T, T), 0), jax.lax.broadcasted_iota(jnp.int32, (T, T), 1)
    return qkv, o, lse, (row >= col) if temporal else (col < N), hd ** -0.5


def _scores(q, k, ok, scale):
    return jnp.where(ok, jnp.dot(q, k.T, preferred_element_type=jnp.float32) * scale, -1e30)


def _fwd_kernel(qkv_ref, o_ref, lse_ref, **dims):
    (qa, ka, va), oa, la, ok, scale = _setup(**dims)
    q, k, v = (_load(qkv_ref, a) for a in (qa, ka, va))
    s = _scores(q, k, ok, scale)
    m = jnp.max(s, axis=1, keepdims=True)
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=1, keepdims=True)
    _store(o_ref, oa, jnp.dot((p / l).astype(v.dtype), v, preferred_element_type=jnp.float32))
    _store(lse_ref, la, m + jnp.log(l))


def _bwd_kernel(qkv_ref, o_ref, do_ref, lse_ref, dqkv_ref, **dims):
    (qa, ka, va), oa, la, ok, scale = _setup(**dims)
    q, k, v = (_load(qkv_ref, a) for a in (qa, ka, va))
    o, do = _load(o_ref, oa), _load(do_ref, oa)
    p = jnp.exp(_scores(q, k, ok, scale) - _load(lse_ref, la))
    dp = jnp.dot(do, v.T, preferred_element_type=jnp.float32)
    delta = jnp.sum(do.astype(jnp.float32) * o.astype(jnp.float32), axis=1, keepdims=True)
    ds = (p * (dp - delta) * scale).astype(q.dtype)
    _store(dqkv_ref, qa, jnp.dot(ds, k, preferred_element_type=jnp.float32))
    _store(dqkv_ref, ka, jnp.dot(ds.T, q, preferred_element_type=jnp.float32))
    _store(dqkv_ref, va, jnp.dot(p.astype(do.dtype).T, do, preferred_element_type=jnp.float32))


def _attention_call(kernel, x, heads, temporal, out, *args):
    """kernel over the programs of x (B, L, N, 3d), on row views of x, args and the outputs out [(shape, dtype)]."""
    B, L, N, d3 = x.shape
    T = L if temporal else max(16, 1 << (N - 1).bit_length())
    view = lambda a: a.reshape((B, L, -1) if temporal else (B * L * N, -1))
    shapes = [jax.ShapeDtypeStruct((B, L, N * s[-1]) if temporal else (B * L * N, s[-1]), dt) for s, dt in out]
    kernel = partial(kernel, d=d3 // 3, heads=heads, N=N, T=T, temporal=temporal)
    grid = (B, N, heads) if temporal else (B * L, heads)
    res = pl.pallas_call(kernel, grid=grid, out_shape=shapes, compiler_params=_params(T * T))(*map(view, (x,) + args))
    return [r.reshape(s) for r, (s, _) in zip(res, out)]


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _attention(qkv, heads, temporal):
    return _attention_fwd(qkv, heads, temporal)[0]


def _attention_fwd(qkv, heads, temporal):
    B, L, N, d3 = qkv.shape
    out = [((B, L, N, d3 // 3), qkv.dtype), ((B, L, N, heads), jnp.float32)]
    o, lse = _attention_call(_fwd_kernel, qkv, heads, temporal, out)
    return o, (qkv, o, lse)


def _attention_bwd(heads, temporal, res, do):
    qkv, o, lse = res
    return tuple(_attention_call(_bwd_kernel, qkv, heads, temporal, [(qkv.shape, qkv.dtype)], o, do, lse))


_attention.defvjp(_attention_fwd, _attention_bwd)


def attention(qkv, heads, temporal):
    """qkv (B, L, N, 3d) -> (B, L, N, d): attention over the tokens of each frame, or causally over the frames."""
    if jax.default_backend() == "gpu":
        return _attention(qkv, heads, temporal)
    B, L, N, d3 = qkv.shape
    q, k, v = jnp.split(qkv.reshape(B, L, N, 3 * heads, -1), 3, axis=3)
    if temporal:
        q, k, v = (t.swapaxes(1, 2).reshape(B * N, L, heads, -1) for t in (q, k, v))
        return jax.nn.dot_product_attention(q, k, v, is_causal=True).reshape(B, N, L, -1).swapaxes(1, 2)
    return jax.nn.dot_product_attention(*(t.reshape(B * L, N, heads, -1) for t in (q, k, v))).reshape(B, L, N, -1)


def _frame(N):
    """The rows of this program's frame in a (B * L * N, d) view and its row in a (B * L, d) view."""
    return (pl.ds(pl.program_id(0) * N, N), slice(None)), (pl.ds(pl.program_id(0), 1), slice(None))


def _resnorm_fwd_kernel(h, g, y, s, t, res, out, *, N):
    rows, frame = _frame(N)
    x = (h[rows].astype(jnp.float32) + g[frame].astype(jnp.float32) * y[rows].astype(jnp.float32)).astype(res.dtype)
    res[rows] = x
    x = x.astype(jnp.float32)
    x = x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
    out[rows] = (x * (1 + s[frame].astype(jnp.float32)) + t[frame].astype(jnp.float32)).astype(out.dtype)


def _resnorm_bwd_kernel(x, g, y, s, dres, dout, dh, dy, dg, ds, dt, *, N):
    rows, frame = _frame(N)
    x, do = x[rows].astype(jnp.float32), dout[rows].astype(jnp.float32)
    r = jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
    xn, dn = x * r, do * (1 + s[frame].astype(jnp.float32))
    d = dres[rows].astype(jnp.float32) + r * (dn - xn * jnp.mean(dn * xn, -1, keepdims=True))
    dh[rows] = d.astype(dh.dtype)
    dy[rows] = (d * g[frame].astype(jnp.float32)).astype(dy.dtype)
    dg[frame] = jnp.sum(d * y[rows].astype(jnp.float32), 0, keepdims=True)
    ds[frame] = jnp.sum(do * xn, 0, keepdims=True)
    dt[frame] = jnp.sum(do, 0, keepdims=True)


def _resnorm_call(kernel, x, out, *args):
    """kernel per frame of x (B, L, N, d), on (rows, d) views of x and args; out [(rows per frame, dtype)]."""
    B, L, N, d = x.shape
    shapes = [jax.ShapeDtypeStruct((B * L * n, d), dt) for n, dt in out]
    res = pl.pallas_call(partial(kernel, N=N), grid=(B * L,), out_shape=shapes, compiler_params=_params(N * d))(
        *(a.reshape(-1, d) for a in (x,) + args))
    return [r.reshape((B, L, n, d)) for r, (n, _) in zip(res, out)]


@jax.custom_vjp
def _resnorm(h, g, y, s, t):
    return tuple(_resnorm_call(_resnorm_fwd_kernel, h, [(h.shape[2], h.dtype)] * 2, g, y, s, t))


def _resnorm_fwd(h, g, y, s, t):
    res, out = _resnorm(h, g, y, s, t)
    return (res, out), (res, g, y, s)


def _resnorm_bwd(saved, grads):
    res, g, y, s = saved
    out = [(res.shape[2], res.dtype)] * 2 + [(1, jnp.float32)] * 3
    dh, dy, dg, ds, dt = _resnorm_call(_resnorm_bwd_kernel, res, out, g, y, s, *grads)
    return dh, dg.astype(g.dtype), dy, ds.astype(s.dtype), dt.astype(s.dtype)


_resnorm.defvjp(_resnorm_fwd, _resnorm_bwd)


def residual_norm(h, g, y, s, t):
    """h + g * y and its RMS norm times (1 + s) plus t, for per-frame g, s, t (B, L, 1, d), both in h's dtype."""
    if jax.default_backend() == "gpu":
        return _resnorm(h, g, y, s, t)
    res = (h + g * y).astype(h.dtype)
    x = res.astype(jnp.float32)
    return res, (x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6) * (1 + s) + t).astype(h.dtype)

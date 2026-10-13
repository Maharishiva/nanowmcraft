"""Pallas (Triton) kernels for the dynamics transformer on NVIDIA GPUs.

Every kernel reads the activations in their natural (B, L, N, width) layout: a program handles a few frames (the
N tokens of each frame (b, l) are rows of a flat (B*L*N, width) view) or one token position (the L frames of
position (b, n) are a column slice of a (B, L, N*width) view). Token counts that are not a multiple of 16 are
handled by masked loads and stores, so 20 tokens cost a 32-row tile and 81 tokens a 128-row tile, no padding copies. Sequences are at most
32 long, so each program keeps its whole tile in registers. On CPU the public functions fall back to plain XLA
implementations of the same math.

attention: per-head softmax attention (spatial: all tokens of a frame; temporal: causal over the frames) with the
query bias applied in the kernel and fp32 scores; the forward writes the log-sum-exp, the backward recomputes the
probabilities and writes one (B, L, N, 3d) gradient tile. Key and value biases are exactly redundant (a key bias
cancels in the softmax, a value bias folds into the output projection bias), so the fused block has none.
norm_mod: adaLN modulation, RMS-normalising in fp32; the backward computes the per-frame scale and shift gradients
inside the program instead of as reductions over the token axis.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import jax.experimental.pallas.triton as plgpu

MASKED = -1e30
EPS = 1e-6
ATTN_FRAMES = 1
NORM_FRAMES = 1
ELEMENTS = 8192
CHUNK = 64

def _call(kernel, grid, out_shape, num_warps=2):
    if jax.default_backend() == "cpu":
        return pl.pallas_call(kernel, grid=grid, out_shape=out_shape, interpret=True)
    return pl.pallas_call(kernel, grid=grid, out_shape=out_shape,
                          compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1))

def _warps(elements):
    """Warps for a tile of `elements` fp32 values: about 64 values per thread, a power of two between 2 and 8."""
    warps = 2
    while warps < 8 and warps * 4096 < elements:
        warps *= 2
    return warps

def _tile(N):
    """Rows of a spatial tile: the next power of two above N (the Triton lowering wants power-of-two arrays)."""
    return max(16, 1 << (N - 1).bit_length())

def _frames(count, frames):
    while count % frames:
        frames //= 2
    return frames

def _index(temporal, ids, N, T, c0, w, width):
    """Indexer and row mask of one tile: (T, w) at columns c0:c0+w of an array whose row width is `width`; ids is
    (frame,) for the rows of one frame or (b, n) for the frames of one token position."""
    if temporal:
        b, n = ids
        return (b, slice(None), pl.ds(pl.multiple_of(n * width + c0, 16), w)), None
    (frame,) = ids
    mask = None if T == N else jax.lax.broadcasted_iota(jnp.int32, (T, w), 0) < N
    cols = pl.ds(c0, w) if isinstance(c0, int) else pl.ds(pl.multiple_of(c0, 16), w)
    return (pl.ds(frame * N, T), cols), mask

def _load(ref, idx, mask):
    return ref[idx] if mask is None else plgpu.load(ref.at[idx], mask=mask, other=0.0)

def _store(ref, idx, mask, val):
    if mask is None:
        ref[idx] = val
    else:
        plgpu.store(ref.at[idx], val, mask=mask)

def _scores_mask(T, n_valid, causal):
    row = jax.lax.broadcasted_iota(jnp.int32, (T, T), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (T, T), 1)
    ok = col < n_valid
    return ok & (row >= col) if causal else ok

def _view(x, temporal):
    B, L, N, w = x.shape
    return x.reshape((B, L, N * w)) if temporal else x.reshape((B * L * N, w))

def _programs(temporal, frames):
    if temporal:
        return [(pl.program_id(0), pl.program_id(1))]
    return [(pl.program_id(0) * frames + f,) for f in range(frames)]

def _head(temporal):
    return pl.program_id(2 if temporal else 1)

def _lse_index(temporal, ids, N, T, h, heads):
    """The (T, 1) log-sum-exp column of head h."""
    if temporal:
        b, n = ids
        return (b, slice(None), pl.ds(n * heads + h, 1)), None
    (frame,) = ids
    mask = None if T == N else jax.lax.broadcasted_iota(jnp.int32, (T, 1), 0) < N
    return (pl.ds(frame * N, T), pl.ds(h, 1)), mask

def _query(qkv_ref, bias_ref, idx, mask, c0, hd):
    x = _load(qkv_ref, idx, mask)
    cols = pl.ds(c0, hd) if isinstance(c0, int) else pl.ds(pl.multiple_of(c0, 16), hd)
    return (x.astype(jnp.float32) + bias_ref[:, cols].astype(jnp.float32)).astype(x.dtype)

def _attn_fwd_kernel(qkv_ref, bias_ref, o_ref, lse_ref, *, d, heads, N, T, temporal, frames):
    hd = d // heads
    scale = hd ** -0.5
    ok = _scores_mask(T, T if temporal else N, temporal)
    for ids in _programs(temporal, frames):
        for h in [_head(temporal)]:
            qi, qm = _index(temporal, ids, N, T, h * hd, hd, 3 * d)
            ki, _ = _index(temporal, ids, N, T, d + h * hd, hd, 3 * d)
            vi, _ = _index(temporal, ids, N, T, 2 * d + h * hd, hd, 3 * d)
            q = _query(qkv_ref, bias_ref, qi, qm, h * hd, hd)
            k = _load(qkv_ref, ki, qm)
            v = _load(qkv_ref, vi, qm)
            s = jnp.where(ok, jnp.dot(q, k.T, preferred_element_type=jnp.float32) * scale, MASKED)
            m = jnp.max(s, axis=1, keepdims=True)
            p = jnp.exp(s - m)
            l = jnp.sum(p, axis=1, keepdims=True)
            o = jnp.dot((p / l).astype(v.dtype), v, preferred_element_type=jnp.float32)
            oi, om = _index(temporal, ids, N, T, h * hd, hd, d)
            _store(o_ref, oi, om, o.astype(o_ref.dtype))
            li, lm = _lse_index(temporal, ids, N, T, h, heads)
            _store(lse_ref, li, lm, m + jnp.log(l))

def _attn_bwd_kernel(qkv_ref, bias_ref, o_ref, do_ref, lse_ref, dqkv_ref, *, d, heads, N, T, temporal, frames):
    hd = d // heads
    scale = hd ** -0.5
    ok = _scores_mask(T, T if temporal else N, temporal)
    for ids in _programs(temporal, frames):
        for h in [_head(temporal)]:
            qi, qm = _index(temporal, ids, N, T, h * hd, hd, 3 * d)
            ki, _ = _index(temporal, ids, N, T, d + h * hd, hd, 3 * d)
            vi, _ = _index(temporal, ids, N, T, 2 * d + h * hd, hd, 3 * d)
            oi, om = _index(temporal, ids, N, T, h * hd, hd, d)
            li, lm = _lse_index(temporal, ids, N, T, h, heads)
            q = _query(qkv_ref, bias_ref, qi, qm, h * hd, hd)
            k = _load(qkv_ref, ki, qm)
            v = _load(qkv_ref, vi, qm)
            o = _load(o_ref, oi, om)
            do = _load(do_ref, oi, om)
            lse = _load(lse_ref, li, lm)
            s = jnp.where(ok, jnp.dot(q, k.T, preferred_element_type=jnp.float32) * scale, MASKED)
            p = jnp.exp(s - lse)
            delta = jnp.sum(do.astype(jnp.float32) * o.astype(jnp.float32), axis=1, keepdims=True)
            dp = jnp.dot(do, v.T, preferred_element_type=jnp.float32)
            ds = (p * (dp - delta) * scale).astype(q.dtype)
            _store(dqkv_ref, qi, qm, jnp.dot(ds, k, preferred_element_type=jnp.float32).astype(dqkv_ref.dtype))
            _store(dqkv_ref, ki, qm, jnp.dot(ds.T, q, preferred_element_type=jnp.float32).astype(dqkv_ref.dtype))
            _store(dqkv_ref, vi, qm, jnp.dot(p.astype(do.dtype).T, do, preferred_element_type=jnp.float32).astype(dqkv_ref.dtype))

def _attn_layout(qkv, heads, temporal):
    B, L, N, d3 = qkv.shape
    if temporal:
        return dict(d=d3 // 3, N=N, T=L, temporal=True, frames=1), (B, N, heads)
    frames = _frames(B * L, ATTN_FRAMES)
    return dict(d=d3 // 3, N=N, T=_tile(N), temporal=False, frames=frames), (B * L // frames, heads)

def _attn_fwd(qkv, bias, heads, temporal):
    B, L, N, d3 = qkv.shape
    d = d3 // 3
    layout, grid = _attn_layout(qkv, heads, temporal)
    kernel = partial(_attn_fwd_kernel, heads=heads, **layout)
    out = (jax.ShapeDtypeStruct(_view(jnp.empty((B, L, N, d), qkv.dtype), temporal).shape, qkv.dtype),
           jax.ShapeDtypeStruct(_view(jnp.empty((B, L, N, heads), jnp.float32), temporal).shape, jnp.float32))
    o, lse = _call(kernel, grid, out, _warps(layout["T"] * layout["T"]))(_view(qkv, temporal), bias.reshape((1, d)))
    return o.reshape((B, L, N, d)), lse.reshape((B, L, N, heads))

def _attn_bwd(qkv, bias, o, do, lse, heads, temporal):
    d = qkv.shape[-1] // 3
    layout, grid = _attn_layout(qkv, heads, temporal)
    kernel = partial(_attn_bwd_kernel, heads=heads, **layout)
    out = jax.ShapeDtypeStruct(_view(qkv, temporal).shape, qkv.dtype)
    dqkv = _call(kernel, grid, out, _warps(layout["T"] * layout["T"]))(_view(qkv, temporal), bias.reshape((1, d)), _view(o, temporal),
                                    _view(do, temporal), _view(lse, temporal))
    return dqkv.reshape(qkv.shape)

@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _attention_pallas(qkv, qbias, heads, temporal):
    """qkv (B, L, N, 3d) and the query bias (d,) -> (B, L, N, d): softmax attention over the tokens of each frame,
    or causally over the frames of each token position."""
    return _attn_fwd(qkv, qbias, heads, temporal)[0]

def _attention_fwd(qkv, qbias, heads, temporal):
    o, lse = _attn_fwd(qkv, qbias, heads, temporal)
    return o, (qkv, qbias, o, lse)

def _attention_bwd(heads, temporal, res, do):
    qkv, qbias, o, lse = res
    dqkv = _attn_bwd(qkv, qbias, o, do, lse, heads, temporal)
    dq = dqkv[..., :qkv.shape[-1] // 3]
    return dqkv, jnp.sum(dq.astype(jnp.float32), axis=(0, 1, 2)).astype(qbias.dtype)

_attention_pallas.defvjp(_attention_fwd, _attention_bwd)

def _chunk(T, d):
    """Columns per pass so that a (T, chunk) fp32 tile stays about 4096 values."""
    return min(d, max(CHUNK, 4096 // T))

def _chunks(T, d):
    chunk = _chunk(T, d)
    return [(c, min(chunk, d - c)) for c in range(0, d, chunk)]

def _row_chunks(refs, frame, N, T, d):
    """Load the frame's rows in column chunks so that the live tiles stay small; a single chunk is kept for
    the second pass, several are re-read (from cache) when needed."""
    chunks = _chunks(T, d)
    loaded = []
    for c0, w in chunks:
        idx, mask = _index(False, (frame,), N, T, c0, w, d)
        loaded.append((c0, w, idx, mask, [_load(ref, idx, mask).astype(jnp.float32) for ref in refs]))
    if len(chunks) == 1:
        return loaded, loaded
    again = [(c0, w, idx, mask, None) for c0, w, idx, mask, _ in loaded]
    return loaded, again

def _reload(refs, entry):
    c0, w, idx, mask, tiles = entry
    if tiles is None:
        tiles = [_load(ref, idx, mask).astype(jnp.float32) for ref in refs]
    return c0, w, idx, mask, tiles

def _norm_fwd_kernel(x_ref, s_ref, t_ref, y_ref, *, d, N, T, frames):
    for (frame,) in _programs(False, frames):
        first, second = _row_chunks([x_ref], frame, N, T, d)
        acc = jnp.zeros((T, 1), jnp.float32)
        for _, _, _, _, (x,) in first:
            acc += jnp.sum(x * x, axis=1, keepdims=True)
        r = jax.lax.rsqrt(acc / d + EPS)
        for entry in second:
            c0, w, idx, mask, (x,) = _reload([x_ref], entry)
            s = s_ref[pl.ds(frame, 1), pl.ds(c0, w)].astype(jnp.float32)
            t = t_ref[pl.ds(frame, 1), pl.ds(c0, w)].astype(jnp.float32)
            _store(y_ref, idx, mask, (x * r * (1.0 + s) + t).astype(y_ref.dtype))

def _norm_bwd_kernel(x_ref, s_ref, dy_ref, dx_ref, ds_ref, dt_ref, *, d, N, T, frames):
    for (frame,) in _programs(False, frames):
        first, second = _row_chunks([x_ref, dy_ref], frame, N, T, d)
        acc = jnp.zeros((T, 1), jnp.float32)
        proj = jnp.zeros((T, 1), jnp.float32)
        for c0, w, _, _, (x, dy) in first:
            g = 1.0 + s_ref[pl.ds(frame, 1), pl.ds(c0, w)].astype(jnp.float32)
            acc += jnp.sum(x * x, axis=1, keepdims=True)
            proj += jnp.sum(dy * g * x, axis=1, keepdims=True)
        r = jax.lax.rsqrt(acc / d + EPS)
        m = proj * r / d
        for entry in second:
            c0, w, idx, mask, (x, dy) = _reload([x_ref, dy_ref], entry)
            g = 1.0 + s_ref[pl.ds(frame, 1), pl.ds(c0, w)].astype(jnp.float32)
            xh = x * r
            _store(dx_ref, idx, mask, (r * (dy * g - xh * m)).astype(dx_ref.dtype))
            ds_ref[pl.ds(frame, 1), pl.ds(c0, w)] = jnp.sum(dy * xh, axis=0, keepdims=True)
            dt_ref[pl.ds(frame, 1), pl.ds(c0, w)] = jnp.sum(dy, axis=0, keepdims=True)

def _norm_layout(h):
    B, L, N, d = h.shape
    frames = _frames(B * L, NORM_FRAMES)
    return dict(d=d, N=N, T=_tile(N), frames=frames), (B * L // frames,), _warps(_tile(N) * _chunk(_tile(N), d))

@jax.custom_vjp
def _norm_mod_pallas(h, scale, shift):
    """h (B, L, N, d), scale and shift (B, L, 1, d) -> rmsnorm(h) * (1 + scale) + shift in h's dtype."""
    B, L, N, d = h.shape
    layout, grid, warps = _norm_layout(h)
    y = _call(partial(_norm_fwd_kernel, **layout), grid, jax.ShapeDtypeStruct((B * L * N, d), h.dtype), warps)(
        h.reshape((B * L * N, d)), scale.reshape((B * L, d)), shift.reshape((B * L, d)))
    return y.reshape(h.shape)

def _norm_mod_fwd(h, scale, shift):
    return norm_mod(h, scale, shift), (h, scale)

def _norm_mod_bwd(res, dy):
    h, scale = res
    B, L, N, d = h.shape
    layout, grid, warps = _norm_layout(h)
    out = (jax.ShapeDtypeStruct((B * L * N, d), h.dtype), jax.ShapeDtypeStruct((B * L, d), jnp.float32),
           jax.ShapeDtypeStruct((B * L, d), jnp.float32))
    dx, ds, dt = _call(partial(_norm_bwd_kernel, **layout), grid, out, warps)(
        h.reshape((B * L * N, d)), scale.reshape((B * L, d)), dy.reshape((B * L * N, d)))
    return dx.reshape(h.shape), ds.reshape((B, L, 1, d)).astype(scale.dtype), dt.reshape((B, L, 1, d)).astype(scale.dtype)

_norm_mod_pallas.defvjp(_norm_mod_fwd, _norm_mod_bwd)

def _gate_bwd_kernel(dout_ref, g_ref, y_ref, dy_ref, dg_ref, *, d, N, T):
    frame = pl.program_id(0)
    for c0, w in _chunks(T, d):
        idx, mask = _index(False, (frame,), N, T, c0, w, d)
        dout = _load(dout_ref, idx, mask).astype(jnp.float32)
        y = _load(y_ref, idx, mask).astype(jnp.float32)
        g = g_ref[pl.ds(frame, 1), pl.ds(c0, w)].astype(jnp.float32)
        _store(dy_ref, idx, mask, (dout * g).astype(dy_ref.dtype))
        dg_ref[pl.ds(frame, 1), pl.ds(c0, w)] = jnp.sum(dout * y, axis=0, keepdims=True)

@jax.custom_vjp
def _gated_add_pallas(h, g, y):
    """h + g * y for a per-frame gate g (B, L, 1, d); the backward computes the gate gradient per frame in one
    pass over the activations instead of as a reduction over the token axis."""
    return h + g.astype(h.dtype) * y

def _gated_add_fwd(h, g, y):
    return gated_add(h, g, y), (g, y)

def _gated_add_bwd(res, dout):
    g, y = res
    B, L, N, d = y.shape
    T = _tile(N)
    out = (jax.ShapeDtypeStruct((B * L * N, d), y.dtype), jax.ShapeDtypeStruct((B * L, d), jnp.float32))
    dy, dg = _call(partial(_gate_bwd_kernel, d=d, N=N, T=T), (B * L,), out, _warps(T * _chunk(T, d)))(
        dout.reshape((B * L * N, d)), g.reshape((B * L, d)), y.reshape((B * L * N, d)))
    return dout, dg.reshape((B, L, 1, d)).astype(g.dtype), dy.reshape(y.shape)

_gated_add_pallas.defvjp(_gated_add_fwd, _gated_add_bwd)

GELU_C = 0.7978845608028654

def _gelu_parts(x):
    """tanh of the gelu argument and the gelu derivative."""
    t = jnp.tanh(GELU_C * (x + 0.044715 * x * x * x))
    return t, 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * GELU_C * (1.0 + 3 * 0.044715 * x * x)

def _flat(ref, i, total):
    idx = (pl.ds(pl.multiple_of(i * ELEMENTS, ELEMENTS), ELEMENTS),)
    mask = None if total % ELEMENTS == 0 else jax.lax.broadcasted_iota(jnp.int32, (ELEMENTS,), 0) + i * ELEMENTS < total
    return idx, mask

def _gelu_fwd_kernel(x_ref, y_ref, *, total):
    idx, mask = _flat(x_ref, pl.program_id(0), total)
    x = _load(x_ref, idx, mask).astype(jnp.float32)
    t, _ = _gelu_parts(x)
    _store(y_ref, idx, mask, (0.5 * x * (1.0 + t)).astype(y_ref.dtype))

def _gelu_bwd_kernel(x_ref, dout_ref, dx_ref, *, total):
    idx, mask = _flat(x_ref, pl.program_id(0), total)
    x = _load(x_ref, idx, mask).astype(jnp.float32)
    dout = _load(dout_ref, idx, mask).astype(jnp.float32)
    _, dgelu = _gelu_parts(x)
    _store(dx_ref, idx, mask, (dout * dgelu).astype(dx_ref.dtype))

@jax.custom_vjp
def _gelu_pallas(x):
    """The tanh gelu, elementwise in fp32; the backward recomputes from the input."""
    total = x.size
    y = _call(partial(_gelu_fwd_kernel, total=total), (-(-total // ELEMENTS),), jax.ShapeDtypeStruct((total,), x.dtype),
              _warps(ELEMENTS))(x.reshape(-1))
    return y.reshape(x.shape)

def _gelu_fwd(x):
    return gelu(x), x

def _gelu_bwd(x, dout):
    total = x.size
    dx = _call(partial(_gelu_bwd_kernel, total=total), (-(-total // ELEMENTS),), jax.ShapeDtypeStruct((total,), x.dtype),
               _warps(ELEMENTS))(x.reshape(-1), dout.reshape(-1))
    return (dx.reshape(x.shape),)

_gelu_pallas.defvjp(_gelu_fwd, _gelu_bwd)

def _attention_xla(qkv, qbias, heads, temporal):
    B, L, N, d3 = qkv.shape
    qkv = qkv.at[..., :d3 // 3].add(qbias.astype(qkv.dtype))
    q, k, v = [x.reshape((B, L, N, heads, d3 // 3 // heads)) for x in jnp.split(qkv, 3, -1)]
    q = q * (q.shape[-1] ** -0.5)
    if temporal:
        s = jnp.einsum("blnhd,bmnhd->bnhlm", q, k, preferred_element_type=jnp.float32)
        s = jnp.where(jnp.arange(L)[:, None] >= jnp.arange(L)[None, :], s, MASKED)
        o = jnp.einsum("bnhlm,bmnhd->blnhd", jax.nn.softmax(s, -1).astype(v.dtype), v)
    else:
        s = jnp.einsum("blnhd,blmhd->blhnm", q, k, preferred_element_type=jnp.float32)
        o = jnp.einsum("blhnm,blmhd->blnhd", jax.nn.softmax(s, -1).astype(v.dtype), v)
    return o.reshape((B, L, N, d3 // 3))

def _norm_mod_xla(h, scale, shift):
    x = h.astype(jnp.float32)
    x = x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + EPS)
    return (x * (1.0 + scale.astype(jnp.float32)) + shift.astype(jnp.float32)).astype(h.dtype)

def _on_gpu():
    return jax.default_backend() == "gpu"

def attention(qkv, qbias, heads, temporal):
    """qkv (B, L, N, 3d) and the query bias (d,) -> (B, L, N, d): softmax attention over the tokens of each frame,
    or causally over the frames of each token position."""
    return (_attention_pallas if _on_gpu() else _attention_xla)(qkv, qbias, heads, temporal)

def norm_mod(h, scale, shift):
    """h (B, L, N, d), scale and shift (B, L, 1, d) -> rmsnorm(h) * (1 + scale) + shift in h's dtype."""
    return (_norm_mod_pallas if _on_gpu() else _norm_mod_xla)(h, scale, shift)

def gated_add(h, g, y):
    """h + g * y for a per-frame gate g (B, L, 1, d)."""
    return _gated_add_pallas(h, g, y) if _on_gpu() else h + g.astype(h.dtype) * y

def gelu(x):
    """The tanh gelu, elementwise in fp32."""
    return _gelu_pallas(x) if _on_gpu() else jax.nn.gelu(x.astype(jnp.float32)).astype(x.dtype)

def _resnorm_fwd_kernel(h, g, y, scale, shift, res, norm, *, N, d):
    f = pl.program_id(0)
    ix = (pl.ds(f * N, N), slice(None))
    gate = g[pl.ds(f, 1), :].astype(jnp.float32)
    value = y[ix].astype(jnp.float32)
    x = (h[ix].astype(jnp.float32) + gate * value).astype(h.dtype)
    res[ix] = x
    x = x.astype(jnp.float32)
    xh = x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + EPS)
    z = xh * (1 + scale[pl.ds(f, 1), :].astype(jnp.float32)) + shift[pl.ds(f, 1), :].astype(jnp.float32)
    norm[ix] = z.astype(norm.dtype)

def _resnorm_bwd_kernel(h, g, y, scale, dres, dnorm, dh, dy, dg, ds, dt, *, N, d):
    f = pl.program_id(0)
    ix = (pl.ds(f * N, N), slice(None))
    x, dn = h[ix].astype(jnp.float32), dnorm[ix].astype(jnp.float32)
    r = jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + EPS)
    xh = x * r
    dn_scaled = dn * (1 + scale[pl.ds(f, 1), :].astype(jnp.float32))
    dx = (r * (dn_scaled - xh * jnp.mean(dn_scaled * xh, -1, keepdims=True))).astype(h.dtype)
    dx = (dres[ix] + dx).astype(h.dtype)
    dh[ix] = dx
    dy[ix] = (dx.astype(jnp.float32) * g[pl.ds(f, 1), :].astype(jnp.float32)).astype(dy.dtype)
    dg[pl.ds(f, 1), :] = jnp.sum(dx.astype(jnp.float32) * y[ix].astype(jnp.float32), 0, keepdims=True)
    ds[pl.ds(f, 1), :] = jnp.sum(dn * xh, 0, keepdims=True)
    dt[pl.ds(f, 1), :] = jnp.sum(dn, 0, keepdims=True)

@jax.custom_vjp
def _resnorm_pallas(h, g, y, scale, shift):
    B, L, N, d = h.shape
    out = jax.ShapeDtypeStruct((B * L * N, d), h.dtype)
    args = (h.reshape(-1, d), g.reshape(-1, d), y.reshape(-1, d), scale.reshape(-1, d), shift.reshape(-1, d))
    result = _call(partial(_resnorm_fwd_kernel, N=N, d=d), (B * L,), (out, out), _warps(N * d))(*args)
    return tuple(v.reshape(h.shape) for v in result)

def _resnorm_fwd(h, g, y, scale, shift):
    out = _resnorm_pallas(h, g, y, scale, shift)
    return out, (out[0], g, y, scale)

def _resnorm_bwd(saved, grad):
    h, g, y, scale = saved
    B, L, N, d = h.shape
    row = jax.ShapeDtypeStruct((B * L * N, d), h.dtype)
    mod = jax.ShapeDtypeStruct((B * L, d), jnp.float32)
    args = tuple(v.reshape(-1, d) for v in (h, g, y, scale, *grad))
    out = _call(partial(_resnorm_bwd_kernel, N=N, d=d), (B * L,), (row, row, mod, mod, mod), _warps(N * d))(*args)
    dh, dy, dg, ds, dt = out
    return dh.reshape(h.shape), dg.reshape(g.shape).astype(g.dtype), dy.reshape(y.shape), ds.reshape(scale.shape).astype(scale.dtype), dt.reshape(scale.shape).astype(scale.dtype)

_resnorm_pallas.defvjp(_resnorm_fwd, _resnorm_bwd)

def residual_norm(h, g, y, scale, shift):
    """Fused gated residual plus next modulated RMSNorm; fallback supports arbitrary layouts."""
    if _on_gpu() and h.shape[2] == 16:
        return _resnorm_pallas(h, g, y, scale, shift)
    res = gated_add(h, g, y)
    return res, norm_mod(res, scale, shift)


def attention_one(qkv, qbias, cache, slot, heads):
    """One temporal query against fixed past K/V plus its own current K/V; causal and differentiable."""
    B, N, width = qkv.shape
    d, hd = qbias.shape[0], qbias.shape[0] // heads
    kw = (width - d) // 2
    expand = lambda x, prefix: jnp.repeat(x.reshape(prefix + (kw // hd, hd)), d // kw, axis=-2)
    q = (qkv[..., :d].astype(jnp.float32) + qbias).astype(qkv.dtype).reshape((B, N, heads, hd))
    kn, vn = [expand(x, (B, N)) for x in jnp.split(qkv[..., d:], 2, -1)]
    k, v = [expand(x, cache.shape[:3]) for x in jnp.split(cache, 2, -1)]
    scores = jnp.einsum("bnhd,blnhd->bnhl", q, k, preferred_element_type=jnp.float32)
    current = jnp.sum(q.astype(jnp.float32) * kn.astype(jnp.float32), -1)
    index = jnp.arange(cache.shape[1])
    scores = jnp.where(index == slot, current[..., None], scores) * hd ** -.5
    p = jax.nn.softmax(jnp.where(index <= slot, scores, MASKED), -1).astype(v.dtype)
    old = jnp.einsum("bnhl,blnhd->bnhd", jnp.where(index != slot, p, 0), v, preferred_element_type=jnp.float32)
    out = old + p[..., slot, None].astype(jnp.float32) * vn.astype(jnp.float32)
    return out.reshape((B, N, d)).astype(qkv.dtype)

def polar_express(g):
    'Five-step Polar Express in bf16, with fp32 normalization and optimizer state.'
    coeffs = ((8.15655, -22.48329, 15.87877), (4.04293, -2.80892, 0.500018),
              (3.89167, -2.77248, 0.506065), (3.28575, -2.36813, 0.464490), (2.34654, -1.70978, 0.423236))
    x = (g / (1.02 * jnp.linalg.norm(g, axis=(-2, -1), keepdims=True) + 1e-7)).astype(jnp.bfloat16)
    tall = x.shape[-2] > x.shape[-1]
    x = jnp.swapaxes(x, -2, -1) if tall else x
    for a, b, c in coeffs:
        aa = x @ jnp.swapaxes(x, -2, -1)
        x = a * x + (b * aa + c * (aa @ aa)) @ x
    return (jnp.swapaxes(x, -2, -1) if tall else x).astype(jnp.float32)

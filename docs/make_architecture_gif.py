"""The architecture GIF: frames -> encoder -> latent tokens over time -> world model (x6) -> decoder, animated from a
real rollout of a checkpoint. Each imagined frame is decoded after every one of its 8 flow-matching Euler steps.

    python docs/make_architecture_gif.py --model runs/base/model.npz --out docs/assets/architecture.gif
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))
NCTX, NGEN, STEPS = 8, 4, 8


def rollout(model_path, data_path, seed=0):
    """Encode 8 real frames of the craft window used by assets/play.gif, then imagine 4 frames, keeping every Euler step."""
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import train
    from make_play_gif import rank_windows
    train.CDT = jnp.float32
    train.MP = dict(dtype=jnp.float32, param_dtype=jnp.float32)
    data = dict(np.load(data_path))
    start, env = rank_windows(data, "craft")[0]
    frames = data["frames"][start:start + NCTX + NGEN, env]
    actions = data["actions"][start - 1:start + NCTX + NGEN - 1, env].astype(np.int32)
    model = train.load_model(model_path)
    gdef, state = nnx.split(model.df)
    ctx = train.encode(model, jnp.asarray(frames[None, :NCTX]))[0]
    win = model.win
    seq = jnp.zeros((1, win) + ctx.shape[1:]).at[0, :NCTX].set(ctx)
    aseq = jnp.zeros((1, win), jnp.int32).at[0, :NCTX].set(jnp.asarray(actions[:NCTX]))

    @jax.jit
    def predict(state, seq, taus, aseq, s, z, tau):
        return nnx.merge(gdef, state)(seq.at[:, s].set(z), taus.at[:, s].set(jnp.full((1,), tau)), aseq)[:, s]

    key, dt, zs_all = jax.random.PRNGKey(seed), 1.0 / STEPS, []
    for f in range(NGEN):
        n = NCTX + f
        aseq = aseq.at[0, n].set(int(actions[n]))
        taus = jnp.where(jnp.arange(win) < n, jnp.where(jnp.arange(win) < NCTX, 1.0, train.CTX_SIGNAL), 0.0)[None]
        key, kc, kz = jax.random.split(key, 3)
        t5 = taus[..., None, None, None]
        seqc = jnp.where(t5 > 0.5, t5 * seq + (1.0 - t5) * jax.random.normal(kc, seq.shape), seq)
        z, zs = jax.random.normal(kz, (1,) + seq.shape[-3:]), []
        for k in range(STEPS):
            zs.append(np.asarray(z[0]))
            z = z + dt * (predict(state, seqc, taus, aseq, n, z, k * dt) - z) / max(1.0 - k * dt, 1e-4)
        zs.append(np.asarray(z[0]))
        seq = seq.at[:, n].set(z)
        zs_all.append(zs)
    Z = np.array(zs_all)
    zdec = np.stack([np.asarray(train.decode(model, jnp.asarray(Z[f]))) for f in range(NGEN)])
    return frames, actions, np.asarray(ctx), Z, zdec


ARGS = argparse.ArgumentParser()
ARGS.add_argument("--model", type=Path, required=True)
ARGS.add_argument("--data", type=Path, default=ROOT / "data" / "eval.npz")
ARGS.add_argument("--out", type=Path, default=ROOT / "docs" / "assets" / "architecture.gif")
ARGS.add_argument("--seed", type=int, default=0)
ARGS = ARGS.parse_args()
FRAMES, ACTIONS, CTX_LAT, Z, ZDEC = rollout(ARGS.model, ARGS.data, ARGS.seed)

# ---------------------------------------------------------------- palette and font (from docs/make_play_gif.py)
BG, FG, DIM, ORANGE = (10, 10, 10), (240, 240, 238), (106, 106, 106), (255, 106, 19)
P1, P2, P3 = (18, 18, 20), (28, 28, 30), (58, 58, 62)
TXT = (196, 196, 194)
NAMES = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE STONE", "PLACE TABLE", "PLACE FURNACE",
         "PLACE PLANT", "WOOD PICKAXE", "STONE PICKAXE", "IRON PICKAXE", "WOOD SWORD", "STONE SWORD", "IRON SWORD"]
KEYS = ["Q", "A", "D", "W", "S", "SPACE", "TAB", "R", "T", "F", "P", "1", "2", "3", "4", "5", "6"]
FONT = {
    "A": "010 101 111 101 101", "B": "110 101 110 101 110", "C": "011 100 100 100 011", "D": "110 101 101 101 110",
    "E": "111 100 110 100 111", "F": "111 100 110 100 100", "G": "011 100 101 101 011", "H": "101 101 111 101 101",
    "I": "111 010 010 010 111", "J": "001 001 001 101 010", "K": "101 101 110 101 101", "L": "100 100 100 100 111",
    "M": "101 111 111 101 101", "N": "110 101 101 101 101", "O": "010 101 101 101 010", "P": "110 101 110 100 100",
    "Q": "010 101 101 110 011", "R": "110 101 110 101 101", "S": "011 100 010 001 110", "T": "111 010 010 010 010",
    "U": "101 101 101 101 111", "V": "101 101 101 101 010", "W": "101 101 111 111 101", "X": "101 101 010 101 101",
    "Y": "101 101 010 010 010", "Z": "111 001 010 100 111", "0": "111 101 101 101 111", "1": "010 110 010 010 111",
    "2": "111 001 111 100 111", "3": "111 001 111 001 111", "4": "101 101 111 001 001", "5": "111 100 111 001 111",
    "6": "111 100 111 101 111", "7": "111 001 001 001 001", "8": "111 101 111 101 111", "9": "111 101 111 001 111",
    " ": "000 000 000 000 000", "=": "000 111 000 111 000", "-": "000 000 111 000 000", ":": "000 010 000 010 000",
    ".": "000 000 000 000 010", "+": "000 010 111 010 000", ">": "100 010 001 010 100", "/": "001 001 010 100 100",
    "(": "010 100 100 100 010", ")": "010 001 001 001 010", "*": "000 101 010 101 000", ",": "000 000 000 010 100",
}


def text(img, x, y, s, color, u=2):
    for ch in s.upper():
        for r, row in enumerate(FONT.get(ch, FONT[" "]).split()):
            for c, bit in enumerate(row):
                if bit == "1":
                    img[y + r * u:y + (r + 1) * u, x + c * u:x + (c + 1) * u] = color
        x += 4 * u
    return x


def tw(s, u=2):
    return 4 * u * len(s) - u


def ctext(img, cx, y, s, color, u=2):
    text(img, cx - tw(s, u) // 2, y, s, color, u)


def mix(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def fill(img, x, y, w, h, c):
    img[y:y + h, x:x + w] = c


def frame(img, x, y, w, h, c, t=2):
    img[y:y + t, x:x + w] = c
    img[y + h - t:y + h, x:x + w] = c
    img[y:y + h, x:x + t] = c
    img[y:y + h, x + w - t:x + w] = c


def up(a, s):
    return np.repeat(np.repeat(a, s, 0), s, 1)


# ---------------------------------------------------------------- latent colours: fixed PCA fitted on the context latents
_flat = CTX_LAT.reshape(-1, 32).astype(np.float64)
_MU = _flat.mean(0)
_, _S, _VT = np.linalg.svd(_flat - _MU, full_matrices=False)
_PC = _VT[:3]
_SD = ((_flat - _MU) @ _PC.T).std(0)


def hsl(h, s, l):
    c = (1 - abs(2 * l - 1)) * s
    hp = (h % 360) / 60
    x = c * (1 - abs(hp % 2 - 1))
    r, g, b = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][int(hp) % 6]
    m = l - c / 2
    return r + m, g + m, b + m


def lat_colors(lat):
    """(4, 4, 32) latent -> (4, 4, 3) uint8 colours, blue family; clean latents coherent, noise speckled."""
    p = ((lat.reshape(-1, 32) - _MU) @ _PC.T) / _SD
    p = p / float(np.clip(p.std(0).mean(), 0.45, 1.0))     # keep full contrast for noisy slots (display only)
    out = []
    for p0, p1, p2 in p:
        l = float(np.clip(0.50 + 0.17 * p0, 0.14, 0.86))
        h = 218 + 28 * float(np.clip(p1, -2.0, 2.0))
        s = float(np.clip(0.62 + 0.15 * p2, 0.25, 0.95))
        out.append([int(round(255 * v)) for v in hsl(h, s, l)])
    return np.array(out, np.uint8).reshape(4, 4, 3)


# ---------------------------------------------------------------- layout
W, H = 1200, 400
BY, BH = 66, 216                           # module boxes
BAR_T, BAR_H, BAR_G = 40, 24, 8            # sub-block rows shared by all three modules
ROWS = [BY + BAR_T + i * (BAR_H + BAR_G) for i in range(5)]
AY = ROWS[2] + BAR_H // 2                  # main axis runs through the middle row
CELL, CG = 8, 1                            # latent token cell, gap
GRID = 4 * CELL + 3 * CG                   # 35
PITCH = GRID + 7
DECK_N, DECK_OFF = 5, 7
widths = dict(deck=126 + (DECK_N - 1) * DECK_OFF, enc=140, row=176, wm=240, zp=GRID, dec=140, out=189)
order = ["deck", "enc", "row", "wm", "zp", "dec", "out"]
L, R = 14, W - 14
free = (R - L) - sum(widths.values())
gaps = [free // 6 + (1 if i < free % 6 else 0) for i in range(6)]
X = {}
x = L
for i, k in enumerate(order):
    X[k] = x
    x += widths[k] + (gaps[i] if i < 6 else 0)
DECK_FX, DECK_FY = X["deck"], AY - 63                     # front frame top-left (2x)
ROW_R = X["row"] + widths["row"]                          # right edge of the latent row
SLOT_X = ROW_R - GRID                                     # newest slot
GY = AY - GRID // 2                                       # grid top
OUT_X, OUT_Y = X["out"], AY - 95
LOOP_Y = 32
KEY_Y = BY + BH + 18
FILM_Y = OUT_Y + 189 + 30
FILM_W = 63 + 4

ENC_BARS = ["TILE EMBED", "TRANSFORMER X6", "STRIDE-2 CONV", "RESBLOCK X4", "1X1 CONV"]
DEC_BARS = ["1X1 CONV", "RESBLOCK X4", "STRIDE-2 TCONV", "TRANSFORMER X6", "TILE READOUT"]
WM_BARS = ["SPATIAL ATTN", "CAUSAL TEMPORAL ATTN", "MLP"]


def lit_col(base, lv, full=ORANGE):
    return base if lv <= 0 else (full if lv >= 2 else mix(base, full, 0.42))


# ---------------------------------------------------------------- primitives: lines with arrowheads, partially lit
def seg(img, a, b, c, t=2):
    (x0, y0), (x1, y1) = a, b
    if y0 == y1:
        xa, xb = sorted((x0, x1))
        img[y0 - 1:y0 - 1 + t, xa:xb + 1] = c
    else:
        ya, yb = sorted((y0, y1))
        img[ya:yb + 1, x0 - 1:x0 - 1 + t] = c


def head(img, tip, d, c, n=5):
    x, y = tip
    for i in range(n):
        if d == "r":
            img[y - 1 - i:y + 1 + i, x - i] = c
        elif d == "l":
            img[y - 1 - i:y + 1 + i, x + i] = c
        elif d == "d":
            img[y - i, x - 1 - i:x + 1 + i] = c
        elif d == "u":
            img[y + i, x - 1 - i:x + 1 + i] = c


def poly_len(pts):
    return sum(abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def sub_poly(pts, s0, s1):
    """Piece of an axis-aligned polyline between arclengths s0 and s1."""
    out, acc = [], 0
    for a, b in zip(pts, pts[1:]):
        n = abs(b[0] - a[0]) + abs(b[1] - a[1])
        lo, hi = max(s0, acc), min(s1, acc + n)
        if hi > lo and n:
            f = lambda s: (a[0] + (b[0] - a[0]) * (s - acc) // n, a[1] + (b[1] - a[1]) * (s - acc) // n)
            out.append((f(lo), f(hi)))
        acc += n
    return out


def edge(img, pts, lit=(), base=DIM, arrow=True):
    """pts: axis-aligned polyline, arrow at the end. lit: list of (f0, f1, level) fractions lit orange."""
    for a, b in zip(pts, pts[1:]):
        seg(img, a, b, base)
    total = poly_len(pts)
    end_lv = 0
    for f0, f1, lv in lit:
        for a, b in sub_poly(pts, int(f0 * total), int(round(f1 * total))):
            seg(img, a, b, lit_col(base, lv))
        if f1 >= 0.999:
            end_lv = max(end_lv, lv)
    if arrow:
        a, b = pts[-2], pts[-1]
        d = "r" if b[0] > a[0] else "l" if b[0] < a[0] else ("d" if b[1] > a[1] else "u")
        head(img, b, d, lit_col(base, end_lv))


# ---------------------------------------------------------------- components
ATT_FILL, ATT_EDGE = (20, 26, 42), (44, 62, 108)


def bar(img, x, y, w, h, label, lv=0, u=2):
    att = "ATTN" in label or "TRANSFORMER" in label
    fill(img, x, y, w, h, lit_col(ATT_FILL if att else P2, lv, (58, 30, 16)))
    frame(img, x, y, w, h, lit_col(ATT_EDGE if att else P3, lv))
    ctext(img, x + w // 2, y + (h - 5 * u) // 2, label, FG if lv >= 2 else TXT, u)


def module(img, x, w, title, bars, lits):
    fill(img, x, BY, w, BH, P1)
    frame(img, x, BY, w, BH, P3)
    ctext(img, x + w // 2, BY + 13, title, FG, 3)
    for i, b in enumerate(bars):
        bar(img, x + 8, ROWS[i], w - 16, BAR_H, b, lits.get(i, 0))


def deck(img, lit=0):
    """Eight real context frames, flat, stacked with a small up-right offset; 9x9 tile grid on the front one."""
    for k in range(DECK_N - 1, -1, -1):                     # back (oldest) first; front = newest real frame
        i = NCTX - 1 - k
        x, y = DECK_FX + k * DECK_OFF, DECK_FY - k * DECK_OFF
        f = up(FRAMES[i], 2)
        if k:
            f = (f.astype(np.float32) * (0.78 - 0.08 * k)).astype(np.uint8)
        frame(img, x - 1, y - 1, 128, 128, BG, 1)
        img[y:y + 126, x:x + 126] = f
    x, y = DECK_FX, DECK_FY
    g = lit_col((14, 14, 16), lit)
    for t in range(1, 9):
        img[y:y + 126, x + 14 * t] = g
        img[y + 14 * t, x:x + 126] = g
    frame(img, x - 1, y - 1, 128, 128, lit_col(BG, lit), 1)


def grid(img, x, y, lat, alpha=1.0, outline=None, cell_hi=None, hi_col=ORANGE, mesh=None):
    cols = lat_colors(lat)
    if alpha < 1:
        cols = (cols.astype(np.float32) * alpha + np.array(BG) * (1 - alpha)).astype(np.uint8)
    for r in range(4):
        for c in range(4):
            fill(img, x + c * (CELL + CG), y + r * (CELL + CG), CELL, CELL, tuple(int(v) for v in cols[r, c]))
    if outline is not None:
        frame(img, x - 4, y - 4, GRID + 8, GRID + 8, outline, 2)
    if mesh is not None:
        for t in range(5):
            img[y - 1:y + GRID + 1, x - 1 + t * (CELL + CG)] = mesh
            img[y - 1 + t * (CELL + CG), x - 1:x + GRID + 1] = mesh
    if cell_hi is not None:                                 # the same token position, attended across time
        r, c = cell_hi
        fill(img, x + c * (CELL + CG) - 1, y + r * (CELL + CG) - 1, CELL + 2, CELL + 2, hi_col)


def key(img, x, y, label, pressed=True, w=None, h=24):
    w = w or max(24, tw(label) + 14)
    fill(img, x, y, w, h, ORANGE if pressed else P2)
    frame(img, x, y, w, h, ORANGE if pressed else P3)
    ctext(img, x + w // 2, y + (h - 10) // 2, label, BG if pressed else (170, 170, 168))
    return w


# ---------------------------------------------------------------- edges (static geometry)
WMX, WMW = X["wm"], widths["wm"]
ZPX = X["zp"]
E = {
    "deck_enc": [(X["deck"] + widths["deck"] + 3, AY), (X["enc"] - 1, AY)],
    "enc_row": [(X["enc"] + widths["enc"] + 2, AY), (SLOT_X - 3 * PITCH - 7, AY)],
    "row_wm": [(ROW_R + 6, AY), (WMX - 1, AY)],
    "wm_zp": [(WMX + WMW + 2, AY), (ZPX - 6, AY)],
    "zp_dec": [(ZPX + GRID + 6, AY), (X["dec"] - 1, AY)],
    "dec_out": [(X["dec"] + widths["dec"] + 2, AY), (OUT_X - 5, AY)],
    "loop": [(ZPX + GRID // 2, GY - 6), (ZPX + GRID // 2, LOOP_Y), (SLOT_X + GRID // 2, LOOP_Y),
             (SLOT_X + GRID // 2, GY - 8)],
}

# world model interior
WB_X, WB_W, WB_H = WMX + 34, WMW - 34 - 30, BAR_H
WB_Y = ROWS[:3]
ADA_Y = ROWS[4]
BUS_X = WB_X + WB_W + 14


def world_model(img, lv):
    fill(img, WMX, BY, WMW, BH, P1)
    frame(img, WMX, BY, WMW, BH, P3)
    ctext(img, WMX + WMW // 2, BY + 13, "WORLD MODEL", FG, 3)
    for i, b in enumerate(WM_BARS):
        bar(img, WB_X, WB_Y[i], WB_W, WB_H, b, lv.get(f"wm{i}", 0))
    # x6 bracket
    bx, y0, y1 = WMX + 26, WB_Y[0], WB_Y[2] + WB_H
    c = P3
    img[y0:y1, bx - 1:bx + 1] = c
    img[y0:y0 + 2, bx - 1:bx + 5] = c
    img[y1 - 2:y1, bx - 1:bx + 5] = c
    text(img, WMX + 7, (y0 + y1) // 2 - 5, "X6", TXT, 2)
    # adaLN block fed by key + tau from below, bus up the right side into every sub-block
    alv = lv.get("ada", 0)
    bar(img, WB_X, ADA_Y, WB_W, WB_H, "ADALN", alv)
    bus = lit_col(DIM, lv.get("bus", 0))
    seg(img, (WB_X + WB_W + 2, ADA_Y + WB_H // 2), (BUS_X, ADA_Y + WB_H // 2), bus)
    seg(img, (BUS_X, ADA_Y + WB_H // 2), (BUS_X, WB_Y[0] + WB_H // 2), bus)
    for i in range(3):
        yy = WB_Y[i] + WB_H // 2
        seg(img, (BUS_X, yy), (WB_X + WB_W + 5, yy), bus)
        head(img, (WB_X + WB_W + 2, yy), "l", bus, 4)


# ---------------------------------------------------------------- one rendered state
def render(st):
    """st: dict describing the scene. Returns HxWx3 uint8."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = BG
    lv = st.get("lit", {})
    text(img, L, FILM_Y + 67 - 20, "TINYWMCRAFT", ORANGE, 4)

    # pixels in: deck of real frames
    deck(img, lv.get("deck", 0))
    ctext(img, DECK_FX + 63, DECK_FY + 126 + 12, "63X63X3", DIM)

    module(img, X["enc"], widths["enc"], "ENCODER", ENC_BARS, {i: lv.get(f"enc{i}", 0) for i in range(5)})
    module(img, X["dec"], widths["dec"], "DECODER", DEC_BARS, {i: lv.get(f"dec{i}", 0) for i in range(5)})
    world_model(img, lv)

    for k, pts in E.items():
        edge(img, pts, st.get("edges", {}).get(k, ()))

    # latent row: entries right-aligned at the slot, sliding left by st['slide'] px
    rx0 = X["row"]
    if not st["row"]:
        for j in range(4):
            frame(img, SLOT_X - j * PITCH, GY, GRID, GRID, (34, 34, 38), 1)
    for j, ent in enumerate(reversed(st["row"])):
        x = SLOT_X - j * PITCH - st.get("slide", 0)
        x3 = SLOT_X - 3 * PITCH                           # leftmost fully visible position
        alpha = round(float(np.clip((x - x3 + PITCH // 2) / (PITCH // 2), 0.0, 1.0)) * 4) / 4
        if alpha <= 0:
            continue
        slot = ent["kind"] == "slot"
        grid(img, x, GY, ent["lat"], alpha, ORANGE if slot else None, st.get("temporal"), ORANGE,
             ORANGE if (slot and st.get("spatial")) else None)
        lab = ent.get("tau")
        if lab is not None and alpha > 0.5:
            ctext(img, x + GRID // 2, GY + GRID + 10, lab, ORANGE if slot else mix(BG, DIM, alpha))
    ctext(img, (SLOT_X - 3 * PITCH + ROW_R) // 2, GY + GRID + 30, "4X4X32", DIM)

    # z after the update, next to the decoder
    if st.get("zp") is not None:
        grid(img, ZPX, GY, st["zp"], 1.0, lit_col(P3, lv.get("zp", 0)) if lv.get("zp", 0) else None)
    else:
        frame(img, ZPX, GY, GRID, GRID, P3, 1)


    # step counter on the loop
    k = st.get("k", 0)
    lab = f"{k}/8"
    cx = (ZPX + SLOT_X + GRID) // 2
    wlab = tw(lab, 3) + 10 + 8 * 8
    x0 = cx - wlab // 2
    fill(img, x0 - 8, LOOP_Y - 10, wlab + 16, 20, BG)
    text(img, x0, LOOP_Y - 7, lab, ORANGE if k else DIM, 3)
    for i in range(8):
        fill(img, x0 + tw(lab, 3) + 10 + i * 8, LOOP_Y - 3, 6, 6, ORANGE if i < k else P3)

    # conditioning under the world model: key + tau
    a = st.get("action")
    cond = [(0, 1, lv["cond"])] if lv.get("cond") else ()
    kcx, tcx = WB_X + WB_W // 4, WB_X + 3 * WB_W // 4
    if a is not None:
        wk = max(24, tw(KEYS[a]) + 14)
        key(img, kcx - wk // 2, KEY_Y, KEYS[a], st.get("pressed", True), wk)
        ctext(img, kcx, KEY_Y + 32, NAMES[a], DIM)
    edge(img, [(kcx, KEY_Y - 3), (kcx, ADA_Y + WB_H + 2)], cond)
    tau = st.get("tau")
    gw = tw("TAU") + 8 + 8 * 8 - 2
    gx = tcx - gw // 2
    text(img, gx, KEY_Y + 7, "TAU", TXT)
    for i in range(8):
        on = tau is not None and i < round(tau * 8)
        fill(img, gx + tw("TAU") + 8 + i * 8, KEY_Y + 6, 6, 12, ORANGE if on else P3)
    edge(img, [(tcx, KEY_Y - 3), (tcx, ADA_Y + WB_H + 2)], cond)

    # output: the slot decoded at every Euler step, large and flat
    out = st.get("out")
    ob = lit_col(P3, lv.get("out", 0))
    frame(img, OUT_X - 3, OUT_Y - 3, 195, 195, ob, 2)
    if out is not None:
        img[OUT_Y:OUT_Y + 189, OUT_X:OUT_X + 189] = up(out, 3)
    ctext(img, OUT_X + 94, OUT_Y + 189 + 12, "63X63X3", DIM)

    # filmstrip of finished imagined frames
    fx0 = R - 4 * FILM_W - 3 * 8
    for i in range(4):
        x = fx0 + i * (FILM_W + 8)
        film = st.get("film", [])
        hi = i < len(film) and st.get("film_hi") == i
        frame(img, x, FILM_Y, FILM_W, FILM_W, ORANGE if hi else (P3 if i < len(film) else (34, 34, 38)), 2)
        if i < len(film):
            img[FILM_Y + 2:FILM_Y + 65, x + 2:x + 65] = film[i]
    return img


# ---------------------------------------------------------------- timeline
PH0, PH = 50, 40            # ms per pipeline phase: first imagined frame slower, then faster
TEMPORAL_POS = [(1, 2), (2, 1), (0, 3), (3, 0), (1, 1), (2, 2), (0, 0), (3, 3)]


def timeline():
    seq = []
    cur = dict(row=[], zp=None, k=0, tau=None, action=int(ACTIONS[NCTX]), pressed=False, out=None, film=[], slide=0)

    def add(dur, **kw):
        if "tag" not in kw:
            kw["tag"] = None
        st = dict(cur)
        st.update(kw)
        seq.append((st, dur))

    real = [dict(kind="real", lat=CTX_LAT[i], tau="1.0") for i in range(NCTX)]
    slot = lambda lat, tau: dict(kind="slot", lat=lat, tau=f"{tau:.2f}")

    # encode the real context once
    add(200, lit={"deck": 2}, edges={"deck_enc": [(0, 1, 2)]})
    for i in range(5):
        add(70, lit={f"enc{i}": 2, f"enc{i - 1}": 1, "deck": 1 if i == 0 else 0},
            edges={"deck_enc": [(0, 1, 1)]} if i == 0 else {})
    cur["row"] = list(real)
    add(80, lit={"enc4": 1}, edges={"enc_row": [(0, 1, 2)]})
    add(80, edges={"enc_row": [(0, 1, 1)]})

    for f in range(NGEN):
        ph = PH0 if f == 0 else PH
        hist = list(cur["row"])
        cur.update(row=hist + [slot(Z[f][0], 0.0)], out=ZDEC[f][0], action=int(ACTIONS[NCTX + f]), pressed=True,
                   k=1, tau=0.0, zp=None, slide=0)
        add(200 if f == 0 else 140, lit={"out": 1})
        for k in range(STEPS):
            cur.update(k=k + 1, tau=k / STEPS, row=hist + [slot(Z[f][k], k / STEPS)])
            add(ph, lit={"cond": 2, "ada": 2}, edges={"row_wm": [(0, 1, 2)]}, tag=f"f{f}k{k}")
            add(ph, lit={"wm0": 2, "ada": 1, "bus": 2, "cond": 1}, edges={"row_wm": [(0, 1, 1)]}, spatial=True)
            add(ph, lit={"wm1": 2, "wm0": 1, "bus": 1}, temporal=TEMPORAL_POS[k])
            add(ph, lit={"wm2": 2, "wm1": 1})
            cur["zp"] = Z[f][k + 1]
            add(ph, lit={"wm2": 1, "zp": 2}, edges={"wm_zp": [(0, 1, 2)]})
            add(ph, lit={"zp": 1, "dec0": 2, "dec1": 2, "dec2": 2},
                edges={"wm_zp": [(0, 1, 1)], "zp_dec": [(0, 1, 2)], "loop": [(0, 0.5, 2)]})
            cur.update(row=hist + [slot(Z[f][k + 1], (k + 1) / STEPS)], out=ZDEC[f][k + 1])
            add(ph, lit={"dec0": 1, "dec1": 1, "dec2": 1, "dec3": 2, "dec4": 2, "out": 2},
                edges={"zp_dec": [(0, 1, 1)], "dec_out": [(0, 1, 2)], "loop": [(0, 0.5, 1), (0.5, 1, 2)]})
            if k < STEPS - 1 and f == 0:
                add(ph, lit={"out": 1}, edges={"dec_out": [(0, 1, 1)], "loop": [(0.5, 1, 1)]})
        # the finished frame: hold, join the context at tau 0.9, slide the window, film it
        cur.update(tau=1.0)
        add(380, lit={"out": 2}, tag=f"fin{f}")
        cur["film"] = cur["film"] + [ZDEC[f][STEPS]]
        cur["row"] = hist + [dict(kind="imag", lat=Z[f][STEPS], tau="0.9")]
        cur.update(pressed=False)
        for s in range(1, 5):
            add(40, slide=PITCH * s // 4, film_hi=f)
        cur["slide"] = 0
    add(1000, film_hi=None)
    return seq


def save(frames, durs, out):
    fixed = [BG, FG, DIM, ORANGE, P1, P2, P3, TXT, (170, 170, 168), (14, 14, 16), (34, 34, 38), (58, 30, 16)]
    fixed += [lit_col(c, 1) for c in (P3, DIM, (14, 14, 16), BG)] + [lit_col(P2, 1, (58, 30, 16))]
    fixed += [mix(BG, DIM, a) for a in (0.25, 0.5, 0.75)] + [ATT_FILL, ATT_EDGE, lit_col(ATT_EDGE, 1),
                                                              lit_col(ATT_FILL, 1, (58, 30, 16))]
    fixed = list(dict.fromkeys(fixed))
    sample = np.concatenate([frames[i] for i in np.linspace(0, len(frames) - 1, 10).astype(int)], 0)
    rest = Image.fromarray(sample).quantize(colors=255 - len(fixed), method=Image.Quantize.MEDIANCUT)
    flat = rest.getpalette()[:3 * (255 - len(fixed))] + [v for c in fixed for v in c]
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat)
    q = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=durs, loop=0, disposal=1)


def main():
    frames, durs = [], []
    for st, d in timeline():
        img = render(st)
        if frames and np.array_equal(img, frames[-1]):
            durs[-1] += d
            continue
        frames.append(img)
        durs.append(d)
    ARGS.out.parent.mkdir(parents=True, exist_ok=True)
    save(frames, durs, ARGS.out)
    print(f"{ARGS.out}: {len(frames)} frames, {sum(durs) / 1000:.2f} s, {ARGS.out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

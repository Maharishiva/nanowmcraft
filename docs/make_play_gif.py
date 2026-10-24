"""Demo GIF: one held-out episode, eight real frames of priming, then the world model free-runs on the recorded
actions. Left pane: real Craftax. Right pane: the world model. Below: the key being pressed, drawn pixel style.

    python docs/make_play_gif.py --model runs/base/model.npz --style craft --out docs/assets/play.gif   # the README gif

CPU is fine: one 32-frame rollout takes about a minute on a laptop.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jax
import jax.numpy as jnp

import train
from clips import streams

train.CDT = jnp.float32                                   # fp32 on CPU, as in play_wm.py
train.MP = dict(dtype=jnp.float32, param_dtype=jnp.float32)

CTX, T = 8, 32
SCALE, U = 4, 4                                           # game pixel -> 4 px; one UI "pixel" -> 4 px
BG, FG, DIM, ORANGE, KEY, KEY_EDGE = (10, 10, 10), (240, 240, 238), (106, 106, 106), (255, 106, 19), (28, 28, 30), (58, 58, 62)
NAMES = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE STONE", "PLACE TABLE", "PLACE FURNACE",
         "PLACE PLANT", "WOOD PICKAXE", "STONE PICKAXE", "IRON PICKAXE", "WOOD SWORD", "STONE SWORD", "IRON SWORD"]
KEYS = ["Q", "A", "D", "W", "S", "SPACE", "TAB", "R", "T", "F", "P", "1", "2", "3", "4", "5", "6"]   # play_wm.py bindings
FONT = {  # 3x5 pixel font
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
}


def text(img, x, y, s, color, u=U):
    """Draw s with the 3x5 font, top-left at (x, y), each font pixel u screen pixels."""
    for ch in s.upper():
        rows = FONT.get(ch, FONT[" "]).split()
        for r, row in enumerate(rows):
            for c, bit in enumerate(row):
                if bit == "1":
                    img[y + r * u:y + (r + 1) * u, x + c * u:x + (c + 1) * u] = color
        x += 4 * u
    return x


def text_width(s, u=U):
    return 4 * u * len(s) - u


def rect(img, x, y, w, h, fill, edge=None):
    img[y:y + h, x:x + w] = fill
    if edge is not None:
        img[y:y + U, x:x + w] = edge
        img[y + h - U:y + h, x:x + w] = edge
        img[y:y + h, x:x + U] = edge
        img[y:y + h, x + w - U:x + w] = edge


def key(img, x, y, label, pressed, w=7 * U, h=7 * U):
    rect(img, x, y, w, h, ORANGE if pressed else KEY, ORANGE if pressed else KEY_EDGE)
    tw = text_width(label, 3)
    text(img, x + (w - tw) // 2, y + (h - 15) // 2, label, BG if pressed else (170, 170, 168), 3)


def up(frame):
    return np.repeat(np.repeat(frame, SCALE, 0), SCALE, 1)


def rank_windows(data, style="explore"):
    """Held-out windows without an episode reset, best first, one per environment, no night or sleep frames.
    explore: varied actions and a moving agent. craft: the agent stays put and crafts, places and hits things,
    so the whole window is terrain the model has seen."""
    frames, actions, dones = data["frames"], data["actions"], data["dones"].astype(bool)
    L = CTX + T
    bright = frames[:, :, :49].mean(axis=(2, 3, 4))                       # night and sleep darken the map
    inv = frames[:, :, 49:].astype(np.int16)
    changed = np.abs(inv[1:] - inv[:-1]).mean(axis=(2, 3, 4)) > 0.5       # inventory strip changed t -> t+1
    best = {}
    for env in range(frames.shape[1]):
        for start in range(1, frames.shape[0] - L, 2):
            if dones[start - 1:start + L - 1, env].any() or bright[start:start + L, env].min() < 62:
                continue
            fut = actions[start + CTX - 1:start + L - 1, env]
            hit = changed[start + CTX - 1:start + L - 1, env]
            clip = frames[start + CTX:start + L, env].astype(np.int16)
            moves = np.isin(fut, [1, 2, 3, 4]).mean()
            motion = np.abs(np.diff(clip[:, :49], axis=0)).mean()
            crafts = (np.isin(fut, [11, 12, 13, 14, 15, 16]) & hit).sum()   # a craft that actually produced an item
            places = (np.isin(fut, [7, 8, 9, 10]) & hit).sum()
            if style == "craft":
                score = 3.0 * min(crafts, 3) + 1.5 * min(places, 3) + 0.5 * hit.sum() + len(set(fut.tolist())) - 6.0 * moves
            else:
                score = 3.0 * min(crafts, 2) + 1.5 * min(places, 2) + len(set(fut.tolist())) + 2.0 * moves + motion / 10
            if score > best.get(env, (None, -1.0))[1]:
                best[env] = (start, score)
    return [(start, env) for env, (start, score) in sorted(best.items(), key=lambda kv: -kv[1][1])]


def rollout(model, ctx_frames, actions, key):
    """train.imagine without its 32-frame limit: the latent rollout slides its window, and the reward head reads
    two 32-frame windows. Returns T imagined frames and the predicted reward for arriving at each."""
    import flax.nnx as nnx
    ctx = train.encode(model, jnp.asarray(ctx_frames)[None])
    ca, fa = jnp.asarray(actions[None, :CTX], jnp.int32), jnp.asarray(actions[None, CTX:], jnp.int32)
    gdef, state = nnx.split(model.df)

    @jax.jit
    def run(state, ctx, ca, fa, key):
        kr, k1, k2 = jax.random.split(key, 3)
        df = nnx.merge(gdef, state)
        zt = train._rollout_latents(df, ctx, ca, fa.T, kr, train.STEPS, model.win).transpose(1, 0, 2, 3, 4)
        z, acts = jnp.concatenate([ctx, zt], 1), jnp.concatenate([ca, fa], 1)
        r_first = train._read_heads(df, z[:, :train.WIN], acts[:, :train.WIN], k1)[0][:, CTX:]
        r_last = train._read_heads(df, z[:, -train.WIN:], acts[:, -train.WIN:], k2)[0]
        return zt, jnp.concatenate([r_first, r_last[:, -(CTX + T - train.WIN):]], 1)
    zt, rewards = run(state, ctx, ca, fa, key)
    return np.asarray(train.decode(model, zt)[0]), np.asarray(rewards[0])


def compose(real, imagined, action, step, priming, real_reward, pred_reward):
    pane = 63 * SCALE
    margin, gap, head, foot = 4 * U, 6 * U, 8 * U, 26 * U
    W, H = margin + pane + gap + pane + margin, head + pane + foot
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = BG
    lx, rx, py = margin, margin + pane + gap, head
    text(img, lx, 2 * U, "REAL", DIM, 3)
    text(img, rx, 2 * U, "WORLD MODEL", ORANGE if not priming else DIM, 3)
    img[py:py + pane, lx:lx + pane] = up(real)
    img[py:py + pane, rx:rx + pane] = up(imagined)
    if real_reward > 0:
        text(img, lx + pane - text_width("+1", 3) - U, py + U, "+1", FG, 3)
    if pred_reward > 0.5:
        text(img, rx + pane - text_width("+1", 3) - U, py + U, "+1", ORANGE, 3)
    # keyboard: W above S, A S D, then the space bar, then the extra key of the pressed non-movement action
    ky = py + pane + 3 * U
    kx = lx
    name = NAMES[action]
    key(img, kx + 8 * U, ky, "W", action == 3)
    key(img, kx, ky + 8 * U, "A", action == 1)
    key(img, kx + 8 * U, ky + 8 * U, "S", action == 4)
    key(img, kx + 16 * U, ky + 8 * U, "D", action == 2)
    key(img, kx + 26 * U, ky + 8 * U, "SPACE", action == 5, w=22 * U)
    other = action not in (1, 2, 3, 4, 5)
    key(img, kx + 50 * U, ky + 8 * U, KEYS[action] if other else "", other, w=9 * U)
    text(img, rx, ky + 2 * U, f"STEP {step:02d}", DIM, 3)
    text(img, rx + 24 * U, ky + 2 * U, "PRIMING" if priming else name, DIM if priming else FG, 3)
    text(img, rx, ky + 9 * U, "PRIMING ON 8 REAL FRAMES" if priming else "SAME KEYS, NO GAME STATE", DIM, 2)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/eval.npz"))
    p.add_argument("--out", type=Path, default=Path("docs/assets/play.gif"))
    p.add_argument("--seed", type=int, default=0, help="sampling noise of the rollout")
    p.add_argument("--rank", type=int, default=0, help="which held-out window, best first")
    p.add_argument("--style", choices=("explore", "craft"), default="explore")
    p.add_argument("--fps", type=float, default=8.0)
    a = p.parse_args()
    data = streams(a.data)
    start, env = rank_windows(data, a.style)[a.rank]
    L = CTX + T
    frames = data["frames"][start:start + L, env]
    actions = data["actions"][start - 1:start + L - 1, env].astype(np.int32)        # incoming action of each frame
    rewards = data["rewards"][start - 1:start + L - 1, env]
    model = train.load_model(a.model)
    imagined, pred_reward = rollout(model, frames[:CTX], actions, jax.random.PRNGKey(a.seed))
    out = []
    for t in range(L):
        priming = t < CTX
        img = compose(frames[t], frames[t] if priming else imagined[t - CTX], int(actions[t]), t, priming,
                      float(rewards[t]), 0.0 if priming else float(pred_reward[t - CTX]))
        out.append(img)
    out = [out[0]] * 4 + out + [out[-1]] * 8
    import imageio.v3 as iio
    a.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(a.out, np.stack(out), duration=int(1000 / a.fps), loop=0)
    print(f"{a.out}: env {env} start {start}, {len(out)} frames, actions {sorted(set(NAMES[i] for i in actions[CTX:]))}")


if __name__ == "__main__":
    main()

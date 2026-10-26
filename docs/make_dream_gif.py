"""Dream GIF: worlds the model imagines from noise, each played by the pixel PPO agent fine-tuned for diamonds (the
agent that drives most of the training data, policies/walker.pkl on Hugging Face, without its goal head). No game runs:
the agent sees only the frames the model draws, and the model draws the next one from the agent's action.

    python docs/make_dream_gif.py --model runs/best/model.npz --policy policies/walker.pkl   # -> docs/assets/dream.gif

The first frame of every world is imagined from nothing; one that shows items or less than full stats is drawn again,
as in the browser player. Shown, 2 rows of 4: the first 100 steps of the worlds that get furthest in them (the items
the model draws in the inventory, on two frames in a row) in daylight, and two worlds at a stretch where night falls
and the agent sleeps. The script also counts the worlds that draw a diamond within all the steps. A GPU helps: 256
worlds of 450 steps take about a minute on an A100.
"""
import argparse
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp

import eval as E
import train
from make_play_gif import BG, ORANGE, daylight

A = 17
GAIN = np.array([0, 0, 0, 0, 0, 1, 2, 2, 4, 8, 2, 3, 5, 1, 1, 1], np.float32)  # HUD slots: wood 1 ... diamond 8, tools


def plain(t):
    return {k: plain(v) for k, v in t.items()} if hasattr(t, "items") else jnp.asarray(getattr(t, "value", t))


def dense(x, p, dt=jnp.float32):
    return x.astype(dt) @ p["kernel"].astype(dt) + p["bias"].astype(dt)


def policy(p, h, obs, pa):
    """collect/collect.py's pixel policy: Impala CNN (bf16) -> GRU 256 with the previous action -> MLP 512 -> logits."""
    bf, relu = jnp.bfloat16, jax.nn.relu
    conv = lambda x, q: jax.lax.conv_general_dilated(x, q["kernel"].astype(bf), (1, 1), "SAME", dimension_numbers=(
        "NHWC", "HWIO", "NHWC")) + q["bias"].astype(bf)
    x = (obs / 255.0).astype(bf)
    for b in ("b1", "b2", "b3"):
        x = jax.lax.reduce_window(conv(x, p[b]["conv"]), -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), "SAME")
        for r in ("r1", "r2"):
            x = x + conv(relu(conv(relu(x), p[b][r + "a"])), p[b][r + "b"])
    e = relu(dense(relu(x).reshape(x.shape[0], -1), p["fc"], bf)).astype(jnp.float32)
    x, g = jnp.concatenate([e, jax.nn.one_hot(pa, A)], -1), p["gru"]
    z, r = (jax.nn.sigmoid(dense(jnp.concatenate([x, h], -1), g[k])) for k in ("wz", "wr"))
    h = (1 - z) * h + z * jnp.tanh(dense(jnp.concatenate([x, r * h], -1), g["wh"]))
    return h, dense(relu(dense(jnp.concatenate([e, h], -1), p["l1"])), p["actor"])


@jax.jit
def act_and_step(model, pol, state, h, frame, pa, key):
    """The agent picks an action from the frame it sees; the model draws the frame that follows it."""
    kp, km = jax.random.split(key)
    h, logits = policy(pol, h, frame, pa)
    a = jax.random.categorical(kp, logits)
    frame, state = train.step(model, state, a, km)
    return frame, state, h, a


def fresh_starts(model, n, hud_tpl, key, tries=8):
    """n worlds whose first frame, imagined from nothing, shows full stats and an empty inventory."""
    keep_f, keep_s = [], []
    for _ in range(tries):
        key, k = jax.random.split(key)
        z = jnp.zeros((4 * n, train.WIN, train.LAT, train.LAT, train.TOK))
        state = (z, jnp.zeros((4 * n, train.WIN), jnp.int32), jnp.int32(0), jnp.int32(0))
        frame, state = train.step(model, state, jnp.zeros(4 * n, jnp.int32), k)
        hud = E.read_hud(np.asarray(frame), hud_tpl)
        ok = np.flatnonzero((hud[:, :4] == 9).all(1) & (hud[:, 4:] == 0).all(1))
        keep_f.append(np.asarray(frame)[ok])
        keep_s.append(jax.tree.map(lambda v: v[ok] if v.ndim else v, state))
        if sum(len(f) for f in keep_f) >= n:
            break
    frames = np.concatenate(keep_f)[:n]
    state = jax.tree.map(lambda *v: jnp.concatenate(v)[:n] if v[0].ndim else v[0], *keep_s)
    return frames, state


def dream(model, pol, hud_tpl, n, steps, seed):
    """frames (n, steps + 1, 63, 63, 3) and actions (n, steps) of n worlds."""
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    frame, state = fresh_starts(model, n, hud_tpl, k)
    h, pa, out, acts = jnp.zeros((n, 256)), jnp.zeros(n, jnp.int32), [frame], []
    frame = jnp.asarray(frame)
    for t in range(steps):
        key, k = jax.random.split(key)
        frame, state, h, pa = act_and_step(model, pol, state, h, frame, pa, k)
        out.append(np.asarray(frame))
        acts.append(np.asarray(pa))
    return np.stack(out, 1), np.stack(acts, 1)


def reached(hud):
    """hud (T, 16) -> (T, 16) bool: the slot shows at least one item on this frame and the one before."""
    has = hud > 0
    return np.concatenate([np.zeros_like(has[:1]), has[1:] & has[:-1]])


def read(frames, d):
    """Per world: the items it holds (two frames in a row), asleep, and the light of every frame."""
    out = []
    for w in frames:
        hud = E.read_hud(w, d["hud_tpl"])
        out.append(dict(got=reached(hud), dead=hud[:, 0] == 0, asleep=E.asleep(w),
                        light=daylight(w[:, None], d["tile_tex"], d["tile_night"])[:, 0]))
    return out


def day_score(r, s, n):
    """Steps s to s + n: the items gained (GAIN), less 20 times the share of dark or asleep frames; None if it dies."""
    seg = slice(s, s + n)
    if r["dead"][seg].any():
        return None
    held = r["got"][s] if s else np.zeros(16, bool)
    dim = (r["light"][seg] < 0.6) | r["asleep"][seg]
    return float(((r["got"][seg].any(0) & ~held) * GAIN).sum()) - 20 * float(dim.mean())


def night_score(r, s, n):
    """Steps s to s + n: night and sleep both shown, the agent falling asleep or waking up counts most."""
    seg = slice(s, s + n)
    if r["dead"][seg].any():
        return None
    sleep, dark = r["asleep"][seg], (r["light"][seg] < 0.5) & ~r["asleep"][seg]
    turns = int((sleep[1:] != sleep[:-1]).sum())
    return min(int(sleep.sum()), 40) + min(int(dark.sum()), 30) + 15 * min(turns, 2) if sleep.sum() <= 0.7 * n else None


def pick(rows, n, show, night, every=5):
    """(world, first step) of `show` worlds: `night` of them at their best stretch with night and sleep, the rest from
    their first frame by the items gained."""
    best = lambda f, starts: [max(((f(r, s, n), s) for s in starts if f(r, s, n) is not None), default=(None, 0))
                              for r in rows]
    nights = best(night_score, range(0, len(rows[0]["light"]) - n + 1, every))
    days = best(day_score, [0])
    by_night = sorted((i for i in range(len(rows)) if nights[i][0] is not None), key=lambda i: -nights[i][0])[:night]
    by_day = sorted((i for i in range(len(rows)) if days[i][0] is not None and i not in by_night),
                    key=lambda i: -days[i][0])[:show - len(by_night)]
    return [(i, 0) for i in by_day] + [(i, nights[i][1]) for i in by_night]


def compose(worlds, marks, length, stride, scale=3, gap=8, cols=4):
    """The GIF frames: worlds (k, T, 63, 63, 3) in rows of `cols`, every stride-th frame; a world with a diamond in its
    inventory gets an orange border from that frame on."""
    k, tile = len(worlds), 63 * scale
    rows = -(-k // cols)
    H, W = rows * tile + (rows + 1) * gap, cols * tile + (cols + 1) * gap
    out = []
    for t in range(0, length, stride):
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = BG
        for i, w in enumerate(worlds):
            f = w[min(t, len(w) - 1)]
            y, x = gap + (i // cols) * (tile + gap), gap + (i % cols) * (tile + gap)
            if 0 <= marks[i] <= t:
                img[y - 3:y + tile + 3, x - 3:x + tile + 3] = ORANGE
            img[y:y + tile, x:x + tile] = np.repeat(np.repeat(f, scale, 0), scale, 1)
        out.append(img)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--policy", type=Path, default=Path("policies/walker.pkl"))
    p.add_argument("--data", type=Path, default=Path("data/eval.npz"), help="for the HUD and tile templates only")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "assets" / "dream.gif")
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--first", type=int, default=100, help="steps shown per world")
    p.add_argument("--show", type=int, default=8)
    p.add_argument("--night", type=int, default=2, help="worlds shown at a stretch with night and sleep")
    p.add_argument("--stride", type=int, default=1, help="one GIF frame per this many steps")
    p.add_argument("--fps", type=float, default=11.25)
    p.add_argument("--scale", type=int, default=3, help="screen pixels per game pixel")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cache", type=Path, help="an npz of the rollout: written if missing, read if there")
    a = p.parse_args()
    z = np.load(a.data)
    d = {k: z[k] for k in ("hud_tpl", "tile_tex", "tile_night")}
    cached = a.cache and a.cache.exists()
    if cached:
        frames = np.load(a.cache)["frames"]
    else:
        pol = plain(pickle.load(open(a.policy, "rb"))["state"]["ac"])
        frames, _ = dream(train.load_model(a.model), pol, d["hud_tpl"], a.worlds, a.steps, a.seed)
    rows = read(frames, d)
    shown = pick(rows, a.first, a.show, a.night)
    if a.cache and not cached:                                         # the worlds shown, to lay out again
        np.savez_compressed(a.cache, frames=frames[[i for i, _ in shown]])
    worlds = [frames[i][s:s + a.first] for i, s in shown]
    marks = [int(np.argmax(rows[i]["got"][s:s + a.first, 9])) if rows[i]["got"][s:s + a.first, 9].any() else -1
             for i, s in shown]
    out = compose(worlds, marks, a.first, a.stride, a.scale)
    out += [out[-1]] * int(a.fps * 2)
    import imageio.v3 as iio
    a.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(a.out, np.stack(out), duration=int(1000 / a.fps), loop=0)
    if shutil.which("ffmpeg"):                     # one palette for all frames and only the changed pixels: about half
        tmp = a.out.with_suffix(".tmp.gif")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.out, "-vf", "split[a][b];[a]palettegen=max_colors="
                        "128:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle", "-gifflags", "+transdiff",
                        tmp], check=True)
        tmp.replace(a.out)
    for i, s in shown:
        r = rows[i]
        print(f"world {i} from step {s}: asleep {r['asleep'][s:s + a.first].sum()}, dark "
              f"{(r['light'][s:s + a.first] < 0.5).sum()}, items {np.flatnonzero(r['got'][s:s + a.first].any(0)).tolist()}")
    got = sum(r["got"][:, 9].any() for r in rows)
    print(f"{a.out}: {len(out)} frames; {got} of {len(rows)} worlds drew a diamond in the inventory")


if __name__ == "__main__":
    main()

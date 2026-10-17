"""Frozen benchmark: open-loop generation, action following, and the reward and termination heads, on 2048 fixed clips
of held-out play.

Each clip in eval.npz holds 8 real frames of context and 64 future frames, with the action, reward and death flag of
the transition into each frame. The 1536 `normal` clips are random windows of play; each of the 512 `event` clips
holds a rare event in its first 16 future frames: a diamond collected, a plant eaten, an iron pickaxe or sword made, a
death in lava or by a mob. A death clip ends at the death: its frames from valid_len on are padding and never scored,
the terminal frame counts.

The model is primed on the 8 real frames and generates one frame per action: 64 frames under the recorded actions, and
16 frames under random actions with the same sampling keys.

  gFVD     Frechet distance between the real and generated future frames 1-64 in the feature space of a fixed random
           CNN (seed 1234, never trained) over stacks of 4 consecutive frames, times 10^4. Lower is better.
  gFVD_16  the same over frames 1-16.
  AF       action following: the share of clips where the rollout under the recorded actions is closer (salient-pixel
           MSE over frames 1-16) to the real future than the rollout under random actions; ties score half. 0.5 = the
           model ignores actions. Higher is better.
  reward_ap / done_ap   the heads teacher-forced on the real frames, each future frame read with 8 frames of context
           before it: average precision of |predicted reward| for the recorded reward events (|r| > 0) and of the done
           probability for the recorded deaths, over all future transitions. reward_base / done_base are the shares of
           positives, the score of a blind guess.
  imagined_return_corr  the reward head read from the rollout under the recorded actions: correlation, over the normal
           clips, of each clip's predicted reward summed over its generated frames with the recorded sum.
  imagined_done_ap      the done head read from the same rollout: average precision of each clip's highest done
           probability over its generated frames for the clips that end in a death; imagined_done_base is their share.

Every metric is also reported on the normal and on the event clips alone (suffixes _normal and _event); a metric with
no positive in its clips is null.

The model is any module (default train.py) with this interface; how it generates a frame is its own business:
  load_model(path), load_meta(path) -> {"cfg": {"seed": ...}, "train_s": ...}
  start(model, frames (B, n0, H, W, 3) uint8, actions (B, n0)) -> state       actions[t] is the action INTO frame t
  step(model, state, action (B,), key) -> (frame (B, H, W, 3) uint8, state)  one frame, never sees future actions
  heads(model, frames (B, L, H, W, 3), actions (B, L), key) -> (reward (B, L), done probability (B, L)), L = 24

eval_s is timed from importing the model module to the last metric, so model loading and compilation count and
reading eval.npz does not; it must stay within 180 s on the maintainer's A100.

    python eval.py --model runs/base/model.npz --tag base
"""
import argparse
import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

H = W = 63
A_N = 17
CTX, FUT, AF_HOR, HEAD_HOR, N_CLIPS = 8, 64, 16, 16, 2048
EVAL_SEED, BATCH, FEAT_BATCH = 0, 64, 16       # clips per rollout batch, clips per feature pass
CLIP, FEAT_SEED, THR = 4, 1234, 16 / 255
GFVD_SCALE, TRAIN_BUDGET_S, EVAL_BUDGET_S = 1e4, 600, 180


class FeatCNN(nnx.Module):
    """Fixed random conv stack over CLIP stacked frames; the final 4x4 map is kept flat so layout counts."""
    def __init__(self, rngs):
        self.c1 = nnx.Conv(3 * CLIP, 32, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c2 = nnx.Conv(32, 64, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c3 = nnx.Conv(64, 128, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c4 = nnx.Conv(128, 32, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)

    def __call__(self, x):
        h = nnx.gelu(self.c1(x))
        h = nnx.gelu(self.c2(h))
        h = nnx.gelu(self.c3(h))
        return self.c4(h).reshape(x.shape[0], -1)


def frame_features():
    """A function from frames (B, T, H, W, 3) uint8 to features (B, T, F): for every frame, the stack of CLIP frames
    ending at it (the first frame repeated before the start) through the fixed CNN, on the device."""
    cnn = FeatCNN(nnx.Rngs(FEAT_SEED))

    @nnx.jit
    def run(cnn, video):
        B, T = video.shape[:2]
        v = video.astype(jnp.float32) / 255
        v = jnp.concatenate([jnp.repeat(v[:, :1], CLIP - 1, 1), v], 1)
        stacks = jnp.concatenate([v[:, t:t + T] for t in range(CLIP)], -1)
        return cnn(stacks.reshape(B * T, H, W, 3 * CLIP)).reshape(B, T, -1)

    def features(video):
        with jax.default_matmul_precision("tensorfloat32"):   # fixed, whatever the module sets
            return np.concatenate([np.asarray(run(cnn, video[i:i + FEAT_BATCH]))
                                   for i in range(0, len(video), FEAT_BATCH)])
    return features


def frechet(fr, fg):
    fr, fg = fr.astype(np.float64), fg.astype(np.float64)
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    cr = np.cov(fr, rowvar=False) + 1e-6 * np.eye(fr.shape[1])
    cg = np.cov(fg, rowvar=False) + 1e-6 * np.eye(fg.shape[1])
    eig = np.linalg.eigvals(cr @ cg)
    return float(((mu_r - mu_g) ** 2).sum() + np.trace(cr) + np.trace(cg) - 2 * np.sum(np.sqrt(np.clip(eig.real, 0, None))))


def salient_mse(pred, real, last_ctx, valid):
    """Per clip, the squared error over the pixels of valid frames where the real future differs from the last
    context frame."""
    real, pred = real.astype(np.float32) / 255, pred.astype(np.float32) / 255
    sal = (np.abs(real - last_ctx[:, None]).max(-1, keepdims=True) > THR) & valid[:, :, None, None, None]
    return ((pred - real) ** 2 * sal).sum((1, 2, 3, 4)) / (sal.sum((1, 2, 3, 4)) + 1e-6)


def average_precision(scores, labels):
    """Area under the precision-recall curve: the mean precision at the rank of each positive; None without one."""
    hits = np.asarray(labels).ravel() > 0.5
    if not hits.any():
        return None
    hits = hits[np.argsort(-np.asarray(scores, np.float64).ravel(), kind="stable")]
    return float(np.mean(np.cumsum(hits)[hits] / (np.flatnonzero(hits) + 1)))


def corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def make_gif(real, predicted, path, k=8, scale=4):
    import imageio.v2 as imageio

    k = min(k, len(real))
    separator = np.full((H, 2, 3), (90, 90, 110), np.uint8)
    images = []
    for t in range(real.shape[1]):
        grid = np.concatenate([np.concatenate([real[i, t], separator, predicted[i, t]], 1) for i in range(k)], 0)
        images.append(np.repeat(np.repeat(grid, scale, 0), scale, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, images, duration=1000 / 6, loop=0)


def rollout(wm, model, context, context_actions, actions, keys):
    """Frames (B, T, H, W, 3) generated one action of actions (B, T) at a time after the real context, one key each."""
    state = wm.start(model, context, context_actions)
    out = []
    for action, key in zip(actions.T, keys):
        frame, state = wm.step(model, state, jnp.asarray(action), key)
        out.append(frame)
    return np.asarray(jnp.stack(out, 1))


def read_heads(wm, model, frames, actions, key):
    """Reward and done probability (B, FUT) of the transition into each future frame, from windows of CTX + HEAD_HOR
    frames every HEAD_HOR frames, each scoring its last HEAD_HOR."""
    reward, done = np.zeros((len(frames), FUT), np.float32), np.zeros((len(frames), FUT), np.float32)
    for w, s in enumerate(range(0, FUT, HEAD_HOR)):
        window = slice(s, s + CTX + HEAD_HOR)
        r, d = wm.heads(model, frames[:, window], actions[:, window], jax.random.fold_in(key, w))
        reward[:, s:s + HEAD_HOR], done[:, s:s + HEAD_HOR] = np.asarray(r)[:, CTX:], np.asarray(d)[:, CTX:]
    return reward, done


def evaluate(wm, model, clips):
    """Both rollouts and the heads, batch by batch -> per-clip arrays, and the first clips for the gif."""
    features = frame_features()
    out = {k: [] for k in ("fr", "fg", "err_true", "err_rand", "r_tf", "d_tf", "r_im", "d_im")}
    preview = None
    for b0 in range(0, len(clips["event"]), BATCH):
        frames, actions = clips["frames"][b0:b0 + BATCH], clips["actions"][b0:b0 + BATCH].astype(np.int32)
        valid = clips["valid"][b0:b0 + BATCH, :AF_HOR]
        key = jax.random.PRNGKey(int(np.random.SeedSequence([EVAL_SEED, b0]).generate_state(1)[0]))
        kg, kt, ki = jax.random.split(key, 3)
        keys = jax.random.split(kg, FUT)
        context, context_actions, real = frames[:, :CTX], actions[:, :CTX], frames[:, CTX:]
        generated = rollout(wm, model, context, context_actions, actions[:, CTX:], keys)
        randomized = rollout(wm, model, context, context_actions, clips["random_actions"][b0:b0 + BATCH], keys[:AF_HOR])
        last = frames[:, CTX - 1].astype(np.float32) / 255
        out["err_true"].append(salient_mse(generated[:, :AF_HOR], real[:, :AF_HOR], last, valid))
        out["err_rand"].append(salient_mse(randomized, real[:, :AF_HOR], last, valid))
        out["fr"].append(features(real))
        out["fg"].append(features(generated))
        for mode, video, k in (("tf", frames, kt), ("im", np.concatenate([context, generated], 1), ki)):
            reward, done = read_heads(wm, model, video, actions, k)     # after generation: the real future comes last
            out[f"r_{mode}"].append(reward)
            out[f"d_{mode}"].append(done)
        if preview is None:
            preview = real[:8].copy(), generated[:8].copy()
    return {k: np.concatenate(v) for k, v in out.items()}, preview


def scores(o, clips, rows):
    """The metrics over the clips `rows`, valid future frames only."""
    m = clips["valid"][rows]
    reward, death = clips["rewards"][rows, CTX:], (clips["dones"][rows, CTX:] > 0) & m
    fd = lambda hi: GFVD_SCALE * frechet(o["fr"][rows, :hi][m[:, :hi]], o["fg"][rows, :hi][m[:, :hi]])
    t, r = o["err_true"][rows], o["err_rand"][rows]
    event, ends = np.abs(reward[m]) > 1e-6, death.any(1)
    return {"gFVD": fd(FUT), "gFVD_16": fd(AF_HOR),
            "AF": float(np.mean((r > t + 1e-8) + 0.5 * (np.abs(r - t) <= 1e-8))),
            "reward_ap": average_precision(np.abs(o["r_tf"][rows][m]), event), "reward_base": float(event.mean()),
            "done_ap": average_precision(o["d_tf"][rows][m], death[m]), "done_base": float(death[m].mean()),
            "imagined_return_corr": corr((o["r_im"][rows] * m).sum(1), (reward * m).sum(1)),
            "imagined_done_ap": average_precision(np.where(m, o["d_im"][rows], 0).max(1), ends),
            "imagined_done_base": float(ends.mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--module", default="train", help="the module with the model interface")
    p.add_argument("--data", type=Path, default=Path("data/eval.npz"))
    p.add_argument("--out", type=Path, default=Path("ledger.jsonl"))
    p.add_argument("--tag")
    p.add_argument("--note", default="")
    p.add_argument("--clips", type=int, default=0, help=f"score this many clips spread over the file, a quick check "
                                                         f"(the benchmark scores all {N_CLIPS})")
    p.add_argument("--gif", action="store_true")
    a = p.parse_args()
    data = np.load(a.data)
    clips = {k: data[k] for k in ("frames", "actions", "rewards", "dones", "valid_len", "event")}
    n = len(clips["event"])
    clips["random_actions"] = np.random.default_rng(EVAL_SEED + 7).integers(0, A_N, (n, AF_HOR)).astype(np.int32)
    if a.clips:
        clips = {k: v[np.linspace(0, n - 1, a.clips).round().astype(int)] for k, v in clips.items()}
        n = a.clips
    clips["valid"] = CTX + np.arange(FUT)[None] < clips["valid_len"][:, None]
    start = time.perf_counter()
    wm = importlib.import_module(a.module)
    model, meta = wm.load_model(a.model), wm.load_meta(a.model)
    o, preview = evaluate(wm, model, clips)
    event = clips["event"] > 0
    groups = {"": np.arange(n), "_normal": np.flatnonzero(~event), "_event": np.flatnonzero(event)}
    metrics = {}
    for suffix, rows in groups.items():
        if len(rows):
            metrics |= {k + suffix: v for k, v in scores(o, clips, rows).items()}
    metrics["imagined_return_corr"] = metrics.get("imagined_return_corr_normal")   # defined on the normal clips
    eval_s = round(time.perf_counter() - start, 1)
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": a.tag or a.model.parent.name,
           "note": a.note, **metrics, "train_seed": meta["cfg"]["seed"], "clips": n, "batch": BATCH,
           "train_s": meta["train_s"], "eval_s": eval_s,
           "in_budget": n == N_CLIPS and meta["train_s"] <= TRAIN_BUDGET_S and eval_s <= EVAL_BUDGET_S,
           "module": a.module, "checkpoint": str(a.model)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    if a.gif:
        make_gif(*preview, a.out.parent / "gifs" / f"{row['tag']}.gif")


if __name__ == "__main__":
    main()

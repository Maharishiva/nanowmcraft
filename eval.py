"""Frozen benchmark: open-loop generation quality from random episode positions of the held-out set.

  gFVD  Frechet distance between real and generated 16-frame clips in the feature space of a fixed random CNN
        (seed 1234, never trained), over stacks of 4 consecutive frames. Lower is better.
  AF    action following: the fraction of windows where the rollout under the true actions is closer (salient-pixel
        MSE) to the real future than a rollout from the same context and noise under random actions; ties score
        half. 0.5 = the model ignores actions. Higher is better.

2048 windows, 8 real context frames, 16 generated frames, 8 Euler steps per frame; fixed sampling seeds.

Reward and termination heads, on the first 512 windows plus one window per recorded death (ending on the reset
frame, so the fatal transition is its last one):
  reward_corr / reward_recall / reward_precision   teacher-forced per-step reward against the record; an event is
        |reward| > 0.05 (recorded rewards are multiples of 0.1).
  done_auc   probability that a recorded death is scored above a recorded survival, teacher-forced.
  imagined_reward_corr / imagined_return_corr / imagined_done_auc   the same heads read from the model's own rollout
        under the true actions, per step, over the 16-step return, and for the deaths at the end of the horizon.

    python eval.py --model runs/base/model.npz --tag base
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train

CTX, HOR, N_WIN = 8, 16, 2048
EVAL_SEED, BATCH = 0, 64
CLIP, FEAT_SEED, THR = 4, 1234, 16 / 255
N_HEAD_WIN, EVENT = 512, 0.05


def sample_indices(data, n):
    """(start, env) of n windows of CTX + HOR frames without an episode reset inside, at random positions."""
    dones = data["dones"].astype(bool)
    t, e = dones.shape
    span = CTX + HOR
    cumulative = np.concatenate([np.zeros((1, e), np.int64), np.cumsum(dones, axis=0)])
    rng = np.random.default_rng(EVAL_SEED)
    indices = []
    while len(indices) < n:
        env = int(rng.integers(0, e))
        start = int(rng.integers(1, t - span + 1))
        if cumulative[start + span - 1, env] == cumulative[start, env]:
            indices.append((start, env))
    return np.asarray(indices, dtype=np.int64)


def death_indices(data):
    """(start, env) of one window per recorded death, ending on the reset frame, with no earlier reset inside."""
    dones = data["dones"].astype(bool)
    span = CTX + HOR
    indices = []
    for t, env in zip(*np.nonzero(dones)):
        start = t - span + 2
        if start >= 1 and not dones[start:t, env].any():
            indices.append((start, env))
    return np.asarray(indices, dtype=np.int64)


def window_batch(data, indices):
    """Frames of each window with the action, reward and done flag of the transition INTO each frame; the transition
    into the first frame of a fresh episode becomes NOOP with zero reward, as in training."""
    starts, envs = indices[:, 0], indices[:, 1]
    times = starts[:, None] + np.arange(CTX + HOR)[None]
    frames = data["frames"][times, envs[:, None]]
    actions = data["actions"][times - 1, envs[:, None]].astype(np.int32)
    rewards = data["rewards"][times - 1, envs[:, None]].astype(np.float32)
    dones = data["dones"][times - 1, envs[:, None]].astype(np.float32)
    fresh = dones[:, 0] > 0.5
    actions[fresh, 0], rewards[fresh, 0], dones[:, 0] = 0, 0.0, 0.0
    return frames, actions, rewards, dones


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


def clip_feats(feat, video):
    horizon = video.shape[1]
    batch = max(1, 256 // horizon)
    out = []
    for start in range(0, len(video), batch):
        v = video[start:start + batch].astype(np.float32) / 255
        stacked = np.concatenate([np.repeat(v[:, :1], CLIP - 1, 1), v], 1)
        clips = np.concatenate([stacked[:, t:t + horizon] for t in range(CLIP)], -1)
        out.append(np.asarray(feat(jnp.asarray(clips.reshape(-1, train.H, train.W, 3 * CLIP)))))
    return np.concatenate(out)


def frechet(fr, fg):
    fr, fg = fr.astype(np.float64), fg.astype(np.float64)
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    cr = np.cov(fr, rowvar=False) + 1e-6 * np.eye(fr.shape[1])
    cg = np.cov(fg, rowvar=False) + 1e-6 * np.eye(fg.shape[1])
    eig = np.linalg.eigvals(cr @ cg)
    return float(((mu_r - mu_g) ** 2).sum() + np.trace(cr) + np.trace(cg) - 2 * np.sum(np.sqrt(np.clip(eig.real, 0, None))))


def salient_mse(pred, real, last_ctx):
    real, pred = real.astype(np.float32) / 255, pred.astype(np.float32) / 255
    sal = np.abs(real - last_ctx[:, None]).max(-1, keepdims=True) > THR
    return ((pred - real) ** 2 * sal).sum((1, 2, 3, 4)) / (sal.sum((1, 2, 3, 4)) + 1e-6)


def make_gif(real, predicted, path, k=8, scale=4):
    import imageio.v2 as imageio

    k = min(k, len(real))
    separator = np.full((train.H, 2, 3), (90, 90, 110), np.uint8)
    images = []
    for t in range(real.shape[1]):
        grid = np.concatenate([np.concatenate([real[i, t], separator, predicted[i, t]], 1) for i in range(k)], 0)
        images.append(np.repeat(np.repeat(grid, scale, 0), scale, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, images, duration=1000 / 6, loop=0)


def evaluate(model, data, indices):
    feat = nnx.jit(lambda x, m=FeatCNN(nnx.Rngs(FEAT_SEED)): m(x))
    gdef, state = nnx.split(model.df)
    roll = jax.jit(lambda state, ctx, ca, fa, key: train._rollout_latents(nnx.merge(gdef, state), ctx, ca, fa, key, win=model.win))

    def predict(frames, actions, future_actions, seed):
        ctx = train.encode(model, jnp.asarray(frames[:, :CTX]))
        pz = roll(state, ctx, jnp.asarray(actions[:, :CTX]), jnp.asarray(future_actions).T, jax.random.PRNGKey(seed))
        return np.asarray(train.decode(model, pz)).transpose(1, 0, 2, 3, 4)

    random_actions = np.random.default_rng(EVAL_SEED + 7).integers(0, train.A_N, size=(len(indices), HOR)).astype(np.int32)
    real_features, predicted_features, true_errors, random_errors = [], [], [], []
    preview = None
    for start in range(0, len(indices), BATCH):
        frames, actions, _, _ = window_batch(data, indices[start:start + BATCH])
        batch_seed = int(np.random.SeedSequence([EVAL_SEED, start]).generate_state(1)[0])
        predicted = predict(frames, actions, actions[:, CTX:], batch_seed)
        randomized = predict(frames, actions, random_actions[start:start + BATCH], batch_seed)
        real = frames[:, CTX:]
        last_context = frames[:, CTX - 1].astype(np.float32) / 255
        real_features.append(clip_feats(feat, real))
        predicted_features.append(clip_feats(feat, predicted))
        true_errors.append(salient_mse(predicted, real, last_context))
        random_errors.append(salient_mse(randomized, real, last_context))
        if preview is None:
            preview = real[:8].copy(), predicted[:8].copy()
    true_errors, random_errors = np.concatenate(true_errors), np.concatenate(random_errors)
    af = float(np.mean((random_errors > true_errors + 1e-8) + 0.5 * (np.abs(random_errors - true_errors) <= 1e-8)))
    return frechet(np.concatenate(real_features), np.concatenate(predicted_features)), af, preview


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel())[0, 1])


def auc(scores, labels):
    """Probability that a random positive scores above a random negative (rank statistic)."""
    scores, labels = np.asarray(scores, np.float64).ravel(), np.asarray(labels).ravel() > 0.5
    ranks = scores.argsort().argsort() + 1
    n_pos, n_neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_heads(model, data, indices, n_uniform):
    """Teacher-forced and imagined reward / termination predictions against the record; the first n_uniform windows
    hold no reset, the rest end on one. The first frame of a window has no predicted transition and is skipped."""
    gdef, state = nnx.split(model.df)
    read = jax.jit(lambda state, z, a, key: train._read_heads(nnx.merge(gdef, state), z, a, key))

    @jax.jit
    def imagined(state, ctx, ca, fa, key):
        kr, kz = jax.random.split(key)
        df = nnx.merge(gdef, state)
        z = train._rollout_latents(df, ctx, ca, fa.T, kr, win=model.win).transpose(1, 0, 2, 3, 4)
        heads = train._read_heads(df, jnp.concatenate([ctx, z], 1), jnp.concatenate([ca, fa], 1), kz)
        return jax.tree.map(lambda v: v[:, CTX:], heads)

    out = {k: [] for k in ("reward", "done", "im_reward", "im_done", "true_reward", "true_done")}
    for start in range(0, len(indices), BATCH):
        frames, actions, rewards, dones = window_batch(data, indices[start:start + BATCH])
        key = jax.random.PRNGKey(int(np.random.SeedSequence([EVAL_SEED, 1, start]).generate_state(1)[0]))
        latents = train.encode(model, jnp.asarray(frames))
        forced = read(state, latents, jnp.asarray(actions), key)
        rolled = imagined(state, latents[:, :CTX], jnp.asarray(actions[:, :CTX]), jnp.asarray(actions[:, CTX:]),
                          jax.random.fold_in(key, 1))
        for k, v in zip(("reward", "done", "im_reward", "im_done"), (*forced, *rolled)):
            out[k].append(np.asarray(v))
        out["true_reward"].append(rewards)
        out["true_done"].append(dones)
    o = {k: np.concatenate(v) for k, v in out.items()}
    reward, true = o["reward"][:n_uniform, 1:], o["true_reward"][:n_uniform, 1:]
    event, hit = np.abs(true) > 1e-6, np.abs(reward) > EVENT
    return {"reward_corr": corr(reward, true), "reward_recall": float((hit & event).sum() / max(event.sum(), 1)),
            "reward_precision": float((hit & event).sum() / max(hit.sum(), 1)),
            "done_auc": auc(o["done"][:, 1:], o["true_done"][:, 1:]),
            "imagined_reward_corr": corr(o["im_reward"][:n_uniform], o["true_reward"][:n_uniform, CTX:]),
            "imagined_return_corr": corr(o["im_reward"][:n_uniform].sum(1), o["true_reward"][:n_uniform, CTX:].sum(1)),
            "imagined_done_auc": auc(o["im_done"], o["true_done"][:, CTX:])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/eval.npz"))
    p.add_argument("--out", type=Path, default=Path("ledger.jsonl"))
    p.add_argument("--tag")
    p.add_argument("--note", default="")
    p.add_argument("--windows", type=int, default=N_WIN, help=f"the frozen protocol uses {N_WIN}")
    p.add_argument("--gif", action="store_true")
    a = p.parse_args()
    model, meta = train.load_model(a.model), train.load_meta(a.model)
    data = dict(np.load(a.data))
    start = time.perf_counter()
    indices = sample_indices(data, a.windows)
    gfvd, af, preview = evaluate(model, data, indices)
    n_uniform = min(N_HEAD_WIN, len(indices))
    heads = evaluate_heads(model, data, np.concatenate([indices[:n_uniform], death_indices(data)]), n_uniform)
    tag = a.tag or a.model.parent.name
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": tag, "note": a.note,
           "gFVD": gfvd, "AF": af, **heads, "train_seed": meta["cfg"]["seed"], "n_win": a.windows, "ctx": CTX,
           "hor": HOR, "train_s": meta["train_s"], "eval_s": round(time.perf_counter() - start, 1),
           "checkpoint": str(a.model)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    if a.gif:
        make_gif(*preview, a.out.parent / "gifs" / f"{tag}.gif")


if __name__ == "__main__":
    main()

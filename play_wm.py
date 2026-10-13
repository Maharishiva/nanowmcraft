"""Play inside one or more world models (run locally).

Panes: real Craftax, optionally the tokenizer's reconstruction of the real frame (what the first WM can
actually SEE), then one pane per world model free-running on your keystrokes. Every WM is primed on the
same real frames, then imagines forward and drifts from real over time.

    python play_wm.py --model runs/base/model.npz
    python play_wm.py --model runs/a/model.npz runs/b/model.npz --labels "a .0015" "b .0016"
    python play_wm.py --model ... --no-recon     # hide the tokenizer pane
    python play_wm.py --model ... --selftest     # headless smoke test -> play_selftest.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name

import train

ENV_NAME = "Craftax-Classic-Pixels-v1"
ACT = {"NOOP": 0, "LEFT": 1, "RIGHT": 2, "UP": 3, "DOWN": 4, "DO": 5, "SLEEP": 6,
       "PLACE_STONE": 7, "PLACE_TABLE": 8, "PLACE_FURNACE": 9, "PLACE_PLANT": 10,
       "MAKE_WOOD_PICKAXE": 11, "MAKE_STONE_PICKAXE": 12, "MAKE_IRON_PICKAXE": 13,
       "MAKE_WOOD_SWORD": 14, "MAKE_STONE_SWORD": 15, "MAKE_IRON_SWORD": 16}
HINTS = [("move", "w a s d"), ("do", "space"), ("place", "t table"), ("", "r stone"),
         ("", "f furnace"), ("", "p plant"), ("craft", "1-6"), ("sleep", "tab"),
         ("noop", "q"), ("", ""), ("reset", "bksp"), ("quit", "esc")]
SAMPLERS = {}


def obs_to_u8(obs):
    return np.asarray(jnp.clip(jnp.asarray(obs) * 255.0, 0, 255).astype(jnp.uint8))


def make_env():
    env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
    ep = env.default_params
    return jax.jit(lambda k: env.reset(k, ep)), jax.jit(lambda k, s, a: env.step(k, s, a, ep))


def recon_of(model, obs_u8):
    return np.asarray(train.decode(model, train.encode(model, jnp.asarray(obs_u8)[None])))[0]


def sampler(gdef, steps):
    """One compiled sampler per architecture: models of the same shape share it, the weights are an argument."""
    if (gdef, steps) not in SAMPLERS:
        SAMPLERS[(gdef, steps)] = jax.jit(lambda state, seq, taus, aseq, slot, key: train._flow_one(
            nnx.merge(gdef, state), seq, taus, aseq, slot, key, steps))
    return SAMPLERS[(gdef, steps)]


class WM:
    """Sliding win-frame latent context; imagines the next frame from your action. Real (primed) frames sit in
    the context at signal 1.0, imagined ones at ctx_tau, as in training-time evaluation."""
    def __init__(self, model, steps, ctx_tau):
        self.m, self.ctx_tau = model, ctx_tau
        self.win = model.win
        self.gdef, self.state = nnx.split(model.df)
        self.sample = sampler(self.gdef, steps)
        self.key = jax.random.PRNGKey(0)

    def reset(self, obs0_u8):
        z0 = train.encode(self.m, jnp.asarray(obs0_u8)[None])[0]
        self.seq = jnp.zeros((1, self.win) + z0.shape).at[0, 0].set(z0)
        self.aseq = jnp.zeros((1, self.win), jnp.int32)
        self.taus = jnp.zeros((1, self.win)).at[0, 0].set(1.0)
        self.n = 1
        return np.asarray(train.decode(self.m, z0[None]))[0]

    def step(self, action, prime_obs=None):
        if self.n >= self.win:
            self.seq = self.seq.at[0, :self.win - 1].set(self.seq[0, 1:])
            self.aseq = self.aseq.at[0, :self.win - 1].set(self.aseq[0, 1:])
            self.taus = self.taus.at[0, :self.win - 1].set(self.taus[0, 1:]).at[0, self.win - 1].set(0.0)
            self.n = self.win - 1
        s = self.n
        self.aseq = self.aseq.at[0, s].set(action)
        if prime_obs is not None:
            z = train.encode(self.m, jnp.asarray(prime_obs)[None])[0]
            level = 1.0
        else:
            self.key, kr = jax.random.split(self.key)
            z = self.sample(self.state, self.seq, self.taus, self.aseq, jnp.asarray(s, jnp.int32), kr)[0]
            level = self.ctx_tau
        self.seq = self.seq.at[0, s].set(z)
        self.taus = self.taus.at[0, s].set(level)
        self.n += 1
        return np.asarray(train.decode(self.m, z[None]))[0]


def load_all(args):
    models = [train.load_model(p) for p in args.model]
    labels = args.labels or [p.parent.name for p in args.model]
    return models, labels, [WM(m, args.steps, args.ctx_tau) for m in models]


def run(args):
    import pygame
    models, labels, wms = load_all(args)
    reset, step = make_env()
    rng = jax.random.PRNGKey(args.seed)
    state = real = recon = None
    imgs = [None] * len(wms)
    t = 0
    dead = False

    def new_episode():
        nonlocal rng, state, real, recon, imgs, t, dead
        rng, r0 = jax.random.split(rng)
        obs, state = reset(r0)
        real = obs_to_u8(obs)
        imgs = [wm.reset(real) for wm in wms]
        dead = False
        for _ in range(args.prime):
            rng, ka, ks = jax.random.split(rng, 3)
            a = int(jax.random.randint(ka, (), 0, train.A_N))
            obs, state, _, done, _ = step(ks, state, a)
            real = obs_to_u8(obs)
            imgs = [wm.step(a, real) for wm in wms]
            if done:
                dead = True
                break
        if args.recon:
            recon = recon_of(models[0], real)
        t = 0

    new_episode()
    pygame.init()
    font = pygame.font.SysFont("menlo,consolas,monospace", 15)
    big = pygame.font.SysFont("menlo,consolas,monospace", 17, bold=True)
    n_panes = 1 + int(args.recon) + len(wms)
    rows = 1 if n_panes <= 3 else 2
    cols = -(-n_panes // rows)
    scale = args.scale or {1: 7, 2: 7, 3: 7, 4: 6, 5: 6, 6: 5, 7: 4, 8: 4}.get(n_panes, 3)
    pane, sidebar, gap, head = 63 * scale, 210, 16, 28
    screen = pygame.display.set_mode((gap + cols * (pane + gap) + sidebar, rows * (head + pane + gap)))
    pygame.display.set_caption("Craftax: REAL vs WM")
    KEY = {pygame.K_q: 0, pygame.K_w: 3, pygame.K_a: 1, pygame.K_s: 4, pygame.K_d: 2,
           pygame.K_SPACE: 5, pygame.K_t: 8, pygame.K_r: 7, pygame.K_f: 9, pygame.K_p: 10,
           pygame.K_TAB: 6, pygame.K_1: 11, pygame.K_2: 12, pygame.K_3: 13, pygame.K_4: 14,
           pygame.K_5: 15, pygame.K_6: 16}

    def blit(img, i, label):
        x, y = gap + (i % cols) * (pane + gap), (i // cols) * (head + pane + gap)
        surf = pygame.surfarray.make_surface(np.asarray(img).transpose(1, 0, 2))
        screen.blit(pygame.transform.scale(surf, (pane, pane)), (x, y + head))
        screen.blit(big.render(label, True, (235, 235, 235)), (x, y + 6))

    def draw():
        screen.fill((18, 18, 20))
        panes = [(real, "REAL")] + ([(recon, "TOKENIZER")] if args.recon else []) + list(zip(imgs, labels))
        for i, (img, label) in enumerate(panes):
            blit(img, i, label)
        sx, y = gap + cols * (pane + gap), 10
        for k, v in HINTS:
            screen.blit(font.render(f"{k:<6}{v}" if k else f"      {v}", True, (160, 170, 180)), (sx, y))
            y += 21
        screen.blit(font.render(f"step {t}", True, (120, 200, 130)), (sx, y + 10))
        if dead:
            screen.blit(big.render("DEAD -- bksp to reset", True, (240, 90, 90)), (gap, head + pane - 24))
        pygame.display.flip()

    draw()
    running = True
    while running:
        ev = pygame.event.wait()
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
            running = False
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_BACKSPACE:
            new_episode(); draw()
        elif ev.type == pygame.KEYDOWN and ev.key in KEY and not dead:
            a = KEY[ev.key]
            rng, ks = jax.random.split(rng)
            obs, state, _, done, _ = step(ks, state, a)
            real = obs_to_u8(obs)
            dead = bool(done)
            if args.recon:
                recon = recon_of(models[0], real)
            imgs = [wm.step(a) for wm in wms]
            t += 1
            draw()
    pygame.quit()


def selftest(args):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import time
    import imageio.v2 as imageio
    models, labels, wms = load_all(args)
    reset, step = make_env()
    rng = jax.random.PRNGKey(args.seed)
    rng, r0 = jax.random.split(rng)
    obs, state = reset(r0)
    real = obs_to_u8(obs)
    for wm in wms:
        wm.reset(real)
    for _ in range(min(args.prime, 6)):
        rng, ka, ks = jax.random.split(rng, 3)
        a = int(jax.random.randint(ka, (), 0, train.A_N))
        obs, state, _, _, _ = step(ks, state, a)
        real = obs_to_u8(obs)
        for wm in wms:
            wm.step(a, real)
    up = lambda im: np.repeat(np.repeat(np.asarray(im), 4, 0), 4, 1)
    rows, t0 = [], time.time()
    for a in [4] * 4 + [2] * 3 + [3] * 3:
        rng, ks = jax.random.split(rng)
        obs, state, _, _, _ = step(ks, state, a)
        real = obs_to_u8(obs)
        sep = np.full((up(real).shape[0], 2, 3), 255, np.uint8)
        cols = [up(real), sep] + ([up(recon_of(models[0], real)), sep] if args.recon else [])
        for wm in wms:
            cols += [up(wm.step(a)), sep]
        rows.append(np.concatenate(cols[:-1], 1))
    gap = np.full((4, rows[0].shape[1], 3), 30, np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.concatenate([out, gap, r], 0)
    imageio.imwrite("play_selftest.png", out)
    print(f"selftest ok -> play_selftest.png (REAL{'|RECON' if args.recon else ''}|{'|'.join(labels)}), "
          f"{(time.time() - t0) / 10:.2f}s per step for {len(wms)} models")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, nargs="+", default=[Path("runs/base/model.npz")])
    p.add_argument("--labels", nargs="*", help="one pane label per model (default: the run directory name)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prime", type=int, default=23, help="silent real-context prefill steps at startup")
    p.add_argument("--steps", type=int, default=train.STEPS, help="ODE sampling steps per frame")
    p.add_argument("--ctx-tau", type=float, default=train.CTX_SIGNAL)
    p.add_argument("--scale", type=int, default=0, help="pixels per game pixel (default: fit the pane count)")
    p.add_argument("--recon", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    for mod in [train]:                     # fp32 on CPU (params are fp32; bf16 is GPU-only)
        mod.CDT = jnp.float32
        mod.MP = dict(dtype=jnp.float32, param_dtype=jnp.float32)
    (selftest if a.selftest else run)(a)

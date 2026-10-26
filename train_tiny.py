import os, sys, time
from functools import partial
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")  # cuBLAS matmuls compile much faster
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

SIZE, PATCH, WIDTH, HEADS, DEPTH, CHANNELS = 63, 7, 256, 8, 6, 128
WINDOW, TOKENS, LATENT, STEPS, BINS, CONTEXT = 32, 16, 32, 8, 41, .9  # CONTEXT: the signal level of past frames
ACTIONS, LR, WARMUP, LOG_EVERY = 17, 1e-3, 200, 500
CODEC_STEPS, CODEC_BATCH, WORLD_STEPS, WORLD_BATCH = 5000, 256, 11000, 64
GRID, PATCHES, PATCH_DIM = SIZE // PATCH, (SIZE // PATCH) ** 2, PATCH * PATCH * 3
REWARDS = (jnp.arange(BINS) - BINS // 2) * .1
Linear, Conv = partial(nnx.Linear, dtype=jnp.bfloat16), partial(nnx.Conv, dtype=jnp.bfloat16)

def rms(value): return value * jax.lax.rsqrt(jnp.mean(value.astype(jnp.float32) ** 2, -1, keepdims=True) + 1e-6)
def noise(value, level, key): return level * value + (1 - level) * random.normal(key, value.shape)

class Attention(nnx.Module):  # along axis 2 of x (B, S, T, WIDTH); with a window's keys and values cached, x is slot
    def __init__(self, rngs):
        self.qkv, self.out = Linear(WIDTH, 3 * WIDTH, rngs=rngs), Linear(WIDTH, WIDTH, rngs=rngs)

    def __call__(self, x, mask=None, cache=None, slot=0):
        q, k, v = jnp.split(self.qkv(x).reshape(x.shape[:3] + (3 * HEADS, WIDTH // HEADS)), 3, axis=3)
        if cache is not None:
            k, v = (jax.lax.dynamic_update_slice_in_dim(old, new, slot, axis=2) for old, new in zip(cache, (k, v)))
        out = jax.nn.dot_product_attention(*(t.reshape((-1,) + t.shape[2:]) for t in (q, k, v)), mask=mask)
        return self.out(out.reshape(x.shape)), (k, v)

class Block(nnx.Module):  # adaLN-zero: attention in space, causal attention in time, MLP
    def __init__(self, rngs):
        self.modulation = Linear(WIDTH, 9 * WIDTH, kernel_init=nnx.initializers.zeros, rngs=rngs)
        self.space, self.time = Attention(rngs), Attention(rngs)
        self.mlp = nnx.Sequential(Linear(WIDTH, 2 * WIDTH, rngs=rngs), nnx.gelu, Linear(2 * WIDTH, WIDTH, rngs=rngs))

    def __call__(self, hidden, condition, mask, cache=None, slot=0):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = jnp.split(self.modulation(nnx.silu(condition))[:, :, None], 9, -1)
        hidden = hidden + g1 * self.space(rms(hidden) * (1 + s1) + b1)[0]
        branch, kv = self.time((rms(hidden) * (1 + s2) + b2).swapaxes(1, 2), mask, cache, slot)
        hidden = hidden + g2 * branch.swapaxes(1, 2)
        return hidden + g3 * self.mlp(rms(hidden) * (1 + s3) + b3), kv

class Residual(nnx.Module):
    def __init__(self, rngs):
        self.first, self.second = (Conv(CHANNELS, CHANNELS, (3, 3), rngs=rngs) for _ in range(2))

    def __call__(self, hidden):
        return hidden + self.second(nnx.gelu(rms(self.first(nnx.gelu(rms(hidden))))))

class Tokenizer(nnx.Module):  # each 7x7 tile one vector of a 9x9 grid, residual convs, a stride-2 conv to 4x4, residual convs
    def __init__(self, rngs):
        residual = lambda: [Residual(rngs) for _ in range(3)]
        down = Conv(CHANNELS, CHANNELS, (3, 3), strides=2, padding="VALID", rngs=rngs)
        up = nnx.ConvTranspose(CHANNELS, CHANNELS, (3, 3), strides=2, padding="VALID", dtype=jnp.bfloat16, rngs=rngs)
        self.encoder = nnx.Sequential(Linear(PATCH_DIM, CHANNELS, rngs=rngs), *residual(), down, *residual(),
                                      Linear(CHANNELS, LATENT, rngs=rngs))
        self.decoder = nnx.Sequential(Linear(LATENT, CHANNELS, rngs=rngs), *residual(), up, *residual(),
                                      Linear(CHANNELS, PATCH_DIM, rngs=rngs), nnx.sigmoid)
        self.mean, self.std = nnx.Variable(jnp.zeros(LATENT)), nnx.Variable(jnp.ones(LATENT))

    def encode(self, frames):
        patches = frames.reshape(-1, GRID, PATCH, GRID, PATCH, 3).swapaxes(2, 3).reshape(-1, GRID, GRID, PATCH_DIM) / 255.
        return ((self.encoder(patches) - self.mean[...]) / self.std[...]).reshape(frames.shape[:-3] + (TOKENS, LATENT))

    def decode(self, latent):
        tiles = self.decoder((latent * self.std[...] + self.mean[...]).reshape(-1, GRID // 2, GRID // 2, LATENT))
        return tiles.reshape(-1, GRID, GRID, PATCH, PATCH, 3).swapaxes(2, 3).reshape(latent.shape[:-2] + (SIZE, SIZE, 3))

class WorldModel(nnx.Module):
    def __init__(self, rngs):
        self.input = Linear(LATENT, WIDTH, rngs=rngs)
        self.position = nnx.Param(.02 * random.normal(rngs.params(), (WINDOW, TOKENS, WIDTH)))
        self.blocks = nnx.List([Block(rngs) for _ in range(DEPTH)])
        self.level, self.action = Linear(WIDTH, WIDTH, rngs=rngs), nnx.Embed(ACTIONS + 1, WIDTH, rngs=rngs)
        self.output = Linear(WIDTH, LATENT, kernel_init=nnx.initializers.zeros, rngs=rngs)
        self.reward = nnx.Sequential(Linear(WIDTH, WIDTH, rngs=rngs), nnx.gelu, Linear(WIDTH, BINS, rngs=rngs))
        self.done = nnx.Sequential(Linear(2 * WIDTH, WIDTH, rngs=rngs), nnx.gelu, Linear(WIDTH, 1, rngs=rngs))

    def __call__(self, latent, levels, actions, resets, cache=None, slot=0):  # with every block's time cache: from slot on
        episode = jnp.cumsum(resets, 1)
        mask = jnp.repeat(jnp.tril(episode[:, :, None] == episode[:, None, :]), TOKENS, 0)[:, None] if cache is None \
            else (jnp.arange(WINDOW) <= slot)[None, None, None]
        frequencies = levels[..., None] * jnp.exp(jnp.linspace(0., jnp.log(1000.), WIDTH // 2))
        action = self.action(jnp.where(resets, ACTIONS, actions))
        condition = self.level(jnp.concatenate([jnp.sin(frequencies), jnp.cos(frequencies)], -1)) + action
        hidden, caches = self.input(latent) + action[:, :, None] + jax.lax.dynamic_slice_in_dim(self.position[...], slot, latent.shape[1]), []
        for block, kv in zip(self.blocks, cache or [None] * DEPTH):
            hidden, kv = block(hidden, condition, mask, kv, slot)
            caches.append(kv)
        hidden = rms(hidden)
        summary = hidden.mean(2)  # the reward read off the frame after the step, the end off the frame before and the action
        heads = jnp.concatenate([self.reward(summary[:, 1:]), self.done(jnp.concatenate([summary[:, :-1], self.action(actions[:, 1:])], -1))], -1)
        return self.output(hidden).astype(jnp.float32), heads.astype(jnp.float32), caches

def tokenizer_loss(tokenizer, frames, key, batch):
    frames = frames[random.randint(key, (batch,), 0, len(frames))]
    return jnp.mean((tokenizer.decode(tokenizer.encode(frames)) - frames / 255.) ** 2)

def world_loss(model, data, key, batch):
    start_key, env_key, level_key, noise_key = random.split(key, 4)
    times = random.randint(start_key, (batch, 1), 0, len(data[0]) - WINDOW + 1) + jnp.arange(WINDOW)
    envs = random.randint(env_key, (batch, 1), 0, data[0].shape[1])
    latent, actions, rewards, dones, truncated = (array[times, envs] for array in data)
    levels = random.uniform(level_key, actions.shape)
    prediction, heads, _ = model(noise(latent, levels[..., None, None], noise_key), levels, actions, jnp.maximum(dones, truncated))
    flow = jnp.mean((prediction - latent) ** 2, axis=(2, 3))
    reward = optax.softmax_cross_entropy_with_integer_labels(heads[..., :-1], jnp.abs(rewards[:, 1:, None] - REWARDS).argmin(-1))
    done = (1 - truncated[:, 1:]) * optax.sigmoid_binary_cross_entropy(heads[..., -1], dones[:, 1:])
    weight = .1 + .9 * levels
    return jnp.mean(weight * flow) + .1 * jnp.mean(weight[:, :-1] * (reward + done))

def fit(model, loss, data, steps, batch, seed):
    schedule = optax.warmup_cosine_decay_schedule(0., LR, WARMUP, steps)
    dims = lambda path, p: optax.contrib.MuonDimensionNumbers(tuple(range(p.ndim - 1)), -1) if "kernel" in str(path) else None
    muon = optax.contrib.muon(schedule, weight_decay=.01, adam_weight_decay=.01, consistent_rms=0.2,
                              muon_weight_dimension_numbers=lambda params: jax.tree_util.tree_map_with_path(dims, params))
    optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.), muon), wrt=nnx.Param)

    @nnx.jit
    @nnx.scan(in_axes=(nnx.Carry, None, 0), out_axes=(nnx.Carry, 0))
    def update(state, data, key):
        model, optimizer = state
        value, gradients = nnx.value_and_grad(loss)(model, data, key, batch)
        optimizer.update(model, gradients)
        return (model, optimizer), value

    for chunk, keys in enumerate(random.split(random.key(seed), (steps // LOG_EVERY, LOG_EVERY)), 1):
        print(f"{loss.__name__} step {chunk * LOG_EVERY}: {update((model, optimizer), data, keys)[1].mean():.4f}")

window = lambda value: jnp.pad(value, [(0, 0), (0, WINDOW - value.shape[1])] + [(0, 0)] * (value.ndim - 2))
start = jax.jit(lambda model, frames, actions: (window(model[0].encode(frames)), window(actions), frames.shape[1]))

@jax.jit
def step(model, state, action, key):  # state: the latent window, its actions, the next slot; the window slides once full
    (tokenizer, world), (latent, actions, slot), (context_key, sample_key) = model, state, random.split(key)
    latent, actions = (jnp.where(slot == WINDOW, jnp.roll(value, -1, 1), value) for value in (latent, actions))
    slot = jnp.minimum(slot, WINDOW - 1)
    actions = actions.at[:, slot].set(action)
    cache = world(noise(latent, CONTEXT, context_key), jnp.full(actions.shape, CONTEXT), actions, 0 * actions)[2]
    current = jax.lax.dynamic_slice_in_dim(actions, slot, 1, 1)
    def euler(sample, level):  # the frame at slot alone, on the cached keys and values of the frames before it
        clean = world(sample[:, None], jnp.full(current.shape, level / STEPS), current, 0 * current, cache, slot)[0][:, 0]
        return sample + (clean - sample) / (STEPS - level), None
    sample = jax.lax.scan(euler, random.normal(sample_key, latent[:, 0].shape), jnp.arange(STEPS))[0]
    return (tokenizer.decode(sample) * 255).round().astype(jnp.uint8), (latent.at[:, slot].set(sample), actions, slot + 1)

@jax.jit
def heads(model, frames, actions, key):  # teacher-forced reward and done probability of the step into each frame, 0 at the first
    latent = noise(model[0].encode(frames), CONTEXT, key)
    logits = jnp.pad(model[1](latent, jnp.full(actions.shape, CONTEXT), actions, 0 * actions)[1], ((0, 0), (1, 0), (0, 0)), constant_values=-1e9)
    return jax.nn.softmax(logits[..., :-1]) @ REWARDS, jax.nn.sigmoid(logits[..., -1])

def load_model(path):
    file, model = np.load(path), (Tokenizer(nnx.Rngs(0)), WorldModel(nnx.Rngs(0)))
    return jax.tree.unflatten(jax.tree.structure(model), [file[f"arr_{index}"] for index in range(len(file.files) - 2)])
load_meta = lambda path: {"cfg": {"seed": int(np.load(path)["seed"])}, "train_s": float(np.load(path)["train_s"])}

def train(data="data/train.npz", out="runs/tiny/model.npz", seed=0):
    began, data, seed = time.time(), np.load(data), int(seed)
    frames = jnp.asarray(data["frames"])
    tokenizer, model = Tokenizer(nnx.Rngs(seed)), WorldModel(nnx.Rngs(seed + 1))
    fit(tokenizer, tokenizer_loss, frames.reshape(-1, SIZE, SIZE, 3), CODEC_STEPS, CODEC_BATCH, seed)
    latent = nnx.jit(lambda codec, images: jax.lax.map(codec.encode, images))(tokenizer, frames)
    tokenizer.mean[...], tokenizer.std[...] = latent.mean((0, 1, 2)), latent.std((0, 1, 2))
    latent = (latent - tokenizer.mean[...]) / tokenizer.std[...]
    pad = lambda array, first=0: jnp.pad(jnp.asarray(array, float), ((1, 0), (0, 0)), constant_values=first)
    truncated = data["truncated"] if "truncated" in data else np.zeros_like(data["dones"])
    windows = latent, pad(data["actions"]).astype(int), pad(data["rewards"]), pad(data["dones"], 1), pad(truncated)
    fit(model, world_loss, windows, WORLD_STEPS, WORLD_BATCH, seed)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, *jax.tree.leaves((tokenizer, model)), seed=seed, train_s=time.time() - began)
    return tokenizer, model

if __name__ == "__main__":  # python train_tiny.py --data data/train.npz --out runs/tiny/model.npz --seed 0
    train(**{flag[2:]: value for flag, value in zip(sys.argv[1::2], sys.argv[2::2])})

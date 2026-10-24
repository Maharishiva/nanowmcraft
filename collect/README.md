# collect: how the training set is made

The 500k training set and the held-out pools come from two files:
`collect.py` (the Craftax parts: policies, the rules for forced attempts, the trip planner, the event list) and
`core.py` (the game-free parts: behaviour schedule, return trips, cutting and packing).

```
python collect/collect.py pool --out data/pool0 --seed 10                                   # 512 worlds x 4000 steps
python collect/collect.py pool --out data/eval --seed 7101 --worlds 256 --steps 2000 --states  # a held-out pool
python collect/collect.py pack --pools data/pool0 data/pool1 --out data                     # train.npz
```

`--policies` is a folder with the four checkpoints named in `POLICIES`, from [policies/](https://huggingface.co/maharishiva/tinywmcraft-v3/tree/main/policies) on Hugging Face.

## Who plays

Four RL policies run on every world at every step, so each one keeps its own memory; one of them acts.

| episodes | policy |
|---:|---|
| 60% | pixel PPO fine-tuned for diamonds (about 21 achievements per episode, a diamond in 75% of them), plus a small goal head; with no goal it plays as the plain policy |
| 15% | the same pixel PPO policy before the diamond fine-tune |
| 10% | a pixel policy distilled from the symbolic one |
| 15% | symbolic PPO |

Each episode draws its driver. About once per 800 steps another policy takes over for 50 to 400 steps.

## Behaviour schedule

Good agents have habits (the sword before the pickaxe, iron crafts right after the furnace is placed), and a world
model learns habits as if they were game rules. The schedule breaks them with generic tools. Each episode is calm
(intensity 0.2: long progress and diamonds) or chaotic (intensity 1: odd events and deaths); the intensity scales
every hazard below.

- **Hot windows.** About every 200 steps: 5 to 30 steps at sampling temperature 2 to 8, or in about a third of them
  3 to 24 uniformly random actions; then 20 to 80 steps at temperature 1 to 1.5. Paused in danger; never sleep.
- **Tool order.** Per tier and episode, a fair coin swaps "make pickaxe" and "make sword" until both tools of the tier
  are held. In half the episodes a tier's crafts wait until the materials for both tools are held. Now and then the
  stone and iron pickaxes and the swords are banned for 100 to 800 steps.
- **Setups.** After a table or a furnace is placed: a detour (hot steps, then no new setup for 60 to 250 steps), or
  the same action again 1 to 40 steps later. In 75% of the episodes the iron crafts wait 32 to 96 steps after a new
  furnace and the setup action itself 96 to 256 steps, so crafts also happen at an old furnace or table.
- **Needs.** Rarely, one need (eating, drinking, sleeping, fighting back) is ignored for 300 to 1500 steps. This gives
  deaths by starvation, thirst, fatigue and mobs.
- **Forced attempts.** Rules on the state propose actions that fail or work by the game logic: mining with a too weak
  tool, crafts at a table, iron crafts with no furnace or at an old one, the tier's second tool, second tables and
  furnaces, furnace presses that fail, plants and saplings, sleep by day, walks toward and into lava. At most one per
  4 to 16 steps.

## Return trips

A world model has to remember what left the view: a few rows (peeks, back-and-forth) and the whole view. When the
player is safe (daylight or dusk, awake, healthy, needs not low, no mob within 3 tiles), about every 3 steps of free
play a trip starts. It draws a walking distance and a tour:

| distance | trips | tours |
|---|---:|---|
| 2-3 tiles (peeks) | 38% | mostly back and forth two or three times (A-B-A-B-A), out and back |
| 4-8 tiles, the start leaves the view (4+ rows or 5+ columns) | 38% | out and back, back and forth, A-B-C-A |
| 9-16 tiles | 17% | out and back, back and forth, A-B-C-A |
| 17-63 tiles | 7% | out and back, A-B-C-A, back to an old place |

A is the start, B a point at that distance and C one at half of it; an old place is a spot seen earlier (distinctive
views are preferred). A breadth-first planner over grass, sand and path (no mobs, not next to lava) walks most legs;
the learned goal walker (the 60% policy steered by its goal head) walks 10% of the 4-8 tile loops, 30% of the 9-16
and half of the longer trips, as it takes about two steps per tile. One planner step in ten is the policy's own
action. Waypoints may get 1 to 6 steps of play (more on long trips), and the end gets 1 to 6 free steps. Danger for
more than 10 steps, no path, being stuck or running out of steps ends a trip. A new table or furnace starts an
out-and-back of 4+ tiles in 30% of the cases, so the agent comes back to it.

## Packer

A pool is one long recording (512 worlds x 4000 steps, about 2M transitions). The packer:

1. marks every transition **busy** (an event from `events()`, a craft, place or sleep press, the do action on anything
   but ground, a trip step, a forced or repeated action) or **quiet** (no reward, no inventory or vitals change, no
   trip); a press while asleep does nothing in the game, so it is no event;
2. protects **complete rollouts** of under 1000 steps, which are never cut: episode start to the first diamond,
   sapling collected to planted to eaten ripe (plants ripen after 600 steps), and whole episodes that end in a death
   (in every 4th world, as there are plenty);
3. drops stretches with nothing busy for over 64 steps (16 steps kept at each end) and the middle of every sleep
   (8 steps kept at each end), and cuts the rest into pieces of 128 to about 512 transitions that end at an episode
   end or on a quiet transition, never on an event; a protected rollout makes its piece as long as it needs;
4. picks pieces to balance events, not frames: each time the piece with the best gain per (300 + its length). Each
   event type is worth up to its target (its supply, at most 400 unless set); counterexamples to known world-model
   flaws and a few rare events weigh 3x, first diamonds, deaths and complete rollouts 6x, sleep onsets 12x. The cells
   coming into view are events too, by how long ago they were last on screen (2-3, 4-7, 8-15, 16-31, 32-127, 128+
   frames), with targets per band; 4-7 and 8-15 weigh most, cells never seen before come by themselves;
5. packs the pieces into 250 streams of 2001 frames (longest first, first fit) and fills the free tails with the starts
   of spare pieces (the most eventful first). Every stream keeps at least one frame for such a tail, so a picked piece
   always keeps its last transition (and its death).

## Data format

`train.npz`, the benchmark set:

| key | shape | meaning |
|---|---|---|
| frames | uint8 [2001, 250, 63, 63, 3] | 250 streams of 2001 frames |
| actions | int16 [2000, 250] | the 17 Craftax-Classic actions |
| rewards | float32 [2000, 250] | |
| dones | bool [2000, 250] | deaths only |
| truncated | bool [2000, 250] | a piece ends here (cut on a quiet transition, or the time limit) |
| starts | bool [2000, 250] | the first frame of a piece that begins at an episode start (step 0): a real game start |
| meta | json string | recipe, policies, event supply and picks |

`actions[t, e]` and `rewards[t, e]` take `frames[t, e]` to `frames[t + 1, e]`. After a done or truncated transition,
`frames[t + 1, e]` is the first frame of the next piece. Many pieces begin mid-game; `starts` marks the ones that
begin at a real game start, so a trainer can sample game starts from them.

`train_labels.npz` has side labels in the same layout, for analysis: per frame the true position, direction, the 7 x 9
view as tile ids, the HUD numbers, achievements, light, the facing block and table, furnace and mobs near; per
transition the death cause, driver, behaviour mode, intensity and trip records; `src` = (pool, world, frame) of every
frame.

A pool directory holds `frames.npy`, `labels.npz` and `records.npz`. With `--states` (held-out pools) it also holds
`states.npz`: the full game state before every action, actions, rewards, dones and the terminal states, so eval clips
can start from any state.

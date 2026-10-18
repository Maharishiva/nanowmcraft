# program.md

How to beat tinywmcraft, for people and for coding agents.

The goal: a lower gFVD than the best so far (iteration 3, 42.5 ± 2.1, shown times 10^4) from one 10-minute A100 run, without making action following, rewards or episode ends worse.
Every iteration is on the [benchmark](https://maharishiva.github.io/tinywmcraft/benchmark.html).

## Rules

- **Data.** Train only on [`train.npz`](https://huggingface.co/datasets/maharishiva/tinywmcraft) (in `data/`), at most 500k transitions. No other data, no pretrained weights.
- **Training.** At most 600 s per run on one A100 40GB: data loading, tokenizer, compilation, everything. No persistent compilation cache. `train.py` stores its own wall clock as `train_s` in the checkpoint.
- **Evaluation.** `eval.py` stays byte-identical to `main`. Its `eval_s`, timed from importing your module to the last metric (model loading and compilation count, reading `eval.npz` does not), is at most 180 s on the same GPU; iteration 3 takes about 103 s. Anything else about generation is yours: the model family, the sampler, the number of steps.
- **One frame per action.** The model generates frame by frame and never sees future actions or frames. The interface below hands it one action at a time; reading the eval data, caching frames across calls or other tricks around it are against the rules and checked in review.
- **Seeds.** Training seeds 0, 1 and 2. Report all three rows.
- **Gates.** Measured in the maintainer's rerun, on the same machine and seeds. Ranked on gFVD alone:
  - gFVD: the best mean minus the new mean must exceed `1.65 * sqrt((sd_best^2 + sd_new^2) / 3)`, where `sd` is the sample SD over the three seeds. Stable recipes need a smaller margin.
  - AF and reward AP: the new three-seed mean stays at or above a fixed floor, 95% of iteration 1's mean (`train_tiny.py`). The floors do not move when the best changes.
  - done AP: reported, not gated. Deaths are rare (142 of the 2048 clips), so it is too noisy to gate on.

| AF | reward_ap |
|---:|---:|
| 0.873 | 0.879 |

One run is noisy: compare three-seed means, never single runs.

## The model interface

`eval.py` imports `train` (or `--module name`) and calls only these:

```python
load_model(path) -> model
load_meta(path)  -> {"cfg": {"seed": int, ...}, "train_s": float, ...}
start(model, frames, actions) -> state            # frames (B, n0, 63, 63, 3) uint8, actions (B, n0): the action INTO each frame
step(model, state, action, key) -> frame, state   # action (B,), frame (B, 63, 63, 3) uint8: exactly one new frame
heads(model, frames, actions, key) -> reward, done  # teacher-forced, (B, L) each; done is a probability, done[:, 0] = 0
```

For every clip, eval calls `start` with 8 real frames, then `step` 64 times, then `heads` on real frames.
`play_wm.py` uses the same interface, so any entry can be played.

## The loop

1. Read `README.md`, this file, and `train.py`.
2. Pick one idea. Change `train.py` and `kernels.py` (new files they import are fine). Never change `eval.py` or the data.
3. Train: `python train.py --seed 0 --out runs/<tag>/model.npz`. Check that `train_s` is at most 600.
4. Score: `python eval.py --model runs/<tag>/model.npz --tag <tag> --note "<the idea in a few words>"`. Every run appends a row to `ledger.jsonl`, kept or not.
5. Keep the change if gFVD drops and AF and the heads hold; otherwise revert it.
6. When an idea survives one seed, run seeds 1 and 2 and check the gates above.

Exploratory runs can use any GPU and any seed; timings from other GPUs are only indicative. Iterations are decided on a Colab A100-SXM4-40GB.

## Submitting

1. Open a pull request with your changes. The template asks for the three `ledger.jsonl` rows and links to the three `model.npz` files on Hugging Face, so anyone can play and score them.
2. The maintainer retrains the three seeds from the pull request on a Colab A100 next to the best so far.
3. If the rerun passes the gates, the maintainer adds it to `docs/records.jsonl` as the next iteration and merges. The benchmark on the site updates with the merge and credits you with a link to the pull request.
4. The crown goes next to your GitHub login, and your model becomes the one the site plays. If it does not export to the browser with `docs/export_web.py`, the maintainer helps.

## Open problems

- **Rare events.** Diamonds, iron tools, eating plants and deaths are a small share of the data, and a quarter of the eval clips hold one. A model that gets them right scores better, and makes the [challenge](https://maharishiva.github.io/tinywmcraft/#challenge) in the browser fairer.
- **Cheaper sampling.** Eight Euler steps per frame. Fewer steps, better solvers, distillation, or a model that is not flow matching at all could free eval time and let a bigger model train.
- **Long horizons.** The benchmark scores 64 frames; the browser plays endless generated worlds. Anything that keeps the world coherent longer shows in the player first.

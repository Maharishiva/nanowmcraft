# program.md

How to beat tinywmcraft, for people and for coding agents.

The goal: a lower gFVD than the current record (14.7 ± 1.9, shown times 10^4) from one 10-minute A100 run, without making action following, rewards or episode ends worse.
The current record and every earlier one are on the [leaderboard](https://maharishiva.github.io/tinywmcraft/#leaderboard).

## Rules

- **Data.** Train only on [`train.npz`](https://huggingface.co/datasets/maharishiva/tinywmcraft_500k) (in `data/`), at most 500k transitions. No other data, no pretrained weights.
- **Training.** At most 600 s per run on one A100 40GB: data loading, tokenizer, compilation, everything. No persistent compilation cache. `train.py` stores its own wall clock as `train_s` in the checkpoint.
- **Evaluation.** `eval.py` stays byte-identical to `main`. Its `eval_s`, timed from importing your module to the last metric (model loading and compilation count, reading `eval.npz` does not), is at most 180 s on the same GPU; the record takes about 155 s. Anything else about generation is yours: the model family, the sampler, the number of steps.
- **One frame per action.** The model generates frame by frame and never sees future actions. The interface below enforces this.
- **Seeds.** Training seeds 0, 1 and 2. Report all three rows.
- **Gates.** Both measured in the maintainer's rerun, on the same machine and seeds:
  - gFVD: the record's mean minus the new mean must exceed `1.65 * sqrt((sd_record^2 + sd_new^2) / 3)`, where `sd` is the sample SD over the three seeds. With the current record (sd 1.9) and a new recipe of sd 1.0, the new mean must be below about 12.6. Stable recipes need a smaller margin.
  - AF and every reward and termination metric in the ledger row: the new mean at least 98% of the record's mean.

One run is noisy: seeds 0, 1 and 2 of the record score 13.5, 16.9 and 13.7. Compare three-seed means, never single runs.

## The model interface

`eval.py` imports `train` (or `--module name`) and calls only these:

```python
load_model(path) -> model
load_meta(path)  -> {"cfg": {"seed": int, ...}, "train_s": float, ...}
start(model, frames, actions) -> state            # frames (B, n0, 63, 63, 3) uint8, actions (B, n0): the action INTO each frame
step(model, state, action, key) -> frame, state   # action (B,), frame (B, 63, 63, 3) uint8: exactly one new frame
heads(model, frames, actions, key) -> reward, done  # teacher-forced, (B, L) each; done is a probability, done[:, 0] = 0
```

Per batch of 64 windows, eval calls `start` with 8 real frames, then `step` 16 times, then `heads` on 24-frame windows of real frames and of 8 real plus 16 generated frames.
`play_wm.py` uses the same interface, so any entry can be played.

## The loop

1. Read `README.md`, this file, and `train.py`.
2. Pick one idea. Change `train.py` and `kernels.py` (new files they import are fine). Never change `eval.py` or the data.
3. Train: `python train.py --seed 0 --out runs/<tag>/model.npz`. Check that `train_s` is at most 600.
4. Score: `python eval.py --model runs/<tag>/model.npz --tag <tag> --note "<the idea in a few words>"`. Every run appends a row to `ledger.jsonl`, kept or not.
5. Keep the change if gFVD drops and AF and the heads hold; otherwise revert it.
6. When an idea survives one seed, run seeds 1 and 2 and check the gates above.

Exploratory runs can use any GPU and any seed; timings from other GPUs are only indicative. Records are decided on a Colab A100-SXM4-40GB.

## Submitting a record

1. Open a pull request with your changes. The template asks for the three `ledger.jsonl` rows and links to the three `model.npz` files on Hugging Face, so anyone can play and score them.
2. The maintainer retrains the three seeds from the pull request on a Colab A100 next to the current record.
3. If the rerun passes the gates, the maintainer adds your record to `docs/records.jsonl` and merges. The leaderboard on the site updates with the merge and credits you with a link to the pull request.

## Open problems

- **Frames against rewards.** The [`normuon`](https://github.com/Maharishiva/tinywmcraft/tree/normuon) branch lowers gFVD by 16% (12.3) with shortcut targets and scaled head gradients, but reward precision drops 8%, so it fails the gates. Keep the frame gain without losing the reward head. See [validation/colab_20261015.md](validation/colab_20261015.md).
- **Cheaper sampling.** The record spends 8 Euler steps per frame. Fewer steps, second-order solvers, distillation, or a model that is not flow matching at all could free eval time and let a bigger model train.
- **Long horizons.** The benchmark scores 16 frames; in play the world drifts after a few dozen. Anything that keeps it coherent longer will show in the browser first.

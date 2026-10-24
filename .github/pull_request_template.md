## What changed

<!-- One or two sentences. -->

## Results

Training seeds 0, 1 and 2, the three rows `eval.py` appended to `ledger.jsonl`:

```jsonl

```

- GPU:
- Checkpoints on Hugging Face (seed 0, 1, 2):

## Checklist

- [ ] `eval.py` and the data are unchanged
- [ ] every training run took at most 600 s cold (`train_s`) and every eval at most 300 s (`eval_s`)
- [ ] the change lives in `train.py`, `kernels.py`, or files they import
- [ ] the gates pass on the public eval: `python docs/make_records.py add --dry-run --ledger ledger.jsonl --tags <seed 0> <seed 1> <seed 2> --n <next> --change "..." --code <url> --checkpoints <url>`

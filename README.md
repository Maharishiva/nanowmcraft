# nanowmcraft: normuon recipe

The code behind the published checkpoint [normuon_head_bridge_s0](https://huggingface.co/maharishiva/nanowmcraft) (seed 0, gFVD 0.00125, AF 0.920, 567 s cold on one A100).
Recipe: causal two-frame tokenizer, shortcut targets, generated-history training, temporal-difference loss, scheduled head-gradient control and NorMuon.
It has not passed the three-seed record gate; the rules and the current record are on [main](https://github.com/Maharishiva/nanowmcraft).

```bash
uv venv --python 3.11 && uv pip install -r pyproject.toml --extra play && source .venv/bin/activate
curl -L --fail --create-dirs -o runs/demo/model.npz https://huggingface.co/maharishiva/nanowmcraft/resolve/main/normuon_head_bridge_s0/model.npz
python play_wm.py --model runs/demo/model.npz
```

#!/usr/bin/env bash
# What a new best model repeats for the site, from its checkpoint and held-out play (CPU is fine, about 5 minutes):
#
#   docs/release_web.sh runs/best/model.npz [data/eval.npz] [train.py]
#
#   1. export_web.py    docs/play.onnx, the slot graph, checked against the training sampler at slots 0, 1, 7, 30, 31
#   2. convert_web.py   docs/play.wgr (fp16, 23 MB) and docs/play_int8.wgr (int8, 12 MB) with docs/play_int8.onnx
#   3. the pictures     docs/assets/play.gif (a held-out window in daylight where the inventory grows, README and
#                       benchmark page), dream.gif (worlds imagined from noise and played inside the model by the
#                       diamond agent, policies/walker.pkl from Hugging Face; a GPU helps), architecture.gif and
#                       architecture_v1.gif (phones), all from this checkpoint
# then the steps it prints: the first frames and the tutorial world in a browser (seeds.html), the records
# (make_records.py) and the pin on Hugging Face (pin_web.py). The gif scripts import the repo's train.py; the export
# takes any module with --train. RUN picks the Python: RUN="uv run --python .venv/bin/python --with ..." and so on.
set -euo pipefail
MODEL=${1:?usage: docs/release_web.sh MODEL.npz [EVAL.npz] [TRAIN.py]}
DATA=${2:-data/eval.npz}
TRAIN=${3:-train.py}
cd "$(dirname "$0")/.."
RUN=${RUN:-uv run --with jax2onnx --with onnx --with onnxruntime python}

$RUN docs/export_web.py --model "$MODEL" --data "$DATA" --train "$TRAIN"
$RUN docs/convert_web.py
$RUN docs/convert_web.py --int8 --int8-onnx docs/play_int8.onnx --out docs/play_int8.wgr
WINDOW="--style progress --steps 48 --rank ${RANK:-5}"     # look at ranks 0-5 and keep the one the model follows best
$RUN docs/make_play_gif.py --model "$MODEL" --data "$DATA" $WINDOW --out docs/assets/play.gif
$RUN docs/make_dream_gif.py --model "$MODEL" --data "$DATA" --policy policies/walker.pkl --worlds 256 --steps 450 --seed 1
$RUN docs/make_architecture_gif.py --model "$MODEL" --data "$DATA" $WINDOW
$RUN docs/make_architecture_gif.py --vertical v1
shasum -a 256 docs/play.wgr docs/play_int8.wgr

cat <<'EOF'

next:
  4. python3 -m http.server 8765 --bind 127.0.0.1 --directory docs, then open
       http://127.0.0.1:8765/seeds.html?model=play_int8.wgr&n=1500
     fresh should be above 0.5 and tries_max under 20; take the first tutorial seed whose thumbnail looks right, and
     play it: http://127.0.0.1:8765/?model=play_int8.wgr&tut=1&seed=<code>#play
  5. python docs/make_records.py add --ledger ledger.jsonl --tags <s0> <s1> <s2> --n <n> --change "..." \
       --code <url at a commit> --checkpoints <url>;  python docs/make_records.py render
  6. python docs/pin_web.py --wgr docs/play_int8.wgr --tut-seed <code>            (a dry run; --yes uploads and pins)
EOF

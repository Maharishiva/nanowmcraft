# train_tiny.py baseline and reward AP, 2026-10-16

`train_tiny.py` (249 lines, no Pallas), training seeds 0, 1 and 2, each a cold process on one Colab A100-SXM4-40GB, and the
three record checkpoints, all scored by the final `eval.py` (with `reward_ap`). Rows: [eval_v3_20261016.jsonl](eval_v3_20261016.jsonl).
Checkpoints: `train_tiny_s{0,1,2}` and `record1_main_s{0,1,2}` on [Hugging Face](https://huggingface.co/maharishiva/tinywmcraft). gFVD times 10^4.

| metric | train_tiny | record 1 | floor (95% of train_tiny) |
|---|---:|---:|---:|
| gFVD ↓ | 16.95 ± 0.50 | 14.69 ± 1.91 | |
| AF | 0.921 | 0.929 | 0.875 |
| reward_corr | 0.931 | 0.942 | 0.884 |
| reward_recall | 0.981 | 0.986 | 0.931 |
| reward_precision | 0.881 | 0.897 | 0.837 |
| reward_ap | 0.939 | 0.945 | 0.892 |
| done_auc | 0.980 | 0.978 | 0.931 |
| imagined_reward_corr | 0.597 | 0.673 | 0.566 |
| imagined_return_corr | 0.793 | 0.847 | 0.753 |
| imagined_done_auc | 0.831 | 0.837 | 0.789 |
| train s (max) | 573 | 581 | 600 |
| eval s (max) | 176 | 156 | 180 |

- Per seed, train_tiny scores 16.64, 16.69 and 17.53. Its eval fits the 180 s limit because it uses cuBLAS matmuls (`--xla_gpu_enable_triton_gemm=false`, as `train.py`): without the flag compilation alone made the eval 250 s.
- The record's scores are unchanged by the final `eval.py` (13.46 / 16.90 / 13.73, same AF): running the imagined rollout before the teacher-forced heads and fixing the feature network's matmul precision do not move them.
- Reward AP of the `normuon` research checkpoint (`normuon_head_bridge_s0`, its own code, teacher-forced on the same 512 windows): 0.920, against precision 0.828 at the 0.05 cutoff. Its reward head ranks events about 2% worse than train_tiny's, not the 6% that the fixed-cutoff precision suggests.

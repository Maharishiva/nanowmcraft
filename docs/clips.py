"""Held-out play as streams, the layout export_web.py, make_play_gif.py and make_architecture_gif.py read:
frames (T, E, 63, 63, 3) uint8, actions (T, E) with actions[t] the action taken in frame t, rewards (T, E) for that
action, dones (T, E) where it ended the episode.

streams() reads a file in that layout as it is, or the benchmark's eval.npz (make_eval.py): there each play clip
(72 frames, actions[:, j] the action INTO frame j) becomes one stream, and the frames a death clip repeats after
its terminal frame count as done, so no window crosses them. Only the play arrays are read.
"""
import numpy as np


def streams(path):
    d = np.load(path)
    if "play_frames" not in d.files:
        return {k: d[k] for k in ("frames", "actions", "rewards", "dones") if k in d.files}
    frames, valid = d["play_frames"], d["play_valid_len"].astype(np.int64)
    out = lambda x: np.concatenate([x[:, 1:], np.zeros_like(x[:, :1])], 1).T    # into frame t + 1 = out of frame t
    T = frames.shape[1]
    pad = np.arange(T)[:, None] + 1 >= valid[None]                               # the next frame is past the end
    return dict(frames=frames.transpose(1, 0, 2, 3, 4), actions=out(d["play_actions"]), rewards=out(d["play_rewards"]),
                dones=out(d["play_dones"]) | pad)

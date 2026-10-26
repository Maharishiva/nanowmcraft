# The challenge: a diamond inside the world model

No code and no GPU needed. The [browser player](https://maharishiva.github.io/tinywmcraft/#play) runs the best model on your own GPU, and every world is imagined from noise. Get a diamond inside it.
The first verified diamond clip wins a paper copy of Sutton & Barto's *Reinforcement Learning*. Every diamond clip and the best glitches go up on the site with your handle.

## What counts

- A diamond in the inventory the model draws, in the browser player, in the model the site plays at the time. The real game does not count.
- The player reads the inventory itself: when a diamond shows there on two frames in a row, it says DIAMOND! and marks the clip with the step it came on. That clip is your entry.
- How the model got there is the model's business. In the real game a diamond needs an iron pickaxe, which needs a furnace, coal and the rest ([the road to a diamond](https://maharishiva.github.io/tinywmcraft/#achievements)); the world model only tries to follow those rules. A diamond it hands out on its own counts too.
- The score counter does not count. The model hands out +1s that can be farmed, which says more about the model than about you.

## How to submit

1. With the diamond in the inventory, press **save clip**. The player writes the run as an MP4 (the last 3000 steps at most). On iPhones and iPads the share sheet opens instead; Save Video puts the clip in Photos.
2. Post the clip on X with a link to this repo, or open an issue here with the clip or a link to it. On a computer, **share** downloads the clip and opens a post with the caption and the links written for you; on a phone it opens the share sheet and copies the caption.
3. Say which browser and GPU you played on. It helps the check below.

## How it is checked

The clip is made by the player and carries what the maintainer needs: the site's address on top, the score strip, the action pressed under every frame, a SEED and STEP line, and an end card with the step the diamond came on.

- The maintainer watches the clip: the frames must follow on from each other, the step counter must run without a gap, and the diamond must sit in the inventory strip the model draws.
- The seed and the actions let the maintainer replay the run: `?seed=XXXX` on the player opens the same world, and the same actions give the same frames on the same device. Other GPUs can round differently, so a replay that drifts is not held against you; a replay that reaches the diamond settles it.
- Verified means checked this way by the maintainer. First means the earliest post or issue.

## What does not count

- An edited video, frames inserted or cut, an edited page, or another model file (the `?model=` parameter): only the model the site plays, as the site plays it.
- A diamond on the map that never reaches the inventory, or one that shows on a single frame and is gone the next.
- The real game, a recording of `play_wm.py`, or anything the player's own clip does not show.

## The glitch category

The model hallucinates in funny ways. Post the weirdest clip the same way: save clip or share, then X or an issue. The best glitches go up on the site with your handle, picked by the maintainer; there is no book for them.

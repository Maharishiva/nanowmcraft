# Diamond Speedrun

Get a diamond inside the world model: in the [browser player](https://maharishiva.github.io/tinywmcraft/#play), not in the real game. No code and no GPU needed: the player runs the model on your own GPU, and every world is imagined from noise.
The fastest run posted before **24 December 2026, 23:59 PST** wins a paper copy of Sutton & Barto's *Reinforcement Learning*.

## What counts

- A diamond in the inventory the model draws, in [iteration 3 on our site](https://maharishiva.github.io/tinywmcraft/?model=iteration3#play) (the file [`web/play.wgr` at 00e272c](https://huggingface.co/maharishiva/tinywmcraft-v3/blob/00e272ceae171d3f8409b3f736c16d6133d8b0c9/web/play.wgr) on Hugging Face). The site plays it by default today; when a newer model takes over the site, that link keeps playing iteration 3 and runs on it still count.
- The player reads the inventory itself: when a diamond shows there on two frames in a row, it says DIAMOND! and marks the clip with the step it came on. That step is your time: the fewest steps win, and a tie goes to the earlier post.
- How the model got there is the model's business. In the real game a diamond needs an iron pickaxe, which needs a furnace, coal and the rest ([the road to a diamond](https://maharishiva.github.io/tinywmcraft/#achievements)); the world model only tries to follow those rules. A diamond it hands out on its own counts too.
- The score counter does not count. The model hands out +1s that can be farmed, which says more about the model than about you.

## How to submit

1. With the diamond in the inventory, press **save clip**. The player writes the run as an MP4 (the last 3000 steps at most). On iPhones and iPads the share sheet opens instead; Save Video puts the clip in Photos.
2. Post the clip in the [Diamond Speedrun discussion](https://github.com/Maharishiva/tinywmcraft/discussions/9): drag the MP4 into a comment and write the seed (it is on the clip), the step the diamond came on, your browser and GPU, and the link to your X post.
3. Post it on X too: the clip as a video, a link to this repo and a tag for [@rldoeswork](https://x.com/rldoeswork). **share** writes the caption and hands you the clip. A run needs both posts; it is judged in the discussion.

## How it is checked

The clip is made by the player and carries what the maintainer needs: the site's address on top, the score strip, the action pressed under every frame, a SEED and STEP line, and an end card with the step the diamond came on.

- The maintainer watches the clip: the frames must follow on from each other, the step counter must run without a gap, and the diamond must sit in the inventory strip the model draws.
- The seed and the actions let the maintainer replay the run: `?seed=XXXX` on the player opens the same world, and the same actions give the same frames on the same device. Other GPUs can round differently, so a replay that drifts is not held against you; a replay that reaches the diamond settles it.

## What does not count

- An edited video, frames inserted or cut, an edited page, or another model file (any other `?model=` than `?model=iteration3`).
- A diamond on the map that never reaches the inventory, or one that shows on a single frame and is gone the next.
- The real game, a recording of `play_wm.py`, or anything the player's own clip does not show.

## Best glitch

The model hallucinates in funny ways. Post your weirdest clip in the [glitch discussion](https://github.com/Maharishiva/tinywmcraft/discussions/10), the same way. There is no prize, just kudos for breaking the world. Every diamond clip and the best glitches go up on the site with your handle.

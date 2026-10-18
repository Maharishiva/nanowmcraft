"""Write docs/hud_tpl.js: the 7 px templates the player's tutorial reads frames with (docs/hud.js), from Craftax-Classic's
own textures (craftax==1.5.0, TEXTURES[7]), as base64 bytes.

  TILES    map tiles next to the player: blocks, and cow / zombie / skeleton on the block they walk on
  PLAYER   the player sprite facing left, right, up, down: RGBA, alpha 0 or 255
  WOOD     the inventory slot of wood (x 5, y 0 of the strip under the map) with counts 0..9, as ds_ach.py renders it
  DIAMOND  the slot of diamonds (x 0, y 1), counts 0..9
  SLOTS    all 16 slots of the strip (status, items, tools; SLOT_NAMES, SLOT_XY), counts 0..9 each, for the player's
           check of an imagined first frame (9/9/9/9 and nothing held)

    python docs/make_hud.py
"""
import base64
import json
from pathlib import Path

import numpy as np
from craftax.craftax_classic.constants import TEXTURES, BlockType

B = 7
T = {k: np.asarray(v) for k, v in TEXTURES[B].items()}
BLOCKS = {"grass": "GRASS", "sand": "SAND", "water": "WATER", "stone": "STONE", "tree": "TREE", "path": "PATH",
          "table": "CRAFTING_TABLE", "plant": "PLANT", "ripe": "RIPE_PLANT", "coal": "COAL", "iron": "IRON",
          "diamond": "DIAMOND", "furnace": "FURNACE", "lava": "LAVA"}
MOBS = {"cow": "GRASS", "zombie": "GRASS", "skeleton": "PATH"}
# the strip under the map as craftax_classic/renderer.py draws it: name, column, row, icon
SLOTS = [("health", 0, 0, "health_texture"), ("food", 1, 0, "hunger_texture"), ("drink", 2, 0, "thirst_texture"),
         ("energy", 3, 0, "energy_texture"), ("sapling", 4, 0, "sapling_texture"), ("wood", 5, 0, "WOOD"),
         ("stone", 6, 0, "STONE"), ("coal", 7, 0, "COAL"), ("iron", 8, 0, "IRON"), ("diamond", 0, 1, "DIAMOND"),
         ("wood_pickaxe", 1, 1, "wood_pickaxe_texture"), ("stone_pickaxe", 2, 1, "stone_pickaxe_texture"),
         ("iron_pickaxe", 3, 1, "iron_pickaxe_texture"), ("wood_sword", 4, 1, "wood_sword_texture"),
         ("stone_sword", 5, 1, "stone_sword_texture"), ("iron_sword", 6, 1, "iron_sword_texture")]


def block(name):
    return T["block_textures"][BlockType[name].value].astype(np.float32)


def slot(icon, n):  # ds_ach.template: the icon in the top-left 5x5 when the count is above 0, the digit over the corner
    L, R, NO = (B - int(0.8 * B)) // 2 - 1, B - int(0.8 * B) - ((B - int(0.8 * B)) // 2 - 1), B - int(B * 0.6)
    p = np.zeros((B, B, 3), np.float32)
    if n > 0:
        p[L:B - R, L:B - R] = icon[..., :3]
    p[NO - 1:B - 1, NO - 1:B - 1] *= 1 - T["number_textures_alpha"][n]
    p[NO - 1:B - 1, NO - 1:B - 1] += T["number_textures"][n]
    return p


def b64(x):
    return base64.b64encode(np.clip(np.round(x), 0, 255).astype(np.uint8).tobytes()).decode()


def main():
    tiles = [block(v) for v in BLOCKS.values()]
    for mob, under in MOBS.items():
        a = T[mob + "_texture_alpha"].astype(np.float32)
        tiles.append(T[mob + "_texture"] * a + block(under) * (1 - a))
    player = T["player_textures"][:4].astype(np.float32)                  # left, right, up, down (4: asleep)
    player[..., 3] = (player[..., 3] > 0.5) * 255
    small = T["smaller_block_textures"]
    wood = np.stack([slot(small[BlockType.WOOD.value], n) for n in range(10)])
    diamond = np.stack([slot(small[BlockType.DIAMOND.value], n) for n in range(10)])
    icon = lambda k: small[BlockType[k].value] if k.isupper() else T[k]
    slots = np.stack([[slot(icon(k), n) for n in range(10)] for _, _, _, k in SLOTS])
    out = {"names": list(BLOCKS) + list(MOBS), "tiles": b64(np.stack(tiles)), "player": b64(player),
           "wood": b64(wood), "diamond": b64(diamond), "slot_names": [n for n, *_ in SLOTS],
           "slot_xy": [[x, y] for _, x, y, _ in SLOTS], "slots": b64(slots)}
    path = Path(__file__).resolve().parent / "hud_tpl.js"
    path.write_text("// made by make_hud.py from Craftax-Classic's textures (github.com/MichaelTMatthews/Craftax, MIT, (c) 2024 "
                    "Michael Matthews)\nexport const TPL = " + json.dumps(out, indent=0) + ";\n")
    print(path, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()

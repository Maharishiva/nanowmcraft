import subprocess
import sys
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

GLYPHS = {
    "n": ["#######.", "########", "##....##", "##....##", "##....##", "##....##", "##....##", "##....##", "##....##"],
    "a": [".######.", "########", "......##", ".#######", "########", "##....##", "##....##", "########", ".#######"],
    "o": [".######.", "########", "##....##", "##....##", "##....##", "##....##", "##....##", "########", ".######."],
    "w": ["##..##..##"] * 7 + ["##########", ".########."],
    "m": ["#########.", "##########"] + ["##..##..##"] * 7,
    "c": [".#######", "########", "##......", "##......", "##......", "##......", "##......", "########", ".#######"],
    "r": ["#######.", "########", "##....##", "##......", "##......", "##......", "##......", "##......", "##......"],
    "f": ["...####", "..#####", "..##...", "#######", "#######", "..##...", "..##...", "..##...", "..##...", "..##...", "..##...", "..##..."],
    "i": ["##", "##", ".."] + ["##"] * 9,
    "y": ["##....##"] * 7 + ["########", ".#######", "......##", "########", "#######."],
    "t": ["..##...", "..##...", "..##...", "#######", "#######", "..##...", "..##...", "..##...", "..##...", "..##...", "..#####", "...####"],
}
WORDS = [("tiny", "tiny"), ("wm", "wm"), ("craft", "craft")]
DESC = {"y": 3}
PALETTES = {
    "light": {"tiny": "#7860a5", "wm": "#ff6a13", "craft": "#2f9e44"},
    "dark": {"tiny": "#a48bd0", "wm": "#ff7a2a", "craft": "#4cc463"},
}
SHADES = {
    "light": {"tiny": ("#a48bd0", "#5b4486"), "wm": ("#ffa04a", "#e84f06"), "craft": ("#62c873", "#1f7f33")},
    "dark": {"tiny": ("#c3b6dd", "#8a70bd"), "wm": ("#ffb070", "#f25f10"), "craft": ("#86e394", "#2f9e44")},
}
ROWS = 12
GAP = 2
CELL = 10
SHADOW = (-1, 1)


def band(top, bottom, y):
    t = min(max((y - 3) / (ROWS - 4), 0.0), 1.0)
    a, b = int(top[1:], 16), int(bottom[1:], 16)
    mix = [round(((a >> k) & 255) * (1 - t) + ((b >> k) & 255) * t) for k in (16, 8, 0)]
    return "#" + "".join(f"{v:02x}" for v in mix)


def layout():
    cells, x = [], 1
    for group, word in WORDS:
        for ch in word:
            g = GLYPHS[ch]
            top = ROWS - len(g) + DESC.get(ch, 0)
            for r, line in enumerate(g):
                for c, v in enumerate(line):
                    if v == "#":
                        cells.append((x + c, top + r, group))
            x += len(g[0]) + GAP
    return cells, x - GAP + 1


def svg(palette, shades, background=None):
    cells, width = layout()
    solid = {(x, y) for x, y, _ in cells}
    m = 2 if background else 1
    w, h = (width + 2 * m) * CELL, (ROWS + 1 + max(DESC.values()) + 2 * m) * CELL
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" shape-rendering="crispEdges">', "<defs>"]
    for group, color in palette.items():
        out.append(f'<pattern id="d-{group}" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="2" height="2" fill="{color}"/><rect x="2" y="2" width="2" height="2" fill="{color}"/></pattern>')
    out.append("</defs>")
    if background:
        out.append(f'<rect width="{w}" height="{h}" fill="{background}"/>')
    out.append(f'<g transform="translate({m * CELL},{m * CELL})">')
    dx, dy = SHADOW
    for x, y, group in cells:
        sx, sy = x + dx, y + dy
        if (sx, sy) not in solid:
            out.append(f'<rect x="{sx * CELL}" y="{sy * CELL}" width="{CELL}" height="{CELL}" fill="url(#d-{group})" opacity="0.55"/>')
    for x, y, group in cells:
        fill = band(*shades[group], y) if shades else palette[group]
        out.append(f'<rect x="{x * CELL}" y="{y * CELL}" width="{CELL}" height="{CELL}" fill="{fill}"/>')
    out.append("</g></svg>")
    return "\n".join(out)


def png(svg_text, out, scale=2):
    w = int(svg_text.split('width="')[1].split('"')[0])
    h = int(svg_text.split('height="')[1].split('"')[0])
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "logo.html"
        page.write_text(f'<html><body style="margin:0;background:transparent">{svg_text}</body></html>')
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--default-background-color=00000000",
                        f"--force-device-scale-factor={scale}", f"--window-size={w},{h}", f"--screenshot={out}", str(page)],
                       check=True, capture_output=True)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in ("light", "dark"):
        png(svg(PALETTES[name], SHADES[name]), out / f"logo-{name}.png")


if __name__ == "__main__":
    main()

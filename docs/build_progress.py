import html
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_logo import svg

HERE = Path(__file__).parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROTOCOL = "2048w_ctx8_h16_nfe8"
W, H = 1040, 440
PAD_L, PAD_R, PAD_T, PAD_B = 70, 24, 24, 46


def banner():
    page = {"tiny": "#f0f0ee", "wm": "#ff6a13", "craft": "#f0f0ee"}
    return svg(page, None).replace('<svg ', '<svg class="banner" ', 1)


def load():
    rows = [json.loads(l) for l in (HERE / "progress.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("protocol") == PROTOCOL and r.get("gFVD") and r["gFVD"] < 0.0025]
    for r in rows:
        r["t"] = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
    rows.sort(key=lambda r: r["t"])
    for i, r in enumerate(rows):
        r["i"] = i
    return rows


def kpis(rows):
    runs = [r for r in rows if r.get("kind") == "run"]
    valid = [r for r in runs if r.get("in_budget")]
    marks = [r for r in rows if r.get("milestone")]
    first, best = marks[0], min(marks, key=lambda r: r["gFVD"])
    drop = (best["gFVD"] / first["gFVD"] - 1) * 100
    days = (rows[-1]["t"] - rows[0]["t"]).total_seconds() / 86400
    cells = [
        ("first recipe", f'{first["gFVD"]:.5f}'),
        ("best 3-seed", f'{best["gFVD"]:.5f}'),
        ("change", f"{drop:+.1f}%"),
        ("runs", str(len(runs))),
        ("in budget", str(len(valid))),
        ("days", f"{days:.1f}"),
    ]
    w = 17
    top = "┌" + "┬".join("─" * w for _ in cells) + "┐"
    mid1 = "│" + "│".join(f" {k:<{w - 1}}" for k, _ in cells) + "│"
    mid2 = "│" + "│".join(f' <span class="{"o" if i in (1, 2) else "w"}">{v:<{w - 1}}</span>' for i, (_, v) in enumerate(cells)) + "│"
    bot = "└" + "┴".join("─" * w for _ in cells) + "┘"
    return "\n".join([top, mid1.replace(" ", " "), mid2, bot])


def chart(rows):
    ys = [r["gFVD"] for r in rows]
    pad = (max(ys) - min(ys)) * 0.04
    lo, hi = min(ys) - pad, max(ys) + pad
    n = len(rows)

    def px(i):
        return PAD_L + (W - PAD_L - PAD_R) * i / max(n - 1, 1)

    def py(v):
        return PAD_T + (H - PAD_T - PAD_B) * (hi - v) / (hi - lo)

    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" font-family="inherit" font-size="11">']
    span = hi - lo
    step = next(s for s in (0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01) if span / s <= 8)
    for k in range(math.ceil(lo / step), math.floor(hi / step) + 1):
        v = k * step
        y = py(v)
        out.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{v:.5f}</text>')
    seen, last_x = None, -1e9
    for r in rows:
        day = r["t"].strftime("%b %d")
        if day != seen and px(r["i"]) - last_x > 60:
            seen, last_x = day, px(r["i"])
            x = px(r["i"])
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PAD_T}" y2="{H - PAD_B}" class="grid"/>')
            out.append(f'<text x="{x:.1f}" y="{H - PAD_B + 18}" text-anchor="middle" class="axis">{day}</text>')
    best, path = math.inf, []
    for r in rows:
        if r.get("kind") == "run" and r.get("in_budget") and r["gFVD"] < best:
            best = r["gFVD"]
            path.append((px(r["i"]), py(best)))
    if path:
        d = f"M{path[0][0]:.1f},{path[0][1]:.1f}" + "".join(f"H{x:.1f}V{y:.1f}" for x, y in path[1:]) + f"H{W - PAD_R}"
        out.append(f'<path d="{d}" class="best"/>')
    marks, low = [], math.inf
    for r in rows:
        if r.get("milestone") and r["gFVD"] < low:
            low = r["gFVD"]
            marks.append(r)
    if marks:
        d = f"M{px(marks[0]['i']):.1f},{py(marks[0]['gFVD']):.1f}" + "".join(f"H{px(r['i']):.1f}V{py(r['gFVD']):.1f}" for r in marks[1:]) + f"H{W - PAD_R}"
        out.append(f'<path d="{d}" class="accepted"/>')
    for r in rows:
        x, y = px(r["i"]), py(r["gFVD"])
        cls = "mark" if r.get("milestone") else ("run" if r.get("in_budget") else "over")
        s = 7 if r.get("milestone") else 4
        tip = html.escape(json.dumps({k: r.get(k) for k in ("tag", "ts", "gFVD", "AF", "seed", "train_s", "kind", "milestone", "source")}))
        out.append(f'<rect x="{x - s / 2:.1f}" y="{y - s / 2:.1f}" width="{s}" height="{s}" class="{cls}" data-tip="{tip}"/>')
    placed = []
    for r in marks:
        x, y = px(r["i"]), py(r["gFVD"])
        text = r["milestone"][:40]
        width = 6.4 * len(text)
        right = x > W - 260
        x0 = x - 6 - width if right else x + 6
        dy = next(dy for dy in (-12, 20, -26, 34, -40, 48, -54) if not any(
            x0 < bx + bw + 8 and bx < x0 + width + 8 and abs(y + dy - by) < 13 for bx, bw, by in placed))
        placed.append((x0, width, y + dy))
        out.append(f'<text x="{x + (-6 if right else 6):.1f}" y="{y + dy:.1f}" text-anchor="{"end" if right else "start"}" class="label">{html.escape(text)}</text>')
    out.append(f'<text x="{PAD_L}" y="{PAD_T - 8}" class="axis">gFVD (lower is better)</text>')
    out.append(f'<text x="{W - PAD_R}" y="{H - 6}" text-anchor="end" class="axis">experiments in time order</text>')
    out.append("</svg>")
    return "\n".join(out)


def table(rows):
    marks = [r for r in rows if r.get("milestone")]
    if not marks:
        return ""
    top, bottom = marks[0]["gFVD"], marks[-1]["gFVD"]
    out = [f'<span class="dimc">{"date":<11} {"gFVD":<9} {"vs start":>8}  {"":<24}  change</span>']
    for r in marks:
        frac = (top - r["gFVD"]) / max(top - bottom, 1e-12)
        bar = "█" * round(frac * 24) + "░" * (24 - round(frac * 24))
        rel = (r["gFVD"] / top - 1) * 100
        out.append(f'{r["t"].strftime("%Y-%m-%d"):<11} {r["gFVD"]:<9.5f} {rel:>+7.1f}%  <span class="o">{bar}</span>  {html.escape(r["milestone"])}')
    return "\n".join(out)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>tinywmcraft progress</title>
<style>
:root {{ --bg:#0a0a0a; --fg:#f0f0ee; --dim:#6a6a6a; --o:#ff6a13; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:"JetBrains Mono","SF Mono",Menlo,Consolas,monospace; font-size:13px; line-height:1.45; }}
main {{ max-width:1100px; margin:0 auto; padding:40px 28px 60px; }}
pre {{ margin:0; font-family:inherit; }}
.banner {{ width:560px; max-width:100%; height:auto; display:block; margin-bottom:6px; }}
.w {{ color:var(--fg); }} .o {{ color:var(--o); }} .dimc {{ color:var(--dim); }}
.box {{ line-height:1.2; }}
.sub {{ color:var(--dim); margin:18px 0 22px; }}
.sub b {{ color:var(--fg); font-weight:normal; }}
h2 {{ font-size:13px; font-weight:normal; color:var(--dim); margin:34px 0 10px; }}
h2::before {{ content:"## "; color:var(--o); }}
.grid {{ stroke:#1c1c1c; stroke-width:1; }}
.axis {{ fill:var(--dim); }}
.best {{ fill:none; stroke:#8a8a8a; stroke-width:1; stroke-dasharray:3 3; }}
.accepted {{ fill:none; stroke:var(--o); stroke-width:2; }}
rect.run {{ fill:var(--fg); opacity:.55; }}
rect.over {{ fill:none; stroke:var(--dim); stroke-width:1; }}
rect.mark {{ fill:var(--o); }}
rect:hover {{ opacity:1; stroke:var(--fg); stroke-width:1.5; cursor:crosshair; }}
.label {{ fill:var(--o); font-size:10.5px; paint-order:stroke; stroke:var(--bg); stroke-width:4px; }}
.legend span {{ margin-right:22px; }}
#tip {{ position:fixed; display:none; background:#111; border:1px solid var(--o); padding:8px 10px; white-space:pre; pointer-events:none; font-size:12px; z-index:2; }}
.foot {{ color:var(--dim); margin-top:30px; }}
a {{ color:var(--o); }}
</style></head><body><main>
{banner}
<p class="sub"><b>gFVD over time.</b> Every dot is one training run scored by the frozen evaluator: 2048 windows, 8 real context frames, 16 generated frames, 8 Euler steps.
Budget: one A100, 10 minutes cold, compilation included. Data: 500k Craftax-Classic transitions.</p>
<pre class="box">{kpis}</pre>
<h2>the fall</h2>
<div class="legend"><span class="o">━━ accepted recipe (multi-seed mean)</span><span style="color:#8a8a8a">┄┄ best single run in budget</span><span>■ run in budget</span><span style="color:var(--dim)">□ over budget</span></div>
{chart}
<h2>accepted changes</h2>
<pre>{table}</pre>
<p class="foot">gFVD: Fréchet distance between real and generated 4-frame clips in a fixed random-CNN feature space; lower is better, and values are not comparable to I3D FVD.
{dropped} earlier rows used a 512-window evaluator and are not shown. Built {built} from <a href="progress.jsonl">progress.jsonl</a>.</p>
</main><div id="tip"></div>
<script>
const tip = document.getElementById("tip");
document.querySelectorAll("rect[data-tip]").forEach(el => {{
  el.addEventListener("mousemove", e => {{
    const d = JSON.parse(el.dataset.tip);
    tip.textContent = Object.entries(d).filter(([k, v]) => v !== null && v !== undefined).map(([k, v]) => k.padEnd(10) + v).join("\\n");
    tip.style.display = "block"; tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 14) + "px";
  }});
  el.addEventListener("mouseleave", () => tip.style.display = "none");
}});
</script></body></html>
"""


def main():
    all_rows = [json.loads(l) for l in (HERE / "progress.jsonl").read_text().splitlines() if l.strip()]
    rows = load()
    page = PAGE.format(
        banner=banner(),
        kpis=kpis(rows),
        chart=chart(rows),
        table=table(rows),
        dropped=sum(1 for r in all_rows if r.get("protocol") != PROTOCOL),
        built=datetime.now().strftime("%Y-%m-%d"),
    )
    (HERE / "progress.html").write_text(page)
    card = page[:page.index("<main>") + 6] + page[page.index('<pre class="box">'):page.index("<h2>accepted changes</h2>")] + "</main></body></html>"
    card_path = HERE / "progress_card.html"
    card_path.write_text(card)
    png = HERE.parent / "assets" / "progress.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--window-size=1100,690", f"--screenshot={png}", str(card_path)], check=True, capture_output=True)
    card_path.unlink()
    print(HERE / "progress.html", png, len(rows), "rows")


if __name__ == "__main__":
    main()

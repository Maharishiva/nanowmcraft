"""The benchmark's records from eval.py's ledger rows: docs/records.jsonl is the one source; the site's benchmark page
and home page draw from it when they load, and this script writes the same numbers into the README table and the home
page's static top three.

    python docs/make_records.py add --ledger ledger.jsonl --tags it3_s0 it3_s1 it3_s2 --n 3 \
        --change "train.py: ..." --code https://github.com/Maharishiva/tinywmcraft/blob/<commit>/train.py \
        --checkpoints https://huggingface.co/maharishiva/tinywmcraft-v3/tree/main/iteration3
    python docs/make_records.py render

add: the means over the three seeds (the score with its sample SD), the slowest train_s and eval_s and each seed's
row, as row n of records.jsonl (replaced if it is there). Prints the gates against the best other row: the score
margin 1.65 * sqrt((sd_best^2 + sd_new^2) / 3), reward AP at or above 95% of iteration 1's, train_s <= 600 and
eval_s <= 300 for every seed, seeds 0, 1 and 2.
render: the rows of the README's benchmark table (the columns its header names), docs/index.html's top three,
best score and next iteration number, and the numbers in the rules (program.md, benchmark.html): the best score
to beat, its eval time and the reward AP floor.
"""
import argparse
import json
import re
import statistics
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS, README, INDEX = ROOT / "docs" / "records.jsonl", ROOT / "README.md", ROOT / "docs" / "index.html"
PROGRAM, BENCH = ROOT / "program.md", ROOT / "docs" / "benchmark.html"
MEANS = ("score", "gFVD", "AF", "logic", "memory", "reward_ap", "done_ap")       # AF only while eval.py reports it
SEED_KEYS = ("score", "gFVD", "gFVD_16", "AF", "logic", "memory", "reward_ap", "done_ap", "train_s", "eval_s")
TRAIN_BUDGET_S, EVAL_BUDGET_S, FLOOR = 600, 300, 0.95
CROWN = '<img class="crown" src="icons/crown.png" alt="crown" onerror="this.remove()">'


def num(v, d):
    """A number with d decimals; a placeholder such as {SCORE_IT3} stays as it is (rows written before the numbers)."""
    return "–" if v is None else f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v)


def rank(r):
    """Best first: the score, then (rows without numbers yet) the newest iteration."""
    return (r["score"] if isinstance(r.get("score"), (int, float)) else float("-inf"), r["n"])


def load():
    return [json.loads(l) for l in RECORDS.read_text().splitlines() if l.strip()] if RECORDS.exists() else []


def add(a):
    ledger = [json.loads(l) for l in Path(a.ledger).read_text().splitlines() if l.strip()]
    rows = [next(r for r in reversed(ledger) if r["tag"] == t) for t in a.tags]      # the last row of each tag
    seeds = [r.get("train_seed", r.get("seed")) for r in rows]
    rec = dict(n=a.n, date=a.date, author=a.author, change=a.change, code=a.code, checkpoints=a.checkpoints)
    for k in MEANS:
        if all(k in r for r in rows):
            rec[k] = round(statistics.mean(r[k] for r in rows), 6)
    rec["score_sd"] = round(statistics.stdev(r["score"] for r in rows), 6)
    rec["train_s"] = round(max(r["train_s"] for r in rows), 1)
    rec["eval_s"] = round(max(r["eval_s"] for r in rows), 1)
    rec["seeds"] = [{"seed": s, **{k: round(r[k], 4) for k in SEED_KEYS if k in r}} for s, r in zip(seeds, rows)]
    old = [r for r in load() if r["n"] != a.n]
    best = max((r for r in old if isinstance(r.get("score"), (int, float))), key=rank, default=None)
    first = next((r for r in old + [rec] if r["n"] == 1 and isinstance(r.get("reward_ap"), (int, float))), None)
    print(json.dumps({k: v for k, v in rec.items() if k != "seeds"}))
    checks = [("seeds 0, 1, 2", sorted(seeds) == [0, 1, 2]),
              (f"train_s <= {TRAIN_BUDGET_S} (max {rec['train_s']})", rec["train_s"] <= TRAIN_BUDGET_S),
              (f"eval_s <= {EVAL_BUDGET_S} (max {rec['eval_s']})", rec["eval_s"] <= EVAL_BUDGET_S)]
    if first is not None:
        floor = round(FLOOR * first["reward_ap"], 3)
        checks.append((f"reward AP {rec['reward_ap']:.3f} >= floor {floor}", rec["reward_ap"] >= floor))
    if best is not None:
        margin = 1.65 * ((best["score_sd"] ** 2 + rec["score_sd"] ** 2) / 3) ** 0.5
        checks.append((f"score {rec['score']:.3f} - best {best['score']:.3f} (#{best['n']}) > margin {margin:.3f}",
                       rec["score"] - best["score"] > margin))
    for text, ok in checks:
        print(("pass  " if ok else "FAIL  ") + text)
    if a.dry_run:
        return
    RECORDS.write_text("".join(json.dumps(r) + "\n" for r in sorted(old + [rec], key=lambda r: r["n"])))
    print(f"wrote row {a.n} to {RECORDS.relative_to(ROOT)}; now: python docs/make_records.py render")


def fmt(r, col):
    """One README cell, by the header's column name."""
    f = lambda k, d: num(r.get(k), d)
    name = col.replace("↑", "").replace("↓", "").strip()
    if name == "#":
        return str(r["n"])
    if name == "change":   # the file the change lives in links to it, as before
        m = re.match(r"(\w+\.py)(:.*)", r["change"])
        return f"[{m[1]}]({m[1]}){m[2]}" if m and (ROOT / m[1]).exists() else r["change"]
    if name == "score":
        return f"{f('score', 3)} ± {f('score_sd', 3)}"
    if name in ("code", "checkpoints"):
        url = r.get(name) or ""
        return f"[{url.rstrip('/').split('/')[-2 if name == 'code' else -1]}]({url})" if url else "–"
    key, d = {"gFVD": ("gFVD", 1), "AF": ("AF", 3), "logic": ("logic", 2), "memory": ("memory", 3),
              "reward AP": ("reward_ap", 3), "done AP": ("done_ap", 3), "train s": ("train_s", 0)}[name]
    return f(key, d)


def render(a):
    recs = sorted(load(), key=lambda r: r["n"])
    lines = README.read_text().split("\n")
    h = next(i for i, l in enumerate(lines) if l.startswith("| # | change |"))
    cols = [c.strip() for c in lines[h].strip("|").split("|")]
    end = h + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    lines[h + 2:end] = ["| " + " | ".join(fmt(r, c) for c in cols) + " |" for r in recs]
    README.write_text("\n".join(lines))
    top = sorted(recs, key=rank, reverse=True)[:3]
    who = lambda r: f'<a href="https://github.com/{r["author"]}">{r["author"]}</a>'
    lead = ' class="lead"'
    li = "\n".join(f'<li{lead if i == 0 else ""}><span class="r">{i + 1}</span><span class="who">'
                   f'{CROWN if i == 0 else ""}{who(r)}</span><span class="g">{num(r["score"], 3)} <span class="sd">± '
                   f'{num(r["score_sd"], 3)}</span></span><span class="ch"><a href="{r["code"]}">iteration {r["n"]}</a> · '
                   f'{r["change"]}</span></li>' for i, r in enumerate(top))
    s = INDEX.read_text()
    s = re.sub(r'(<ol class="top" id="benchmark">\n).*?(\n</ol>)', lambda m: m[1] + li + m[2], s, count=1, flags=re.S)
    s = re.sub(r'<span id="best">[^<]*</span>', f'<span id="best">{num(top[0]["score"], 3)}</span>', s, count=1)
    s = re.sub(r'<span id="next">[^<]*</span>', f'<span id="next">{max(r["n"] for r in recs) + 1}</span>', s, count=1)
    s = re.sub(r'(<p class="np" id="np">).*?(</p>)', lambda m: f'{m[1]}now playing: <a href="{top[0]["code"]}">iteration '
               f'{top[0]["n"]}</a> by {CROWN}{who(top[0])}{m[2]}', s, count=1)
    INDEX.write_text(s)
    # the numbers in the rules: the best score to beat, its eval time, the reward AP floor (95% of iteration 1)
    best, ap1 = top[0], next(r for r in recs if r["n"] == 1)["reward_ap"]
    floor = num(FLOOR * ap1, 3) if isinstance(ap1, (int, float)) else "{REWARD_AP_FLOOR}"
    score, sd, eval_s = num(best["score"], 3), num(best["score_sd"], 3), num(best["eval_s"], 0)
    N = r"(?:[\d.]+|\{[A-Z0-9_]+\})"      # a number or a placeholder
    subs = {PROGRAM: [(rf"(The goal: a higher score than the best so far \(iteration )\d+, {N} ± {N}",
                       rf"\g<1>{best['n']}, {score} ± {sd}"),
                      (rf"iteration \d+ takes about {N} s", f"iteration {best['n']} takes about {eval_s} s"),
                      (rf"(\| reward_ap \|\n\|---:\|\n\| ){N}", rf"\g<1>{floor}")],
            BENCH: [(rf"reward AP ≥ {N}", f"reward AP ≥ {floor}"),
                    (rf"\(iteration \d+: about {N} s\)", f"(iteration {best['n']}: about {eval_s} s)")]}
    for path, rules in subs.items():
        s = path.read_text()
        for pat, rep in rules:
            s, k = re.subn(pat, rep, s, count=1)
            if not k:
                print(f"note: no match for {pat!r} in {path.name}")
        path.write_text(s)
    print(f"README table: {len(recs)} rows; index.html: best {score} (iteration {best['n']}); "
          f"program.md and benchmark.html: reward AP floor {floor}, eval about {eval_s} s")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("add")
    q.add_argument("--ledger", required=True)
    q.add_argument("--tags", nargs=3, required=True, help="the ledger tags of seeds 0, 1 and 2")
    q.add_argument("--n", type=int, required=True, help="the iteration number")
    q.add_argument("--change", required=True)
    q.add_argument("--code", required=True, help="the code at a fixed commit")
    q.add_argument("--checkpoints", required=True)
    q.add_argument("--author", default="Maharishiva", help="GitHub login")
    q.add_argument("--date", default=date.today().isoformat())
    q.add_argument("--dry-run", action="store_true", help="print the row and the gates, write nothing")
    sub.add_parser("render")
    a = p.parse_args()
    add(a) if a.cmd == "add" else render(a)


if __name__ == "__main__":
    main()

"""Put the player's model on Hugging Face and pin it in play.js: the file, its revision and its sha256, and the
tutorial's world (TUT_SEED, picked with seeds.html for this file).

    python docs/pin_web.py --wgr docs/play_int8.wgr --tut-seed 0066                 # a dry run: what would change
    python docs/pin_web.py --wgr docs/play_int8.wgr --tut-seed 0066 --yes           # upload, check, rewrite play.js
    python docs/pin_web.py --wgr docs/play_int8.wgr --revision <commit> --yes       # already uploaded: pin only

The upload is one commit to --repo (default maharishiva/tinywmcraft-v3) with the .wgr at web/<name> and, if given,
other files next to it (--also: the matching .onnx, the fp16 file to fall back to). The pinned file is downloaded
back at that revision and must have the same sha256. The site can only load it from a public repo: the script says so
when the repo is private. Needs huggingface_hub and a token with write access (hf auth login).
"""
import argparse
import hashlib
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent
PLAY = DOCS / "play.js"
SEED_ABC = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"     # play.js: 4-character seed codes


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seed_number(s):
    """A seed code (0066) or number (198) as its number."""
    if s.isdigit() and len(s) != 4:
        return int(s)
    s = s.upper()
    assert len(s) == 4 and all(c in SEED_ABC for c in s), f"not a seed code: {s}"
    n = 0
    for c in s:
        n = 32 * n + SEED_ABC.index(c)
    return n


def rewrite(js, url, sha, tut):
    """play.js with the new HF_MODEL, the hash in the comment above it and TUT_SEED."""
    js, n = re.subn(r"const HF_MODEL = '[^']*';", f"const HF_MODEL = '{url}';", js, count=1)
    assert n == 1, "no HF_MODEL line in play.js"
    head, tail = js.split("const HF_MODEL", 1)
    head = re.sub(r"sha256 [0-9a-f]{8}(?=[^\n]*\n$)", f"sha256 {sha[:8]}", head)   # the comment line just above
    js = head + "const HF_MODEL" + tail
    if tut is not None:
        js, n = re.subn(r"const TUT_SEED = \d+;", f"const TUT_SEED = {tut};", js, count=1)
        assert n == 1, "no TUT_SEED line in play.js"
    return js


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wgr", type=Path, default=DOCS / "play_int8.wgr")
    p.add_argument("--also", type=Path, nargs="*", default=[], help="also upload these next to it (web/<name>)")
    p.add_argument("--repo", default="maharishiva/tinywmcraft-v3")
    p.add_argument("--revision", help="pin this commit of the repo instead of uploading")
    p.add_argument("--tut-seed", help="the tutorial's world for this file, a seed code or number (seeds.html)")
    p.add_argument("--yes", action="store_true", help="upload and rewrite play.js; without it, a dry run")
    a = p.parse_args()
    sha, path = sha256(a.wgr), f"web/{a.wgr.name}"
    tut = None if a.tut_seed is None else seed_number(a.tut_seed)
    print(f"{a.wgr}: {a.wgr.stat().st_size / 1e6:.2f} MB, sha256 {sha}; tutorial seed {tut}")
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi()
    private = api.repo_info(a.repo).private
    if private:
        print(f"note: {a.repo} is private; the site cannot load the model until it is public")
    rev = a.revision or "<new commit>"
    url = f"https://huggingface.co/{a.repo}/resolve/{rev}/{path}"
    if not a.yes:
        print(f"dry run: would upload {path}{''.join(', web/' + o.name for o in a.also)} to {a.repo}" if not a.revision
              else f"dry run: would pin {a.repo} at {a.revision}")
        new = rewrite(PLAY.read_text(), url, sha, tut).splitlines()
        for o, n in zip(PLAY.read_text().splitlines(), new):
            if o != n:
                print(f"- {o}\n+ {n}")
        return
    if not a.revision:
        ops = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(a.wgr))]
        ops += [CommitOperationAdd(path_in_repo=f"web/{o.name}", path_or_fileobj=str(o)) for o in a.also]
        rev = api.create_commit(a.repo, ops, commit_message=f"web: {path}, sha256 {sha[:8]}").oid
        url = f"https://huggingface.co/{a.repo}/resolve/{rev}/{path}"
    got = sha256(hf_hub_download(a.repo, path, revision=rev))
    assert got == sha, f"the pinned file differs: {got[:8]} != {sha[:8]}"
    PLAY.write_text(rewrite(PLAY.read_text(), url, sha, tut))
    print(f"pinned {url} (sha256 {sha[:8]}) in {PLAY.name}" + ("" if tut is None else f", TUT_SEED {tut}"))


if __name__ == "__main__":
    main()

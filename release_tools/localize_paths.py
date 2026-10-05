#!/usr/bin/env python3
"""Replace the path placeholders in this release with your own paths.

Each placeholder (see PATH_MAP.md) is read from an environment variable of the same name without the
angle brackets, e.g. DATA_ROOT=/data/maxtoki. REPO_ROOT defaults to this checkout, EVAL_ROOT to
REPO_ROOT/controlled-evaluation and TMP to /tmp. <HOME> is read from RELEASE_HOME (HOME is always set
in a shell). Placeholders without a value are left alone.

    python release_tools/localize_paths.py            # show what would change
    python release_tools/localize_paths.py --apply    # change the files in this checkout
"""
import argparse, os, pathlib

EXT = {".py", ".md", ".json", ".csv", ".txt", ".sh", ".yaml", ".yml", ".log", ".tsv", ".jsonl",
       ".tex", ".bib", ".cfg", ".toml", ".ini", ".r", ".ipynb", ".diff", ".patch", ".js", ".mjs",
       ".cjs", ".ts"}
NAMES = ["REPO_ROOT", "EVAL_ROOT", "DATA_ROOT", "CODE_ROOT", "LOCAL_VOLUME", "HF_CACHE", "CONDA_ROOT",
         "CLAUDE_CLI", "CLAUDE_HOME", "HOME", "AGENT_TMP", "SYS_TMP", "TMP", "HOMEBREW_PREFIX"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    vals = {n: os.environ.get(n) for n in NAMES if n != "HOME"}
    vals["HOME"] = os.environ.get("RELEASE_HOME")   # HOME is always set, so use RELEASE_HOME for it
    vals["REPO_ROOT"] = vals["REPO_ROOT"] or str(root)
    vals["EVAL_ROOT"] = vals["EVAL_ROOT"] or str(root / "controlled-evaluation")
    vals["TMP"] = vals["TMP"] or "/tmp"
    subs = {f"<{k}>": v for k, v in vals.items() if v}
    n_files = n_subs = 0
    bsubs = {k.encode(): v.encode() for k, v in subs.items()}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXT or "release_tools" in p.parts or ".git" in p.parts:
            continue
        if p.parent == root:
            continue          # the top-level documents explain the placeholders; leave them as they are
        try:
            b = p.read_bytes()        # bytes, so line endings and encodings stay exactly as they are
        except OSError:
            continue
        new = b
        for k, v in bsubs.items():
            new = new.replace(k, v)
        if new != b:
            n_files += 1
            n_subs += sum(b.count(k) for k in bsubs)
            if a.apply:
                p.write_bytes(new)
    print(("changed" if a.apply else "would change"), n_files, "files,", n_subs, "placeholders")
    for k, v in sorted(subs.items()):
        print(f"  {k} -> {v}")

if __name__ == "__main__":
    main()

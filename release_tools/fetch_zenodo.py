#!/usr/bin/env python3
"""Download the files of this release that are kept on Zenodo and put them back at their paths.

ZENODO_MANIFEST.csv lists every file that is not in the git repository. For each file it gives the
Zenodo file that holds it: the file itself (large files, named after their path with "/" written as
"__") or a zip archive (one per area; inside it, each file keeps its path in this release). Rows whose
`deposited` column is not "yes" are not on Zenodo; that column says how to rebuild them.

Text files on Zenodo carry the same path placeholders as the repository (see PATH_MAP.md); run
release_tools/localize_paths.py after fetching.

    python release_tools/fetch_zenodo.py                        # show what would be downloaded
    python release_tools/fetch_zenodo.py --get                  # download, check sha256, place files
    python release_tools/fetch_zenodo.py --get --only attention-grn-217M   # only paths containing this text

A full fetch needs about 83 GB of free disk: about 41 GB of downloads, kept in .zenodo_cache/ (use --cache
to put them elsewhere; they can be deleted afterwards), plus about 42 GB of placed files. .gitignore lists
every placed path, so git does not pick them up.
"""
import argparse, csv, hashlib, pathlib, shutil, sys, urllib.request, zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
RECORD = "23169989"   # Zenodo record id


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def download(record, name, dest):
    url = f"https://zenodo.org/records/{record}/files/{name}?download=1"
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 24)
    part.replace(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", default=RECORD, help="Zenodo record id (default: the one of this release)")
    ap.add_argument("--get", action="store_true", help="download and place the files")
    ap.add_argument("--only", default="", help="only rows whose path contains this text")
    ap.add_argument("--cache", default=str(ROOT / ".zenodo_cache"), help="where downloads are kept")
    a = ap.parse_args()
    with open(ROOT / "ZENODO_MANIFEST.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if a.only in r["path"]]
    absent = [r for r in rows if r["deposited"] != "yes"]
    todo = [r for r in rows if r["deposited"] == "yes"
            and not ((ROOT / r["path"]).exists() and sha256(ROOT / r["path"]) == r["sha256"])]
    names = sorted({r["zenodo_file"] for r in todo})
    print(f"{len(rows)} rows; {len(todo)} files to fetch from {len(names)} Zenodo files; "
          f"{len(absent)} not on Zenodo (see the deposited column).")
    for n in names:
        print("  " + n)
    if not a.get:
        return
    if not a.record or a.record.startswith("__"):
        sys.exit("No Zenodo record id is set; pass --record <id>.")
    cache = pathlib.Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    bad = []
    for name in names:
        local = cache / name
        if not local.exists():
            print("downloading " + name, flush=True)
            download(a.record, name, local)
        group = [r for r in todo if r["zenodo_file"] == name]
        for r in group:
            dest = ROOT / r["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if r["in_zip"] == "yes":
                with zipfile.ZipFile(local) as z, z.open(r["path"]) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 24)
            else:
                shutil.copyfile(local, dest)
            ok = sha256(dest) == r["sha256"]
            print(("ok   " if ok else "BAD  ") + r["path"])
            if not ok:
                bad.append(r["path"])
    if bad:
        sys.exit(f"{len(bad)} files do not match their sha256 in ZENODO_MANIFEST.csv")
    print("done")


if __name__ == "__main__":
    main()

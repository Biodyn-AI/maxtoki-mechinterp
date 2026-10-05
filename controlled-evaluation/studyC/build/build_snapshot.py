"""Build the pre-audit snapshot of the MaxToki deployment for the blind re-audit (Study C).

Copies (never moves) files from
  <REPO_ROOT>/projects/maxtoki
  <REPO_ROOT>/pipelines
into
  <EVAL_ROOT>/studyC/snapshot/maxtoki/

State wanted: the project as it was before the audit started
(audit checklist written 2026-05-07 16:57 local time).

Rules (all decisions are recorded in build/snapshot_inventory.json):
  * a file is a candidate only if its mtime < CUTOFF;
  * files edited after CUTOFF are reconstructed only where the audit-added
    text can be identified with confidence (see RECONSTRUCT below);
  * per-file size: < 5 MB copied; 5-20 MB copied only if on ALLOW_MID;
    everything else is listed (size, shape or columns/rows) in a
    per-folder `_large_files_listing.tsv`;
  * excluded always: audits/, paper*/, any audit_* script, __pycache__, ._*,
    node_modules, the web-atlas build (sae-atlas-217M/atlas, atlas-data),
    model weights.
Only this build folder and the snapshot folder are written.
"""
from __future__ import annotations

import csv
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

SRC = Path("<REPO_ROOT>/projects/maxtoki")
PIPE = Path("<REPO_ROOT>/pipelines")
BIOM = Path("<DATA_ROOT>")
STUDYC = Path("<EVAL_ROOT>/studyC")
DST = STUDYC / "snapshot" / "maxtoki"
BUILD = STUDYC / "build"
ALT = BUILD / "alternatives"
CUTOFF = dt.datetime(2026, 5, 7, 16, 57, 0).timestamp()
RECON_MTIME = dt.datetime(2026, 5, 7, 16, 56, 0).timestamp()

RUNS = ["attention-grn-217M", "circuit-tracing-217M", "exhaustive-mapping-217M",
        "longevity-mechinterp-217M", "manifold-discovery-217M", "sae-atlas-217M",
        "spectral-geometry-217M", "topology-141-217M"]
SMALL = 5_000_000
BIG = 20_000_000
ALLOW_MID = {
    "runs/attention-grn-217M/outputs/phase3/full_pair_dataset.csv",
    "runs/attention-grn-217M/outputs/phase3_k562_1b/full_pair_dataset.csv",
    "runs/attention-grn-217M/outputs/phase3_adamson/full_pair_dataset.csv",
    "runs/attention-grn-217M/outputs/phase3/matched_pair_dataset.csv",
    "runs/attention-grn-217M/outputs/phase2_adamson/pair_dataset.csv",
    "runs/manifold-discovery-217M/artifacts/anchors/centroids_internal.npy",
    "runs/manifold-discovery-217M/artifacts/anchors/centroids_zeroshot.npy",
    "runs/exhaustive-mapping-217M/outputs/experiment3/state_signatures.npz",
    "runs/sae-atlas-217M/outputs/phase0/layer_05/gene_names.json",
}
SKIP_DIRS = {"node_modules", "__pycache__", ".git"}
SETUP_FILES = ["maxtoki_adapter.py", "dataset_loader.py", "topk_sae.py", "token_dictionary.json",
               "MaxToki-217M-HF/config.json", "MaxToki-217M-HF/generation_config.json",
               "MaxToki-1B-HF/config.json", "MaxToki-1B-HF/generation_config.json"]
SPECS = ["attention-grn-extraction-and-evaluation.md", "residual-stream-spectral-geometry.md",
         "topology-geometry-141-hypotheses.md", "manifold-discovery-extraction-compactification.md",
         "longevity-mechinterp-donor-aware.md", "sparse-autoencoders/01-sae-atlas.md",
         "sparse-autoencoders/02-causal-circuit-tracing.md",
         "sparse-autoencoders/03-exhaustive-mapping-and-steering.md",
         "sparse-autoencoders/README.md"]
# small public reference files that pre-audit scripts read from absolute paths
REFS = ["biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv",
        "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json",
        "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/go_bp_gene_sets.json",
        "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/reactome_gene_sets.json",
        "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/kegg_gene_sets.json",
        "biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl"]

inventory = {"included": [], "reconstructed": [], "listed_not_copied": [], "excluded": [], "omitted_docs": []}
listings: dict[Path, list[list]] = {}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(SRC))


def describe(p: Path) -> str:
    """Shape/columns description for a file that is listed but not copied."""
    try:
        suf = p.suffix.lower()
        if suf == ".npy":
            with open(p, "rb") as f:
                ver = np.lib.format.read_magic(f)
                shape, fortran, dtype = np.lib.format._read_array_header(f, ver)
            return f"npy shape={tuple(shape)} dtype={dtype}"
        if suf == ".npz":
            out = []
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    with z.open(n) as f:
                        ver = np.lib.format.read_magic(f)
                        shape, fortran, dtype = np.lib.format._read_array_header(f, ver)
                    out.append(f"{n}:{tuple(shape)}:{dtype}")
            return "npz " + "; ".join(out)
        if suf in (".csv", ".tsv"):
            with open(p, "rb") as f:
                header = f.readline().decode("utf-8", "replace").strip()
                n = sum(1 for _ in f)
            return f"table rows={n} columns={header[:300]}"
        if suf == ".json":
            return "json (not parsed)"
        if suf == ".pt":
            return "torch checkpoint (not loaded)"
        return suf.lstrip(".") or "file"
    except Exception as e:  # pragma: no cover
        return f"(could not describe: {e!r})"


def add_listing(dst_dir: Path, src: Path, reason: str):
    st = src.stat()
    listings.setdefault(dst_dir, []).append([
        src.name,
        str(st.st_size), dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"), describe(src), reason, str(src.relative_to(SRC))])
    inventory["listed_not_copied"].append({"src": rel(src), "bytes": st.st_size, "reason": reason})


def copy_file(src: Path, dst: Path, category="copied"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    inventory["included"].append({"src": str(src), "dst": str(dst.relative_to(DST)), "bytes": src.stat().st_size,
                                  "mtime": dt.datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds"),
                                  "category": category})


# ---------------------------------------------------------------- reconstruction helpers
def strip_sections(lines: list[str], start_rx: str, stop_rx: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Remove every block that starts at a line matching start_rx and runs up to (not including)
    the next line matching stop_rx (searched after the start line)."""
    out, removed, i = [], [], 0
    srx, erx = re.compile(start_rx), re.compile(stop_rx)
    while i < len(lines):
        if srx.search(lines[i]):
            j = i + 1
            while j < len(lines) and not erx.search(lines[j]):
                j += 1
            removed.append((i + 1, j))  # 1-based inclusive start, exclusive end
            i = j
            continue
        out.append(lines[i]); i += 1
    return out, removed


def squeeze_blank(lines):
    out = []
    for l in lines:
        if l.strip() == "" and out and out[-1].strip() == "":
            continue
        out.append(l)
    return out


def recon_circuit(src: Path) -> tuple[str, dict]:
    L = src.read_text().splitlines(keepends=True)
    L, r1 = strip_sections(L, r"^\*\*Chance-baseline anchor \(added 2026-05-07", r"^## ")
    L, r2 = strip_sections(L, r"^### .*\(added 2026-05-07", r"^##+ ")
    return "".join(squeeze_blank(L)), {"removed_line_ranges": r1 + r2}


def recon_sae(src: Path) -> tuple[str, dict]:
    L = src.read_text().splitlines(keepends=True)
    L, r1 = strip_sections(L, r"^### Positive-TF case study RE-RUN .*\(added 2026-05-07", r"^## ")
    txt = "".join(L)
    ins = (" **Bootstrap 95% CI (added 2026-05-07, audit action A3): [1.011, 1.019]** over 10,000 resamples of the 50 features. "
           "The CI is exceptionally tight and decisively excludes the paper's 2.36× — confirming a real non-replication, "
           "not a sample-size artifact. (`projects/maxtoki/audits/bootstrap_cis_summary.json`)")
    assert txt.count(ins) == 1, "sae CI sentence not found exactly once"
    txt = txt.replace(ins, "")
    return "".join(squeeze_blank(txt.splitlines(keepends=True))), {"removed_line_ranges": r1, "removed_inline": [ins.strip()[:80] + "..."]}


def recon_manifold_summary(src: Path) -> tuple[str, dict]:
    L = src.read_text().splitlines(keepends=True)
    L, r1 = strip_sections(L, r"^\*\*Bootstrap 95% CIs \(added 2026-05-07, audit action A3\)", r"^\*\*All four gate thresholds")
    # drop the closing sentence of that block too
    L2, r1b = [], []
    skip_next_blank = False
    for i, l in enumerate(L):
        if l.startswith("**All four gate thresholds (trust ≥ 0.80, rand/donor/branch ≥ 0.20) are crossed by every bootstrap CI"):
            r1b.append(i + 1); skip_next_blank = True; continue
        L2.append(l)
    L, r2 = strip_sections(L2, r"^### Cell-level benchmark — scoped subset \(added 2026-05-07, audit residual", r"^### ")
    return "".join(squeeze_blank(L)), {"removed_line_ranges": r1 + r2, "removed_single_lines": r1b}


def recon_spectral(src: Path, old: Path) -> tuple[str, dict]:
    cur = src.read_text().splitlines(keepends=True)
    prev = old.read_text().splitlines(keepends=True)
    sm = difflib.SequenceMatcher(a=prev, b=cur, autojunk=False)
    out, reverted = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        old_txt = "".join(prev[i1:i2])
        if tag == "replace" and ("attractor" in old_txt or "geometry encodes **time**" in old_txt):
            out.extend(prev[i1:i2]); reverted.append({"current_lines": [j1 + 1, j2], "restored_from_old_lines": [i1 + 1, i2]})
        else:
            out.extend(cur[j1:j2])
    assert len(reverted) == 3, f"expected 3 wording reverts, got {len(reverted)}"
    txt = "".join(out)
    ci = " [bootstrap 95% CI 0.380, 0.384]"
    assert txt.count(ci) == 1
    txt = txt.replace(ci, "")
    return txt, {"reverted_wording_edits": reverted, "removed_inline": [ci.strip()],
                 "old_copy_used": str(old)}


def recon_topology(src: Path, keep_non_audit: bool) -> tuple[str, dict]:
    L = src.read_text().splitlines(keepends=True)
    start = r"^### .*added 2026-05-07, audit" if keep_non_audit else r"^### .*added 2026-05-07"
    L, r = strip_sections(L, start, r"^##+ ")
    return "".join(squeeze_blank(L)), {"removed_line_ranges": r}


def recon_longevity(src: Path) -> tuple[str, dict]:
    L = src.read_text().splitlines(keepends=True)
    L, r = strip_sections(L, r"^## Data-availability blocker \(audit residual exposure", r"^## (?!Data-availability)")
    return "".join(squeeze_blank(L)).rstrip("\n") + "\n", {"removed_line_ranges": r}


def write_recon(text: str, dst: Path, src: Path, info: dict, method: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    os.utime(dst, (RECON_MTIME, RECON_MTIME))
    inventory["reconstructed"].append({"src": str(src), "dst": str(dst.relative_to(DST)), "method": method,
                                       "src_mtime": dt.datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds"),
                                       "src_sha256": sha256(src), "dst_sha256": sha256(dst), **info})


RECONSTRUCT = {
    "runs/topology-141-217M/FINAL_SUMMARY.md": ("strip every '### ...' section whose heading says 'added 2026-05-07' (6 audit-marked + 4 other May-7 sections)",
                                                lambda s: recon_topology(s, keep_non_audit=False)),
    "runs/longevity-mechinterp-217M/FINAL_SUMMARY.md": ("strip the '## Data-availability blocker (audit residual exposure #7 ...)' section", recon_longevity),
    "summaries/circuit-tracing-217M-FINAL_SUMMARY.md": ("strip the A4 paragraph and the A2 re-run section", recon_circuit),
    "summaries/sae-atlas-217M-FINAL_SUMMARY.md": ("strip the A8 v2/A8 v1/A4 sections and the inline A3 CI sentence", recon_sae),
    "summaries/manifold-discovery-217M-FINAL_SUMMARY.md": ("strip the A3 bootstrap block and the residual-#6 section", recon_manifold_summary),
    "summaries/spectral-geometry-217M-FINAL_SUMMARY.md": ("revert the 3 A6 wording edits using the 2026-04-17 copy; strip the inline A3 CI",
                                                          lambda s: recon_spectral(s, SRC / "runs/spectral-geometry-217M/outputs/FINAL_SUMMARY.md")),
}
OMIT_POST = {"README.md": "edited in place 2026-05-07 18:50 (A5/A7/residual #5); A5 rewrote parts of the topology row and the cross-pipeline synthesis with no copy of the old wording, so the pre-audit text cannot be reconstructed with confidence",
             "requirements.txt": "created 2026-05-07 17:52 by audit action A7 (did not exist before the audit)"}


# ---------------------------------------------------------------- main build
def main(rebuild: bool):
    if DST.exists() and any(DST.iterdir()):
        if not rebuild:
            sys.exit(f"{DST} exists; pass --rebuild to recreate it")
        shutil.rmtree(DST, ignore_errors=True)  # only the snapshot this script created
        if DST.exists():
            shutil.rmtree(DST, ignore_errors=True)
    DST.mkdir(parents=True)
    ALT.mkdir(parents=True, exist_ok=True)

    # 1. runs
    for run in RUNS:
        root = SRC / "runs" / run
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if d not in SKIP_DIRS and not d.startswith("._"))
            dpp = Path(dp)
            r = rel(dpp)
            for f in sorted(fn):
                if f.startswith("._") or f == ".DS_Store":
                    continue
                p = dpp / f
                rp = rel(p)
                st = p.stat()
                if rp in RECONSTRUCT:
                    continue
                if st.st_mtime >= CUTOFF:
                    inventory["excluded"].append({"src": rp, "reason": "modified at/after cutoff",
                                                  "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")})
                    continue
                if f.startswith("audit_"):
                    inventory["excluded"].append({"src": rp, "reason": "audit script"}); continue
                # web atlas build
                if rp.startswith("runs/sae-atlas-217M/atlas/") or rp.startswith("runs/sae-atlas-217M/atlas-data/"):
                    if rp == "runs/sae-atlas-217M/atlas-data/cross_layer_graph.json":
                        copy_file(p, DST / rp); continue
                    top = "runs/sae-atlas-217M/" + rp.split("/")[2]
                    inventory["listed_not_copied"].append({"src": rp, "bytes": st.st_size, "reason": "web atlas build/data (not an analysis output)"})
                    continue
                if re.match(r"runs/sae-atlas-217M/outputs/phase0/layer_\d\d/gene_names.json", rp) and "layer_05" not in rp:
                    add_listing(DST / r, p, "identical copy of phase0/layer_05/gene_names.json (not copied)"); continue
                if rp.startswith("runs/manifold-discovery-217M/artifacts/operators/") and p.suffix == ".npy":
                    add_listing(DST / r, p, "per-head operator export (not copied to keep the snapshot small)"); continue
                if st.st_size >= BIG:
                    add_listing(DST / r, p, "larger than 20 MB (not copied)"); continue
                if st.st_size >= SMALL and rp not in ALLOW_MID:
                    add_listing(DST / r, p, "5-20 MB (not copied to keep the snapshot small)"); continue
                copy_file(p, DST / rp)
    # summary line for the web atlas (one listing row per top folder)
    atlas_rows = [x for x in inventory["listed_not_copied"] if x["reason"].startswith("web atlas")]
    for top in ("atlas", "atlas-data"):
        rows = [x for x in atlas_rows if x["src"].startswith(f"runs/sae-atlas-217M/{top}/")]
        if rows:
            listings.setdefault(DST / "runs/sae-atlas-217M", []).append(
                [f"{top}/ ({len(rows)} files)", str(sum(x["bytes"] for x in rows)), "", "folder", "web atlas build/data (not copied)", f"runs/sae-atlas-217M/{top}/"])

    # 2. reconstructed files
    for rp, (method, fn) in RECONSTRUCT.items():
        src = SRC / rp
        text, info = fn(src)
        write_recon(text, DST / rp, src, info, method)
    # alternative topology version (keeps the 4 non-audit May-7 sections) -> outside the snapshot
    text, info = recon_topology(SRC / "runs/topology-141-217M/FINAL_SUMMARY.md", keep_non_audit=True)
    (ALT / "topology-141-217M_FINAL_SUMMARY.keep_non_audit_may7_sections.md").write_text(text)
    inventory["alternatives"] = [{"file": "alternatives/topology-141-217M_FINAL_SUMMARY.keep_non_audit_may7_sections.md", **info}]

    # 3. summaries/ (pre-cutoff as-is)
    for p in sorted((SRC / "summaries").iterdir()):
        if p.name.startswith("._") or not p.is_file():
            continue
        rp = rel(p)
        if rp in RECONSTRUCT:
            continue
        if p.stat().st_mtime >= CUTOFF:
            inventory["excluded"].append({"src": rp, "reason": "modified at/after cutoff, not reconstructable"}); continue
        copy_file(p, DST / rp)

    # 4. project-level docs that cannot be reconstructed
    for f, why in OMIT_POST.items():
        inventory["omitted_docs"].append({"src": f, "reason": why})

    # 5. setup code and configs
    for f in SETUP_FILES:
        p = SRC / "setup" / f
        if p.exists() and p.stat().st_mtime < CUTOFF:
            copy_file(p, DST / "setup" / f)
        else:
            inventory["excluded"].append({"src": f"setup/{f}", "reason": "missing or modified after cutoff"})
    for p in sorted((SRC / "setup").iterdir()):
        if p.is_file() and not p.name.startswith("._") and p.name not in SETUP_FILES and p.name not in ("",):
            inventory["excluded"].append({"src": rel(p), "reason": "created after cutoff" if p.stat().st_mtime >= CUTOFF else "not needed"})
    for w in ("MaxToki-217M-HF/model.safetensors", "MaxToki-1B-HF/model.safetensors"):
        p = SRC / "setup" / w
        if p.exists():
            listings.setdefault(DST / "setup", []).append([w, str(p.stat().st_size), dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                                                          "model weights (public: theodoris-lab/MaxToki on Hugging Face)", "model weights (not copied)", f"setup/{w}"])

    # 6. pipeline specs
    for s in SPECS:
        p = PIPE / s
        assert p.stat().st_mtime < CUTOFF, s
        copy_file(p, DST / "pipelines" / s, category="spec")

    # 7. small reference data read by pre-audit scripts (paths mirror the originals under biomechinterp/)
    for r in REFS:
        p = BIOM / r
        copy_file(p, DST / "reference_data" / "biomechinterp" / r, category="reference_data")

    # 8. listings inside the snapshot
    for d, rows in listings.items():
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "_large_files_listing.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["name", "bytes", "mtime", "shape_or_columns", "why_not_copied", "path_in_project"])
            for row in rows:
                w.writerow(row)

    # 9. strip absolute paths that reveal paper or revision folders (text files only)
    path_rx = re.compile(r"(?:(?:/Volumes|/Users|~)[^\s\"'`]*?|projects/maxtoki)/(?:paper[-\w]*|revision|framework-eval)\b[^\s\"'`]*")
    n_path_hits = 0
    for p in DST.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".md", ".py", ".json", ".txt", ".log", ".csv", ".tsv", ".sh", ".yaml", ".yml"):
            if p.stat().st_size > 20_000_000:
                continue
            t = p.read_text(errors="replace")
            if path_rx.search(t):
                n_path_hits += len(path_rx.findall(t))
                mt = p.stat().st_mtime
                p.write_text(path_rx.sub("[path removed]", t))
                os.utime(p, (mt, mt))
                inventory.setdefault("paths_stripped", []).append(str(p.relative_to(DST)))
    inventory["n_path_hits_stripped"] = n_path_hits

    # 10. remove macOS AppleDouble sidecar files ('._*') that the OS writes on exFAT for the
    #     com.apple.provenance attribute of every file this script creates
    n_ad = 0
    for p in list(DST.rglob("._*")):
        if p.is_file():
            p.unlink(); n_ad += 1
    inventory["appledouble_removed"] = n_ad

    # 11. totals
    total = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file())
    inventory["totals"] = {"files": sum(1 for p in DST.rglob("*") if p.is_file()), "bytes": total,
                           "built": dt.datetime.now().isoformat(timespec="seconds"),
                           "cutoff": dt.datetime.fromtimestamp(CUTOFF).isoformat(timespec="minutes")}
    with open(BUILD / "snapshot_inventory.json", "w") as f:
        json.dump(inventory, f, indent=1)
    print(json.dumps(inventory["totals"]), "reconstructed:", len(inventory["reconstructed"]),
          "listed:", len(inventory["listed_not_copied"]), "excluded:", len(inventory["excluded"]))


if __name__ == "__main__":
    main(rebuild="--rebuild" in sys.argv)

"""Write studyC/SNAPSHOT_MANIFEST.md (outside the snapshot), a per-file sha256 list,
and a partial README reconstruction candidate (outside the snapshot).
Reads build/snapshot_inventory.json and build/leak_scan_results.json.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

STUDYC = Path("<EVAL_ROOT>/studyC")
SNAP = STUDYC / "snapshot" / "maxtoki"
BUILD = STUDYC / "build"
SRC = Path("<REPO_ROOT>/projects/maxtoki")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def mb(x):
    return f"{x/1e6:.1f} MB"


def partial_readme():
    """Candidate only (NOT placed in the snapshot): remove the blocks the audit added
    as whole paragraphs/sections. Lines rewritten by A5 inside the status table and the
    synthesis bullets cannot be reverted and are left as they are."""
    t = (SRC / "README.md").read_text().splitlines(keepends=True)
    out, skip, removed = [], False, []
    for i, l in enumerate(t, 1):
        if l.startswith("## Environment"):
            skip = True; removed.append(i); continue
        if skip and l.startswith("## "):
            skip = False
        if skip:
            removed.append(i); continue
        if l.startswith("**Cross-pipeline scope qualifier (load-bearing).**") or l.startswith("**Field-wide regulatory-specificity context (load-bearing).**"):
            removed.append(i); continue
        out.append(l)
    txt = "".join(out)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    p = BUILD / "alternatives" / "README.partial_reconstruction.md"
    p.write_text(txt)
    return p, removed


def main():
    inv = json.load(open(BUILD / "snapshot_inventory.json"))
    leak = json.load(open(BUILD / "leak_scan_results.json"))
    # per-file hashes of the snapshot
    rows = []
    for p in sorted(SNAP.rglob("*")):
        if p.is_file():
            rows.append((str(p.relative_to(SNAP)), p.stat().st_size, sha(p)))
    with open(BUILD / "snapshot_files_sha256.tsv", "w") as f:
        f.write("path\tbytes\tsha256\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\n")
    readme_alt, readme_removed = partial_readme()

    inc = inv["included"]
    by_area = collections.defaultdict(lambda: [0, 0])
    for x in inc:
        parts = x["dst"].split("/")
        key = "/".join(parts[:2]) if parts[0] in ("runs",) else parts[0]
        by_area[key][0] += 1; by_area[key][1] += x["bytes"]
    listed = inv["listed_not_copied"]
    lr = collections.Counter(x["reason"] for x in listed)
    lb = collections.Counter()
    for x in listed:
        lb[x["reason"]] += x["bytes"]
    exc = inv["excluded"]
    exc_post = [x for x in exc if "after cutoff" in x["reason"]]
    audit_named = sorted(x["src"] for x in exc_post if "audit" in x["src"])
    exc_mt = collections.Counter(x.get("mtime", "")[:10] for x in exc_post)
    exc_may7 = [x for x in exc_post if x.get("mtime", "").startswith("2026-05-07")]
    exc_later = [x for x in exc_post if x.get("mtime", "") and not x.get("mtime", "").startswith("2026-05-07")]

    L = []
    A = L.append
    A("# Study C pre-audit snapshot: manifest")
    A("")
    A("This manifest is for the study team. It is kept outside the snapshot folder so blind auditors do not see it.")
    A("")
    A("## 1. Numbers first")
    t = inv["totals"]
    A("")
    A(f"- Snapshot folder: `{SNAP}`")
    A(f"- {t['files']:,} files, {mb(t['bytes'])} (limit about 300 MB). Built {t['built']}. Cutoff: files last changed before {t['cutoff']} local time (the audit checklist was written at 16:57 on 2026-05-07).")
    A(f"- Copied as they were: {len(inc):,} files. Rebuilt with audit text removed: {len(inv['reconstructed'])} files. "
      f"Listed but not copied: {len(listed):,} files ({mb(sum(x['bytes'] for x in listed))}). "
      f"Left out because they changed at or after the cutoff: {len(exc_post)} files.")
    A(f"- Leak scan: {leak['files_scanned']:,} text files. No hit for audit files, audit actions, paper folders, the evaluation workspace, the corrected CRISPRi numbers, repair outputs, re-run names, hook-bug wording, sign-bug wording or A100. All strong-term hits are pre-audit wording (section 7).")
    A(f"- macOS wrote {inv.get('appledouble_removed', 0):,} '._*' sidecar files (exFAT stores the com.apple.provenance attribute that way); the build deleted them.")
    A("")
    A("## 2. What is in the snapshot")
    A("")
    A("Counts below are the 1,806 files copied unchanged. The snapshot also holds the 6 rebuilt summaries (section 3) and 50 `_large_files_listing.tsv` files, which gives the total of 1,862.")
    A("")
    A("| area | files | size |")
    A("|---|---:|---:|")
    for k, (n, b) in sorted(by_area.items(), key=lambda x: -x[1][1]):
        A(f"| {k} | {n} | {mb(b)} |")
    A("")
    A("- `runs/<8 runs>/`: scripts, logs, configs, run-level READMEs, STATUS, EXTENDED_FINDINGS, MANUAL_REVIEW, and outputs under 5 MB, "
      "plus nine 5-20 MB files chosen because they hold per-pair or per-anchor data behind headline numbers "
      "(attention-grn pair datasets, manifold internal and zero-shot centroids, steering state signatures, one copy of the SAE gene-name list).")
    A("- `summaries/`: the pre-audit files as they were, plus 4 rebuilt files (section 3).")
    A("- `pipelines/`: the 8 deployed specs and `sparse-autoencoders/README.md` (the SAE composition guide). All are dated 2026-04-15 and were not changed later. "
      "Left out: `audit-recurring-review-issues.md` (the audit checklist) and `sparse-autoencoders/04-atlas-deployment.md` (web atlas, not one of the 8).")
    A("- **Additions beyond the task list** (remove if not wanted): `setup/` (the adapter, dataset loader and TopK-SAE code that every run imports, the token dictionary, "
      "and the two model config.json files; no weights) and `reference_data/biomechinterp/...` (6 small public reference files that the run scripts read from absolute paths: "
      "TRRUST, STRING pairs, GO BP / Reactome / KEGG gene sets, the Geneformer symbol-to-Ensembl dictionary; the folder mirrors the original path under "
      "`<DATA_ROOT>/`). DoRothEA files were not added, because only the post-audit repair used them.")
    A("- `_large_files_listing.tsv` in each folder that had large or duplicate files: name, bytes, mtime, shape or columns/rows, and why it was not copied.")
    A("")
    A("## 3. Rebuilt files (audit-added text removed)")
    A("")
    A("File times of rebuilt files are set to 2026-05-07 16:56 (the nominal snapshot time). Their real history is below.")
    A("")
    for r in inv["reconstructed"]:
        A(f"- `{r['dst']}` (source last changed {r['src_mtime']}). Method: {r['method']}.")
        det = {k: v for k, v in r.items() if k in ("removed_line_ranges", "removed_single_lines", "reverted_wording_edits", "removed_inline")}
        A(f"  Detail: `{json.dumps(det, ensure_ascii=False)}`")
    A("")
    A("Confidence, file by file:")
    A("- **circuit-tracing summary: high.** The two removed blocks are marked '(added 2026-05-07, audit action A4/A2)'. The rest keeps the pre-audit text (54.6% 'near-chance').")
    A("- **sae-atlas summary: high.** Removed the A8 v2, A8 v1 and A4 sections (one contiguous block) and one inline A3 sentence. "
      "A clean pre-audit sibling also exists and is included as-is (`sae-atlas-217M-FINAL_SUMMARY_12layer.md`, 2026-04-19).")
    A("- **manifold summary (summaries/ copy): high.** After removing the A3 block and the residual-#6 section, it differs from the pre-audit runs/ copy (2026-05-07 16:41) only by the backfill sections that were never copied into summaries/. The stale 'n/t' rows are therefore genuinely pre-audit.")
    A("- **spectral summary: high.** The audit made exactly 3 wording edits (A6) and one inline CI (A3). A diff against the older 2026-04-17 copy shows those 3 edits as the only changed passages in the pre-Phase-8 text; they were restored from that copy. All other differences are Phase 8/9b additions from 2026-05-03.")
    A("- **longevity run summary: medium-high.** Removed the '## Data-availability blocker (audit residual exposure #7 ...)' section at the end. The rest reads as the pre-audit write-up (it refers to the report rebuilt at 11:40 that day). No older copy exists, so small in-place edits elsewhere cannot be ruled out; the audit completion files list only this addition for longevity.")
    A("- **topology run summary: medium.** Every '### ...' section whose heading contains 'added 2026-05-07' was removed (this is broader than the literal marker '(added 2026-05-07': only 1 of the 4 non-audit headings below contains that exact string): 6 audit sections (GroupKFold, Louvain seeds, Phase-0 seed, cross-layer replication, synthetic control, bootstrap CIs) and 4 other May-7 sections (H139, Phase 4 vs scGPT, Phase 8 chain, Phase 11 autoloop). "
      "Those 4 were most likely written before the audit (the Phase 11 section still has the pre-audit 'NOVEL POSITIVE' framing, none of the 4 mentions the audit, and the pre-audit manifold summary of 16:41 shows the executor itself used '(added 2026-05-07)' markers before the audit), and their content is in the snapshot anyway via `runs/topology-141-217M/EXTENDED_FINDINGS.md` and `outputs/phase11_autoloop/MANUAL_REVIEW.md` (both 16:37). "
      "Verifier's view: the evidence favours the version that keeps them. "
      f"A version that keeps those 4 sections is at `{BUILD / 'alternatives/topology-141-217M_FINAL_SUMMARY.keep_non_audit_may7_sections.md'}` if the team prefers it.")
    A("")
    A("## 4. Left out on purpose")
    A("")
    for d in inv["omitted_docs"]:
        A(f"- `{d['src']}`: {d['reason']}.")
    A(f"  A partial rebuild of the README (Environment section, 'Cross-pipeline scope qualifier' and 'Field-wide regulatory-specificity context' paragraphs removed; source lines {readme_removed[0]}-{readme_removed[-1]} range, {len(readme_removed)} lines) "
      f"is at `{readme_alt}`. It still contains A5 wording in the topology row and the cross-pipeline synthesis bullets. It is **not** in the snapshot.")
    A("- `audits/` (all six files), every `paper*/` folder, the public repo copy, `review_plans/`, `prompts/` (reviewer role prompts) and the audit checklist spec.")
    A(f"- {len(exc_may7)} files changed on 2026-05-07 at or after 16:57: the audit repair scripts and their outputs"
      f" ({len(audit_named)} of them have 'audit' in the name), plus the edited summaries handled in section 3.")
    A(f"- {len(exc_later)} files created or changed after 2026-05-07 (the current re-runs: `v2_*` scripts and outputs, `setup/hooks_v2.py`, `setup/test_hooks_v2.py`, `attention-grn-217M/outputs/v2_eval/`, and similar). Dates seen: {dict(sorted(exc_mt.items()))}.")
    n_atlas = sum(1 for x in listed if x["reason"].startswith("web atlas"))
    b_atlas = sum(x["bytes"] for x in listed if x["reason"].startswith("web atlas"))
    A(f"- The web atlas build (`runs/sae-atlas-217M/atlas/`, `atlas-data/`: {n_atlas} files, {mb(b_atlas)}, not counting node_modules, which was skipped entirely), except `atlas-data/cross_layer_graph.json`, which a circuit-tracing script reads.")
    A("- Model weights and SAE checkpoints (listed with sizes), raw datasets (h5ad files, 0.6-30 GB each; not listed).")
    A("")
    A("## 5. Listed, not copied")
    A("")
    A("| reason | files | size |")
    A("|---|---:|---:|")
    for k, n in lr.most_common():
        A(f"| {k} | {n} | {mb(lb[k])} |")
    A("")
    A("Key files an auditor cannot open (they can see them in the listings): `circuit-tracing-217M/outputs/circuit_edges.csv` (79.9 MB), "
      "`attention-grn-217M/outputs/phase0*/attention_edges_layer_mean.npy` and per-head arrays, `manifold-discovery-217M/artifacts/anchors/centroids_external.npy` (35.5 MB), "
      "`manifold-discovery-217M/outputs/phase1/cells_*.npz`, the 13 SAE checkpoints (48.6 MB each), spectral `layer_gene_embeddings.npy` files. "
      "So auditors can read all code and small outputs but cannot re-run the heaviest re-analyses (edge counts by layer, donor bootstrap on the external panel, curveball with attention scores) from the snapshot alone.")
    A("")
    A("## 6. What blind auditors can and cannot find from this snapshot")
    A("")
    A("- Can find (evidence is in the snapshot): the Phase 11 sign bug (code + JSON), the triplet overwrite (code + ratios), the block-deleting hooks (code; Exp-1 per-feature files show 0 edges at L6), steering invariance (steering JSONs), the curveball non-mixing (code + TRRUST file + gene lists), the 0/48 detection problem, the manifold gate/null/zero-shot issues (reports + anchor metadata), the spectral CKA/disjoint issues, spec deviations.")
    A("- Cannot find: errors that exist only in the audit files, the repair outputs or the paper (for example the corrected-CRISPRi base-rate error, the GATA1 matched-null problem, the 80% subsampling CIs). Those rows in the error ledger have `present_in_pre_audit_state` = no and should not be scored in Study C.")
    A("- Cannot fully find: errors whose only pre-audit trace was the project README (it is left out).")
    A("- Note: the deployed circuit-tracing spec itself warns (pitfall 6) that HF `output_hidden_states` indexing is off by one. This is genuine pre-audit content and stays.")
    A("")
    A("## 7. Leak scan (build/leak_scan.py -> build/leak_scan_results.json)")
    A("")
    A("| term group | hits | judgement |")
    A("|---|---:|---|")
    judge = {
        "added 2026-05-07": "pre-audit May-7 work: runs/manifold-discovery-217M/FINAL_SUMMARY.md (16:41) and runs/topology-141-217M/README.md (16:38); no audit content",
        "reviewer": "spec text about the executor-reviewer loop and a brainstormer prompt ('not a conservative reviewer'); pre-audit",
        "revision": "commit subjects of pinned source repos ('Add revision analyses', 'BMC Genomics revision') and 'selection revision' in the spectral summary; pre-audit",
        "investigation": "spec text ('focused investigation'); pre-audit",
        "54.61 / 53.48 / 53.42": "numeric coincidences inside CSV/log values (e.g. 0.53422433); no CRISPRi result",
        "V2_ / v2 re-run names": "all are 'SV2_SV4'-style singular-vector names in spectral outputs; no re-run file",
        "off-by-one": "deployed spec 02-causal-circuit-tracing.md pitfall 6 (2026-04-15), a genuine part of the contract",
        "spot-check": "the topology H139 'spot-check' phase name; pre-audit",
    }
    for k, v in leak["strong"].items():
        A(f"| {k} | {v['n_hits']} | {judge.get(k, 'none' if v['n_hits'] == 0 else 'see results file')} |")
    A("")
    A("Weak terms (innocent uses are common; every hit is in the results file):")
    A("")
    A("| term | hits | files | judgement |")
    A("|---|---:|---:|---|")
    wj = {
        "audit (any)": "pipeline vocabulary only: 'strict max-null audit' (H141), 'boundary-condition audits', 'Spec §8 validation audit' (spectral, 2026-05-03); no reference to the 2026-05-07 audit",
        "bug (word)": "spec pitfalls ('a common bug is ...'), a fixed implementation bug in the pre-audit topology summary, the exhaustive summary's own 'not a code bug' claim (pre-audit), STATUS notes",
        "2026-05-07 (date)": "pre-audit May-7 work (manifold STATUS/FINAL_SUMMARY, topology README, longevity status line)",
        "GATA1": "gene names in perturbation lists and gene tables",
        "ChIP": "spec reference lists (DoRothEA as a database); no ChIP-seq result",
        "corrected": "'BH-corrected' and similar",
        "overwrite": "spec text on how to apply the ablation delta",
        "power": "spec text and one manifold sentence",
    }
    for k, v in leak["weak"].items():
        A(f"| {k} | {v['n_hits']} | {v['n_files']} | {wj.get(k, '')} |")
    A("")
    A("## 8. Suggestions for running Study C")
    A("")
    A("- Point auditors at `snapshot/maxtoki/` only, from a workspace with no project instruction file or memory index; block reads outside it and scan transcripts afterwards.")
    A("- Do not give the original audit checklist as is: it names this project's cases (GATA1 ChIP-seq, H123 positive control, '54.6% near-chance'). Use a neutral rewrite of P1-P10 for the checklist arm.")
    A("- Score only ledger rows with `present_in_pre_audit_state` = yes or partly.")
    A("")
    A("## 9. Files")
    A("")
    A(f"- Build script: `{BUILD / 'build_snapshot.py'}`; inventory: `{BUILD / 'snapshot_inventory.json'}`; per-file hashes: `{BUILD / 'snapshot_files_sha256.tsv'}` ({len(rows):,} rows).")
    A(f"- Leak scan: `{BUILD / 'leak_scan.py'}` -> `{BUILD / 'leak_scan_results.json'}`.")
    A(f"- Alternatives (not in the snapshot): `{BUILD / 'alternatives'}`.")
    A("")
    A("## Plain-words summary")
    A("")
    A(f"The snapshot is a {mb(t['bytes'])} copy of the MaxToki project as it stood just before the audit began. It has every run's code, logs, configs and small results, "
      "the pipeline specs, and the run summaries. Six summaries that the audit edited were rebuilt by cutting out the audit's additions; for five of them we are confident, for the topology one we cut a little more than needed. "
      "The project README could not be rebuilt and is left out. Audit files, paper folders, repair outputs and the new re-run files are all left out. A word scan found no leak of the audit, the later checks or the known answers; the hits are ordinary pipeline words.")
    (STUDYC / "SNAPSHOT_MANIFEST.md").write_text("\n".join(L) + "\n")
    cfg = {"task": "Study C pre-audit snapshot", "cutoff_local_time": inv["totals"]["cutoff"],
           "source_roots": [str(SRC), "<REPO_ROOT>/pipelines",
                            "<DATA_ROOT> (reference_data only)"],
           "snapshot": str(SNAP), "snapshot_files": len(rows), "snapshot_bytes": sum(r[1] for r in rows),
           "seeds": "none (no random steps)",
           "scripts_sha256": {n: sha(BUILD / n) for n in ("build_snapshot.py", "leak_scan.py", "build_manifest.py")},
           "outputs_sha256": {n: sha(BUILD / n) for n in ("snapshot_inventory.json", "leak_scan_results.json", "snapshot_files_sha256.tsv")},
           "reconstructed_sources_sha256": {r["src"]: r["src_sha256"] for r in inv["reconstructed"]},
           "note": "Copied files keep their source bytes; their sha256 is listed in snapshot_files_sha256.tsv."}
    (BUILD / "run_config.json").write_text(json.dumps(cfg, indent=1))
    print("manifest written; files hashed:", len(rows))


if __name__ == "__main__":
    main()

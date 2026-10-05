"""Scan the Study C snapshot for text that could reveal the audit, the later
investigation, or the known errors. Read-only. Writes leak_scan_results.json.

Strong terms should not appear at all. Weak terms are common words that also
have innocent uses; every hit is listed so a person can judge it.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

SNAP = Path("<EVAL_ROOT>/studyC/snapshot/maxtoki")
OUT = Path(__file__).resolve().parent / "leak_scan_results.json"
TEXT_EXT = {".md", ".py", ".json", ".txt", ".log", ".csv", ".tsv", ".sh", ".yaml", ".yml", ".html", ".tex"}
MAX_BYTES = 20_000_000

STRONG = {
    "added 2026-05-07": r"added 2026-05-07",
    "audit action / residual": r"audit (action|residual)",
    "audits/ folder": r"audits/|audit-20260507",
    "audit spec file": r"audit-recurring-review-issues",
    "reviewer": r"\breviewer",
    "revision": r"\brevision\b|/revision/",
    "investigation": r"\binvestigation\b",
    "paper folders": r"paper-(plos-one|biosystems|jbi|cbac|deanon)|projects/maxtoki/paper\b",
    "framework-eval / studyC": r"framework-eval|studyC",
    "54.61 / 53.48 / 53.42": r"54\.61|53\.48|53\.42|0\.5348|0\.5342",
    "repair outputs": r"groupkfold_crispri|phase8t_positive_tf|phase8t_chance_baseline|edge_density_chance_baseline|synthetic_positive_control|seed_stability|iter_04_replication|bootstrap_pearson|external_validation_external_bootstrap|cell_level_benchmark_lite",
    "V2_ / v2 re-run names": r"V2_|v2_hooks|v2_zero_edit|hooks_v2|v2_eval",
    "hook artefact wording": r"artefact of the hook|artifact of the hook|block deletion|deletes? (a|the) (whole )?(transformer )?block",
    "sign bug wording": r"sign bug|ignored predicted sign|dropped sign",
    "off-by-one": r"off-by-one|off by one",
    "A100": r"\bA100\b",
    "spot-check": r"spot-check",
    "Opus / Claude version": r"Opus 4\.7|Opus 5\.5|claude-opus",
}
WEAK = {
    "audit (any)": r"(?i)\baudit",
    "bug (word)": r"(?i)\bbugs?\b",
    "2026-05-07 (date)": r"2026-05-07",
    "GATA1": r"GATA1",
    "ChIP": r"(?i)chip-seq|dorothea",
    "corrected": r"(?i)\bcorrected\b",
    "overwrite": r"(?i)overwrit",
    "power": r"(?i)\bpower\b",
}


def scan():
    strong_hits = defaultdict(list)
    weak_counts = defaultdict(lambda: defaultdict(int))
    weak_examples = defaultdict(list)
    n_files = 0
    for p in sorted(SNAP.rglob("*")):
        if not p.is_file() or p.name.startswith("._") or p.suffix.lower() not in TEXT_EXT or p.stat().st_size > MAX_BYTES:
            continue
        n_files += 1
        rp = str(p.relative_to(SNAP))
        with open(p, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                for k, rx in STRONG.items():
                    if re.search(rx, line):
                        strong_hits[k].append({"file": rp, "line": i, "text": line.strip()[:200]})
                for k, rx in WEAK.items():
                    if re.search(rx, line):
                        weak_counts[k][rp] += 1
                        if len(weak_examples[k]) < 400:
                            weak_examples[k].append({"file": rp, "line": i, "text": line.strip()[:160]})
    return n_files, strong_hits, weak_counts, weak_examples


if __name__ == "__main__":
    n, strong, wc, wex = scan()
    res = {"files_scanned": n,
           "strong": {k: {"n_hits": len(strong.get(k, [])), "hits": strong.get(k, [])[:200]} for k in STRONG},
           "weak": {k: {"n_hits": sum(wc[k].values()), "n_files": len(wc[k]), "by_file": dict(sorted(wc[k].items(), key=lambda x: -x[1])[:60]),
                        "examples": wex[k][:120]} for k in WEAK}}
    OUT.write_text(json.dumps(res, indent=1))
    print("files scanned:", n)
    for k in STRONG:
        print(f"STRONG {k:28s} {len(strong.get(k, []))}")
    for k in WEAK:
        print(f"weak   {k:28s} {sum(wc[k].values()):6d} hits in {len(wc[k])} files")

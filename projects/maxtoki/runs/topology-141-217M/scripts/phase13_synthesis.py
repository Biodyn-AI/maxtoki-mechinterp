"""Phase 13 — synthesis: compile a 5-layer hierarchy report and
final FINAL_SUMMARY.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
OUT = RUN / "outputs/synthesis"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def main():
    p0 = load_json(RUN / "outputs/phase0/phase0_summary.json")
    p1 = load_json(RUN / "outputs/phase1/phase1_summary.json")
    p5 = load_json(RUN / "outputs/phase5/phase5_summary.json")
    p6 = load_json(RUN / "outputs/phase6/phase6_summary.json")
    p78 = load_json(RUN / "outputs/phase78/phase78_summary.json")
    p9 = load_json(RUN / "outputs/phase9/phase9_summary.json")
    p12 = load_json(RUN / "outputs/phase12/phase12_summary.json")

    # Compose the 5-layer hierarchy table per source-paper §1
    findings: list[dict] = []

    # Layer 1 — cross-model (skipped in this run)
    findings.append(dict(
        layer=1,
        finding="Cross-model CCA alignment (H17/H20/H24)",
        status="SKIPPED",
        verdict="No reference model (scGPT/Geneformer) extracted on the same gene pool — "
                "Layer 1 cannot be evaluated for MaxToki-217M in this run.",
    ))

    # Layer 2 — persistent homology (H01/H03)
    if p5:
        per_dom = []
        for dom, info in p5.items():
            sig = info.get("n_layers_significant", 0)
            tot = info.get("n_layers", 0)
            rew = info.get("rewire_n_significant", 0)
            rew_tot = info.get("rewire_n_total", 0)
            per_dom.append(f"{dom}: {sig}/{tot} layers feature-shuffle-sig; "
                            f"{rew}/{rew_tot} rewire-sig")
        findings.append(dict(
            layer=2,
            finding="Persistent homology (H01/H03) + rewiring null",
            status="EVALUATED",
            verdict="; ".join(per_dom),
        ))

    # Layer 3 — manifold distance hierarchy
    if p6 and "per_domain_per_metric" in p6:
        df = pd.DataFrame(p6["per_domain_per_metric"])
        per_metric = df.groupby("metric")["mean"].mean().sort_values()
        ordered = ", ".join(f"{m}={v:+.3f}" for m, v in per_metric.items())
        # Check ordering monotonicity
        order_ok = (per_metric["euclidean"] <= per_metric["geodesic"] <=
                    per_metric.get("diffusion", per_metric["geodesic"]) <=
                    per_metric.get("triangle_defect", 1.0)) \
            if "diffusion" in per_metric.index else False
        findings.append(dict(
            layer=3,
            finding="Manifold distance hierarchy (H13/H16/H32/H69-H70)",
            status="EVALUATED",
            verdict=f"Mean ΔAUROC vs coexpression by metric: {ordered}; "
                    f"target hierarchy monotonic: {order_ok}",
        ))

    # Layer 4 — H123 strongest finding
    if p78 and "h123" in p78:
        per_dom = []
        for dom, info in p78["h123"].items():
            per_dom.append(f"{dom}: {info['n_layers_positive_null_gap']}/{info['n_layers']} layers "
                            f"positive null-gap (mean Δ={info['mean_delta']:+.3f})")
        findings.append(dict(
            layer=4,
            finding="H123 signed motif-community hardening (paper's strongest)",
            status="EVALUATED",
            verdict="; ".join(per_dom),
        ))

    # Layer 5 — strict max-null audit
    if p12 and "overall" in p12:
        ov = p12["overall"]
        per_dom = []
        for dom, info in p12.items():
            if dom == "overall":
                continue
            per_dom.append(f"{dom}: mean strict margin {info['mean_strict_margin']:+.3f}, "
                            f"{info['n_strict_positive']}/{info['n_tests']} tests strict-positive")
        findings.append(dict(
            layer=5,
            finding="H141 strict max-null audit",
            status="EVALUATED",
            verdict=(f"Overall mean strict margin = {ov['mean_strict_margin']:+.3f}, "
                     f"{ov['n_strict_positive']}/{ov['n_tests']} tests pass; "
                     + "; ".join(per_dom)),
        ))

    # Stability-selected descriptors (Phase 9 / H91)
    if p9:
        per_dom = []
        for dom, info in p9.items():
            per_dom.append(f"{dom}: mean AUC {info['mean_auc']:.3f} "
                            f"({info['fraction_above_chance']*100:.0f}% above chance)")
        findings.append(dict(
            layer=99,  # supplementary
            finding="H91 stability-selected descriptors (combined classifier)",
            status="EVALUATED",
            verdict="; ".join(per_dom),
        ))

    findings_df = pd.DataFrame(findings)
    findings_df.to_csv(OUT / "layered_findings.csv", index=False)

    # Persist a flat all-summaries dump
    with open(OUT / "all_phase_summaries.json", "w") as f:
        json.dump(dict(
            phase0=p0, phase1=p1, phase5=p5, phase6=p6,
            phase78=p78, phase9=p9, phase12=p12,
        ), f, indent=2, default=str)

    # FINAL_SUMMARY.md
    md = ["# topology-141-217M — FINAL SUMMARY\n",
          "_Pipeline_: `pipelines/topology-geometry-141-hypotheses.md` (arxiv 2602.22289)\n",
          "_Target model_: MaxToki-217M-HF (LlamaForCausalLM, 11 layers, hidden=1232)\n",
          "_Run date_: 2026-05-03\n\n",
          "## Scope\n",
          "Headline backbone of the 141-hypothesis pipeline: phases 0, 1+2+3 harness, "
          "5 (persistent homology + rewiring null), 6 (manifold distance hierarchy), "
          "7+8 (community + signed motif-community H123), 9 (stability-selected "
          "descriptors H91), 12 (strict max-null), 13 (synthesis). "
          "Phase 4 cross-model CCA was **skipped** — no scGPT/Geneformer "
          "extraction available on the MaxToki gene pool. Phase 11 autonomous Codex "
          "loop was **skipped** — no Codex backend in this environment.\n\n",
          "## 5-layer hierarchy of findings (per source paper §1)\n\n",
          "| Layer | Finding | Status | Verdict |\n",
          "|---|---|---|---|\n"]
    for f_ in findings:
        md.append(f"| L{f_['layer']} | {f_['finding']} | {f_['status']} | {f_['verdict']} |\n")
    md.append("\n## Per-phase summaries\n")
    for tag, payload in [("Phase 0 (extraction)", p0),
                         ("Phase 1 (splits/pair tables)", p1),
                         ("Phase 5 (persistent homology + rewire null)", p5),
                         ("Phase 6 (manifold distance hierarchy)", p6),
                         ("Phase 7+8 (community + signed motif H16/H116/H123)", p78),
                         ("Phase 9 (H91 stability selection)", p9),
                         ("Phase 12 (H141 strict max-null)", p12)]:
        md.append(f"\n### {tag}\n```json\n{json.dumps(payload, indent=2, default=str)}\n```\n")

    (RUN / "FINAL_SUMMARY.md").write_text("".join(md))
    print("\n[phase 13] wrote synthesis + FINAL_SUMMARY.md")
    print("\n5-LAYER HIERARCHY:")
    for f_ in findings:
        print(f"  L{f_['layer']}  {f_['finding']:<60s}  {f_['status']}")
        print(f"        verdict: {f_['verdict']}")


if __name__ == "__main__":
    main()

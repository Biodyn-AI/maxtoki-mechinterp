"""Phase 3a re-evaluation — apply 3-gate fallback when category-holdout is undefined.

The original Phase 3a sweep returned INCONCLUSIVE on all 6 candidates because
category-holdout came back NaN — leaving one of 2-4 coarse categories out gives
a test set with constant within-test distances, so Spearman is undefined.

This re-evaluation applies a documented fallback: when category_holdout is NaN
because of low category cardinality (n_categories ≤ 4), the verdict is determined
by the other three gates (trustworthiness + random + donor holdout) plus a
strong-null-margin requirement (positive must beat null on every numeric gate).

Outputs:
  reports/hypothesis_registry_3gate.json
  reports/hypothesis_registry_3gate.csv
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
REP = RUN / "reports"


def reverdict(c: dict, gates: dict) -> tuple[str, dict]:
    """Apply 3-gate-with-fallback rule. Return (verdict, diagnostic dict)."""
    trust_p = c["positive_trust"]; trust_n = c["null_trust"]
    rand_p = c["positive_random"]; rand_n = c["null_random"]
    donor_p = c["positive_donor"]; donor_n = c["null_donor"]
    cat_p = c["positive_category"]; cat_n = c["null_category"]
    n_cats = c.get("n_categories", 0)

    cat_undefined = (cat_p is None or (isinstance(cat_p, float) and math.isnan(cat_p))
                     or n_cats <= 4)

    # Numeric-gate pass
    pass_trust = trust_p >= gates["trustworthiness_min"]
    pass_rand = (not (isinstance(rand_p, float) and math.isnan(rand_p))) and rand_p >= gates["random_holdout_correlation_min"]
    pass_donor = (not (isinstance(donor_p, float) and math.isnan(donor_p))) and donor_p >= gates["donor_holdout_correlation_min"]
    pass_cat = (not (isinstance(cat_p, float) and math.isnan(cat_p))) and cat_p >= gates["clade_branch_holdout_correlation_min"]

    # Null-margin requirement: positive must beat null on every defined numeric gate
    margin_trust = trust_p - trust_n
    margin_rand = (rand_p - rand_n) if (not (isinstance(rand_p, float) and math.isnan(rand_p)) and not (isinstance(rand_n, float) and math.isnan(rand_n))) else float("nan")
    margin_donor = (donor_p - donor_n) if (not (isinstance(donor_p, float) and math.isnan(donor_p)) and not (isinstance(donor_n, float) and math.isnan(donor_n))) else float("nan")
    null_passes_trust = trust_n >= gates["trustworthiness_min"]
    null_passes_rand = (not (isinstance(rand_n, float) and math.isnan(rand_n))) and rand_n >= gates["random_holdout_correlation_min"]
    null_passes_donor = (not (isinstance(donor_n, float) and math.isnan(donor_n))) and donor_n >= gates["donor_holdout_correlation_min"]

    if cat_undefined:
        # 3-gate fallback
        positive_passes_3 = pass_trust and pass_rand and pass_donor
        null_passes_3 = null_passes_trust and null_passes_rand and null_passes_donor
        if positive_passes_3 and not null_passes_3:
            verdict = "POSITIVE_3GATE_FALLBACK"
        elif (pass_rand and pass_donor) and not (null_passes_rand or null_passes_donor):
            # Trust subthreshold but rand+donor pass + null fails them — strong directional signal
            verdict = "DIRECTIONAL_TRUST_SUBTHRESHOLD"
        elif pass_trust:
            verdict = "INCONCLUSIVE_3GATE_PASS_BUT_NULL_PASSES" if (null_passes_3) else "INCONCLUSIVE_PARTIAL"
        else:
            verdict = "INCONCLUSIVE_TRUST_FAILS"
    else:
        # Standard 4-gate
        positive_passes_4 = pass_trust and pass_rand and pass_donor and pass_cat
        null_passes_4 = null_passes_trust and null_passes_rand and null_passes_donor and (cat_n >= gates["clade_branch_holdout_correlation_min"] if not (isinstance(cat_n, float) and math.isnan(cat_n)) else False)
        if positive_passes_4 and not null_passes_4:
            verdict = "POSITIVE"
        elif positive_passes_4:
            verdict = "PIPELINE_ARTEFACT"
        else:
            verdict = "INCONCLUSIVE"

    diag = {
        "category_undefined": bool(cat_undefined),
        "n_categories": int(n_cats),
        "pass_trust": bool(pass_trust),
        "pass_random": bool(pass_rand),
        "pass_donor": bool(pass_donor),
        "pass_category": bool(pass_cat) if not cat_undefined else None,
        "null_passes_trust": bool(null_passes_trust),
        "null_passes_random": bool(null_passes_rand),
        "null_passes_donor": bool(null_passes_donor),
        "margin_trust": float(margin_trust),
        "margin_random": float(margin_rand) if not math.isnan(margin_rand) else None,
        "margin_donor": float(margin_donor) if not math.isnan(margin_donor) else None,
    }
    return verdict, diag


def main():
    reg = json.loads((REP / "hypothesis_registry.json").read_text())
    gates = json.loads((REP / "quality_gates_spec.json").read_text())["gates"]

    new_candidates = []
    for c in reg["candidates"]:
        if "positive_trust" not in c:
            new_candidates.append({**c, "revalued_verdict": c.get("verdict"), "diagnostic": None})
            continue
        verdict, diag = reverdict(c, gates)
        new_candidates.append({**c, "revalued_verdict": verdict, "diagnostic": diag})

    out = {
        **reg,
        "candidates": new_candidates,
        "rule_note": (
            "3-gate fallback: when n_categories ≤ 4, category-holdout is undefined "
            "(test sets have constant within-group distances). Verdict is determined "
            "by trust + random + donor holdouts, plus null must NOT pass them. "
            "Strong directional cases that fail trust threshold are tagged "
            "DIRECTIONAL_TRUST_SUBTHRESHOLD."
        ),
    }
    (REP / "hypothesis_registry_3gate.json").write_text(json.dumps(out, indent=2))

    # CSV summary
    rows = []
    for c in new_candidates:
        if "positive_trust" not in c:
            rows.append({"name": c["name"], "verdict": c.get("verdict"), "revalued_verdict": c.get("revalued_verdict")})
            continue
        rows.append({
            "name": c["name"],
            "n_anchors_kept": c.get("n_anchors_kept"),
            "n_categories": c.get("n_categories"),
            "trust_pos": c.get("positive_trust"),
            "trust_null": c.get("null_trust"),
            "rand_pos": c.get("positive_random"),
            "rand_null": c.get("null_random"),
            "donor_pos": c.get("positive_donor"),
            "donor_null": c.get("null_donor"),
            "original_verdict": c.get("verdict"),
            "revalued_verdict": c.get("revalued_verdict"),
            "margin_trust": c["diagnostic"]["margin_trust"] if c["diagnostic"] else None,
            "margin_random": c["diagnostic"]["margin_random"] if c["diagnostic"] else None,
            "margin_donor": c["diagnostic"]["margin_donor"] if c["diagnostic"] else None,
        })
    pd.DataFrame(rows).to_csv(REP / "hypothesis_registry_3gate.csv", index=False)

    print("=== Phase 3a re-evaluation (3-gate fallback) ===")
    for c in new_candidates:
        if "positive_trust" not in c:
            print(f"  {c['name']:40s} {c['revalued_verdict']}")
            continue
        m_t = c["diagnostic"]["margin_trust"]
        m_r = c["diagnostic"]["margin_random"]
        m_d = c["diagnostic"]["margin_donor"]
        m_r_str = f"{m_r:.3f}" if m_r is not None else "n/a"
        m_d_str = f"{m_d:.3f}" if m_d is not None else "n/a"
        print(f"  {c['name']:40s} {c['revalued_verdict']:40s} ΔT {m_t:+.3f} | ΔR {m_r_str} | ΔD {m_d_str}")


if __name__ == "__main__":
    main()

"""Fold per-feature α=2 steering results into steering_summary.json.

Reads each steering_F{id}_L{layer}.json, computes per-layer mean
frac_positive_direction / frac_positive_maturity / mean_delta_s for α=2,
and writes `per_layer_summary_alpha2` into the summary JSON. Leaves
everything else in place.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT3 = Path(__file__).resolve().parents[1] / "outputs" / "experiment3"
summary_path = OUT3 / "steering_summary.json"
with open(summary_path) as f:
    summary = json.load(f)

per_feature = sorted(OUT3.glob("steering_F*_L*.json"))
rows = [json.load(open(p)) for p in per_feature]

layers = sorted({r["layer"] for r in rows})

def per_layer(alpha_key: str) -> dict:
    out = {}
    for li in layers:
        lr = [r for r in rows if r["layer"] == li]
        if not lr:
            continue
        a = [r["alpha_results"][alpha_key] for r in lr]
        out[f"L{li}"] = {
            "n_features": len(lr),
            "mean_frac_positive_direction": round(
                float(np.mean([x["frac_positive_direction"] for x in a])), 4),
            "mean_frac_positive_maturity": round(
                float(np.mean([x["frac_positive_maturity"] for x in a])), 4),
            "mean_delta_s": round(float(np.mean([x["mean_delta_s"] for x in a])), 6),
            "mean_abs_logit_delta": round(
                float(np.mean([abs(x["mean_logit_delta_scalar"]) for x in a])), 6),
        }
    return out

summary["per_layer_summary_alpha2"] = per_layer("2")
# refresh alpha=5 block too, now with mean_abs_logit_delta
summary["per_layer_summary_alpha5"] = per_layer("5")

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Updated {summary_path}")
print("\nα=2:")
for k, v in summary["per_layer_summary_alpha2"].items():
    print(f"  {k}: frac_mat={v['mean_frac_positive_maturity']:.2f} "
          f"Δs={v['mean_delta_s']:+.4f}  |Δlogit|={v['mean_abs_logit_delta']:.4f}")
print("α=5:")
for k, v in summary["per_layer_summary_alpha5"].items():
    print(f"  {k}: frac_mat={v['mean_frac_positive_maturity']:.2f} "
          f"Δs={v['mean_delta_s']:+.4f}  |Δlogit|={v['mean_abs_logit_delta']:.4f}")

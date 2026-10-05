"""Build atlas-ready JSON data files from MaxToki Stage-1 outputs.

Produces the schema in `pipelines/sparse-autoencoders/04-atlas-deployment.md` §4
into `projects/maxtoki/runs/sae-atlas-217M/atlas-data/`. The structure mirrors
the geneformer-atlas template's `public/data/` so the React frontend can be
swapped in directly.

Reads:
  phase1/layer_XX/results.json
  phase2/layer_XX/feature_catalog.json
  phase2/layer_XX/significant_enrichments.csv
  phase2/layer_XX/annotation_summary.json
  phase3/svd_comparison.csv
  phase4/cross_layer_tracking.csv
  phase5/layer_XX/module_labels.npy
  phase6/causal_patching.csv + summary.json
  phase7/highways_*.json
  phase8/perturbation_response.csv + perturbation_summary.json
  phase11/celltype_enrichments.csv + summary.json
  phase12/layer_XX/umap_coords.npy + feature_metadata.csv

Writes 12 layer-files × 3 + ~9 global JSONs into atlas-data/.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/sae-atlas-217M"
P = RUN / "outputs"
OUT = RUN / "atlas-data"
OUT.mkdir(parents=True, exist_ok=True)

N_LAYERS = 12
LAYERS = list(range(N_LAYERS))
D_SAE = 4928
D_MODEL = 1232
K = 32
PROBE_LAYER = 5  # phase11/celltype + phase8/perturbation probe at L5

PRIMARY_LAYERS = [0, 5, 11]  # layers that have full Stage-1 coverage incl. cross-layer

print(f"  output dir: {OUT}")


def _json_default(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return round(float(obj), 6)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.bool_): return bool(obj)
    raise TypeError(f"not serialisable: {type(obj)}")


def save_json(data, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"), default=_json_default)
    print(f"    {path.name}  {os.path.getsize(path)/1024:.0f} KB")


# ── load helpers ───────────────────────────────────────────────────────────
def load_catalog(layer: int) -> list[dict]:
    return json.load(open(P / f"phase2/layer_{layer:02d}/feature_catalog.json"))


def load_enrichments(layer: int) -> pd.DataFrame:
    return pd.read_csv(P / f"phase2/layer_{layer:02d}/significant_enrichments.csv")


def load_modules(layer: int) -> np.ndarray:
    return np.load(P / f"phase5/layer_{layer:02d}/module_labels.npy")


def load_umap(layer: int) -> np.ndarray:
    return np.load(P / f"phase12/layer_{layer:02d}/umap_coords.npy")


def load_results(layer: int) -> dict:
    return json.load(open(P / f"phase1/layer_{layer:02d}/results.json"))


def load_annot_summary(layer: int) -> dict:
    return json.load(open(P / f"phase2/layer_{layer:02d}/annotation_summary.json"))


def parse_term(term: str) -> tuple[str, str]:
    """'GO_BP:Foo Bar (GO:0001234)' → ('GO_BP', 'Foo Bar')."""
    if ":" in term:
        ont, rest = term.split(":", 1)
    else:
        ont, rest = "OTHER", term
    rest = rest.strip()
    # strip trailing GO/KEGG accession in parentheses if present
    if rest.endswith(")") and "(" in rest:
        i = rest.rfind("(")
        rest = rest[:i].strip()
    return ont, rest


# ── per-layer files: features, positions, annotations ─────────────────────
def process_layer(layer: int) -> tuple[dict, list[dict]]:
    print(f"\n  layer {layer}...")
    cat = load_catalog(layer)  # list[{feature_id, top20_genes, max_activation, activation_frequency}]
    enr = load_enrichments(layer)
    mods = load_modules(layer)
    umap = load_umap(layer)

    n_alive = len(cat)
    catalog_by_id: dict[int, dict] = {f["feature_id"]: f for f in cat}

    enr_by_feat: dict[int, list[dict]] = defaultdict(list)
    for _, row in enr.iterrows():
        ont, term = parse_term(str(row["term"]))
        enr_by_feat[int(row["feature_id"])].append({
            "o": ont,
            "t": term,
            "p": float(row["q_bh"]),
            "or": round(float(row["overlap"]) * float(row["N"])
                        / max(float(row["K"]) * float(row["n"]), 1.0), 2),
            "n": int(row["overlap"]),
            "g": [],  # overlap genes not stored per-row in our CSV
        })
    # sort each feature's anns by p
    for k in enr_by_feat:
        enr_by_feat[k].sort(key=lambda a: a["p"])

    compact: list[dict] = []
    annotations_out: dict[str, dict] = {}
    positions_out: list[list[float]] = [[0.0, 0.0]] * D_SAE

    for fi in range(D_SAE):
        in_cat = fi in catalog_by_id
        feat = catalog_by_id.get(fi)
        anns = enr_by_feat.get(fi, [])
        best = anns[0] if anns else None
        # top genes from catalog top20
        if feat is not None:
            top_genes_names = feat.get("top20_genes", [])[:5]
            top_genes_compact = [{"n": g, "a": 1.0} for g in top_genes_names]
            max_act = float(feat.get("max_activation", 0.0))
            act_freq = float(feat.get("activation_frequency", 0.0))
        else:
            top_genes_compact = []
            max_act = 0.0
            act_freq = 0.0

        compact.append({
            "i": fi,
            "d": 0 if in_cat else 1,
            "f": round(act_freq, 6),
            "ma": round(max_act, 4),
            "fc": int(round(act_freq * 1_000_000)),
            "na": len(anns),
            "m": int(mods[fi]) if fi < len(mods) else -1,
            "sv": 0,  # SVD-aligned flag not computed per-feature in MaxToki phase 3
            "lb": best["t"] if best else "",
            "to": best["o"] if best else "none",
            "tg": top_genes_compact,
        })
        if anns:
            # Full-detail annotation block (for FeaturePage lazy load)
            all_genes = [{"n": g, "a": 1.0, "fc": 0}
                         for g in (feat.get("top20_genes", []) if feat else [])[:20]]
            annotations_out[str(fi)] = {
                "genes": all_genes,
                "anns": anns,
            }
        if fi < len(umap):
            positions_out[fi] = [round(float(umap[fi, 0]), 5), round(float(umap[fi, 1]), 5)]

    save_json(compact, OUT / f"layer_{layer:02d}_features.json")
    save_json(annotations_out, OUT / f"layer_{layer:02d}_annotations.json")
    save_json(positions_out, OUT / f"layer_{layer:02d}_positions.json")

    n_modules = int(np.unique(mods[mods >= 0]).size)
    return {
        "n_alive": n_alive,
        "n_modules": n_modules,
        "annotation_rate": round(load_annot_summary(layer)["annotation_rate"], 4),
        "n_annotated": load_annot_summary(layer)["n_annotated"],
        "total_enrichments": load_annot_summary(layer)["total_enrichments"],
    }, compact


# ── global_summary.json — adapt the existing 12-layer summary ─────────────
def build_global_summary(per_layer_stats):
    print("\n  global_summary.json")
    layers_out = []
    total_alive = total_annot = total_modules = total_enrichments = 0
    for li in LAYERS:
        st = per_layer_stats[li]
        res = load_results(li)
        ann = load_annot_summary(li)
        ont_counts = defaultdict(int)
        enr = load_enrichments(li)
        for term in enr["term"]:
            ont, _ = parse_term(str(term))
            ont_counts[ont] += 1

        layers_out.append({
            "layer": li,
            "alive": st["n_alive"],
            "dead": D_SAE - st["n_alive"],
            "annotated": ann["n_annotated"],
            "annotation_rate": round(ann["annotation_rate"], 4),
            "n_modules": st["n_modules"],
            "n_svd_aligned": 0,
            "n_novel": st["n_alive"],
            "ontology_counts": {
                "GO_BP": ont_counts.get("GO_BP", 0),
                "KEGG": ont_counts.get("KEGG", 0),
                "Reactome": ont_counts.get("Reactome", 0),
                "STRING": ont_counts.get("STRING", 0),
                "TRRUST": ont_counts.get("TRRUST_TF", 0) + ont_counts.get("TRRUST", 0),
            },
            "variance_explained": round(res["variance_explained"], 4),
            "mean_feature_cosine": round(res["mean_abs_decoder_cosine"], 4),
        })
        total_alive += st["n_alive"]
        total_annot += ann["n_annotated"]
        total_modules += st["n_modules"]
        total_enrichments += st["total_enrichments"]

    summary = {
        "model": "MaxToki-217M",
        "total_features": D_SAE * N_LAYERS,
        "total_alive": total_alive,
        "total_annotated": total_annot,
        "total_enrichments": total_enrichments,
        "n_layers": N_LAYERS,
        "features_per_layer": D_SAE,
        "d_model": D_MODEL,
        "d_sae": D_SAE,
        "k": K,
        "total_modules": total_modules,
        "total_novel": total_alive,
        "n_features_per_layer": D_SAE,  # legacy key for template compatibility
        "layers": layers_out,
    }
    save_json(summary, OUT / "global_summary.json")


# ── modules.json ──────────────────────────────────────────────────────────
def build_modules(all_features: dict[int, list[dict]]):
    print("\n  modules.json")
    out = []
    for li in LAYERS:
        mods = load_modules(li)
        feats_by_id = {f["i"]: f for f in all_features[li]}
        for module_id in sorted(set(int(m) for m in mods if m >= 0)):
            mod_feats = [int(i) for i, m in enumerate(mods) if m == module_id]
            ann_counts = defaultdict(int)
            for fi in mod_feats[:200]:  # sample for speed
                f = feats_by_id.get(fi)
                if f and f["lb"]:
                    ann_counts[f["lb"]] += 1
            top_anns = sorted(ann_counts.items(), key=lambda x: -x[1])[:5]
            out.append({
                "layer": li,
                "id": module_id,
                "n": len(mod_feats),
                "features": mod_feats[:100],  # cap for size
                "top_anns": [{"t": t, "c": c} for t, c in top_anns],
            })
    save_json(out, OUT / "modules.json")


# ── gene_index.json ───────────────────────────────────────────────────────
def build_gene_index(all_features: dict[int, list[dict]]):
    print("\n  gene_index.json")
    idx: dict[str, list] = defaultdict(list)
    for li in LAYERS:
        for f in all_features[li]:
            for rank, g in enumerate(f["tg"]):
                idx[g["n"]].append({
                    "l": li,
                    "i": f["i"],
                    "r": rank,
                    "lb": f["lb"][:60] if f["lb"] else "",
                    "m": f["m"],
                })
    save_json(dict(idx), OUT / "gene_index.json")


# ── ontology_index.json ───────────────────────────────────────────────────
def build_ontology_index():
    print("\n  ontology_index.json")
    idx: dict[str, list] = defaultdict(list)
    for li in LAYERS:
        enr = load_enrichments(li)
        for _, row in enr.iterrows():
            ont, term = parse_term(str(row["term"]))
            idx[term].append({
                "l": li,
                "i": int(row["feature_id"]),
                "p": float(row["q_bh"]),
                "o": ont,
            })
    filtered = {k: v for k, v in idx.items() if len(v) >= 2}
    save_json(filtered, OUT / "ontology_index.json")


# ── cross_layer_tracking.json ─────────────────────────────────────────────
def build_cross_layer_tracking():
    print("\n  cross_layer_tracking.json")
    df = pd.read_csv(P / "phase4/cross_layer_tracking.csv")
    pairs = []
    for _, r in df.iterrows():
        pairs.append({
            "src_layer": int(r["src_layer"]),
            "tgt_layer": int(r["tgt_layer"]),
            "n_src_alive": int(r["n_src_alive"]),
            "n_matches": int(r["n_matches"]),
            "persistence_rate": round(float(r["persistence_rate"]), 4),
        })
    save_json({"pairs": pairs, "primary_layers": PRIMARY_LAYERS}, OUT / "cross_layer_tracking.json")


# ── cross_layer_graph.json (highways → atlas summary + top-deps stub) ─────
def build_cross_layer_graph():
    print("\n  cross_layer_graph.json")
    out = {}
    for fname in sorted((P / "phase7").glob("highways_*.json")):
        d = json.load(open(fname))
        key = f"L{d['src_layer']:02d}_L{d['tgt_layer']:02d}"
        out[key] = {
            "summary": {
                "src_layer": d["src_layer"],
                "tgt_layer": d["tgt_layer"],
                "n_positions": d["n_positions"],
                "n_alive_src": d["n_alive_src"],
                "n_highways": d["n_highways"],
                "highway_rate": round(d["highway_rate"], 4),
                "mean_max_pmi": round(d["mean_max_pmi"], 3),
                "median_max_pmi": round(d["median_max_pmi"], 3),
                "max_max_pmi": round(d["max_max_pmi"], 3),
            },
            "deps": [],  # per-feature dep edges not exported in MaxToki phase 7
        }
    save_json(out, OUT / "cross_layer_graph.json")


# ── svd_comparison.json ───────────────────────────────────────────────────
def build_svd_comparison():
    print("\n  svd_comparison.json")
    df = pd.read_csv(P / "phase3/svd_comparison.csv")
    per_layer = {}
    for _, r in df.iterrows():
        li = int(r["layer"])
        svd_var = float(r["svd_top50_var_explained"])
        sae_var = float(r["sae_var_explained"])
        per_layer[f"L{li:02d}"] = {
            "svd_variance": round(svd_var, 4),
            "sae_variance": round(sae_var, 4),
            "gain": round(sae_var / svd_var, 2) if svd_var > 0 else 0,
            "n_aligned": int(r["n_svd_aligned"]),
            "n_novel": int(r["n_novel"]),
            "pct_novel": round(float(r["pct_novel"]), 4),
        }
    save_json({
        "aggregate": {
            "n_layers_compared": len(per_layer),
            "primary_layers": PRIMARY_LAYERS,
            "mean_sae_variance": round(df["sae_var_explained"].mean(), 4),
            "mean_svd_variance": round(df["svd_top50_var_explained"].mean(), 4),
        },
        "per_layer": per_layer,
    }, OUT / "svd_comparison.json")


# ── causal_patching.json ──────────────────────────────────────────────────
def build_causal_patching():
    print("\n  causal_patching.json")
    df = pd.read_csv(P / "phase6/causal_patching.csv")
    summary = json.load(open(P / "phase6/summary.json"))
    out = {
        "summary": {
            "n_features": int(summary["n_features"]),
            "median_specificity": round(float(summary["median_specificity"]), 4),
            "pct_above_2x": float(summary["pct_above_2x"]),
            "pct_above_10x": float(summary["pct_above_10x"]),
            "mean_target_delta": round(float(summary["mean_target_delta"]), 4),
            "mean_other_delta": round(float(summary["mean_other_delta"]), 4),
            "probe_layer": PROBE_LAYER,
        },
        "features": [{
            "i": int(r["feature_id"]),
            "lb": "",  # label populated by frontend from gene_index lookup
            "td": round(float(r["mean_target_delta"]), 4),
            "od": round(float(r["mean_other_delta"]), 4),
            "sr": round(float(r["specificity_ratio"]), 3),
        } for _, r in df.iterrows()],
    }
    save_json(out, OUT / "causal_patching.json")


# ── perturbation_response.json ────────────────────────────────────────────
def build_perturbation_response():
    print("\n  perturbation_response.json")
    df = pd.read_csv(P / "phase8/perturbation_response.csv")
    summary = json.load(open(P / "phase8/perturbation_summary.json"))
    out = {
        "summary": {
            "probe_layer": int(summary["probe_layer"]),
            "n_targets": int(summary["n_targets"]),
            "n_tfs": int(summary["n_tfs"]),
            "detection_rate": round(float(summary["detection_rate"]), 4),
            "n_detected": int(summary["n_detected"]),
            "n_tf_specific": int(summary["n_tf_specific"]),
            "tf_specificity_rate": round(float(summary["tf_specificity_rate"]), 4),
            "note": summary.get("note", ""),
        },
        "targets": [{
            "gene": str(r["target"]),
            "tf": bool(r["is_tf"]),
            "nk": int(r["n_pos"]),
            "nr": int(r["n_responding"]),
            "ns": int(r.get("n_responding", 0)) if bool(r["is_specific"]) else 0,
            "detected": bool(r["detected"]),
            "is_specific": bool(r["is_specific"]),
            "top": [],  # per-target responding-feature list not exported by phase 8 CSV
        } for _, r in df.iterrows()],
    }
    save_json(out, OUT / "perturbation_response.json")


# ── celltype enrichments (one layer only — best effort) ───────────────────
def build_celltype():
    print(f"\n  layer_{PROBE_LAYER:02d}_celltypes.json (only layer with celltype data)")
    enr_path = P / "phase11/celltype_enrichments.csv"
    summary_path = P / "phase11/summary.json"
    if not enr_path.exists():
        print("    (no celltype data)")
        return
    df = pd.read_csv(enr_path)
    summ = json.load(open(summary_path))
    by_feat: dict[str, dict] = defaultdict(lambda: {"ct": [], "ti": [], "tc": []})
    for fi, grp in df.groupby("feature_id"):
        ranked = grp.sort_values("p").head(10)
        by_feat[str(int(fi))]["ct"] = [
            {"c": str(r["cell_type"]), "p": float(r["p"]), "or": 1.0, "n": 0}
            for _, r in ranked.iterrows()
        ]
    out = {
        "summary": {
            "n_cells": 0,
            "tissues": [],
            "n_cell_types": int(summ["n_cell_types"]),
            "n_features_with_enrichment": int(summ["n_features_enriched"]),
            "probe_layer": PROBE_LAYER,
        },
        "cell_type_meta": {},
        "features": dict(by_feat),
    }
    save_json(out, OUT / f"layer_{PROBE_LAYER:02d}_celltypes.json")


# ── main ──────────────────────────────────────────────────────────────────
def main():
    per_layer_stats: dict[int, dict] = {}
    all_features: dict[int, list[dict]] = {}
    for li in LAYERS:
        per_layer_stats[li], all_features[li] = process_layer(li)

    build_global_summary(per_layer_stats)
    build_modules(all_features)
    build_gene_index(all_features)
    build_ontology_index()
    build_cross_layer_tracking()
    build_cross_layer_graph()
    build_svd_comparison()
    build_causal_patching()
    build_perturbation_response()
    build_celltype()

    print("\n  --- file inventory ---")
    total = 0
    for f in sorted(OUT.glob("*.json")):
        sz = os.path.getsize(f); total += sz
        print(f"    {f.name:38s} {sz/1024:8.1f} KB")
    print(f"    {'TOTAL':38s} {total/1024/1024:8.1f} MB")


if __name__ == "__main__":
    main()

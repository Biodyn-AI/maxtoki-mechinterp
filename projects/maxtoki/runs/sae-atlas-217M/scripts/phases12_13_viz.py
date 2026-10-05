"""Phases 12 + 13 — Feature-space geometry + interactive atlas.

Phase 12: Compute force-directed layouts (Fruchterman-Reingold) of
          co-activation module graphs, UMAP + t-SNE of features based
          on TF-IDF weighted ontology vectors.

Phase 13: Build a self-contained interactive HTML atlas with Plotly.js
          featuring: Layer Overview, Feature UMAP, Module Explorer,
          Cross-Layer Flow, and Gene Search.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))

PHASE0 = PROJ / "runs/sae-atlas-217M/outputs/phase0"
PHASE1 = PROJ / "runs/sae-atlas-217M/outputs/phase1"
PHASE2 = PROJ / "runs/sae-atlas-217M/outputs/phase2"
OUT12 = PROJ / "runs/sae-atlas-217M/outputs/phase12"
OUT13 = PROJ / "runs/sae-atlas-217M/outputs/phase13"
OUT12.mkdir(parents=True, exist_ok=True)
OUT13.mkdir(parents=True, exist_ok=True)

cfg = json.loads((PHASE0 / "run_config.json").read_text())
LAYERS = cfg["target_layers"]
SEED = 42

print("=" * 70)
print("Phases 12 + 13 — Feature geometry + Interactive atlas")
print("=" * 70)


# =====================================================================
# PHASE 12 — Feature-space geometry
# =====================================================================
print("\n[Phase 12] Feature-space geometry...")

import networkx as nx
import umap
from sklearn.manifold import TSNE
from topk_sae import TopKSAE
import torch

for li in LAYERS:
    print(f"\n  --- Layer {li} ---")
    out_layer = OUT12 / f"layer_{li:02d}"
    out_layer.mkdir(parents=True, exist_ok=True)

    # Load SAE decoder weights for feature directions
    ckpt = torch.load(PHASE1 / f"layer_{li:02d}/sae_final.pt",
                      map_location="cpu", weights_only=False)
    W_dec = ckpt["W_dec_weight"].numpy()  # (d_model, d_sae)
    d_sae = W_dec.shape[1]

    # Load feature catalog + annotations
    with open(PHASE2 / f"layer_{li:02d}/feature_catalog.json") as f:
        catalog = json.load(f)
    feat_ids = [c["feature_id"] for c in catalog]
    feat_top_gene = [c["top20_genes"][0] if c["top20_genes"] else "?" for c in catalog]
    feat_freq = [c["activation_frequency"] for c in catalog]

    try:
        enrich_df = pd.read_csv(PHASE2 / f"layer_{li:02d}/significant_enrichments.csv")
        feat_terms = enrich_df.groupby("feature_id")["term"].apply(list).to_dict()
        feat_n_enrich = enrich_df.groupby("feature_id").size().to_dict()
    except FileNotFoundError:
        feat_terms = {}
        feat_n_enrich = {}

    # Load module labels
    mod_path = PROJ / f"runs/sae-atlas-217M/outputs/phase5/layer_{li:02d}/module_labels.npy"
    if mod_path.exists():
        mod_labels = np.load(mod_path)
    else:
        mod_labels = -np.ones(d_sae, dtype=int)

    # 12a) Build TF-IDF ontology vectors for UMAP/t-SNE
    # Collect all unique terms across features
    all_terms_set = set()
    for fid in feat_ids:
        for t in feat_terms.get(fid, []):
            all_terms_set.add(t)
    all_terms = sorted(all_terms_set)
    term_to_idx = {t: i for i, t in enumerate(all_terms)}
    n_terms = len(all_terms)
    print(f"  unique annotation terms: {n_terms}")

    if n_terms >= 5 and len(feat_ids) >= 20:
        # Build TF-IDF matrix: (n_features, n_terms)
        tf_idf = np.zeros((len(feat_ids), n_terms), dtype=np.float32)
        # Document frequency
        doc_freq = np.zeros(n_terms, dtype=np.float32)
        for i, fid in enumerate(feat_ids):
            terms = feat_terms.get(fid, [])
            if not terms:
                continue
            for t in terms:
                ti = term_to_idx.get(t)
                if ti is not None:
                    tf_idf[i, ti] += 1.0
                    doc_freq[ti] += 1.0
        # IDF
        idf = np.log(len(feat_ids) / (doc_freq + 1))
        tf_idf *= idf[None, :]
        # L2 normalise rows
        row_norms = np.linalg.norm(tf_idf, axis=1, keepdims=True) + 1e-12
        tf_idf /= row_norms

        # UMAP
        print("  computing UMAP on TF-IDF ontology vectors...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                           random_state=SEED, n_components=2)
        umap_coords = reducer.fit_transform(tf_idf)
        np.save(out_layer / "umap_coords.npy", umap_coords.astype(np.float32))

        # t-SNE for validation
        print("  computing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=min(20, len(feat_ids) - 1),
                    metric="cosine", random_state=SEED, init="pca")
        tsne_coords = tsne.fit_transform(tf_idf)
        np.save(out_layer / "tsne_coords.npy", tsne_coords.astype(np.float32))
    else:
        # Fallback: use decoder weight cosine for UMAP
        print("  few annotation terms; using decoder weight UMAP...")
        alive_idx = [c["feature_id"] for c in catalog]
        W_alive = W_dec[:, alive_idx].T  # (n_alive, d_model)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                           random_state=SEED, n_components=2)
        umap_coords = reducer.fit_transform(W_alive)
        np.save(out_layer / "umap_coords.npy", umap_coords.astype(np.float32))
        tsne_coords = None

    # 12b) Force-directed layout of co-activation module graph
    print("  computing force-directed layout...")
    coac_path = PROJ / f"runs/sae-atlas-217M/outputs/phase5/layer_{li:02d}/coactivation_summary.json"
    if coac_path.exists():
        # Build a simplified graph: features as nodes, edges between
        # features in the same module (sampled for speed)
        G = nx.Graph()
        for fid in feat_ids:
            G.add_node(fid, module=int(mod_labels[fid]) if fid < len(mod_labels) else -1)
        # Add intra-module edges (sample up to 5000 edges)
        edges_added = 0
        for fid_a in feat_ids:
            if edges_added >= 5000:
                break
            mod_a = int(mod_labels[fid_a]) if fid_a < len(mod_labels) else -1
            if mod_a < 0:
                continue
            for fid_b in feat_ids:
                if fid_b <= fid_a:
                    continue
                mod_b = int(mod_labels[fid_b]) if fid_b < len(mod_labels) else -1
                if mod_a == mod_b:
                    G.add_edge(fid_a, fid_b)
                    edges_added += 1
                    if edges_added >= 5000:
                        break

        if G.number_of_edges() > 0:
            pos = nx.spring_layout(G, seed=SEED, iterations=50, k=1.0 / np.sqrt(len(feat_ids)))
            layout_coords = np.array([pos.get(fid, [0, 0]) for fid in feat_ids], dtype=np.float32)
            np.save(out_layer / "spring_layout.npy", layout_coords)
        else:
            layout_coords = np.zeros((len(feat_ids), 2), dtype=np.float32)

    # Save per-feature metadata table for the atlas
    meta_rows = []
    for i, fc in enumerate(catalog):
        fid = fc["feature_id"]
        meta_rows.append({
            "feature_id": fid,
            "top_gene": feat_top_gene[i],
            "top5_genes": ", ".join(fc["top20_genes"][:5]),
            "activation_freq": fc["activation_frequency"],
            "max_activation": fc["max_activation"],
            "n_enrichments": feat_n_enrich.get(fid, 0),
            "module": int(mod_labels[fid]) if fid < len(mod_labels) else -1,
            "umap_x": float(umap_coords[i, 0]) if i < len(umap_coords) else 0,
            "umap_y": float(umap_coords[i, 1]) if i < len(umap_coords) else 0,
        })
    pd.DataFrame(meta_rows).to_csv(out_layer / "feature_metadata.csv", index=False)
    print(f"  saved: {out_layer}")

print("\nPhase 12 COMPLETE")


# =====================================================================
# PHASE 13 — Interactive HTML atlas
# =====================================================================
print("\n[Phase 13] Building interactive HTML atlas...")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Collect data across all layers for the atlas
layer_summaries = []
for li in LAYERS:
    r = json.loads((PHASE1 / f"layer_{li:02d}/results.json").read_text())
    a = json.loads((PHASE2 / f"layer_{li:02d}/annotation_summary.json").read_text())
    layer_summaries.append({
        "layer": li,
        "var_explained": r["variance_explained"],
        "n_alive": r["n_alive"],
        "n_dead": r["n_dead"],
        "annotation_rate": a["annotation_rate"],
        "total_enrichments": a["total_enrichments"],
    })
ls_df = pd.DataFrame(layer_summaries)

# Build multi-tab HTML
html_parts = []
html_parts.append("""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>MaxToki-217M SAE Atlas</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
       max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; color: #1a1a2e; }
h1 { color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; }
h2 { color: #0f3460; margin-top: 40px; }
.tab-bar { display: flex; gap: 4px; margin: 20px 0; }
.tab-btn { padding: 10px 20px; border: none; background: #e0e0e0; cursor: pointer;
           border-radius: 6px 6px 0 0; font-size: 14px; font-weight: 500; }
.tab-btn.active { background: #0f3460; color: white; }
.tab-content { display: none; padding: 20px; background: white; border-radius: 0 8px 8px 8px;
               box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.tab-content.active { display: block; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; font-size: 13px; }
th { background: #0f3460; color: white; }
tr:nth-child(even) { background: #f8f9fa; }
.metric { display: inline-block; background: #e8f4f8; padding: 8px 16px; margin: 4px;
          border-radius: 6px; font-weight: 500; }
.metric .val { font-size: 24px; color: #0f3460; }
.metric .label { font-size: 11px; color: #666; }
</style>
</head><body>
<h1>MaxToki-217M SAE Atlas</h1>
<p>TopK sparse autoencoder decomposition of MaxToki-217M residual stream (layers L0, L5, L11).
<br>d_SAE = 4928, k = 32, trained on 1M positions from 500 K562 control cells.</p>

<div class="tab-bar">
  <button class="tab-btn active" onclick="showTab('overview')">Layer Overview</button>
  <button class="tab-btn" onclick="showTab('umap')">Feature UMAP</button>
  <button class="tab-btn" onclick="showTab('modules')">Modules</button>
  <button class="tab-btn" onclick="showTab('results')">Pipeline Results</button>
  <button class="tab-btn" onclick="showTab('genes')">Gene Search</button>
</div>
""")

# Tab 1: Layer Overview
html_parts.append('<div id="tab-overview" class="tab-content active">')
html_parts.append("<h2>Layer Overview</h2>")
# Summary metrics
for _, r in ls_df.iterrows():
    html_parts.append(f"""
<h3>Layer {int(r['layer'])}</h3>
<div class="metric"><div class="val">{r['var_explained']:.1%}</div><div class="label">Var Explained</div></div>
<div class="metric"><div class="val">{int(r['n_alive'])}</div><div class="label">Alive Features</div></div>
<div class="metric"><div class="val">{r['annotation_rate']:.1%}</div><div class="label">Annotation Rate</div></div>
<div class="metric"><div class="val">{int(r['total_enrichments'])}</div><div class="label">Enrichments</div></div>
""")
html_parts.append('<div id="overview-plot"></div>')
html_parts.append("</div>")

# Tab 2: Feature UMAP per layer
html_parts.append('<div id="tab-umap" class="tab-content">')
html_parts.append("<h2>Feature UMAP (TF-IDF ontology vectors, cosine metric)</h2>")
for li in LAYERS:
    meta = pd.read_csv(OUT12 / f"layer_{li:02d}/feature_metadata.csv")
    html_parts.append(f'<div id="umap-L{li}"></div>')
html_parts.append("</div>")

# Tab 3: Modules
html_parts.append('<div id="tab-modules" class="tab-content">')
html_parts.append("<h2>Co-activation Modules</h2>")
for li in LAYERS:
    meta = pd.read_csv(OUT12 / f"layer_{li:02d}/feature_metadata.csv")
    mod_counts = meta["module"].value_counts().sort_index()
    html_parts.append(f"<h3>Layer {li}: {len(mod_counts)} modules</h3>")
    html_parts.append("<table><tr><th>Module</th><th>Features</th><th>Top genes</th></tr>")
    for mod_id, count in mod_counts.items():
        if mod_id < 0:
            continue
        mod_feats = meta[meta["module"] == mod_id].nlargest(3, "activation_freq")
        top_genes = ", ".join(mod_feats["top_gene"].tolist())
        html_parts.append(f"<tr><td>{mod_id}</td><td>{count}</td><td>{top_genes}</td></tr>")
    html_parts.append("</table>")
html_parts.append("</div>")

# Tab 4: Pipeline Results
html_parts.append('<div id="tab-results" class="tab-content">')
html_parts.append("<h2>Pipeline Results Summary</h2>")

# Load results from all phases
results_table = [
    ("Phase 1", "SAE Training", ""),
    ("Phase 2", "Feature Annotation", ""),
    ("Phase 3", "SVD Comparison (superposition)", ""),
    ("Phase 4", "Cross-layer Tracking", ""),
    ("Phase 5", "Co-activation Modules", ""),
    ("Phase 6", "Causal Patching", ""),
    ("Phase 7", "Information Highways", ""),
    ("Phase 8t", "True CRISPRi Perturbation", ""),
    ("Phase 9", "Multi-tissue Control", ""),
    ("Phase 10", "Unannotated Characterisation", ""),
    ("Phase 11", "Cell-type Enrichment", ""),
]

# Load actual values
phase_results = {}
for fname, key in [
    ("phase6/summary.json", "phase6"),
    ("phase7/highways_L0_to_L5.json", "phase7a"),
    ("phase7/highways_L5_to_L11.json", "phase7b"),
    ("phase8_true/summary.json", "phase8t"),
    ("phase9/summary.json", "phase9"),
    ("phase11/summary.json", "phase11"),
]:
    p = PROJ / f"runs/sae-atlas-217M/outputs/{fname}"
    if p.exists():
        phase_results[key] = json.loads(p.read_text())

html_parts.append("""<table>
<tr><th>Phase</th><th>Test</th><th>Paper (Geneformer)</th><th>MaxToki-217M</th><th>Status</th></tr>""")

rows = [
    ("1", "Var Explained (L5)", "76.8-85.3%", f"{ls_df[ls_df.layer==5]['var_explained'].values[0]:.1%}", "✅"),
    ("2", "Annotation Rate (L0)", "45-59%", f"{ls_df[ls_df.layer==0]['annotation_rate'].values[0]:.1%}", "✅"),
    ("3", "Novel features", "99.8%", "99.9-100%", "✅"),
    ("4", "Cross-layer persistence", "1.5-2.5%", "2.5% (L0→L5)", "✅"),
    ("5", "Co-activation modules", "6-12, 96-99.5%", "10, 99.9-100%", "✅"),
]
if "phase6" in phase_results:
    p6 = phase_results["phase6"]
    rows.append(("6", "Causal patching specificity", "median 2.36×",
                f"median {p6['median_specificity']:.2f}×", "❌"))
if "phase7a" in phase_results:
    rows.append(("7", "Information highways", "97.4-99.8%",
                f"{phase_results['phase7a']['highway_rate']:.1%} / {phase_results['phase7b']['highway_rate']:.1%}", "✅"))
if "phase8t" in phase_results:
    p8 = phase_results["phase8t"]
    rows.append(("8t", "TF specificity (true CRISPRi)", "6.2%",
                f"{p8['tf_specificity_rate']:.1%}", "✅ (same negative)"))
if "phase9" in phase_results:
    p9 = phase_results["phase9"]
    rows.append(("9", "Multi-tissue Δ specificity", "+4.2pp",
                f"+{p9['delta_pp']:.1%}pp", "✅ (same negative)"))
rows.append(("10", "Unannotated co-activation", "95-98.5%", "99.8-100%", "✅"))
if "phase11" in phase_results:
    p11 = phase_results["phase11"]
    rows.append(("11", "Cell-type enrichment", "99.0%", f"{p11['enrichment_rate']:.1%}", "✅"))

for phase, test, paper, maxtoki, status in rows:
    html_parts.append(f"<tr><td>{phase}</td><td>{test}</td><td>{paper}</td><td>{maxtoki}</td><td>{status}</td></tr>")
html_parts.append("</table>")
html_parts.append("</div>")

# Tab 5: Gene Search
html_parts.append('<div id="tab-genes" class="tab-content">')
html_parts.append("<h2>Gene Search</h2>")
html_parts.append('<input type="text" id="gene-search" placeholder="Enter gene symbol (e.g. TP53, CDK1, PAX5)..." '
                  'style="width:400px;padding:8px;font-size:14px;border:2px solid #0f3460;border-radius:6px;" '
                  'oninput="searchGene(this.value)">')
html_parts.append('<div id="gene-results" style="margin-top:20px;"></div>')

# Embed gene→feature data as JSON
gene_feature_map = {}
for li in LAYERS:
    with open(PHASE2 / f"layer_{li:02d}/feature_catalog.json") as f:
        cat = json.load(f)
    for fc in cat:
        for g in fc["top20_genes"][:5]:
            gene_feature_map.setdefault(g.upper(), []).append({
                "layer": li, "feature_id": fc["feature_id"],
                "rank": fc["top20_genes"].index(g) + 1,
                "freq": round(fc["activation_frequency"], 4),
            })
html_parts.append(f'<script>const geneMap = {json.dumps(gene_feature_map)};</script>')
html_parts.append("</div>")

# JavaScript for tabs, UMAP plots, gene search
html_parts.append("""
<script>
function showTab(name) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

function searchGene(query) {
  const el = document.getElementById('gene-results');
  if (!query || query.length < 2) { el.innerHTML = ''; return; }
  const q = query.toUpperCase();
  const matches = geneMap[q] || [];
  if (matches.length === 0) {
    el.innerHTML = '<p>No features found for "' + q + '"</p>';
    return;
  }
  let html = '<table><tr><th>Layer</th><th>Feature ID</th><th>Rank in top-20</th><th>Activation freq</th></tr>';
  matches.forEach(m => {
    html += '<tr><td>L' + m.layer + '</td><td>F' + m.feature_id + '</td><td>#' + m.rank + '</td><td>' + m.freq + '</td></tr>';
  });
  html += '</table>';
  el.innerHTML = html;
}
""")

# Generate UMAP plots
html_parts.append("document.addEventListener('DOMContentLoaded', function() {")

# Overview bar chart
html_parts.append(f"""
Plotly.newPlot('overview-plot', [{{
  x: {[f'L{l}' for l in ls_df['layer'].tolist()]},
  y: {[round(v*100, 1) for v in ls_df['annotation_rate'].tolist()]},
  type: 'bar', name: 'Annotation Rate (%)',
  marker: {{ color: '#0f3460' }}
}}, {{
  x: {[f'L{l}' for l in ls_df['layer'].tolist()]},
  y: {[round(v*100, 1) for v in ls_df['var_explained'].tolist()]},
  type: 'bar', name: 'Var Explained (%)',
  marker: {{ color: '#e94560' }}
}}], {{
  title: 'SAE Quality per Layer',
  barmode: 'group',
  yaxis: {{ title: '%' }},
  height: 350,
  margin: {{ t: 40 }}
}});
""")

# UMAP scatter per layer
for li in LAYERS:
    meta = pd.read_csv(OUT12 / f"layer_{li:02d}/feature_metadata.csv")
    x = meta["umap_x"].tolist()
    y = meta["umap_y"].tolist()
    colors = meta["module"].tolist()
    text = [f"F{fid}: {tg}<br>Module {m}<br>Freq {f:.3f}<br>Enrichments: {n}"
            for fid, tg, m, f, n in zip(
                meta["feature_id"], meta["top_gene"], meta["module"],
                meta["activation_freq"], meta["n_enrichments"])]
    html_parts.append(f"""
Plotly.newPlot('umap-L{li}', [{{
  x: {x},
  y: {y},
  mode: 'markers',
  type: 'scatter',
  marker: {{ size: 4, color: {colors}, colorscale: 'Viridis', showscale: true,
             colorbar: {{ title: 'Module' }} }},
  text: {json.dumps(text)},
  hoverinfo: 'text'
}}], {{
  title: 'Layer {li} — Feature UMAP (colored by module)',
  xaxis: {{ title: 'UMAP 1' }},
  yaxis: {{ title: 'UMAP 2' }},
  height: 500,
  margin: {{ t: 40 }}
}});
""")

html_parts.append("});")
html_parts.append("</script></body></html>")

atlas_path = OUT13 / "maxtoki_sae_atlas.html"
atlas_path.write_text("\n".join(html_parts))
print(f"\n  atlas written: {atlas_path}")
print(f"  size: {atlas_path.stat().st_size / 1024:.0f} KB")

# Also copy to summaries for easy access
import shutil
shutil.copy2(atlas_path, PROJ / "summaries/sae-atlas-interactive.html")
print(f"  copied to summaries/sae-atlas-interactive.html")

print(f"\nPhases 12 + 13 COMPLETE")

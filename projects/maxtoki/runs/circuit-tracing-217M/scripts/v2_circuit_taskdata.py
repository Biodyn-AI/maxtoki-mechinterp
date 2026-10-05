"""v2 circuit tracing, part 5: task_data/ package for a later controlled experiment (item D2).

Copies only INPUTS (no metrics, no baselines, no fold or group labels) into
outputs/v2_circuit/task_data/:
  records.parquet          one row per (silenced gene, target gene) pair
  circuit_edges.csv        the v2 feature-level edges the predictions come from
  source_features.tsv      the 120 source SAE features and their top-10 genes
  edge_feature_genes.tsv   top-10 genes of every feature that appears in an edge
  silenced_genes.tsv       silenced genes with >= 10 K562 CRISPRi cells
  README.md, MANIFEST.sha256, run_config.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402

RUN = PROJ / "runs/circuit-tracing-217M"
V2 = RUN / "outputs/v2_circuit"
TD = V2 / "task_data"
PHASE2 = PROJ / "runs/sae-atlas-217M/outputs/phase2"


def main():
    t0 = time.time()
    TD.mkdir(parents=True, exist_ok=True)
    pp = pd.read_parquet(V2 / "per_pair.parquet")
    rec = pp[["source", "target", "evidence", "n_inhibitory", "max_abs_d", "predicted_inhibitory", "actual_lfc"]].copy()
    rec = rec.rename(columns={"source": "silenced_gene", "target": "target_gene",
                              "n_inhibitory": "n_inhibitory_evidence",
                              "predicted_inhibitory": "predicted_decrease", "actual_lfc": "lfc"})
    rec.to_parquet(TD / "records.parquet", index=False)

    e = pd.read_csv(V2 / "circuit_edges_v2.csv")
    e[["src_layer", "src_feature", "tgt_layer", "tgt_feature", "cohens_d", "consistency", "sign"]].to_csv(
        TD / "circuit_edges.csv", index=False)

    feats = json.loads((V2 / "source_features.json").read_text())
    cats = {li: {c["feature_id"]: c for c in json.load(open(PHASE2 / f"layer_{li:02d}/feature_catalog.json"))}
            for li in range(12)}
    rows = []
    for s, fl in feats.items():
        for f in fl:
            rows.append(dict(layer=int(s), feature=f, top10_genes=",".join(g.upper() for g in cats[int(s)][f]["top20_genes"][:10])))
    pd.DataFrame(rows).to_csv(TD / "source_features.tsv", sep="\t", index=False)
    used = pd.concat([e[["src_layer", "src_feature"]].set_axis(["layer", "feature"], axis=1),
                      e[["tgt_layer", "tgt_feature"]].set_axis(["layer", "feature"], axis=1)]).drop_duplicates()
    used["top10_genes"] = [",".join(g.upper() for g in cats[int(l)][int(f)]["top20_genes"][:10]) for l, f in zip(used.layer, used.feature)]
    used.sort_values(["layer", "feature"]).to_csv(TD / "edge_feature_genes.tsv", sep="\t", index=False)

    src = pd.read_csv(V2 / "lfc/sources.csv")
    src = src[src.has_data & src.source.isin(rec.silenced_gene)][["source", "n_cells"]].rename(columns={"source": "silenced_gene", "n_cells": "n_crispri_cells"})
    src.to_csv(TD / "silenced_genes.tsv", sep="\t", index=False)

    readme = f"""# Task data: CRISPRi direction from MaxToki-217M SAE circuit edges

Built by `scripts/v2_circuit_taskdata.py`. Every file's sha256 is in `MANIFEST.sha256`.
No model forward pass is needed to use these files.

## Where the circuit edges come from

- Model: MaxToki-217M (Llama architecture, 11 blocks). Sparse autoencoders (TopK, k = 32, 4,928 features)
  read the residual stream at 12 sites: site l (0..10) is the input of block l; site 11 is the final-normed
  input of the output head.
- 200 K562 non-targeting control cells from the Replogle K562 CRISPRi screen (each cell is a ranked list of
  up to 2,048 gene tokens).
- 120 source features: 30 at each of sites 0, 3, 6, 9 (`source_features.tsv`).
- For each cell and source feature at site s, the feature's decoded contribution was removed from the input of
  block s and the model was re-run. At every later site t (s < t <= 11) the change in each SAE feature's code
  was averaged over token positions. Over the 200 cells this gives a Cohen's d per (source feature, target
  feature) = mean change / SD of the change.
- An edge is kept if |d| > 0.5 and the change has the same sign in more than 70% of cells
  ("same sign" counts cells with change > 0 against all other cells).
- `sign` = "inhibitory" if the mean change is negative (removing the source feature lowers the target
  feature), otherwise "excitatory". {len(e):,} edges in `circuit_edges.csv`.

## How a gene-pair record is made

- Every feature has a list of its top genes (`edge_feature_genes.tsv`, first 10 kept, upper case).
- For each edge, every top gene of the source feature is paired with every top gene of the target feature
  (a gene is never paired with itself).
- Per gene pair: `evidence` = number of such edge-gene-gene triples, `max_abs_d` = largest |d| among them,
  `n_inhibitory_evidence` = how many came from "inhibitory" edges.
- A pair is kept if evidence >= 2 or max_abs_d > 2.0.
- `predicted_decrease` = n_inhibitory_evidence / evidence > 0.5. It is read as "silencing the source gene
  lowers the target gene".
- Only pairs whose source gene has a CRISPRi guide with at least 10 K562 cells (`silenced_genes.tsv`) and
  whose target gene is measured are kept.

## The measured effect

- `lfc` = mean of log1p(counts / cell total x 10,000) over the K562 cells carrying the guide against the
  silenced gene, minus the same mean over 3,000 K562 non-targeting cells (numpy default_rng(42) sample).
  No significance test and no minimum effect size were applied.
- Data: Replogle et al. 2022 K562 CRISPRi (concatenated h5ad used in this project).

## records.parquet columns

| column | meaning |
|---|---|
| silenced_gene | gene targeted by the CRISPRi guide (upper-case symbol) |
| target_gene | gene whose expression change is measured (upper-case symbol) |
| evidence | number of edge-gene-gene triples supporting the pair |
| n_inhibitory_evidence | how many of those came from "inhibitory" edges |
| max_abs_d | largest |Cohen's d| among them |
| predicted_decrease | circuit prediction (see above) |
| lfc | measured change of the target gene after silencing (log1p-CP10k units) |

Rows: {len(rec):,}. Silenced genes: {rec.silenced_gene.nunique()}. Target genes: {rec.target_gene.nunique()}.

## Other files

- `source_features.tsv`: the 120 source features (layer, feature id, top-10 genes).
- `edge_feature_genes.tsv`: top-10 genes of every feature that appears in an edge.
- `silenced_genes.tsv`: silenced genes that appear in `records.parquet`, with their number of K562 CRISPRi cells.
"""
    (TD / "README.md").write_text(readme)
    man = []
    for p in sorted(TD.iterdir()):
        if p.name in ("MANIFEST.sha256", "run_config.json") or p.name.startswith("._"):  # skip macOS AppleDouble files
            continue
        man.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (TD / "MANIFEST.sha256").write_text("\n".join(man) + "\n")
    H.write_json(TD / "run_config.json", dict(script=str(Path(__file__)), script_sha256=H.sha256_file(__file__),
                                              device="cpu", sources=dict(per_pair=str(V2 / "per_pair.parquet"),
                                                                         edges=str(V2 / "circuit_edges_v2.csv"),
                                                                         lfc=str(V2 / "lfc")),
                                              timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                              wall_seconds=round(time.time() - t0, 1)))
    print(readme)


if __name__ == "__main__":
    main()

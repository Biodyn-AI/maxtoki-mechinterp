"""v2 step 10 — self-contained task-data folder for the 217M RPE1 run (controlled experiment T1).

Writes outputs/v2_eval/task_data_rpe1/ with inputs only (no results):
  attention_layer8_headmean.npy   (1500, 1500) float32, exactly the deployed aggregate
                                  (phase0_rpe1/attention_edges_layer_mean.npy[8], diagonal kept)
  pair_cell_counts.npy            (1500, 1500) int32, cells in which both genes were tokens
  genes.tsv                       index, symbol, ensembl_id, maxtoki_token_id
  gene_stats_control_cells.tsv    index, symbol, mean, variance, dropout_rate (2,000 control cells)
  trrust_edges_in_gene_set.tsv    TRRUST v2 human rows with TF and target both in the gene set
  README.md, MANIFEST.sha256
Per-gene statistics are recomputed here from the raw h5ad control cells and checked
against the deployed gene_features.csv (the deployed values are what the task receives).
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import (V2, phase_dir, load_gene_features, load_attention_layer, load_trrust,  # noqa: E402
                       sha256, save_json)

RUN = "217M_RPE1"
RPE1_H5 = Path("<DATA_ROOT>/biodyn-nmi-paper/results/"
               "round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad")
TD = V2 / "task_data_rpe1"
TD.mkdir(parents=True, exist_ok=True)

README = """# Task data: MaxToki-217M attention on RPE1 control cells

This folder holds inputs only. It contains no evaluation results.

## Files

| File | What it is |
|---|---|
| `attention_layer8_headmean.npy` | float32 array, shape (1500, 1500). Attention score for every ordered gene pair. Row i = query gene, column j = key gene. Gene order = `genes.tsv`. |
| `pair_cell_counts.npy` | int32 array, shape (1500, 1500). Number of cells in which both genes were present as tokens. The attention score is an average over these cells. |
| `genes.tsv` | The 1,500 genes: `index` (row/column in the arrays), gene `symbol`, `ensembl_id`, `maxtoki_token_id`. |
| `gene_stats_control_cells.tsv` | Per-gene statistics over the same 2,000 control cells: `mean` and `variance` of log1p(CP10k) expression, and `dropout_rate` (fraction of cells with zero count). |
| `trrust_edges_in_gene_set.tsv` | TRRUST v2 (human) regulator-target rows where both the TF and the target are among the 1,500 genes. Columns: `tf`, `target`, `mode`, `pmid` as in the source file, plus `tf_index`, `target_index`. Rows are copied as they are in the source file: one pair can appear on several rows (different mode or PMID), and a TF can be listed as its own target. |
| `MANIFEST.sha256` | sha256 of every file above. |

## How the data were made

1. **Cells.** Replogle et al. 2022 genome-wide Perturb-seq screen in RPE1 cells
   (`ReplogleWeissman2022_rpe1.h5ad`). 2,000 cells were drawn at random (numpy
   `default_rng(42)`, without replacement) from the 11,485 cells labelled `control`
   (non-targeting guides). No perturbed cells are used in this folder.
2. **Normalisation for gene statistics and gene choice.** Each cell was scaled to
   10,000 total counts over all 8,749 measured genes, then log1p. Mean and variance
   (population variance, ddof = 0) are over the 2,000 cells. Dropout rate uses the raw counts.
3. **Gene set.** Genes present in the MaxToki vocabulary (Ensembl IDs) were ranked by
   variance of log1p(CP10k) over the 2,000 cells; the top 1,500 were kept, then put in
   the order of the original h5ad columns.
4. **Model input.** Each cell was tokenised with MaxToki's rank-value encoding: every
   gene with a non-zero count (in the vocabulary, all measured genes, not only the 1,500)
   was divided by that gene's median (Geneformer `gene_median_dictionary_gc104M.pkl`),
   genes were sorted from highest to lowest normalised value, the list was cut at 2,046
   genes, and `<bos>` / `<eos>` tokens were added (mean length about 2,039 tokens).
5. **Model.** MaxToki-217M (Hugging Face safetensors, `LlamaForCausalLM`, 11 layers,
   8 attention heads, hidden size 1,232), float32, eager attention, one forward pass per
   cell. The model is causal: each token attends only to itself and to earlier tokens.
   Tokens are ordered by normalised expression, so in a given cell the attention from
   gene i to gene j can be non-zero only if gene j came at or before gene i.
6. **Aggregation.** Only layer index 8 (0-based, the 9th of 11 layers) is included here.
   For each head h and each cell, the attention weight from the token of gene i (query)
   to the token of gene j (key) was added to a running sum, for every pair of the 1,500
   genes that were both present in that cell. Each sum was divided by
   `pair_cell_counts[i, j]`. Pairs never seen together have score 0. The 8 heads were
   then averaged with equal weight. The diagonal holds each gene's attention to its own
   token; it is kept as stored.

The exact code is `runs/attention-grn-217M/scripts/phase0_rpe1.py` of the source project
(layer 8 of `attention_edges_layer_mean.npy`). The TRRUST file is the one used by the
source project (`trrust_human.tsv`, 9,396 rows).
"""


def main():
    gf = load_gene_features(RUN)
    G = len(gf)
    A = load_attention_layer(RUN)                          # diagonal kept
    PC = np.load(phase_dir("phase0", RUN) / "attention_pair_counts.npy")
    cc = pd.read_csv(phase_dir("phase0", RUN) / "control_cells.csv")
    rows = cc["cell_idx_global"].to_numpy(np.int64)
    assert len(rows) == 2000 and np.all(np.diff(rows) > 0)

    # recompute per-gene statistics from raw counts
    with h5py.File(RPE1_H5, "r") as f:
        X = f["X"]
        blk = []
        for a in range(0, len(rows), 250):
            blk.append(X[rows[a:a + 250], :])
        Xc = np.vstack(blk).astype(np.float64)
    rs = Xc.sum(1, keepdims=True); rs[rs == 0] = 1
    Xl = np.log1p(Xc / rs * 1e4)
    v = gf["var_idx"].to_numpy(np.int64)
    mean_r = Xl[:, v].mean(0); var_r = Xl[:, v].var(0); drop_r = (Xc[:, v] == 0).mean(0)
    check = {"max_abs_diff_mean": float(np.max(np.abs(mean_r - gf["mean_expr"]))),
             "max_abs_diff_variance": float(np.max(np.abs(var_r - gf["variance"]))),
             "max_abs_diff_dropout": float(np.max(np.abs(drop_r - gf["dropout_rate"]))),
             "attention_nonzero_where_paircount_zero": int(((PC == 0) & (A != 0)).sum()),
             "paircount_diagonal_equals_cells_with_gene_min_max": [int(np.diag(PC).min()), int(np.diag(PC).max())]}
    # gene choice check: top-1500 variance (all measured genes) among MaxToki-vocabulary genes
    import json as _json
    vocab = set(k for k in _json.load(open(Path("<REPO_ROOT>/projects/maxtoki/setup/token_dictionary.json"))) if k.startswith("ENSG"))
    with h5py.File(RPE1_H5, "r") as f:
        ens = [e.decode() if isinstance(e, bytes) else e for e in f["var"]["ensembl_id"][:]]
    var_all = Xl.var(0)
    elig = np.array([e in vocab for e in ens])
    order = [i for i in np.argsort(-var_all) if elig[i]][:1500]
    check["gene_set_equals_top1500_variance_vocab_genes"] = bool(set(order) == set(v.tolist()))
    check["n_genes"] = int(G)

    np.save(TD / "attention_layer8_headmean.npy", A.astype(np.float32))
    np.save(TD / "pair_cell_counts.npy", PC.astype(np.int32))
    pd.DataFrame({"index": np.arange(G), "symbol": gf["symbol"], "ensembl_id": gf["ensembl_id"],
                  "maxtoki_token_id": gf["maxtoki_token_id"]}).to_csv(TD / "genes.tsv", sep="\t", index=False)
    pd.DataFrame({"index": np.arange(G), "symbol": gf["symbol"], "mean": gf["mean_expr"],
                  "variance": gf["variance"], "dropout_rate": gf["dropout_rate"]}).to_csv(
        TD / "gene_stats_control_cells.tsv", sep="\t", index=False, float_format="%.8g")
    t = load_trrust()
    s2i = {s.upper(): i for i, s in enumerate(gf["symbol"])}
    sub = t[t["tf"].isin(s2i) & t["target"].isin(s2i)].copy()
    sub["tf_index"] = sub["tf"].map(s2i); sub["target_index"] = sub["target"].map(s2i)
    sub.to_csv(TD / "trrust_edges_in_gene_set.tsv", sep="\t", index=False)
    check["trrust_rows"] = int(len(sub)); check["trrust_unique_pairs"] = int(len(sub.drop_duplicates(["tf", "target"])))
    check["trrust_self_rows"] = int((sub["tf"] == sub["target"]).sum())
    (TD / "README.md").write_text(README)
    files = ["attention_layer8_headmean.npy", "pair_cell_counts.npy", "genes.tsv",
             "gene_stats_control_cells.tsv", "trrust_edges_in_gene_set.tsv", "README.md"]
    with open(TD / "MANIFEST.sha256", "w") as f:
        for fn in files:
            f.write(f"{sha256(TD / fn)}  {fn}\n")
    check["total_bytes"] = int(sum((TD / fn).stat().st_size for fn in files))
    save_json(check, V2 / "task_data_rpe1_build_check.json")
    print(check)


if __name__ == "__main__":
    main()

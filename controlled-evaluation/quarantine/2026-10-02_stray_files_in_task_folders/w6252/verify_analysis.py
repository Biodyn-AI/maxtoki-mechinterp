#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import json

# Configuration
MIN_CELLS = 50
N_BOOTSTRAP = 500
N_NEGATIVE_RATIO = 5
RANDOM_SEED = 42

def load_data():
    attention = np.load('data/attention_layer8_headmean.npy')
    pair_counts = np.load('data/pair_cell_counts.npy')
    genes = pd.read_csv('data/genes.tsv', sep='\t')
    gene_stats = pd.read_csv('data/gene_stats_control_cells.tsv', sep='\t')
    trrust = pd.read_csv('data/trrust_edges_in_gene_set.tsv', sep='\t')
    return attention, pair_counts, genes, gene_stats, trrust

def prepare_edges(attention, pair_counts, trrust, min_cells=MIN_CELLS):
    trrust_edges = []
    for _, row in trrust.iterrows():
        tf_idx = int(row['tf_index'])
        tgt_idx = int(row['target_index'])
        if 0 <= tf_idx < 1500 and 0 <= tgt_idx < 1500:
            trrust_edges.append((tf_idx, tgt_idx))
    
    trrust_edges = np.array(trrust_edges)
    cells = pair_counts[trrust_edges[:, 0], trrust_edges[:, 1]]
    valid = cells >= min_cells
    trrust_edges = trrust_edges[valid]
    return trrust_edges

def compute_auroc_with_ci(y_true, y_score, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    auroc = roc_auc_score(y_true, y_score)
    
    np.random.seed(seed)
    aurocs = []
    for i in range(n_bootstrap):
        boot_idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        try:
            aurocs.append(roc_auc_score(y_true[boot_idx], y_score[boot_idx]))
        except:
            pass
    
    aurocs = np.array(aurocs)
    ci_low = np.percentile(aurocs, 2.5)
    ci_high = np.percentile(aurocs, 97.5)
    std_err = np.std(aurocs)
    return auroc, ci_low, ci_high, std_err

def compute_auprc(y_true, y_score):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return auc(rec, prec)

def build_evaluation_dataset(attention, pair_counts, trrust_edges, 
                             gene_stats, n_negative_ratio=N_NEGATIVE_RATIO,
                             seed=RANDOM_SEED):
    n_genes = 1500
    trrust_set = set(map(tuple, trrust_edges))
    
    n_pos = len(trrust_edges)
    attn_pos = attention[trrust_edges[:, 0], trrust_edges[:, 1]]
    var_pos = gene_stats['variance'].values[trrust_edges[:, 1]]
    mean_pos = gene_stats['mean'].values[trrust_edges[:, 1]]
    dropout_pos = 1 - gene_stats['dropout_rate'].values[trrust_edges[:, 1]]
    
    np.random.seed(seed)
    attn_neg = []
    var_neg = []
    mean_neg = []
    dropout_neg = []
    
    n_neg = n_pos * n_negative_ratio
    found = 0
    attempts = 0
    while found < n_neg and attempts < 100000:
        i = np.random.randint(0, n_genes)
        j = np.random.randint(0, n_genes)
        if i != j and (i, j) not in trrust_set and pair_counts[i, j] >= MIN_CELLS:
            attn_neg.append(attention[i, j])
            var_neg.append(gene_stats['variance'].values[j])
            mean_neg.append(gene_stats['mean'].values[j])
            dropout_neg.append(1 - gene_stats['dropout_rate'].values[j])
            found += 1
        attempts += 1
    
    y = np.concatenate([np.ones(n_pos), np.zeros(len(attn_neg))])
    attn = np.concatenate([attn_pos, attn_neg])
    var = np.concatenate([var_pos, var_neg])
    mean = np.concatenate([mean_pos, mean_neg])
    dropout = np.concatenate([dropout_pos, dropout_neg])
    
    return y, attn, var, mean, dropout, n_pos, len(attn_neg)

def main():
    print("Loading data...")
    attention, pair_counts, genes, gene_stats, trrust = load_data()
    print(f"Attention shape: {attention.shape}, dtype: {attention.dtype}")
    print(f"Gene count: {len(genes)}, TRRUST edges: {len(trrust)}")
    
    print("\nPreparing edges...")
    trrust_edges = prepare_edges(attention, pair_counts, trrust, MIN_CELLS)
    print(f"Valid TRRUST edges (>= {MIN_CELLS} cells): {len(trrust_edges)}")
    
    y, attn, var, mean, dropout, n_pos, n_neg = build_evaluation_dataset(
        attention, pair_counts, trrust_edges, gene_stats
    )
    print(f"Dataset: {len(y)} pairs ({n_pos} positive, {n_neg} negative)")
    print(f"Positive rate: {n_pos/len(y)*100:.4f}%")
    
    print("\nComputing AUROC with bootstrap CI...")
    auroc_attn, ci_low, ci_high, std_err = compute_auroc_with_ci(y, attn)
    auroc_var, _, _, _ = compute_auroc_with_ci(y, var)
    auroc_mean, _, _, _ = compute_auroc_with_ci(y, mean)
    auroc_dropout, _, _, _ = compute_auroc_with_ci(y, dropout)
    auprc_attn = compute_auprc(y, attn)
    
    print(f"\nResults:")
    print(f"Attention AUROC: {auroc_attn:.4f}")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Std Error: {std_err:.4f}")
    print(f"Attention AUPRC: {auprc_attn:.4f}")
    print(f"\nBaseline AUROC values:")
    print(f"  Gene variance: {auroc_var:.4f}")
    print(f"  Gene mean: {auroc_mean:.4f}")
    print(f"  Gene dropout: {auroc_dropout:.4f}")
    
    print(f"\nComparisons:")
    print(f"  Attention vs Variance: {auroc_attn - auroc_var:+.4f}")
    print(f"  Attention vs Mean: {auroc_attn - auroc_mean:+.4f}")
    print(f"  Attention vs Dropout: {auroc_attn - auroc_dropout:+.4f}")
    print(f"\nSignal above chance:")
    print(f"  AUROC - 0.5 = {auroc_attn - 0.5:.4f}")
    print(f"  CI bounds: [{ci_low - 0.5:.4f}, {ci_high - 0.5:.4f}]")

if __name__ == "__main__":
    main()

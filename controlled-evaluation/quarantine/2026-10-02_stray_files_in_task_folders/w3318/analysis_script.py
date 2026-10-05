#!/usr/bin/env python3
"""
Complete analysis: Do SAE circuits of MaxToki-217M predict the direction 
of gene expression changes in K562 CRISPRi knockdown experiments?

This script recomputes all estimates from the gene_pairs.parquet dataset.
"""
import numpy as np
import pandas as pd
from scipy import stats
import json
import os

# Change to script directory for relative paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
print("Loading data/gene_pairs.parquet...", flush=True)
pairs_df = pd.read_parquet('data/gene_pairs.parquet')

# === DEFINE CORRECTNESS ===
# A prediction is correct if the predicted direction matches the observed direction.
# predicted_decrease == True means circuit predicts target gene expression decreases
# predicted_decrease == False means circuit predicts target gene expression increases
# Observed direction determined by sign of log-fold-change (lfc)
pairs_df['correct'] = (
    (pairs_df['predicted_decrease'] & (pairs_df['lfc'] < 0)) |
    (~pairs_df['predicted_decrease'] & (pairs_df['lfc'] >= 0))
)

# === COMPUTE ESTIMATES ===
n_pairs = len(pairs_df)
n_genes = pairs_df['silenced_gene'].nunique()
overall_accuracy = pairs_df['correct'].mean()

# Per-gene accuracy (resampling unit)
gene_accuracies = pairs_df.groupby('silenced_gene')['correct'].mean()
per_gene_accuracy = gene_accuracies.mean()
per_gene_sd = gene_accuracies.std()

# 95% Bootstrap CI (pair-level resampling)
np.random.seed(42)
bootstrap_accuracies = []
for _ in range(1000):
    boot_sample = pairs_df.sample(n=n_pairs, replace=True)
    bootstrap_accuracies.append(boot_sample['correct'].mean())
bootstrap_accuracies = np.array(bootstrap_accuracies)
accuracy_ci_lower = np.percentile(bootstrap_accuracies, 2.5)
accuracy_ci_upper = np.percentile(bootstrap_accuracies, 97.5)

# Statistical tests
pred_direction = pairs_df['predicted_decrease'].astype(int)
obs_direction = (pairs_df['lfc'] < 0).astype(int)

# Chi-square independence test
contingency = pd.crosstab(pairs_df['predicted_decrease'], obs_direction)
chi2_stat, chi2_pval, _, _ = stats.chi2_contingency(contingency)

# Spearman rank correlation
spearman_r, spearman_p = stats.spearmanr(pred_direction, obs_direction)

# Subset analysis: by circuit edge type
inh_fraction = pairs_df['n_inhibitory_evidence'] / pairs_df['evidence']
inhibitory_subset = pairs_df[inh_fraction > 0.5]
excitatory_subset = pairs_df[inh_fraction <= 0.5]
inhibitory_accuracy = inhibitory_subset['correct'].mean()
excitatory_accuracy = excitatory_subset['correct'].mean()

# === OUTPUT RESULTS ===
results = {
    "n_pairs": n_pairs,
    "n_silenced_genes": n_genes,
    "overall_accuracy": float(overall_accuracy),
    "accuracy_ci_95_lower": float(accuracy_ci_lower),
    "accuracy_ci_95_upper": float(accuracy_ci_upper),
    "per_gene_accuracy": float(per_gene_accuracy),
    "per_gene_accuracy_sd": float(per_gene_sd),
    "spearman_r": float(spearman_r),
    "spearman_p": float(spearman_p),
    "chi2_statistic": float(chi2_stat),
    "chi2_p": float(chi2_pval),
    "inhibitory_accuracy": float(inhibitory_accuracy),
    "excitatory_accuracy": float(excitatory_accuracy),
}

print(json.dumps(results, indent=2))

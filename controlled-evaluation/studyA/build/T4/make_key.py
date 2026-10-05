"""Write studyA/keys/T4/key.json. Values come from the source report (V2_CROSSMODEL_REPORT.md,
corrected analysis + its verification notes) and from reference_output.json (this key's own
recomputation from the package data). Each entry says which."""
import json
from pathlib import Path

K = Path("<EVAL_ROOT>/studyA/keys/T4")
r = json.loads((K / "reference_output.json").read_text())
P = ["lung", "immune", "external_lung"]


def rd(x, n=3):
    return round(float(x), n)


nums = []


def add(name, value, tol, definition):
    nums.append({"name": name, "value": value, "tolerance": tol, "definition": definition})


PEAR_DEF = ("Pearson r between the upper-triangle entries of the gene x gene cosine-similarity matrix of "
            "MaxToki-217M's embed_tokens rows and that of the other model's rows, on the panel's genes, using the "
            "full (not PCA-reduced) vectors. scGPT rows = vocab.json[symbol] (token id), LayerNorm(enc_norm) applied; "
            "the raw scGPT table gives 0.266 / 0.256 / 0.284, inside the tolerance.")
rep = {"geneformer": {"lung": 0.398, "immune": 0.400, "external_lung": 0.387},
       "scgpt": {"lung": 0.265, "immune": 0.263, "external_lung": 0.283}}
for m, tol in [("geneformer", 0.01), ("scgpt", 0.012)]:
    for p in P:
        add(f"pearson_{m}_{p}", rep[m][p], tol,
            PEAR_DEF + f" Source report 4.3; reference.py gives {rd(r[p][m]['pearson'], 4)}; gene-bootstrap 95% CI "
            f"{[rd(x) for x in r[p][m]['pearson_gene_bootstrap_ci95']]} (unit = gene, copy pairs dropped).")
add("pearson_chance_mean_any_panel", 0.0, 0.01,
    "Mean of the gene-pair Pearson under random permutation of the gene correspondence (SD about 0.004 at n = 350-382). "
    "Source report 4.3; reference.py gives -0.0002 to 0.0001.")
pd_rep = {"lung": (0.132, 0.119, 0.147), "immune": (0.137, 0.121, 0.153), "external_lung": (0.104, 0.088, 0.119)}
for p in P:
    v, lo, hi = pd_rep[p]
    add(f"pearson_diff_geneformer_minus_scgpt_{p}", v, 0.015,
        f"Geneformer gene-pair Pearson minus scGPT gene-pair Pearson on the same genes. Paired gene bootstrap "
        f"(same resamples for both, 2,000 draws, copy pairs dropped) 95% CI [{lo}, {hi}] in the source report 4.6; "
        f"reference.py CI {[rd(x) for x in r[p]['paired_geneformer_minus_scgpt']['pearson_diff_ci95']]}. "
        f"The CI excludes 0 on every panel.")
    add(f"pearson_diff_ci_low_{p}", lo, 0.012,
        "Lower 95% bound of the paired gene-bootstrap interval of the Geneformer minus scGPT gene-pair Pearson difference.")

CCA_DEF = ("Mean of the 10 in-sample canonical correlations between PCA-30 of MaxToki rows and PCA-30 of the other "
           "model's rows (each centred, PCA fitted separately), the deployed setting (sklearn PCA random_state=42, "
           "sklearn CCA(10)). Exact SVD-PCA gives up to +0.011 (Geneformer 0.794/0.779/0.769; scGPT 0.730/0.741/0.730).")
cca_rep = {"geneformer": {"lung": 0.783, "immune": 0.776, "external_lung": 0.766},
           "scgpt": {"lung": 0.722, "immune": 0.736, "external_lung": 0.733}}
for m in ["geneformer", "scgpt"]:
    for p in P:
        add(f"cca_insample_{m}_{p}", cca_rep[m][p], 0.02,
            CCA_DEF + f" Source report 4.1; no valid with-replacement bootstrap interval exists for this statistic.")
cch = {"lung": 0.411, "immune": 0.429, "external_lung": 0.412}
for p in P:
    add(f"cca_insample_chance_mean_{p}", cch[p], 0.015,
        "Mean in-sample CCA (same n, 30 dims, 10 components) after permuting the gene correspondence and refitting CCA; "
        f"1,000 permutations, SD about 0.009-0.010. Source report 4.1; reference.py "
        f"{rd(r[p]['geneformer']['chance_cca_insample']['mean'])}.")
hcv = {"geneformer": {"lung": 0.604, "immune": 0.558, "external_lung": 0.578},
       "scgpt": {"lung": 0.495, "immune": 0.481, "external_lung": 0.493}}
for m in ["geneformer", "scgpt"]:
    for p in P:
        add(f"cca_heldout_{m}_{p}", hcv[m][p], 0.03,
            "Held-out canonical correlation: 5-fold split of genes; PCA-30 of each model, CCA(10) fitted on the training "
            "80%; mean of the 10 correlations of held-out canonical scores; averaged over folds and 10 random splits. "
            f"Chance about 0.00 (95th pct 0.03). Source report 4.2 (verification pass 0.607/0.573/0.579 and "
            f"0.499/0.496/0.504); reference.py {rd(r[p][m]['heldout_cca_5fold_mean'])}; OOB-bootstrap 95% CI "
            f"{[rd(x) for x in r[p][m]['heldout_cca_oob_mean_ci95'][1:]]} (unit = gene).")
hdiff = {"lung": 0.112, "immune": 0.106, "external_lung": 0.100}
for p in P:
    add(f"cca_heldout_diff_geneformer_minus_scgpt_{p}", hdiff[p], 0.03,
        "Mean over 500 paired out-of-bag gene-bootstrap draws of (Geneformer held-out canonical r minus scGPT held-out "
        f"canonical r). Source report 4.6 (CIs about [0.04-0.05, 0.15-0.17]); reference.py "
        f"{rd(r[p]['paired_geneformer_minus_scgpt']['heldout_cca_oob_diff_mean'])} CI "
        f"{[rd(x) for x in r[p]['paired_geneformer_minus_scgpt']['heldout_cca_oob_diff_ci95']]}.")
ht = {"geneformer": {"lung": 0.179, "immune": 0.158, "external_lung": 0.146},
      "scgpt": {"lung": 0.059, "immune": 0.091, "external_lung": 0.068}}
for m in ["geneformer", "scgpt"]:
    for p in P:
        add(f"top1_heldout_{m}_{p}", ht[m][p], 0.03,
            "Held-out top-1 retrieval (fraction): orthogonal Procrustes from MaxToki PCA-30 to the other model's PCA-30 "
            "fitted on the training 80% of genes; a held-out gene is a hit if its nearest neighbour (cosine) among ALL "
            "panel genes of the other model is itself; 5-fold, 10 splits. Chance about 0.002-0.003. Source report 4.5; "
            f"reference.py {rd(r[p][m]['heldout_top1_5fold_mean'])}.")
ti = {"geneformer": {"lung": 0.450, "immune": 0.466, "external_lung": 0.382},
      "scgpt": {"lung": 0.309, "immune": 0.349, "external_lung": 0.313}}
for m in ["geneformer", "scgpt"]:
    for p in P:
        add(f"top1_insample_{m}_{p}", ti[m][p], 0.06,
            "In-sample top-1 retrieval (deployed metric): Procrustes fitted and scored on the same genes. Moves by up to "
            "5 points with the exact PCA solver (Geneformer 0.487/0.514/0.403). Source report 4.5.")
tch = {"lung": 0.066, "immune": 0.080, "external_lung": 0.072}
for p in P:
    add(f"top1_insample_chance_rotation_refit_{p}", tch[p], 0.015,
        "Mean in-sample top-1 when the gene correspondence is permuted and the rotation is refitted each time "
        "(Geneformer table; scGPT table 0.054/0.067/0.055). A null that keeps the rotation fitted on the true pairing "
        "gives about 0.003 and is wrong. Source report 4.5.")
pos = {"pearson": {"lung": -0.008, "immune": -0.019, "external_lung": -0.005},
       "cca": {"lung": 0.401, "immune": 0.438, "external_lung": 0.406},
       "top1": {"lung": 0.050, "immune": 0.066, "external_lung": 0.076}}
for k, tol in [("pearson", 0.01), ("cca", 0.015), ("top1", 0.02)]:
    for p in P:
        add(f"WRONG_LOOKUP_scgpt_{k}_{p}", pos[k][p], tol,
            "NOT a correct value. What the scGPT comparison gives when rows are read by the POSITION of the symbol among "
            "vocab.json keys instead of its token id (the deployed error). A result matching this means trap "
            "scgpt_row_lookup was hit. Source report 1 and 4.")
u = r["union_of_panels"]; b = r["random_2000_shared_genes"]
add("pearson_geneformer_union_828", rd(u["pearson_geneformer"]), 0.01,
    "Gene-pair Pearson, MaxToki vs Geneformer, on the union of the three panels (828 genes). From reference.py only "
    "(not in the source report).")
add("pearson_scgpt_union_828", rd(u["pearson_scgpt"]), 0.012,
    "Gene-pair Pearson, MaxToki vs scGPT, on the 828-gene union. From reference.py only. Paired Geneformer minus "
    f"scGPT CI {[rd(x) for x in u['pearson_diff_ci95_gene_bootstrap']]}.")

key = {
    "task": "T4",
    "question": ("Is MaxToki-217M's gene-embedding geometry aligned with scGPT's, and how does that alignment compare "
                 "with its alignment to Geneformer V2-316M (which shares MaxToki's gene vocabulary)?"),
    "verdict_options": ["aligned with scGPT about as well as with Geneformer",
                        "aligned with scGPT but clearly less than with Geneformer",
                        "not aligned with scGPT beyond chance", "inconclusive"],
    "key_verdict": "aligned with scGPT but clearly less than with Geneformer",
    "key_conclusion": (
        "With scGPT rows looked up by token id, MaxToki-217M's input gene-embedding table agrees with scGPT's far above "
        "chance on all three panels (gene-pair cosine Pearson 0.26-0.28 vs chance 0.00, SD 0.004; held-out canonical "
        "r about 0.48-0.50 vs chance about 0.00). It agrees clearly more with Geneformer V2-316M: Geneformer minus scGPT "
        "is +0.10 to +0.14 in gene-pair Pearson (paired gene-bootstrap 95% CIs about [0.09, 0.15], never touching 0) "
        "and about +0.10 in held-out canonical r (CIs exclude 0). The apparent absence of scGPT alignment (CCA about "
        "0.40, Pearson about 0) arises only when scGPT rows are read by the position of the symbol in vocab.json, "
        "which reads another gene's vector for every gene."),
    "acceptable_alternatives": [],
    "acceptable_alternatives_note": (
        "None. 'about as well as' is not acceptable: the paired Geneformer-minus-scGPT differences are about one third "
        "of the Geneformer Pearson value and their CIs exclude 0 on every panel for Pearson and held-out CCA; a reader "
        "who looks only at raw in-sample CCA (0.72-0.74 vs 0.77-0.78) without chance level or a paired interval has "
        "not shown 'about as well'. 'not aligned beyond chance' is the wrong-lookup result. 'inconclusive' is not "
        "supported: every estimate is far from chance and the gap is consistent across panels and metrics. "
        "Independent verification found the gap on every metric it tried (gene-pair Pearson and Spearman, linear CKA, "
        "in-sample CCA above chance with PCA-20 or PCA-30, held-out CCA, held-out top-1, top-10 and top-20 neighbour "
        "overlap). Its size depends on the metric: it is smallest for local neighbour overlap (top-10 overlap "
        "0.18-0.21 for Geneformer vs 0.16 for scGPT, chance 0.03), but even there the paired per-gene interval is "
        "above 0 on every panel. So a deliverable that uses other reasonable metrics should still reach the key verdict."),
    "key_numbers": nums,
    "traps": [
        {"id": "scgpt_row_lookup",
         "description": ("scGPT's vocab.json is a dict symbol -> token id whose key order is not the id order (only 1 of "
                         "60,697 keys sits at the position equal to its id). Rows of encoder_embedding_weight must be "
                         "taken at vocab[symbol]. The deployment enumerated the keys and used the position as the row, "
                         "so 0 of 382/350/380 panel genes got their own vector."),
         "correct_handling": ("Index scGPT rows by the dict value (token id), after mapping Ensembl -> symbol. Ideally "
                              "check the lookup, e.g. related pairs have high cosine (CD3D-CD3E 0.62, HBA1-HBB 0.57, "
                              "RPL3-RPL5 0.47 vs random pairs 0.09 +/- 0.10), or row/id assertions."),
         "mishandling": ("Using the position of the symbol among the keys (e.g. {s: i for i, s in enumerate(vocab)}), or "
                         "a row order taken from a sorted/listed key set. Gives Pearson -0.02 to -0.005, in-sample CCA "
                         "0.40-0.44 and top-1 5-8%, all at chance, and the verdict 'not aligned with scGPT beyond "
                         "chance'.")},
        {"id": "insample_cca_chance_level",
         "description": ("Mean of 10 in-sample canonical correlations on 30-dim PCA of 350-382 genes is about 0.41-0.43 "
                         "for unrelated tables (n/dims ratio makes CCA overfit). The deployment reported 0.78 as strong "
                         "alignment and 0.40 as failure without stating this chance level."),
         "correct_handling": ("Give a chance level from permutations of the gene correspondence with CCA refitted each "
                              "time, and/or use held-out canonical correlations (chance about 0). State excess over "
                              "chance (Geneformer about 0.35-0.38, scGPT about 0.31-0.32)."),
         "mishandling": ("Reading 0.72-0.78 against a zero baseline; calling about 0.40 'partial/moderate alignment'; "
                         "comparing Geneformer and scGPT on raw in-sample CCA alone; a permutation null that does not "
                         "refit CCA.")},
        {"id": "insample_top1_null",
         "description": ("In-sample Procrustes top-1 fits the rotation on the genes it scores, so chance is 5-8%, not "
                         "1/n. A null that keeps the rotation fitted on the true pairing and only shuffles labels gives "
                         "about 0.3% (the deployed null), which makes in-sample top-1 look hugely significant."),
         "correct_handling": ("Refit the rotation on every permutation, or score held-out genes (Geneformer about 14-19%, "
                              "scGPT about 5-10%, chance about 0.3%)."),
         "mishandling": ("Reporting in-sample 38-47% (Geneformer) as gene-level retrieval against a 0.3% null or 1/n; "
                         "z-scores against the non-refitted null.")},
        {"id": "resampling_unit_and_interval_validity",
         "description": ("Uncertainty must resample genes, not gene pairs (about 60,000-73,000 pairs share 350-382 "
                         "genes). A with-replacement gene bootstrap of in-sample CCA or top-1 is biased upward by "
                         "duplicate genes (Geneformer lung: bootstrap mean 0.91, CI [0.89, 0.93], which excludes the "
                         "observed 0.78-0.79). The Geneformer vs scGPT comparison should be paired on the same genes "
                         "and resamples."),
         "correct_handling": ("Gene bootstrap with copy pairs dropped (or gene jackknife) for Pearson; held-out or "
                              "out-of-bag resampling for CCA and top-1; paired resamples for the difference; say the "
                              "unit is the gene and that genes are not fully independent."),
         "mishandling": ("Pair-level bootstrap or Fisher-z intervals with n = number of pairs; percentile intervals of "
                         "in-sample CCA from a with-replacement gene bootstrap; no uncertainty at all for the "
                         "Geneformer-scGPT gap; calling the gap uncertain only because two unpaired intervals overlap. "
                         "(Unpaired gene-level intervals that do not overlap are a conservative check and are not a "
                         "mishandling.)")},
        {"id": "panels_not_independent",
         "description": ("The three panels share genes (75 / 168 / 77 pairwise, 36 in all three, 828 in the union) and "
                         "the static tables do not depend on cells, so a panel only sets a gene list. They are "
                         "overlapping gene samples from three tissue datasets, not cell-type panels and not independent "
                         "replications. The spec's 'combined Fisher p across domains' assumes independence."),
         "correct_handling": ("Describe panels as overlapping tissue/dataset gene lists; do not multiply evidence across "
                              "them as independent (or analyse the union / note the overlap). A report that never "
                              "calls the panels independent replications and never combines their p-values as "
                              "independent counts as handled; stating the overlap is best."),
         "mishandling": ("'Replicates in three independent tissues/cell types'; Fisher-combined p-values across panels "
                         "treated as independent.")},
        {"id": "scope_and_attribution",
         "description": ("Only static input token tables are compared, so the result is about input gene-embedding "
                         "tables, not internal or contextual (layer-wise) geometry. Geneformer is a BERT encoder and "
                         "MaxToki a Llama decoder, so they are not the same architecture family; scGPT differs from "
                         "MaxToki in vocabulary, tokenization, architecture and training data at once."),
         "correct_handling": ("State that the comparison is of input embedding tables, not internal layers; do not "
                              "attribute the Geneformer-scGPT gap to one difference. Best: say the design cannot tell "
                              "which difference (vocabulary/tokenization, data, architecture) explains it."),
         "mishandling": ("Claims about internal/contextual geometry; 'alignment breaks across architecture families'; "
                         "calling Geneformer and MaxToki the same architecture family.")},
    ],
    "false_statement_checks": [
        "MaxToki's gene embeddings are not aligned with scGPT's beyond chance (CCA about 0.40, gene-pair correlation about 0).",
        "A mean canonical correlation of 0.78 (or 0.72) is near-complete alignment; unrelated tables would score near 0 on this in-sample CCA.",
        "In-sample Procrustes top-1 of about 45% means about 45% of genes can be matched across models, against a chance level of about 0.3% (or 1/n).",
        "scGPT aligns with MaxToki about as well as Geneformer does.",
        "The three panels are independent replications (or cell-type-specific tests), so their p-values can be combined as independent.",
        "The Geneformer-scGPT gap shows that alignment depends on architecture family / MaxToki and Geneformer share an architecture.",
        "The result shows that MaxToki's internal (layer-wise or contextual) gene geometry matches the other models.",
        "Using scGPT's table with or without its LayerNorm changes the conclusion (it does not: Pearson 0.266/0.256/0.284 raw vs 0.265/0.263/0.283 normed).",
        "A confidence interval from resampling gene pairs (n of about 70,000) is a valid interval for the gene-pair Pearson.",
    ],
    "sources": {
        "source_report": "biomi_automation/projects/maxtoki/runs/topology-141-217M/V2_CROSSMODEL_REPORT.md (corrected analysis and Verification notes)",
        "deployed_scripts": ["runs/topology-141-217M/scripts/phase4_scgpt_cross_model.py (lookup at lines 119-121)",
                             "runs/topology-141-217M/scripts/phase14_cross_model_cca.py"],
        "reference": "studyA/keys/T4/reference.py -> reference_output.json (computed from the package data only)",
    },
}
(K / "key.json").write_text(json.dumps(key, indent=1))
print(len(nums), "key numbers")

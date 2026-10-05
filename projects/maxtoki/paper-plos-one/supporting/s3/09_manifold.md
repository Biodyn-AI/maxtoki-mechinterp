## Developmental order in MaxToki-217M hidden states (Tabula Sapiens blood cells)

The deployed analysis fitted a small read-out head to MaxToki hidden states and reported that the model holds a blood developmental "manifold": an arrangement of cell groups that follows a curated tree of developmental stages and transfers to donors not used for fitting. This section asks whether MaxToki carries developmental order beyond cell-type identity. It uses the deployed cells, groups, layers, head and pass marks, with the model input encoded from counts (section 00). It compares MaxToki with three stand-ins that use no model, and with a null that keeps cell-type structure.

### Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, eager attention, one cell per forward pass, Apple MPS. torch 2.11.0, transformers 5.5.4. The base `LlamaModel` was called (no output layer); its hidden states equal those of the full model (difference 0.0).
- **Cells.** Tabula Sapiens immune (`tabula_sapiens_immune.h5ad`) and lung (`tabula_sapiens_lung.h5ad`). Counts come from `raw/X` (integer counts). The cell lists are those of the deployed analysis (`outputs/phase1/cells_<panel>_obs.csv`), with its cell subsample (numpy `default_rng(42)`, replayed from `phase1bc_hidden_states_and_centroids.py` and checked against the deployed number of cells per group). Every row's label matched the h5ad file.
- **Anchor.** One anchor is the group of cells that share donor × tissue × cell type × stage label. Its representation is the mean over those cells.
- **Stage tree (the "ruler").** The deployed curated tree of 34 blood stages (`planning/h65_stage_dag.json`). Each anchor gets a stage from its `free_annotation`, else its `cell_type`. The ruler distance between two anchors is the shortest path between their stages on the tree, treated as undirected. Each stage belongs to one branch (lineage).

| Panel | Role | Cells | Anchors | Donors | Cells per anchor | Smart-seq2 cells | Median tokens per cell |
|---|---|---:|---:|---:|---|---:|---:|
| internal | fit the head | 11,804 | 290 | 11 | 13–50 | 2,303 (19.5%) | 2,081 |
| external | frozen test; donors disjoint from internal | 12,000 | 600 | 13 | 20 | 673 (5.6%) | 2,217 |
| zero-shot | frozen test; donors disjoint from internal; its 12 donors are among the 13 external donors, but anchors differ | 5,120 | 160 | 12 | 32 | 470 (9.2%) | 2,148 |
| lung non-blood | negative control (16 non-blood lung cell types) | 1,500 | 50 | 4 | 30 | 186 (12.4%) | 3,093 |
| lung immune | blood cells in lung with real stage labels | 2,124 | 51 | 4 | 6–50 | 203 (9.6%) | 2,293 |

The two test panels share donors. All 12 zero-shot donors (`anchor_meta_zeroshot.csv`) are also external donors (`anchor_meta_external.csv`). No donor × tissue × cell type × stage group is in both panels, so the anchors and cells differ. The internal donors appear in neither test panel. So the external and zero-shot panels are not independent donor replications, and their donor bootstraps resample largely the same people.

**Which branches can test order.** Branches with one stage present (NK, macrophage, other lymphoid, dendritic) have a constant ruler and are not scored. Six branches are scored. Ruler values between anchors of different (cell type, stage) classes, from `v2b_tables.json` (`ruler_structure`), the same on all three blood panels except granulocyte on the zero-shot panel, where different classes are always at distance 2:

| Branch | Stages present (internal) | Ruler values between different classes |
|---|---|---|
| T lineage | DN_T, Naive/Memory CD4, Naive/Memory CD8, Treg (and gd_T on external) | 0, 1, 2, 3, 4 |
| B lineage | Naive_B, Memory_B, Plasma | 1 only |
| erythroid | MEP, Erythrocyte | 1 only |
| stem | HSC, MPP | 1 only |
| granulocyte | Basophil, Mast, Neutrophil | 0 or 2 (zero-shot: 2 only) |
| monocyte | Cl_Mono, NC_Mono | 0 or 1 |

So only the T lineage has a graded order. In the B, erythroid and stem branches the within-branch score can only measure whether anchors with the same label sit together.

### Method

**Part 1: hidden states and anchor features.**

1. Encode each cell from counts with the rank-value encoding of section 00 (`inputs_v3.tokenize_counts(check=True)`), context 4,096 tokens (at most 4,094 genes plus `<bos>` and `<eos>`).
2. Forward pass, batch 1. Keep all 12 hidden states (0 = embedding, 1–10 = outputs of blocks 0–9, 11 = after the final RMSNorm). Average each over gene positions 1 to L − 2 (no `<bos>`/`<eos>`).
3. Memory: from the fourth extraction call on, sequences were right-padded with `<pad>` (attention mask 0) to a multiple of 256 tokens. In a causal model padding does not change the real positions (checked below). The first 1,200 internal cells are unpadded.
4. Anchor centroid = mean of its cells' states (n × 12 × 1,232 per panel).
5. MaxToki feature (the deployed "pooled drift", 2,464 numbers): for blocks early (hidden states 1–4), mid (5–8) and late (9–11), average the states in the block and multiply by the block's operator A (the mean over the block's layers of the transposed attention output projection, 1,232 × 1,232). Feature = [y_early − y_mid, y_mid − y_late].
6. All features are standardised with the internal panel's mean and SD, so a head fitted on internal anchors applies unchanged to the other panels.

**Part 2: comparison features** (one row per anchor, the same cells as the MaxToki centroids). MaxToki PCA-64 is a reduced MaxToki feature and uses the model. The lookup, the token bag and HVG are the three stand-ins that use no model.

| Name | What it is |
|---|---|
| MaxToki PCA-64 (uses the model) | PCA to 64 numbers of the MaxToki feature (fitted on internal anchors), whitened; the same size as the stand-ins |
| Lookup (cell type), seeds 0–4 | one random 64-number Gaussian code per `cell_type` string, plus 0.5 × Gaussian noise per anchor. Knows the label and nothing else |
| Lookup (cell type + stage), seeds 0–2 | the same per (cell type, stage) class |
| Token bag | for each cell, each gene token at rank r of n gets weight 1 − r/n (the model's own input order); anchor mean; log1p; genes with SD > 0 on internal; standardise; PCA-64 fitted on internal anchors, whitened |
| HVG | `raw/X` counts per 10,000 + log1p per cell; anchor mean; 2,000 highly variable genes (Seurat v1 dispersion, 20 mean bins, internal cells only); standardise; PCA-64 fitted on internal anchors, whitened |

**Part 3: the read-out head and the gates** (the deployed code: `phase5_let_anchor.train_let`; frozen-panel code of `phase7_external_validation.py` and `phase8_zeroshot_transfer.py`).

1. Head: z = W(x − b), with W a 10 × d matrix. Predicted distance = β · arccos(cos(z_i, z_j)), β learnable. Loss = mean squared error to the ruler distances + 0.1 × reconstruction error. Adam, learning rate 5 × 10⁻³, 1,500 full-batch epochs, torch seed 42 (the deployed seed), 1 CPU thread.
2. Gates on the internal panel (head fitted on internal anchors):
   - trust = trustworthiness (scikit-learn, k = 15) of the 10-number output relative to the head's input feature. It checks that anchors among each other's 15 nearest neighbours in the output were also close in the input: it penalises output neighbours that were far apart in the input, weighted by their input rank. It never looks at the ruler.
   - random holdout = mean Spearman correlation between output distances and ruler distances on the held-out 20% of anchors, over 10 random 80/20 splits (head fitted on each 80%).
   - donor holdout = the same, leaving out one donor at a time (groups with ≥ 3 anchors).
   - branch holdout = the same, leaving out one branch at a time; plain mean over the 6 scored branches.
3. Gates on frozen panels (head fitted once on internal anchors): trust as above; random = mean Spearman over 20 random 20% subsets; donor = mean Spearman within each donor; within-branch = mean Spearman within each scored branch; global = Spearman over all anchor pairs of the panel.
4. Pass marks (fixed by the deployed analysis): trust ≥ 0.80; random, donor and branch Spearman ≥ 0.20.
5. "Different cell type" (diff): the same Spearman using only pairs of anchors whose `cell_type` differs. It averages fewer branches: 3 of 6 on internal and external, 2 on zero-shot, because in the other branches every such pair has the same ruler value.

**Part 4: nulls.**

1. Deployed null: stages shuffled among anchors within each branch (`default_rng(42)`). This also splits anchors of one cell type across stages.
2. Structured null (keeps cell-type structure). A class is a (cell type, stage) pair, over all blood panels. Each draw permutes the class → stage map among the classes of one branch. Every anchor keeps its branch, and all anchors of one class share one stage, so same-class pairs stay at distance 0, as in the real ruler. Only the developmental order between classes is broken.
   - Fitted null: the whole pipeline on the permuted ruler (internal head fitted again, then applied frozen to external and zero-shot under the same map). 40 draws (`default_rng([2026, d])`); internal branch holdout uses the first 20.
   - Evaluation-only null: the held-out branch heads fitted on the real ruler are kept; only the held-out ruler is drawn again. 2,000 draws (`default_rng([2027, d])`).
   - p = (1 + number of null values ≥ observed) / (1 + draws), one-sided.
3. Lung negative control, two designs, frozen head, 2,000 draws each: deployed design (one random stage per lung anchor, `default_rng([6161, d])`) and structured design (one random stage per lung cell type, `default_rng([6262, d])`).

**Part 5: intervals and seed spread.**

1. Donor cluster bootstrap: each replicate draws the panel's donors with replacement and keeps all their anchors; copies of one anchor are never paired; heads stay frozen (seed 42). 2,000 replicates (`default_rng([8080, panel, b])`), percentile 95% interval. Unit: donor (13 external, 12 zero-shot, 11 internal). Lookup = mean of the 5 cell-type seeds inside each replicate. Global numbers use pair weights, which equal expanding the resample.
2. Head seeds: the full internal head fitted with torch seeds 0–19 and scored with the same gate code. Seed-paired contrasts (same seed for both heads) use seeds 0–9.

**Part 6: the single head singled out by the deployment (L10H6).** Feature = hidden state 11 centroid × the 154 output-projection columns of attention head 6 in layer 10. Control: 30 Gaussian 154 × 1,232 projections of the same hidden state (`default_rng([6060, 11, seed])`). Scoring follows the deployed head screen (`phase9_head_attribution.py`): in-sample trust, 3 random splits, 6-branch holdout, composite = 0.5 trust + 0.25 random + 0.25 branch.

**Part 7: Smart-seq2 sensitivity.** Centroids rebuilt from 10x cells only. Anchors with no 10x cell are dropped (internal 290 → 247, external 600 → 587, zero-shot 160 → 152). The comparison is with all-cell centroids of the same kept anchors, over 10 head seeds (42 and 0–8).

### Checks

**Input encoding.**
- Every cell passed `inputs_v3.tokenize_counts(check=True)`, and every chunk of 50 cells passed `assert_encoding_batch`, before any forward pass: 653 chunks, 32,548 of 32,548 cells, 0 genes out of order. All count rows were whole numbers.
- A separate tokeniser (h5py `raw/X`, own Ensembl → token and gene-median maps, stable sort) gave the stored tokens for 108 of 108 cells, equal up to exact ties.
- Next-gene loss (mean cross-entropy of the model's prediction of each next gene token), 41 random 10x cells (`default_rng(20261003)`): 3.09 nats with the count-based input. With the deployed analysis's gene order (genes ranked by log1p(CP10k) divided by the gene median) it was 4.03 nats; difference −0.95 (95% percentile bootstrap interval over cells, 5,000 resamples: −1.18 to −0.73), lower in 41 of 41 cells. A uniform guess over the 20,275-token vocabulary scores 9.92. In 4 Smart-seq2 cells the loss was 7.82 nats.

**Forward passes and centroids.**

| Check | Result |
|---|---|
| Fresh full-model forward, no padding, against stored per-cell states (108 cells, padded and unpadded chunks) | largest difference 3.8 × 10⁻⁶ |
| Padding: 30 test cells, padded against unpadded | largest difference 3.8 × 10⁻⁶ (float32 rounding) |
| Fresh centroids of 3 whole anchors (13, 20, 30 cells) against stored | largest difference 4.8 × 10⁻⁷ |
| Deployed extraction code in this environment against a deployed centroid | difference 1.9 × 10⁻⁶ |
| Deployed tokens rebuilt from the stored `X` (5 cells per panel) | exact, which proves the row mapping |

**Analysis code.**
- The driver scripts import the shared analysis code: the task list (2,078 head fits), the task code, the seeds and the analysis arguments. The deployed head and internal gate code is imported read-only from `phase5_let_anchor.py`. The frozen-panel gate code of `phase7`/`phase8` sits inside their `main()`, so it is copied line for line.
- Rulers rebuilt from labels equal the deployed ruler files (and the deployed within-branch null ruler) on 4 panels.
- Own gate code, applied to the deployed centroids of the lung-immune panel, gives the deployed values exactly.

### Results

**Main findings.**
- MaxToki-217M shows no blood developmental order beyond cell-type identity that these tests detect. Only the T lineage has graded stages. Under the structured null, every branch and global number sits inside the null (one-sided p from 0.15 to 0.76). Two kinds of number are above it: zero-shot within-donor order, and internal and external trust at head seed 42 (structured-null table below).
- On pairs of anchors with different cell types, the branch numbers average near 0: 0.067 internal, 0.006 external, 0.016 zero-shot. These are means over 3, 3 and 2 branches; per branch they range from −0.28 to +0.30. In the T lineage, the only graded branch, held-out order is 0.038. On the frozen panels it is 0.205 (external) and 0.288 (zero-shot), below a cell-type lookup (0.347 and 0.484).
- A random code per cell-type label passes the three correlation gates on all three blood panels (5 of 5 seeds) and all four gates on external (4 of 5 seeds). Its trust is below 0.80 on the internal panel (0.754–0.790) and the zero-shot panel (0.710–0.770).
- On the zero-shot panel MaxToki fails the 0.20 within-branch gate: 0.180 at seed 42; 0.147 ± 0.048 over seeds 0–19; 3 of 20 seeds pass. This failure depends on the anchor set (Smart-seq2 paragraph below).
- Among the order numbers, MaxToki is clearly ahead of the stand-ins (donor-bootstrap interval above 0) only in cross-lineage (global) order. It beats a label lookup and HVG there by 0.06–0.11. Its lead over a bag of its own input tokens is small and depends on how the bag is built. On within-branch numbers the donor intervals include 0. MaxToki's trust is the highest of all representations on every blood panel, but trust does not use the ruler.

**H65 gates by representation** (head seed 42; lookup = cell-type seed 0; `table_gates_by_representation.csv`):

| Panel | Number | MaxToki | Lookup | Token bag | HVG | MaxToki PCA-64 |
|---|---|---:|---:|---:|---:|---:|
| internal (head fitted here) | trust | 0.845 | 0.785 | 0.705 | 0.717 | 0.686 |
| | random / donor | 0.849 / 0.783 | 0.739 / 0.661 | 0.674 / 0.556 | 0.726 / 0.676 | 0.796 / 0.575 |
| | branch holdout | 0.414 | 0.364 | 0.429 | 0.354 | 0.487 |
| | branch holdout, diff | 0.067 | 0.056 | −0.078 | 0.088 | 0.075 |
| | branch holdout, deployed null | −0.008 | −0.084 | −0.016 | 0.011 | – |
| external (frozen) | trust | 0.905 | 0.851 | 0.859 | 0.859 | 0.820 |
| | within-branch / diff | 0.417 / 0.006 | 0.312 / 0.070 | 0.386 / −0.039 | 0.367 / −0.011 | 0.381 / −0.014 |
| | global / global diff | 0.865 / 0.835 | 0.826 / 0.782 | 0.849 / 0.820 | 0.763 / 0.720 | 0.863 / 0.833 |
| zero-shot (frozen) | trust | 0.823 | 0.770 | 0.750 | 0.768 | 0.735 |
| | within-branch / diff | **0.180** / 0.016 | 0.319 / 0.145 | 0.133 / −0.085 | 0.030 / −0.060 | 0.457 / 0.160 |
| | global / global diff | 0.867 / 0.851 | 0.811 / 0.782 | 0.819 / 0.807 | 0.757 / 0.746 | 0.855 / 0.835 |
| lung non-blood (frozen) | trust / branch | 0.719 / −0.045 | 0.656 / −0.017 | 0.892 / −0.060 | 0.755 / −0.144 | 0.811 / −0.157 |

- Gate verdicts for MaxToki: internal and external pass all four; zero-shot fails the branch gate; lung non-blood fails all four.
- Lookup (cell type), 5 seeds: internally all pass random, donor and branch, and all fail trust (0.754–0.790). On external, 4 of 5 seeds pass all four gates (seed 4 fails trust, 0.781). On zero-shot, all 5 pass random, donor and branch, and all fail trust (0.710–0.770).
- Token bag and HVG pass all four gates on external and fail only trust internally.
- The deployed null fails for every representation, including the lookup (−0.084), because it splits cell types. It cannot tell order from identity.

**Per branch, internal held out** (MaxToki; `v2b_tables.json`, `summaries` → `h65|maxtoki|pos` → `per_group`):

| Branch | Anchors | MaxToki | MaxToki, diff |
|---|---:|---:|---:|
| T lineage | 101 | 0.038 | −0.027 |
| B lineage | 45 | 0.282 | – |
| monocyte | 34 | 0.020 | 0.018 |
| granulocyte | 25 | 0.530 | 0.211 |
| erythroid | 7 | 0.783 | – |
| stem | 4 | 0.828 | – |
| plain mean | | 0.414 | 0.067 |

The plain mean is carried by the small branches. Weighted by anchors it is 0.182, below the gate.

**T lineage, the only graded branch** (`table_h65_per_branch.csv`):

| Panel | MaxToki | Lookup (seed 0) | Token bag | HVG |
|---|---:|---:|---:|---:|
| internal, held out | 0.038 | 0.145 | 0.023 | 0.087 |
| external, frozen | 0.205 | 0.347 | 0.094 | 0.087 |
| zero-shot, frozen | 0.288 | 0.484 | −0.038 | 0.101 |

On frozen panels the head was fitted on internal T labels, and a lookup reproduces that learned order better than MaxToki.

**Structured null** (MaxToki; fitted null 40 draws, internal branch 20; evaluation-only null 2,000 draws; `v2b_tables.json` → `structured_null_refit`, `v2b_evalnull_internal_branch.json`):

| Number | Observed | Null mean (SD) | p |
|---|---:|---:|---:|
| internal branch holdout (fitted null) | 0.414 | 0.384 (0.073) | 0.38 |
| internal branch holdout (evaluation-only) | 0.414 | 0.402 (0.054) | 0.39 |
| internal branch, diff (evaluation-only) | 0.067 | 0.007 (0.099) | 0.27 |
| internal trust (fitted null) | 0.845 | 0.805 (0.013) | 0.024 |
| external within-branch | 0.417 | 0.332 (0.071) | 0.15 |
| external within-branch, diff | 0.006 | 0.099 (0.117) | 0.71 |
| external global | 0.865 | 0.843 (0.033) | 0.29 |
| external trust | 0.905 | 0.885 (0.011) | 0.024 |
| zero-shot within-branch | 0.180 | 0.124 (0.101) | 0.27 |
| zero-shot global | 0.867 | 0.831 (0.033) | 0.15 |
| zero-shot trust | 0.823 | 0.804 (0.012) | 0.098 |
| zero-shot within-donor | 0.880 | 0.764 (0.073) | 0.024 |

- Every branch and global number is inside its null.
- The internal and external trust rows reach the smallest p that 40 draws allow, but these use head seed 42. Using the seed mean over seeds 0–19, trust is 1.0 null SD above the null mean internally, 1.8 externally and 1.4 on zero-shot. Only 2 of 20 seeds (10%) are above the largest internal null value. Trust also never looks at the ruler.
- Zero-shot within-donor order is above its null (p = 0.024, one seed). Within-donor pairs include pairs from different branches, so this number mixes in cross-lineage order. It is also the one number that the Smart-seq2 cells move (below).
- The lookup, token bag and HVG are inside their own nulls on every branch number (p 0.12–0.85 across the three).

**Head-seed spread** (seeds 0–19; `seeds_summary.json`). Mean ± SD [min, max]:

| Number | 20 seeds | Seed 42 |
|---|---|---:|
| internal trust | 0.818 ± 0.009 [0.809, 0.844] | 0.845 |
| external trust | 0.904 ± 0.002 | 0.905 |
| external within-branch | 0.422 ± 0.011 | 0.417 |
| external within-branch, diff | 0.013 ± 0.016 | 0.006 |
| external global | 0.856 ± 0.004 | 0.865 |
| zero-shot trust | 0.822 ± 0.004 | 0.823 |
| zero-shot within-branch | 0.147 ± 0.048 [0.057, 0.248]; 3 of 20 ≥ 0.20 | 0.180 |
| zero-shot global | 0.871 ± 0.005 | 0.867 |
| lung non-blood trust | 0.725 ± 0.018 | 0.719 |

Seed 42 gives the highest internal trust of the 21 seeds. The zero-shot branch failure holds across seeds.

**Cross-lineage (global) order and contrasts** (donor bootstrap, frozen heads, 2,000 replicates, percentile 95% interval; lookup = mean of 5 seeds; `v2b_boot_<panel>.json`, `v2b_boot_<panel>_global.json`):

| Representation | External global [95% CI] | Zero-shot global [95% CI] |
|---|---|---|
| MaxToki | 0.865 [0.839, 0.896] | 0.867 [0.828, 0.888] |
| Lookup (mean of 5 seeds) | 0.801 [0.779, 0.821] | 0.776 [0.700, 0.822] |
| Token bag | 0.849 [0.809, 0.882] | 0.819 [0.691, 0.866] |
| HVG | 0.763 [0.714, 0.808] | 0.757 [0.704, 0.821] |

| Contrast | Panel | Within-branch | Within-branch, diff | Global | Global, diff |
|---|---|---|---|---|---|
| MaxToki − lookup | external | +0.075 [−0.038, 0.138] | −0.021 [−0.187, 0.075] | **+0.064 [0.050, 0.091]** | **+0.086 [0.068, 0.119]** |
| MaxToki − token bag | external | +0.031 [−0.067, 0.094] | +0.044 [−0.016, 0.184] | +0.016 [0.002, 0.046] | +0.015 [−0.000, 0.047] |
| MaxToki − HVG | external | +0.050 [−0.062, 0.111] | +0.017 [−0.087, 0.167] | **+0.102 [0.073, 0.140]** | **+0.115 [0.086, 0.156]** |
| MaxToki − lookup | zero-shot | −0.197 [−0.567, 0.005] | −0.210 [−0.366, 0.153] | **+0.091 [0.027, 0.163]** | **+0.109 [0.041, 0.186]** |
| MaxToki − token bag | zero-shot | +0.047 [−0.332, 0.363] | +0.100 [−0.059, 0.424] | +0.048 [0.009, 0.149] | +0.044 [0.008, 0.140] |
| MaxToki − HVG | zero-shot | +0.149 [−0.177, 0.298] | +0.076 [−0.125, 0.342] | **+0.110 [0.038, 0.151]** | **+0.105 [0.036, 0.154]** |
| MaxToki − lookup | internal (held out) | +0.021 [−0.393, 0.063] | +0.013 [−0.438, 0.087] | – | – |
| MaxToki − token bag | internal (held out) | −0.016 [−0.358, 0.063] | +0.146 [−0.025, 0.594] | – | – |
| MaxToki − HVG | internal (held out) | +0.060 [−0.347, 0.083] | −0.021 [−0.131, 0.082] | – | – |

MaxToki within-branch values: external 0.417 [0.297, 0.532]; zero-shot 0.180 [−0.103, 0.369]. Internal rows resample evaluation donors only (held-out heads fixed), so they are too narrow.

**Seed-paired contrasts, global order** (seeds 0–9, mean ± SD; `seeds_summary.json` → `seed_paired_contrasts`):

| Contrast | External | Zero-shot |
|---|---|---|
| MaxToki − token bag | +0.004 ± 0.005 (7 of 10 > 0) | +0.053 ± 0.005 (10 of 10) |
| MaxToki − HVG | +0.090 ± 0.007 | +0.108 ± 0.010 |
| MaxToki − lookup (seed 0) | +0.028 ± 0.003 | +0.059 ± 0.004 |

**Token bag built from the deployed gene order.** A second token bag, built the same way from the gene order the deployed analysis fed the model (genes ranked by log1p(CP10k) divided by the gene median), is also model-free. Its global order is 0.876 [0.837, 0.903] external and 0.879 [0.823, 0.902] zero-shot. MaxToki minus this bag: −0.010 [−0.023, 0.016] external and −0.012 [−0.031, 0.022] zero-shot (own bootstrap code, 1,000 replicates, unit donor, percentile; `verify/verify_gates.json`). So the zero-shot lead depends on which gene order the bag uses: zero-shot global order is 0.819 for the count-based bag, 0.879 for the deployed-order bag and 0.867 for MaxToki.

Reading:
- On every within-branch number, MaxToki is not separable from a lookup or from the expression baselines.
- On global order it beats the lookup and HVG by 0.06–0.11, with intervals clear of 0.
- Against a bag of its own input tokens it ties on external over head seeds (+0.004 ± 0.005) and leads on zero-shot by about 0.05. A bag of the deployed gene order matches it on both panels.
- Global order depends mostly on lineage and cell-type grouping. The structured null keeps both and breaks only the stage order between classes within a branch, and it keeps most of global order (null means 0.843 and 0.831 against 0.865 and 0.867; p = 0.29 and 0.15). How much of global order comes from pairs across lineages was not measured directly.

**Lung negative control** (`v2b_lung_control_6161.json`, `v2b_lung_control_6262.json`).
- Deployed design (one random stage per lung anchor): random, donor and branch gates pass together in 0 of 2,000 draws for all 12 representations, MaxToki and pure label codes alike. Branch alone passes in 9.5–12.0% of draws. This control cannot fail any representation.
- Structured design (one random stage per lung cell type), frozen head:

| Representation | Branch mean | Pass branch | Pass random | Pass donor | Pass random + donor + branch |
|---|---:|---:|---:|---:|---:|
| MaxToki | 0.470 | 93.0% | 15.5% | 5.8% | 5.2% |
| Lookup (seed 0) | 0.508 | 96.5% | 14.6% | 3.1% | 2.7% |
| Token bag | 0.563 | 97.2% | 19.0% | 5.3% | 5.2% |
| HVG | 0.430 | 89.9% | 12.1% | 4.8% | 4.2% |

Once random labels follow lung cell types, the branch gate passes for meaningless labels in about 93% of draws. MaxToki's lung trust is 0.719, so "all four" passes in 0%.
- Lung immune panel (blood cells in lung with real stage labels), frozen head, own code: trust 0.799, random 0.878, donor 0.899, branch 0.470, global 0.901. It passes the three correlation gates and is just below the trust gate.

**L10H6 against random projections of the same hidden state** (`v2b_n60_random_projection.json`):

| Number | L10H6 | 30 random projections, mean ± SD | Share of random ≥ L10H6 |
|---|---:|---|---:|
| trust | 0.875 | 0.809 ± 0.011 | 0 / 30 |
| random holdout | 0.789 | 0.776 ± 0.016 | 17% |
| branch holdout | 0.419 | 0.403 ± 0.071 | 50% (15 / 30) |
| branch holdout, diff | 0.066 | 0.049 ± 0.061 | 43% |
| composite | 0.739 | 0.699 ± 0.021 | 0 / 30 |

L10H6 leads only through trust, which does not use the ruler, and so through the composite, which is half trust. Its features are more concentrated (participation ratio 12.2 against 14.7, range 13.7–16.1, for the random projections). Whether this explains the trust lead was not tested. The scoring code is the deployed head screen's: trust and branch holdout are computed as for the H65 gates (trust of the head fitted on all internal anchors, relative to its own input feature), but random holdout uses 3 splits instead of 10.

**Smart-seq2 cells** (`verify/verify_ss2.json`). Smart-seq2 is a plate-based method. `raw/X` holds read counts for its cells, and the model predicts their genes poorly (next-gene loss 7.82 nats in 4 cells against 3.09 in 10x cells; Checks). Paired change, 10x-only minus all cells on the same kept anchors, mean over 10 head seeds: internal trust +0.010; external trust +0.002, global +0.003, within-branch −0.001; zero-shot global −0.005, within-branch −0.016, within-donor −0.078. Every other paired change was at most 0.016 in size. The paired check covers trust on all three panels and the frozen-panel numbers; it does not cover the internal random, donor or branch holdout. So with the anchors held fixed, the Smart-seq2 cells barely matter, except for zero-shot within-donor order. The anchor set matters more: on the kept anchors (247 internal, 152 zero-shot), zero-shot within-branch is 0.34 ± 0.01 (all cells, 10 seeds) instead of 0.147 on the full panels. Dropping the 8 zero-shot anchors made only of Smart-seq2 cells (and 43 such internal anchors) changes it by about 0.2.

**Other orderings tested by the deployed analysis** (head seed 42; structured null = class → category map permuted; 30 draws, except H95 within-category, which has 22 valid draws on external and 28 on zero-shot):

| Ordering | Result |
|---|---|
| H38 (7 signalling categories) | Internal MaxToki trust / random / donor / category holdout 0.801 / 0.758 / 0.599 / 0.229. External category: MaxToki 0.608, token bag 0.701, HVG 0.640, lookup (seed 0) 0.826. MaxToki 0.608 against null mean 0.331, p = 0.032; token bag and HVG reach the same p. Any expression-based representation follows these categories, and the lookups do better. |
| H95 (4 effector categories) | Internal trust 0.8018 with 1 CPU thread and 0.7998 with 4, so its pass of the 0.80 gate depends on the thread count. External within-category 0.206 against a null mean of 0.364, p = 0.78. |
| H103 (B-cell maturation) | Only two labels exist ("B cell", "plasma cell"), so the test only asks whether two labels separate. Internal random / donor 0.861 / 0.861 for MaxToki and for each of the 3 cell-type lookup seeds. |

### Verification

No separate verification by a second agent is recorded for this analysis. The run includes these second-way checks (script `scripts/v3_devorder_verify.py`, outputs `outputs/v3_devorder/verify/`; fresh model passes on MPS, the rest on CPU):

| Number | First way | Second way | Agreement |
|---|---|---|---|
| Model input | `inputs_v3` tokeniser | own tokeniser (h5py, own maps, stable sort) | 108 / 108 cells equal up to exact ties |
| Per-cell hidden states | stored chunks | fresh full-model forward, no padding | largest difference 3.8 × 10⁻⁶ (108 cells) |
| Anchor centroids | stored | fresh, 3 whole anchors | largest difference 4.8 × 10⁻⁷ |
| Pooled-drift features | feature files | own code | largest difference 7.5 × 10⁻⁶ |
| Rulers | stored ruler files | rebuilt from labels | identical, 4 panels |
| Internal branch, evaluation-only null | 0.414 vs null mean 0.402, p = 0.39 | own code, new stream (`default_rng([31338, d])`, 2,000 draws): null mean 0.398, p = 0.38; diff 0.067 vs 0.004, p = 0.27 | same reading |
| Donor bootstrap, global contrasts | pair weights, 2,000 replicates | explicit resampling, new stream, 1,000 replicates | external MaxToki − token bag +0.016 [0.002, 0.044] (main [0.002, 0.046]); zero-shot +0.048 [0.011, 0.136] (main [0.009, 0.149]); MaxToki − HVG and − lookup within 0.006 of the main interval ends |
| Structured lung control | analysis code | own code, new stream (`default_rng([62620, d])`) | MaxToki passes branch in 93.7% of draws (main 93.0%) |

One check failed in an informative way. Fitting the seed-42 head from the own features, which differ from the feature files by at most 7.5 × 10⁻⁶, gave internal trust 0.813 instead of 0.845. The frozen-panel numbers moved by up to 0.014 (for example, external within-branch 0.422 against 0.417). So single-seed numbers carry optimiser noise. The head-seed table measures it, and the main text quotes seed means where they matter.

### Limits

- **No random-initialised MaxToki** and no other cell model (scVI, Geneformer, Palantir, CellTypist) was compared.
- **The deployed 88-head screen was not run on these centroids.** So it is not known whether L10H6 would still be the top head. The L10H6 comparison only asks whether that head beats random projections.
- **Most numbers use one head seed (42).** Seeds 0–19 cover the full internal fit only (seeds 0–9 for the paired contrasts; seeds 42 and 0–8 for the Smart-seq2 check), not the held-out branch fits, the nulls or the bootstraps.
- **Fitted nulls are coarse:** 40 draws (20 for internal branch holdout), so the smallest p is 0.024 (0.048).
- **Donor intervals cover the within-branch and global numbers and the contrasts, not trust.** Internal bootstrap intervals are too narrow (evaluation donors only, heads fixed).
- **The two test panels share donors.** All 12 zero-shot donors are external donors; only the anchors differ. So the zero-shot panel is not an independent donor replication of the external panel.
- **Only the T lineage has graded stages.** In the other scored branches, the within-branch numbers can only show whether anchors with the same label sit together.
- **Only a curated stage tree was tested.** No real non-blood differentiation ruler and no new anchors.
- **The lookup noise level (0.5) and code size (64) were fixed, not tuned.** Lookup trust changes with them.
- **The thread-count effect** (H95) was checked at 1 and 4 threads only.
- **Smart-seq2 cells were kept** in the main analysis (`raw/X` holds read counts for them, and the model predicts their genes poorly). The sensitivity analysis above shows their effect.
- **Process memory** went above the 8 GB target (about 8.5 GB) in the first three extraction calls, which used unpadded sequences. Results are not affected (padding check above).
- The raw h5ad files (19.8 GB, 3.2 GB) were fingerprinted (size, time, sha256 of the first and last 16 MB), not hashed in full. The rows used are listed in `run_config_centroids.json`.

### Files

- Scripts (`projects/maxtoki/runs/manifold-discovery-217M/scripts/`):
  - drivers: `v3_devorder_centroids.py` (cell plans, encoding checks, forward passes, centroids), `v3_devorder_features.py` (MaxToki, lookups, token bag, HVG), `v3_devorder_pool.py` (all 2,078 head fits; thread check), `v3_devorder_analyze.py` (tables, nulls, lung control, bootstraps, random projections), `v3_devorder_seeds.py` (head seeds), `v3_devorder_verify.py` (second-way checks), `v3_devorder_common.py` (paths);
  - analysis code they import: `v2b_devorder_common.py`, `v2b_devorder_01_features.py` (feature recipes), `v2b_devorder_02_pool.py` (task list and task code), `v2b_devorder_03_analyze.py` (tables, nulls, bootstraps);
  - deployed code used read-only: `phase5_let_anchor.py` (head and internal gates), `phase7_external_validation.py` and `phase8_zeroshot_transfer.py` (frozen-panel gates), `phase1bc_hidden_states_and_centroids.py` (cell subsample, pooling), `phase9_head_attribution.py` (single-head scoring), `phase13_h38_lite.py`, `phase3a_manifold_sweep.py`, `phase3a_sweep2_ordinal.py` (H38, H95, H103);
  - shared input code: `projects/maxtoki/setup/inputs_v3.py`.
- Stage tree: `projects/maxtoki/runs/manifold-discovery-217M/planning/h65_stage_dag.json`. Deployed cell lists: `projects/maxtoki/runs/manifold-discovery-217M/outputs/phase1/cells_<panel>_obs.csv`.
- Outputs (`projects/maxtoki/runs/manifold-discovery-217M/outputs/v3_devorder/`):
  - centroids and metadata: `artifacts/anchors/centroids_<panel>.npy` (n × 12 × 1,232), `anchor_meta_<panel>.csv`, `d_target_<panel>.npy` (rulers); operators: `artifacts/operators/`; per-cell states: `percell/<panel>/chunk_*.npz`; tokens: `cells/tokens_<panel>.npz`; cell plans: `cells/plan_<panel>.csv`, `cells/plan.json`;
  - features: `features/<representation>__<panel>.npy`, `features_manifest.json`;
  - head fits: `pool/results.jsonl`, `heads/`, `thread_check/results.jsonl`;
  - analysis: `v2b_tables.json` (summaries, ruler structure, fitted structured nulls), `table_gates_by_representation.csv`, `table_h65_per_branch.csv`, `v2b_evalnull_internal_branch.json`, `v2b_lung_control_6161.json`, `v2b_lung_control_6262.json`, `v2b_boot_external.json`, `v2b_boot_zeroshot.json`, `v2b_boot_internal_branch.json`, `v2b_boot_external_global.json`, `v2b_boot_zeroshot_global.json`, `v2b_n60_random_projection.json`, `v2b_n60_effective_dimension.json`;
  - head seeds: `seeds/results.jsonl`, `seeds_summary.json`;
  - checks: `verify/verify_gates.json`, `verify/verify_ss2.json`, `verify/ss2_per_seed.csv`, `verify/fresh_nll.json`, `verify/fresh_nll_cells.csv`, `verify/fresh_anchors.json`, `verify/fresh_anchor_cells.csv`;
  - provenance: `run_config_centroids.json` (device, versions, seeds, every cell row and label, encoding record and check result per panel, sha256 of inputs, code and outputs, wall time per chunk), `run_config.json` (analysis seeds and sha256), `run_config_features_*.json`, `verify/run_config.json`.
- Run time: 30,157 s of forward passes on MPS (0.84–1.27 s per cell, shared GPU), in 74 calls of at most about 7.8 minutes.

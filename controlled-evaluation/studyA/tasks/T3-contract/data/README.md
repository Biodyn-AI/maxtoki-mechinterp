# Data for this task

All files are in this folder. Paths below are relative to the package root. Gene symbols are upper case.
TSV files are tab-separated with a header row; `cell_manifest.csv` is comma-separated.

## Where the data come from

- **Cells.** K562 cells from the Replogle et al. (2022) genome-scale CRISPRi Perturb-seq screen. Each cell
  carries either a non-targeting control guide or a guide that knocks down one gene.
- **Model.** MaxToki-217M, a single-cell foundation model. It is a decoder-only transformer (Llama
  architecture, 11 blocks, hidden size 1,232). A cell is given to the model as a sequence of gene tokens
  ordered by the gene's expression rank in that cell (highest first; rank-value encoding), with `<bos>` at
  the start and `<eos>` at the end, at most 2,048 tokens. A gene appears at most once per cell.
- **Sparse autoencoder (SAE).** A TopK SAE with 4,928 features and k = 32 (at each token position at most 32
  features are non-zero). It was trained on the model's residual-stream activations entering decoder block 5
  (blocks numbered from 0; `hidden_states[5]` in Hugging Face `transformers` terms), using the 1,019,996
  token positions of the 500 `catalog` control cells (see below; 917,996 positions for training, 102,000
  held out). Feature activations are non-negative and have no physical unit.
- **Forward passes.** Every cell listed in `cell_manifest.csv` was tokenised (maximum length 2,048), run
  through the model up to the input of block 5, and encoded with the SAE.

## Files

### `cell_manifest.csv` (9,200 rows)

| column | meaning |
|---|---|
| `row` | cell identifier (row index of the cell in the source dataset); matches `rows` in `cell_feature_means.npz` |
| `group` | `ref`, `pool`, `kd` or `catalog` (below) |
| `tf` | the knocked-down gene, for `kd` cells; empty otherwise |
| `barcode` | cell barcode in the source dataset |
| `gem_group` | 10x GEM group (sequencing lane / batch) in the source dataset |
| `UMI_count` | total UMI count of the cell |

Groups. All cells were drawn at random. The three control groups do not overlap.
- `ref`: 400 non-targeting control cells.
- `pool`: 800 further non-targeting control cells.
- `catalog`: the 500 non-targeting control cells used to train the SAE and to build the top-20 gene lists.
- `kd`: 7,500 knockdown cells for 87 transcription factors (TFs), up to 100 cells per TF (34 to 100). The 87
  TFs are all genes knocked down in this K562 screen that are listed as a regulator in TRRUST or DoRothEA.

### `cell_feature_means.npz`

NumPy archive with three arrays (load with `numpy.load`; about 180 MB in memory):

| array | shape | meaning |
|---|---|---|
| `rows` | (9,200,) int64 | cell identifier, same as `row` in the manifest |
| `mean_gene` | (9,200, 4,928) float32 | per-cell feature value: the mean of the feature's SAE activation over all gene tokens of the cell (every position except `<bos>` and `<eos>`; positions where the feature is not active count as 0) |
| `n_tokens` | (9,200,) int32 | number of tokens of the cell, including `<bos>` and `<eos>` |

### `feature_top20.tsv` (97,492 rows)

The top genes of each SAE feature, built from the 500 `catalog` cells. For each (feature, gene) pair, the
mean activation of the feature over the token positions where the feature is active (> 0) and the token is
that gene. The 20 genes with the highest mean are listed. No minimum number of positions was required.
4,807 features have 20 genes; 121 have fewer (features that are active on fewer than 20 genes). The token
`<SPECIAL>` stands for `<bos>`/`<eos>` and appears in the lists of 59 features; it is not a gene.

| column | meaning |
|---|---|
| `feature_id` | 0 to 4,927 |
| `rank` | 1 = highest mean activation |
| `gene` | gene symbol (or `<SPECIAL>`) |
| `mean_act_when_active` | the mean activation used for the ranking |
| `n_active_positions` | number of positions (in the 500 catalog cells) where the feature is active on this gene |

### `feature_info.tsv` (4,928 rows)

| column | meaning |
|---|---|
| `feature_id` | 0 to 4,927 |
| `n_genes_listed` | number of entries in `feature_top20.tsv` |
| `has_special_token` | whether `<SPECIAL>` is in the list |
| `n_active_positions` | number of the 1,019,996 catalog token positions where the feature is active |

### `gene_universe.tsv` (6,324 rows)

All genes that appear as a token in at least one of the 500 `catalog` cells. These are the only genes that
can appear in a top-20 list.

| column | meaning |
|---|---|
| `gene` | gene symbol |
| `detection_count` | number of the 500 catalog cells in which the gene is a token (1 to 500) |
| `gene_length_bp` | genomic span of the gene (end minus start, base pairs); empty for 54 genes without coordinates |
| `chr` | chromosome; empty when unknown |

### `targets_trrust.tsv` (2,151 rows)

TRRUST v2 (human) TF-target pairs whose target is in the gene universe. Columns: `tf`, `target`, `mode`
(Activation / Repression / Unknown; several values joined with `;` when sources differ), `n_pmid` (number of
supporting PubMed IDs).

### `targets_dorothea.tsv` (106,815 rows)

DoRothEA (human, all confidence levels) TF-target pairs whose target is in the gene universe.

| column | meaning |
|---|---|
| `tf`, `target` | the pair |
| `confidence` | DoRothEA confidence level, A (highest) to D; a few pairs carry two levels (for example `B;D`) |
| `chip_flag` | True if the pair is also in DoRothEA's ChIP-seq-supported subset |

**Which TFs have targets.** Both target files list every TF of the source database that has at least one
target in the gene universe (TRRUST 482 TFs; DoRothEA 459 TFs, 296 of them with ChIP-seq-flagged pairs), not
only the 87 knocked-down TFs. Of the 87 knocked-down TFs, 58 have at least one TRRUST target in the universe
and 30 have at least one DoRothEA target (20 of them with ChIP-seq-flagged pairs). The other 22 have no
target in the universe in either file.

### `tf_info.tsv` (87 rows)

| column | meaning |
|---|---|
| `tf` | knocked-down TF |
| `n_kd_cells` | number of `kd` cells for this TF in this package |
| `tf_gene_measured` | whether the TF's own gene is in the expression panel of the source dataset |
| `kd_remaining_frac` | mean expression of the TF's gene in its knockdown cells divided by its mean in the 400 `ref` cells (linear scale, from the dataset's normalised log1p values); empty when the gene is not measured |

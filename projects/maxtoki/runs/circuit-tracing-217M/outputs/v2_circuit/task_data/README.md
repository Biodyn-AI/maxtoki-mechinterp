# Task data: CRISPRi direction from MaxToki-217M SAE circuit edges

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
  feature), otherwise "excitatory". 272,905 edges in `circuit_edges.csv`.

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

Rows: 698,624. Silenced genes: 227. Target genes: 6324.

## Other files

- `source_features.tsv`: the 120 source features (layer, feature id, top-10 genes).
- `edge_feature_genes.tsv`: top-10 genes of every feature that appears in an edge.
- `silenced_genes.tsv`: silenced genes that appear in `records.parquet`, with their number of K562 CRISPRi cells.

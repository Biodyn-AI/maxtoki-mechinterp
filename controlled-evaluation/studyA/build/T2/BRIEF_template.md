# Task brief

## 1. Question

Do the SAE feature circuits of MaxToki-217M predict the direction in which a gene's expression
changes when another gene is silenced by CRISPRi in K562 cells?

Give exactly one verdict:

- **supported**
- **not supported**
- **inconclusive**

## 2. Background

**The model.** MaxToki-217M is a transformer model of single-cell gene expression
(Gómez Ortega & Theodoris, bioRxiv 2026). It uses the Llama decoder architecture:
11 layers, 8 attention heads per layer, hidden size 1,232, about 217 million parameters.
It was trained on about 175 million human single-cell transcriptomes, first on single
cells and then on sequences of cell states. A cell is given to the model as a sequence
of gene tokens. Each gene's count is divided by that gene's median across a large
reference corpus, and genes are put in order from highest to lowest value (rank-value
encoding). Here each cell was given to the model on its own.

**Sparse autoencoders (SAEs).** Twelve TopK SAEs (4,928 features each, k = 32), one per
site. Site l (0 to 10) is the residual stream at the input of block l; site 11 is the
final normalised hidden state that enters the output head. Each SAE was trained on the
residual-stream vectors of 500 K562 non-targeting control cells. Each feature has a list
of top genes: the genes at whose token positions the feature is most active, on average,
in those 500 cells.

**The circuits.** 120 source features were traced: at each of the sites 0, 3, 6 and 9,
the 30 features whose top genes had the strongest gene-set enrichment. 200 K562
non-targeting control cells were used. For each cell and source feature at site s, the
feature's contribution to the SAE reconstruction was removed from the input of block s,
and the model was run on. At every later site, the change in every SAE feature's
activation (averaged over the cell's token positions) was recorded. Over the 200 cells,
each (source feature, target feature) pair gets a Cohen's d (mean change / SD of the
change) and a consistency (the share of cells whose change has the more common sign). A
pair is a circuit edge if |d| > 0.5 and consistency > 0.7. An edge is "inhibitory" if
removing the source feature lowers the target feature on average, and "excitatory"
otherwise.

**From circuits to genes (the method's rule).** The method pairs each top gene of an
edge's source feature with each top gene of its target feature. For each (source gene,
target gene) pair it counts the supporting edges, keeps pairs with enough support, and
predicts that silencing the source gene lowers the target gene when more than half of the
support comes from inhibitory edges, and raises it otherwise. `data/README.md` gives the
exact rule. `data/gene_pairs.parquet` holds the pairs it produces.

**The knockdowns.** K562 cells from the Replogle et al. 2022 CRISPRi Perturb-seq screen.
For a silenced gene and a measured gene, the log-fold change (LFC) is the mean of
log1p(counts per 10,000) over the cells carrying a guide against the silenced gene, minus
the same mean over 3,000 non-targeting control cells. 227 silenced genes (32 to 575 cells
each) are top genes of source features and appear in the gene pairs.

## 3. Materials

All paths are relative to this package folder.

| Path | What it is |
|---|---|
| `data/circuit_edges.csv` | 272,905 circuit edges: `src_layer`, `src_feature`, `tgt_layer`, `tgt_feature`, `cohens_d`, `consistency`, `sign`. |
| `data/source_features.tsv` | The 120 source features (site, feature id) with their top-10 genes. |
| `data/feature_top_genes.tsv` | Top-10 genes of every feature that appears in an edge (46,705 features). |
| `data/gene_pairs.parquet` | 698,624 (silenced gene, target gene) pairs made by the method's rule: support counts, the method's prediction (`predicted_decrease`) and the observed `lfc`. |
| `data/knockdown_lfc.npz` | LFC of all 6,546 measured genes for each of the 227 silenced genes, with cells per silenced gene and the control cells' mean expression. |
| `data/silenced_genes.tsv` | The 227 silenced genes and their number of CRISPRi cells. |
| `data/README.md` | How each data file was made, step by step, and every column. |
| `data/MANIFEST.sha256` | sha256 of the data files. |
| `methods/source_method_paper.pdf` | The source paper for this kind of analysis. It applies the analysis to other single-cell models. |

<<CONTRACT_SENTENCE>>
## 4. Deliverable

Return two things.

**(a) `report`**: a Markdown report with these sections, in this order:

1. **Question**
2. **Data**
3. **Methods**
4. **Results**: give every estimate with its uncertainty (for example a 95% confidence
   interval) and say what the unit of resampling was.
5. **Verdict**: one of supported / not supported / inconclusive, then 2-4 sentences of
   justification.
6. **Limitations**

**(b) `results`**: a JSON object with these fields:

- `verdict`: one of `"supported"`, `"not supported"`, `"inconclusive"`.
- `estimates`: a list of objects, each with `name`, `value`, `ci_low`, `ci_high`,
  `method_of_uncertainty`.
- `analysis_script`: a complete, self-contained Python script. When it is run from the
  package root with
  `<CODE_ROOT>/analysis-tasks/bin/python`, it must
  recompute every number reported in `estimates` from `data/` and print them as JSON.

## 5. Limits

- CPU only. Python with numpy, pandas (with pyarrow, for `.parquet` files), scipy and
  scikit-learn is available at
  `<CODE_ROOT>/analysis-tasks/bin/python`.
- Each command must finish within 9 minutes.
- Aim to finish within about 2 hours of work.
- Do not write files. Run Python inline.
- Read only inside this package folder.

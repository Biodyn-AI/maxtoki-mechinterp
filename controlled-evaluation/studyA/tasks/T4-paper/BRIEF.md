# Task brief

## 1. Question

Is MaxToki-217M's gene-embedding geometry aligned with scGPT's, and how does that alignment
compare with its alignment to Geneformer V2-316M (which shares MaxToki's gene vocabulary)?

Give exactly one verdict:

- **aligned with scGPT about as well as with Geneformer**
- **aligned with scGPT but clearly less than with Geneformer**
- **not aligned with scGPT beyond chance**
- **inconclusive**

## 2. Background

**The three models.** All three are transformer models of single-cell gene expression. Each
gives every gene its own input token, and each has a table with one learned vector per token
(the gene embedding table). The tables are compared here.

- **MaxToki-217M** (Gómez Ortega & Theodoris, bioRxiv 2026). Llama decoder: 11 layers,
  8 attention heads, hidden size 1,232, about 217 million parameters. It was trained on about
  175 million human single-cell transcriptomes, first on single cells and then on sequences of
  cell states. A cell is given to the model as a list of gene tokens: each gene's count is
  divided by that gene's median across a large reference corpus, and genes are put in order
  from highest to lowest value (rank-value encoding). Genes are identified by Ensembl gene ID.
- **Geneformer V2-316M** (Theodoris et al. 2023; V2 release). BERT encoder: 18 layers,
  18 attention heads, hidden size 1,152, about 316 million parameters. Trained by masked-gene
  prediction on about 104 million human single-cell transcriptomes, with the same rank-value
  encoding. Its gene tokens and token ids are the same as MaxToki's.
- **scGPT whole-human** (Cui et al. 2024). Transformer: 12 layers, 8 attention heads,
  embedding size 512. Trained on about 33 million human cells from CELLxGENE. Genes are
  identified by gene symbol. A gene's expression value is binned and encoded by a separate
  value encoder, which is added to the gene's token vector; the table here is the gene token
  part only.

**The tables.** Read directly from each released checkpoint (no model was run):

| Model | File | Shape |
|---|---|---|
| MaxToki-217M | `data/maxtoki_217m/embed_tokens.npy` | (20275, 1232) |
| Geneformer V2-316M | `data/geneformer_v2_316m/word_embeddings.npy` | (20275, 1152) |
| scGPT whole-human | `data/scgpt_whole_human/encoder_embedding_weight.npy` (+ its LayerNorm in `encoder_enc_norm.npz`) | (60697, 512) |

All are float32 and unitless. Row i of a table is the vector of that model's token id i. Each
model's own token dictionary or vocabulary file is next to its table. `data/gene_ids/ensembl_symbol.csv`
maps Ensembl gene IDs to gene symbols.

**The genes.** Three gene panels in `data/panels/` (382, 350 and 380 genes, as Ensembl IDs).
Each was built from one single-cell dataset (Tabula Sapiens lung, Tabula Sapiens immune, and a
separate lung dataset) by picking expressed genes with regulatory or marker annotations. Use
these panels as the gene sets for the comparison.

`data/README.md` describes every file and how it was made.

## 3. Materials

All paths are relative to this package folder.

| Path | What it is |
|---|---|
| `data/maxtoki_217m/` | MaxToki-217M token-embedding table, token dictionary (Ensembl ID to token id), model config. |
| `data/geneformer_v2_316m/` | Geneformer V2-316M token-embedding table, token dictionary (Ensembl ID to token id), model config. |
| `data/scgpt_whole_human/` | scGPT token-embedding table, its LayerNorm parameters, the checkpoint's `vocab.json` (gene symbol to token id) and `args.json`. |
| `data/gene_ids/ensembl_symbol.csv` | Ensembl gene ID to gene symbol table. |
| `data/panels/` | The three gene panels. |
| `data/README.md` | How each data file was made. |
| `data/MANIFEST.sha256` | sha256 of the data files. |
| `methods/source_method_paper.pdf` | The source paper for this kind of analysis. It applies the analysis to other single-cell models (scGPT and Geneformer); its cross-model alignment analysis is the relevant part. |

## 4. Deliverable

Return two things.

**(a) `report`**: a Markdown report with these sections, in this order:

1. **Question**
2. **Data**
3. **Methods**
4. **Results**: give every estimate with its uncertainty (for example a 95% confidence
   interval) and say what the unit of resampling was.
5. **Verdict**: one of the four verdicts above, then 2-4 sentences of justification.
6. **Limitations**

**(b) `results`**: a JSON object with these fields:

- `verdict`: one of `"aligned with scGPT about as well as with Geneformer"`,
  `"aligned with scGPT but clearly less than with Geneformer"`,
  `"not aligned with scGPT beyond chance"`, `"inconclusive"`.
- `estimates`: a list of objects, each with `name`, `value`, `ci_low`, `ci_high`,
  `method_of_uncertainty`.
- `analysis_script`: a complete, self-contained Python script. When it is run from the
  package root with
  `<EVAL_ROOT>/bin/python`, it must
  recompute every number reported in `estimates` from `data/` and print them as JSON.

## 5. Limits

- CPU only. Python with numpy, pandas, scipy and scikit-learn is available at
  `<EVAL_ROOT>/bin/python`.
- Each command must finish within 9 minutes.
- Aim to finish within about 2 hours of work.
- Do not write files. Run Python inline.
- Read only inside this package folder.

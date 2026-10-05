# Data

All paths are relative to this `data/` folder. No model was run to make these files.
The three embedding tables are parameter tensors read directly from the released model
checkpoints and saved as NumPy arrays (float32, unitless). Row i of a table is the vector
for token id i of that model.

## maxtoki_217m/

| File | What it is |
|---|---|
| `embed_tokens.npy` | float32, shape (20275, 1232). The tensor `model.embed_tokens.weight` of the MaxToki-217M Hugging Face checkpoint (`LlamaForCausalLM`). This is the input token-embedding table. Llama adds no position vector at the input, so this row is what the first layer receives for a token. |
| `token_dictionary.json` | The model's token dictionary: JSON object, key = token, value = token id. 23,277 entries: 20,271 genes keyed by Ensembl gene ID (`ENSG...`), 6 special tokens (`<pad>`, `<mask>`, `<bos>`, `<eos>`, `<boq>`, `<eoq>`), and 3,000 numeric tokens (the integers -1500 to 1499). Token ids 20,275 and above (all numeric tokens, `<boq>`, `<eoq>`) have no row in this table. |
| `config.json` | The checkpoint's model configuration (11 layers, 8 heads, hidden size 1,232, vocab size 20,275). |

## geneformer_v2_316m/

| File | What it is |
|---|---|
| `word_embeddings.npy` | float32, shape (20275, 1152). The tensor `bert.embeddings.word_embeddings.weight` of the Geneformer V2-316M checkpoint (Hugging Face `ctheodoris/Geneformer`, subfolder `Geneformer-V2-316M`, `BertForMaskedLM`). This is the token-embedding table before position embeddings and the embedding LayerNorm are added. |
| `token_dictionary.json` | The model's token dictionary (release file `token_dictionary_gc104M.pkl`, written as JSON; content unchanged). Key = Ensembl gene ID or special token, value = token id. 20,275 entries: 20,271 genes and 4 special tokens (`<pad>`, `<mask>`, `<cls>`, `<eos>`). |
| `config.json` | The checkpoint's model configuration (18 layers, 18 heads, hidden size 1,152, vocab size 20,275). |

## scgpt_whole_human/

| File | What it is |
|---|---|
| `encoder_embedding_weight.npy` | float32, shape (60697, 512). The tensor `encoder.embedding.weight` of the scGPT whole-human pretrained checkpoint (`best_model.pt`). |
| `encoder_enc_norm.npz` | The parameters of the LayerNorm `encoder.enc_norm` from the same checkpoint: `weight` (512,), `bias` (512,), and `eps` (1e-5, the PyTorch default used by this layer). In scGPT, the gene encoder takes a gene token's row of `encoder_embedding_weight` and applies this LayerNorm. Expression values enter the model through a separate value encoder; they are not part of this table. |
| `vocab.json` | The vocabulary file distributed with the checkpoint, copied byte for byte. JSON object, key = gene symbol (or one of the special tokens `<pad>`, `<cls>`, `<eoc>`), value = token id. 60,697 entries. |
| `args.json` | The training configuration file distributed with the checkpoint, copied byte for byte (12 layers, 8 heads, embedding size 512). |

## gene_ids/

| File | What it is |
|---|---|
| `ensembl_symbol.csv` | 63,675 rows, columns `ensembl_id`, `symbol`. Taken from the Geneformer release file `gene_name_id_dict_gc104M.pkl` (gene symbol to Ensembl gene ID), written as CSV. In this file each symbol has one Ensembl ID and each Ensembl ID has one symbol. |

## panels/

Three gene panels, one CSV each, with one column `ensembl_id`. Row order has no meaning.

| File | Genes | Source dataset |
|---|---|---|
| `lung.csv` | 382 | Tabula Sapiens, lung (Jones et al. 2022) |
| `immune.csv` | 350 | Tabula Sapiens, immune |
| `external_lung.csv` | 380 | Human lung, Smart-seq2 (Krasnow lab; Travaglini et al. 2020). A lung dataset separate from Tabula Sapiens. |

How the panels were made. For each dataset, cells were sampled at random. Candidate
genes had to be in MaxToki's vocabulary, be detected in the sampled cells, and be annotated
in at least one of: TRRUST v2 (as a transcription factor or as a target), a STRING
protein-protein interaction edge list, or a short list of canonical cell-type marker genes.
- lung and external_lung: 1,500 sampled cells; genes detected in at least 1% of them with
  non-zero variance. Chosen in priority order: up to 80 TRRUST transcription factors, then up
  to 150 TRRUST targets, then STRING genes until 350 genes were reached (each group ranked by
  expression variance), then every canonical marker gene that qualified.
- immune: 2,000 sampled cells; the 1,500 most variable genes of those cells were taken first,
  then restricted to annotated genes observed in the sample, then capped at 350 with priority
  transcription factor > target > STRING > marker only.

## MANIFEST.sha256

sha256 of every file in this folder.

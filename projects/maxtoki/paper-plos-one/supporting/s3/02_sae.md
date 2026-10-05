## Sparse autoencoders on correctly encoded inputs (MaxToki-217M, 12 layers)

A sparse autoencoder (SAE) rewrites each hidden-state vector as a sum of a few learned directions ("features"). This analysis trains one TopK SAE at each of the 12 residual-stream sites of MaxToki-217M, on hidden states from correctly encoded K562 control cells. It reports how well each SAE reconstructs held-out hidden states. Sections 03, 04 and 05 use these SAEs.

### Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, Apple MPS. torch 2.11.0, transformers 5.5.4.
- **Cells.** 500 K562 non-targeting control cells from `replogle_concat.h5ad`, drawn with numpy `default_rng(42)` from a pool of 10,691 and sorted.
- **Input.** Counts rebuilt and encoded with `inputs_v3` (section 00): counts / gene median, high to low, `<bos>` + up to 2,046 genes + `<eos>` (at most 2,048 tokens). 489 of 500 cells are cut at 2,046 genes. In total 1,019,996 token positions. One cell per forward pass.
- **Sites.** `hidden_states[0]` to `hidden_states[11]` (section 00), at every position, including `<bos>` and `<eos>`. Hidden size 1,232.
- **Split.** The first 90% of positions train the SAE (917,996 positions). The last 102,000 positions are held out. These are cells 450–499; cell 450 is only partly held out (1,648 of its 2,048 positions).

### Method

Each SAE encodes a hidden-state vector into 4,928 non-negative feature values, of which at most 32 are non-zero, and decodes them back. It is trained to make the decoded vector close to the input.

1. **SAE** (`projects/maxtoki/setup/topk_sae.py`). z = TopK₃₂(W_enc (x − μ) + b_enc), with negative kept values set to 0. x̂ = W_dec z + μ. μ is the mean of the training rows (summed in float64, stored as float32). d_model = 1,232, d_sae = 4,928, k = 32. W_dec starts as the transpose of W_enc, and its columns are scaled to unit L2 norm after every step.
2. **Training.** Loss = mean squared error between x and x̂ (no L1 term). Adam, learning rate 3 × 10⁻⁴, batch 4,096, 4 epochs (900 steps), `torch.manual_seed(42)`, shuffled by a PyTorch DataLoader. Training saves a checkpoint each epoch so it can resume; layers 0, 3, 4, 5, 6, 8 and 11 were resumed at least once. Training rows are read from disk by index to keep memory under about 8 GB; the shuffle order still comes from the same torch random generator.
3. **Evaluation on the 102,000 held-out positions.**
   - Fraction of variance explained (FVE) = 1 − (sum of squared reconstruction errors) / (sum of squared distances to the held-out mean). Sums run over all 1,232 dimensions and all held-out positions.
   - Dead features = features that never fire on the held-out positions. Also counted on all 1,019,996 positions.
   - Mean L0 = average number of non-zero features per position.
   - Intervals: 95% percentile interval from a cluster bootstrap over the 50 held-out cells (2,000 resamples, seed 20261001).
4. **Loading in later analyses.** The SAEs are saved as `layer_XX/sae_final.pt` in the format that `hooks_v2.load_saes(..., sae_dir=...)` reads.

### Checks

| Check | Result |
|---|---|
| Encoding check before every forward pass | `inputs_v3.assert_encoding_batch` on every cell of all 20 batches of 25 cells: 500 / 500 pass, 0 order violations (68 positions are exact ties in another order, which the check allows). The count rows are whole-number multiples of their smallest value in all 500 cells. Tokens ranked from the stored log1p values fail the same check in 500 / 500 cells. |
| Same cell rows as the atlas design | Encoding these 500 rows from the stored log1p values rebuilds the stored gene list of the atlas (`outputs/phase0/layer_00/gene_names.json`) at all 1,019,996 positions. |
| Hooks see the training activations | `hooks_v2.load_saes(range(12), sae_dir=outputs/v3_sae)` loads all 12 SAEs. On 3 held-out cells (451, 475, 499), the live input captured at every hook site equals the stored training activation exactly (largest difference 0.0, all 12 sites, including the input of `lm_head`). Reconstruction error from the live codes matches the evaluation within 3.4 × 10⁻⁹ (relative). Ablating all features at layer 5 changes the input by exactly the SAE reconstruction (largest error 9.5 × 10⁻⁷ on values up to 352). |
| Training code | Given the layer-0 inputs of the deployed atlas (rebuilt from its tokens without a forward pass, because layer 0 is the embedding lookup), the same training code reproduced the deployed layer-0 SAE: held-out FVE within 0.0001, the same number of dead features, epoch losses within 0.011%, and decoder columns that moved during training matched at median \|cos\| 0.998 (min 0.970). Weights are not bit-identical (relative difference 3–5%), most likely from float rounding that grows over 900 steps. |
| Second computation | Separate code (CPU, float64 sums, two-pass variance) reproduces every FVE within 1.5 × 10⁻⁸ and every L0 and dead count exactly. The FVE stored at the end of training equals the summary value (difference < 10⁻¹⁵). |
| Overfitting | FVE on the training positions is 0.701–0.929, which is 0.001–0.010 above the held-out FVE at each layer. |

### Results

Held-out reconstruction (n = 102,000 positions from 50 cells; 95% percentile interval, cluster bootstrap over the 50 held-out cells, 2,000 resamples):

| Layer | FVE [95% CI] | Dead features, held-out / all positions (of 4,928) | Mean L0 |
|---|---|---|---|
| 0 | 0.700 [0.696, 0.704] | 2,431 / 2,431 | 32.00 |
| 1 | 0.910 [0.908, 0.913] | 0 / 0 | 32.00 |
| 2 | 0.928 [0.926, 0.930] | 0 / 0 | 32.00 |
| 3 | 0.912 [0.909, 0.915] | 0 / 0 | 32.00 |
| 4 | 0.900 [0.897, 0.903] | 0 / 0 | 32.00 |
| 5 | 0.862 [0.858, 0.866] | 0 / 0 | 32.00 |
| 6 | 0.847 [0.842, 0.851] | 0 / 0 | 32.00 |
| 7 | 0.847 [0.842, 0.851] | 5 / 0 | 32.00 |
| 8 | 0.842 [0.836, 0.847] | 18 / 0 | 32.00 |
| 9 | 0.859 [0.853, 0.863] | 12 / 0 | 32.00 |
| 10 | 0.869 [0.864, 0.874] | 4 / 0 | 32.00 |
| 11 | 0.871 [0.865, 0.876] | 4 / 0 | 32.00 |

- The SAEs explain 70% of the held-out variance at layer 0 and 84–93% at layers 1–11.
- No feature is dead on all positions at layers 1–11. On the held-out positions alone, 4–18 features are dead at layers 7–11.
- Layers 6 and 7 have the same FVE by chance (0.84715 and 0.84717). Their data and SAEs differ (hashes in `eval/meta.json`).
- Layer 0 is special. `hidden_states[0]` is the embedding row of each gene token, and the 500 cells contain 6,330 distinct tokens. So layer 0 sees only 6,330 distinct vectors, and 2,431 features never fire.
- Mean L0 is exactly 32.00 at every layer and in every bootstrap resample. A TopK SAE keeps 32 features per position unless some of the top 32 are ≤ 0, and that never happened. So L0 carries no information here.
- For context, the SAEs of the deployed atlas, run on the same held-out tokens, explain 0.667 at layer 0, 0.848 at layer 1, and less than zero at layers 8–10 (−0.041, −0.131, −0.012).

### Verification

No separate verification by a second agent is recorded for this analysis. The report itself describes these cross-checks (all in the Checks table): an independent recomputation of every FVE, L0 and dead count with separate CPU code (`scripts/v3_sae_verify.py`, `verify/recompute.json`); the hook check on 3 held-out cells (`hooks_check.json`); and the layer-0 reproduction of the training protocol (`verify/l0_protocol_compare.json`). All agreed.

### Limits

- One training seed. There is no seed-to-seed spread for FVE, dead counts or which features exist.
- One cell type. Training and held-out positions come from the same 500 K562 control cells. The intervals cover cell-to-cell variation among the 50 held-out cells only. They do not cover other cell types or perturbed cells.
- One SAE size (4,928 features, k = 32). Other sizes and sparsity levels were not tried.
- Only reconstruction was measured. Whether features are interpretable was not tested here.
- The training-code check covers layer 0 only. Other layers would need forward passes on the deployed inputs, which were not run.

### Files

- Code: `projects/maxtoki/runs/sae-atlas-217M/scripts/v3_sae.py` (sub-commands: prepare, extract, train, evaluate, hooks check, summary), `projects/maxtoki/runs/sae-atlas-217M/scripts/v3_sae_verify.py`; shared code `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`, `projects/maxtoki/setup/topk_sae.py`.
- SAEs: `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/layer_XX/sae_final.pt` (with `results.json` and `training_log.json`), XX = 00 to 11.
- Evaluation: `.../outputs/v3_sae/eval/summary.json`, `eval/table.csv`, `eval/layer_XX.npz` (per-cell sums), `eval/meta.json` (data and SAE hashes).
- Checks: `.../outputs/v3_sae/hooks_check.json`, `verify/recompute.json`, `verify/l0_protocol_compare.json`, `verify/l0_protocol/layer_00/`.
- Inputs and activations: `.../outputs/v3_sae/cells.npz` (rows, offsets, tokens), `gene_names_v3.json`, `activations/layer_XX.npy` (training activations, 60.3 GB) and `activations_eval/layer_XX.npy` (held-out rows, 6.0 GB), which are not deposited because of their size (ZENODO_MANIFEST.csv says how to rebuild them), and `codes/layer_XX_topk.npz` (top-32 codes at all 1,019,996 positions, 2.35 GB).
- Provenance: `.../outputs/v3_sae/run_config.json` (device, versions, seeds, cell rows, encoding-check result, input and code sha256, wall time per chunk), `prepare.json`, `chunks.json`, `verify/run_config.json`, `logs/`.
- Run time (MPS): extraction 17.5 min, training about 65 min for 12 layers, evaluation 13 min, checks about 6 min.

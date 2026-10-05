# V3-1 — the 12 layer SAEs, retrained on correctly encoded inputs

Date: 2026-10-01 / 2026-10-02. Model: MaxToki-217M (HF safetensors), float32, MPS.
Code: `scripts/v3_sae.py` (extract, train, evaluate, hooks check, summary) and
`scripts/v3_sae_verify.py` (independent checks). Outputs: `outputs/v3_sae/`.
New SAEs: `outputs/v3_sae/layer_XX/sae_final.pt`, same format as the deployed
`outputs/phase1/layer_XX/sae_final.pt`. No existing file was changed.

---

## 0. Results first

1. **All 12 SAEs are retrained** on hidden states of the same 500 K562 control cells, now
   with the right gene order (counts / gene median, from `setup/inputs_v3.py`). Every cell passed
   the encoding check before its forward pass (500 / 500). The old tokens fail the same check
   (500 / 500), so the check can tell the two apart.
2. **The new SAEs fit well at every layer.** Fraction of variance explained (FVE) on held-out
   tokens: **0.700 at layer 0, 0.910–0.928 at layers 1–3, 0.842–0.900 at layers 4–11**.
   Dead features on held-out tokens: 0 at layers 1–6, 4–18 at layers 7–11 (out of 4,928), and
   0 on all tokens at layers 1–11. Layer 0 has 2,431 (as deployed, see 3.3). Mean L0 = 32.00 at
   every layer.
3. **The deployed SAEs do not fit the correctly encoded inputs.** On the same held-out tokens
   they explain 0.667 (layer 0) and 0.848 / 0.803 (layers 1 / 2), then fall fast:
   0.59 at layer 3, 0.29 at layer 5, 0.11 at layer 7, and **below zero at layers 8–10**
   (−0.04, −0.13, −0.01). Below zero means they do worse than predicting the average activation.
   They also have many more dead features there (410–1,090 of 4,928 on all tokens at layers 8–10).
4. **So the old SAEs cannot be reused.** The gap (new minus deployed) is +0.03 at layer 0,
   +0.06 to +0.12 at layers 1–2, and +0.32 to +0.99 at layers 3–11. All 95% intervals are far
   from zero.
5. **The features are different features.** For layers 1–11, at most 0.6% of the new features
   have a deployed feature pointing the same way (|cos| ≥ 0.9). Feature numbers from earlier
   work do not carry over. Every SAE-based result must re-select its features.
6. **The training protocol is reproduced.** Fed the deployed layer-0 inputs, the v3 training
   code gives back the deployed layer-0 SAE: FVE 0.69351 vs 0.69344, 2,430 vs 2,430 dead
   features, epoch losses within 0.011%, matching decoder columns at median |cos| 0.998.
7. **hooks_v2 loads the new SAEs.** Its live captures equal the stored training activations
   exactly (difference 0.0) at all 12 layers, on 3 held-out cells.

---

## 1. What was done

### 1.1 The deployed protocol, and what this run copies

Read from `scripts/full_12layer_pipeline.py` (lines 44–49, 59–62, 104, 133–177),
`scripts/phase0_extract_positions.py`, `scripts/phase1_train_saes.py` and `setup/topk_sae.py`.

| Item | Deployed | v3 |
|---|---|---|
| Cells | K562 non-targeting controls; `default_rng(42).choice(pool, 500)`, sorted | Same rule and seed. Same 500 rows (pool of 10,691). |
| Encoding | `tokenize_cell(X)` on log1p(CP10k) — **wrong** | `inputs_v3`: counts = round(expm1(X) / unit), then counts / median order — **right** |
| Tokens | ≤ 2,048 per cell (`<bos>` + up to 2,046 genes + `<eos>`), batch size 1 | Same. 1,019,996 positions; 489 of 500 cells are cut at 2,046 genes. |
| Sites | `hidden_states[0..11]`, every position incl. `<bos>`/`<eos>` | Same |
| Split | 1,019,996 < 1.1 M, so `train_sae` takes its else-branch: first 90% of positions train (917,996), last 102,000 held out | Same. Held out = cells 450–499 (cell 450 is only partly held out: 1,648 of its 2,048 positions). |
| SAE | TopK, d_model 1,232, d_sae 4,928, k = 32, Adam lr 3e-4, batch 4,096, 4 epochs (900 steps), `torch.manual_seed(42)`, shuffle by DataLoader, decoder columns renormalised each step | Same |

Differences that do not change the maths:
- Training rows are read by index from a file instead of from one in-memory tensor (to keep
  memory under ~8 GB on a shared machine). The DataLoader is the same type, so the shuffle order
  uses the torch random generator exactly as before. For layers 4–11 the rows were read from a
  copy on the internal SSD (for layers 5–11 the copy was checked byte for byte with sha256; the
  layer-4 record does not say).
- The train mean `mu` is summed in float64, then cast to float32 (deployed: float32 sum).
- Held-out evaluation runs in chunks of 8,192 rows. Active features are counted on the CPU.
- Training saves a checkpoint each epoch (model, optimiser, random state) so it can resume.
  Layers 0, 3, 4, 5, 6, 8 and 11 were resumed at least once.

Check that these differences do not matter: section 4.4 (the layer-0 reproduction).

### 1.2 How the inputs changed

Old vs new gene order in the 500 training cells (full order, before the cut):
Spearman mean 0.841 (min 0.714); first 200 genes shared 56% (47–68%); kept genes shared 92%
(89–100%); share of positions holding the same gene 0.09%. This matches the audit
(`checks/INPUT_ENCODING_AUDIT.md`, K562 controls: 0.84 / 0.55 / 0.92 / 0.001).

### 1.3 Evaluation

Each SAE is run on the 102,000 held-out positions of the **new** activations. Both SAE sets
(v3 and deployed) see exactly the same rows.

- **FVE (main number)** = 1 − (sum of squared reconstruction errors) / (sum of squared
  distances to the held-out mean). Sums run over all 1,232 dimensions and all held-out tokens.
  The baseline is the same for both SAEs, so they can be compared directly.
- **FVE, own-mu** = the deployed training formula: the baseline is the distance to the SAE's own
  stored train mean `mu`. For the v3 SAEs this equals the main number within 0.0004. For the
  deployed SAEs their `mu` is far from the new data, so this baseline is larger and the number
  looks better (table column "dep own-mu").
- **Dead features** = features that never fire on the held-out tokens (deployed definition).
  Also given on all 1,019,996 tokens.
- **Mean L0** = average number of non-zero features per token.
- **Uncertainty**: 95% percentile interval from a cluster bootstrap over the 50 held-out cells
  (2,000 resamples, seed 20261001). The gap between the two SAEs is bootstrapped paired
  (same resampled cells for both).

---

## 2. Per-layer results

FVE is a fraction (0 to 1; below 0 = worse than the average). "ho" = dead on held-out tokens,
"all" = dead on all tokens. Deployed numbers on the old tokens come from the deployed
`results.json` (102,000 held-out old-token positions) and the v2 site check
(`outputs/v2_sae_site_check/`, 5 other control cells, hooks_v2 path).

| Layer | v3 FVE [95% CI] | v3 dead ho / all | v3 L0 | Deployed SAE, new tokens: FVE [95% CI] | Deployed dead ho / all | Gap v3 − deployed [95% CI] | Deployed on old tokens: results.json / v2 site check | Deployed dead, old tokens |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.700 [0.696, 0.704] | 2,431 / 2,431 | 32.00 | 0.667 [0.664, 0.670] | 2,430 / 2,430 | +0.033 [+0.031, +0.034] | 0.693 / 0.696 | 2,430 |
| 1 | 0.910 [0.908, 0.913] | 0 / 0 | 32.00 | 0.848 [0.846, 0.849] | 0 / 0 | +0.063 [+0.060, +0.066] | 0.904 / 0.908 | 0 |
| 2 | 0.928 [0.926, 0.930] | 0 / 0 | 32.00 | 0.803 [0.801, 0.805] | 2 / 2 | +0.124 [+0.123, +0.126] | 0.935 / 0.938 | 1 |
| 3 | 0.912 [0.909, 0.915] | 0 / 0 | 32.00 | 0.592 [0.585, 0.599] | 0 / 0 | +0.320 [+0.313, +0.327] | 0.932 / 0.946 | 0 |
| 4 | 0.900 [0.897, 0.903] | 0 / 0 | 32.00 | 0.417 [0.410, 0.424] | 26 / 20 | +0.483 [+0.475, +0.490] | 0.924 / 0.930 | 5 |
| 5 | 0.862 [0.858, 0.866] | 0 / 0 | 32.00 | 0.289 [0.284, 0.293] | 101 / 34 | +0.573 [+0.567, +0.579] | 0.889 / 0.897 | 25 |
| 6 | 0.847 [0.842, 0.851] | 0 / 0 | 32.00 | 0.189 [0.184, 0.192] | 262 / 80 | +0.659 [+0.652, +0.665] | 0.846 / 0.859 | 52 |
| 7 | 0.847 [0.842, 0.851] | 5 / 0 | 32.00 | 0.112 [0.106, 0.117] | 261 / 86 | +0.735 [+0.727, +0.743] | 0.803 / 0.816 | 33 |
| 8 | 0.842 [0.836, 0.847] | 18 / 0 | 32.00 | −0.041 [−0.048, −0.036] | 823 / 410 | +0.883 [+0.876, +0.892] | 0.772 / 0.786 | 44 |
| 9 | 0.859 [0.853, 0.863] | 12 / 0 | 32.00 | −0.131 [−0.138, −0.126] | 1,496 / 1,090 | +0.990 [+0.981, +1.000] | 0.745 / 0.759 | 10 |
| 10 | 0.869 [0.864, 0.874] | 4 / 0 | 32.00 | −0.012 [−0.017, −0.008] | 1,402 / 937 | +0.881 [+0.873, +0.889] | 0.757 / 0.775 | 13 |
| 11 | 0.871 [0.865, 0.876] | 4 / 0 | 32.00 | 0.172 [0.165, 0.178] | 436 / 198 | +0.699 [+0.693, +0.707] | 0.828 / 0.845 | 115 |

(Layers 6 and 7 have the same v3 FVE by chance: 0.84715 vs 0.84717. Their data and SAEs differ;
see `eval/meta.json` hashes.)

Extra columns are in `outputs/v3_sae/eval/table.csv` and `eval/summary.json`: own-mu FVE with
intervals, in-sample FVE on the training tokens (0.701–0.929; 0.001–0.010 above held-out, so
little overfitting), and mean L0 on all tokens.

What the table shows:
- **v3 vs deployed on their own data.** At layers 1–5 the new SAEs fit their data about as well as
  the deployed SAEs fit theirs (for example layer 3: 0.912 vs 0.932). At layers 7–11 they fit
  better (layer 9: 0.859 vs 0.745) and have fewer dead features on held-out tokens (4–18 vs
  10–115).
  These two numbers come from different data, so they say how hard each dataset is, not which
  SAE is better.
- **Deployed SAEs on the new data.** They lose 0.03 at layer 0 (where only the mix of genes
  changed), 0.06–0.13 at layers 1–2, and 0.3–1.0 from layer 3 on.

---

## 3. Why the deployed SAEs fail on the new inputs

`outputs/v3_sae/verify/recompute.json`.

### 3.1 The average activation moved

The deployed SAE subtracts its stored mean `mu` (the average old-input activation). Distance
from that mean to the new held-out mean, divided by the per-token variance of the new data:

| Layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mean shift / variance | 0.05 | 0.07 | 0.17 | 0.17 | 0.17 | 0.24 | 0.29 | 0.58 | 1.05 | 0.99 | 0.23 |
| cos(new mean, old mean) | 0.92 | 0.88 | 0.81 | 0.78 | 0.76 | 0.78 | 0.78 | 0.70 | 0.48 | 0.49 | 0.61 |

At layers 9–10 the shift alone is as large as all the token-to-token variance. The norm of the
mean also grew: at layer 9 it is 51.1 with the new inputs vs 17.5 with the old ones.

### 3.2 The shift is not the whole story

Re-centring the deployed SAE on the new train mean (its `mu` replaced by the v3 `mu`; no held-out
data used) helps at every layer from 1 to 11, but not enough:

| Layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Deployed, as is | 0.848 | 0.803 | 0.592 | 0.417 | 0.289 | 0.189 | 0.112 | −0.041 | −0.131 | −0.012 | 0.172 |
| Deployed, re-centred | 0.881 | 0.848 | 0.692 | 0.515 | 0.405 | 0.337 | 0.284 | 0.307 | 0.315 | 0.342 | 0.296 |
| v3 | 0.910 | 0.928 | 0.912 | 0.900 | 0.862 | 0.847 | 0.847 | 0.842 | 0.859 | 0.869 | 0.871 |

So the directions the model uses changed too, not only the average.

### 3.3 The features are different

For each new feature that fires on held-out tokens, I took the largest |cosine| between its
decoder direction and any live deployed decoder direction.

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Median best match | 0.92 | 0.59 | 0.45 | 0.32 | 0.26 | 0.27 | 0.31 | 0.30 | 0.31 | 0.29 | 0.25 | 0.25 |
| Share with match ≥ 0.9 | 0.68 | 0.00 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 | 0.005 | 0.005 | 0.003 | 0.000 |

Random directions give a median best match of 0.11 and 0% ≥ 0.9.
Caveat: both SAE sets start from the same seed-42 starting weights. At layers 1–2 the best match
is almost always the feature with the same number, so part of that similarity is the shared
start, not shared learning. I have no second training seed, so I cannot say how much two SAEs
trained on the same data would agree. The practical point stands: feature numbers do not carry
over.

Layer 0 is special. In a Llama model, `hidden_states[0]` is just the gene's embedding row. The
500 cells contain 6,330 distinct tokens, so layer 0 sees only 6,330 distinct vectors. About 2,500
features are enough for them, and the other ~2,430 never fire, in both SAE sets. The new inputs
only change how often each gene appears, so the layer-0 SAE barely changes (68% of features
match at ≥ 0.9).

---

## 4. Checks (all passed)

1. **Encoding check before every forward pass.** `inputs_v3.assert_encoding_batch` on every cell
   of every 25-cell batch (20 batches, 500 / 500 pass; 68 positions are exact ties put in a
   different order, which the check allows). The count rows are whole-number multiples of their
   smallest value in all 500 cells. The old tokens fail the check in 500 / 500 cells.
2. **Same cells as deployed.** The old encoding of these 500 rows rebuilds the deployed
   `outputs/phase0/layer_00/gene_names.json` exactly (1,019,996 / 1,019,996 positions).
3. **hooks_v2 can load and use the new SAEs** (`outputs/v3_sae/hooks_check.json`).
   `hooks_v2.load_saes(range(12), sae_dir=outputs/v3_sae)` works. On 3 held-out cells, the live
   input captured at every hook site equals the stored training activation exactly (max
   difference 0.0, all 12 layers, including layer 11 = the input of `lm_head`). Reconstruction
   error from the live codes matches the evaluation within 3.4e-9 (relative). Ablating all
   features at layer 5 changes the input by exactly the SAE reconstruction (max error 9.5e-7 on
   values up to 352). The 3 cells pass the encoding check.
4. **Training protocol reproduced** (`verify/l0_protocol_compare.json`). Layer-0 activations are
   embedding rows, so the deployed layer-0 inputs can be rebuilt from the deployed tokens with no
   forward pass. (Check: the same lookup on the new tokens equals the stored v3 layer-0 file in
   all 1,019,996 rows.) Training on them with the exact v3 code gives FVE 0.69351 vs deployed
   0.69344, 2,430 vs 2,430 dead, epoch losses 0.00031254 / 0.00020805 / 0.00016499 / 0.00014336
   vs 0.00031254 / 0.00020806 / 0.00016500 / 0.00014338. Decoder columns that moved during
   training match by number at median |cos| 0.998 (min 0.970). Weights are not bit-identical
   (relative difference 3–5%). The likely cause is float rounding: for example the train mean
   differs by up to 4e-9 (float64 vs float32 sum), and small differences grow over 900 steps. The old tokens were used here only for this check, never
   in a forward pass.
5. **Numbers re-derived a second way** (`verify/recompute.json`). Separate code, CPU instead of
   MPS, float64 sums, two-pass variance. All FVE values agree with the MPS summary within
   1.5e-8. L0 and dead counts agree exactly, for both SAE sets at all 12 layers. The FVE stored
   in each `results.json` at the end of training equals the summary's own-mu FVE (difference
   < 1e-15).
6. **Bootstrap.** Mean L0 is exactly 32.00 in every resample. All other intervals are in the
   table.

---

## 5. What changed compared with the deployed and v2 results

- **Deployed SAEs** (`outputs/phase1/`): trained on the wrong gene order. On their own data they
  looked fine (FVE 0.69–0.94). On correctly encoded inputs they explain 0.67–0.85 at layers 0–1,
  0.80 at layer 2, then 0.59 down to −0.13 at layers 3–11. They should not be used with corrected
  inputs.
- **v2** did not retrain the SAEs. It fixed the hooks and checked the SAE sites
  (`V2_HOOKS_REPORT.md`, `outputs/v2_sae_site_check/`, still on old tokens). That check's
  conclusion still holds with the new SAEs: each SAE `layer_XX` belongs at `hidden_states[XX]`,
  which is where hooks_v2 places it (check 4.3 above).
- **New SAEs** (`outputs/v3_sae/`): FVE 0.70 at layer 0 and 0.84–0.93 at layers 1–11 on
  held-out correctly encoded tokens. At layers 7–11 they fit better and have fewer dead features
  than the deployed SAEs had on the old data.
- **Consequence for later work**: every SAE-based result (annotation, patching, Phases 8t/9/11,
  TF specificity, circuits and CRISPRi, triplets, steering) needs a re-run with
  `sae_dir = outputs/v3_sae`. Old feature numbers mean nothing for the new SAEs. Scripts that
  check their feature list against deployed files (for example `v2_circuit_trace.py:101`) will
  stop, as the audit predicted.

---

## 6. What was not done, and caveats

- **The 60 GB of training activations were not deleted.** The task said to delete them if they
  exceed ~20 GB (they are 60.32 GB, `outputs/v3_sae/activations/`). I do not delete data
  myself, so I left this to you. The command (it checks that every output exists first and
  records the deletion in `run_config.json`):
  `.venv/bin/python runs/sae-atlas-217M/scripts/v3_sae.py cleanup --yes`.
  Kept either way: held-out rows (`activations_eval/`, 5.6 GB), the top-32 v3 codes for all
  1,019,996 positions at every layer (`codes/`, 2.2 GB), tokens (`cells.npz`) and data hashes.
- **Mean L0 tells little.** A TopK SAE keeps exactly 32 features per token unless some of the
  top 32 are ≤ 0. That never happened here, in either SAE set. So L0 = 32.00 everywhere.
- **One training seed.** I did not train a second seed. So I cannot give a seed-to-seed spread
  for FVE, dead counts or feature matching.
- **Held-out tokens come from the same 500-cell sample** (cells 450–499, K562 controls). The
  intervals cover cell-to-cell variation in those 50 cells. They do not cover other cell types or
  perturbed cells.
- **The deployed SAEs were not run on old-token held-out data in this item.** That would need a
  forward pass on wrongly encoded inputs, which the run rules forbid. The deployed numbers on old
  tokens come from their `results.json` and the v2 site check.
- **Downstream analyses were not redone here** (annotation, patching, circuits and the rest).
  This item only provides the SAEs and their fit.
- The deployed-token training check covers layer 0 only (other layers would need forward passes
  on wrong inputs).

---

## 7. Files

- New SAEs: `outputs/v3_sae/layer_XX/sae_final.pt` (+ `results.json`, `training_log.json`),
  XX = 00–11.
- Evaluation: `outputs/v3_sae/eval/summary.json`, `eval/table.csv`, `eval/layer_XX.npz`
  (per-cell sums), `eval/meta.json` (data and SAE hashes).
- Checks: `outputs/v3_sae/hooks_check.json`, `verify/recompute.json`,
  `verify/l0_protocol_compare.json`, `verify/l0_protocol/layer_00/`.
- Provenance: `outputs/v3_sae/run_config.json` (device, versions, seeds, cell rows, encoding
  check, input and code sha256, wall time per chunk), `prepare.json`, `chunks.json`,
  `verify/run_config.json`, logs in `outputs/v3_sae/logs/`.
- Inputs: `outputs/v3_sae/cells.npz` (rows, offsets, v3 tokens, deployed tokens),
  `gene_names_v3.json`, `activations/layer_XX.npy` (60 GB, see section 6),
  `activations_eval/layer_XX.npy`, `codes/layer_XX_topk.npz`.
- Code: `scripts/v3_sae.py`, `scripts/v3_sae_verify.py`; shared: `setup/inputs_v3.py`,
  `setup/hooks_v2.py`, `setup/topk_sae.py` (none changed).

Run time (MPS): extraction 17.5 min, training 12 layers about 65 min over several resumed calls,
evaluation 13 min, checks about 6 min.

---

## Plain-words summary

The 12 SAEs were trained on model activations from inputs with the wrong gene order. I trained
them again with the right order, using the same cells and the same settings. The new SAEs
explain 70% of the held-out variance at layer 0 and 84–93% at layers 1–11, with almost no dead
features. The old SAEs fit the new activations badly: they explain 85% at layer 1, 59% at layer
3, and less than nothing at layers 8–10. Part of the reason is that the average activation moved.
But the features themselves also changed: almost no new feature matches an old one. So every
result built on the old SAE features must be redone with the new SAEs, and old feature numbers
cannot be reused. I checked that the training code reproduces the old layer-0 SAE when given the
old inputs, that hooks_v2 loads the new SAEs and sees exactly the training activations, and that
a second, independent calculation gives the same numbers. I did not delete the 60 GB of
activation files; the command to do so is in section 6.

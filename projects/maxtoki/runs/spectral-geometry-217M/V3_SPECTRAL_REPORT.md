# V3 spectral report — effective rank and cross-sample CKA with correct inputs and a random-init control (item V3-5)

Date: 2026-10-02. Model: MaxToki-217M (HF safetensors, float32, eager attention, Apple MPS). torch 2.11.0,
transformers 5.5.4, numpy 1.26.4. Inputs: `setup/inputs_v3.py` (raw counts / gene median order). Hooks: `setup/hooks_v2.py`
(memory guard, model loader, and the layer check in V7; no edits are made in this item).
Code: `scripts/v3_spectral.py` (cells, extraction, metrics, CKA, checks) and `scripts/v3_spectral_intervals.py`
(jackknife intervals, tables). Every number below comes from the files in section 9.

Words used:
- **Effective rank (ER)**: how many directions a set of vectors really uses. Here: exp of the entropy of the
  squared singular values of the gene × hidden matrix (1,500 genes × 1,232 dimensions), columns centred. It is a
  count of dimensions, with no unit. Same code as the deployed `phase1_svd.py`.
- **CKA** (linear centred kernel alignment): how similar two sets of vectors are for the same genes.
  1 = same geometry. Same code as the deployed `phase8_stability.py`.
- **Per-gene vector**: the hidden state at a gene's token, averaged over all cells in which the gene was among the
  2,046 kept genes. L0 = token embedding; L1–L10 = output of blocks 0–9; L11 = final norm applied to the output of
  block 10.

## 0. Results first

1. **The deployed numbers come back exactly with the same code.** On the stored deployed embeddings: ER
   560.8 (L0) → 93.6 (L11), feature-shuffle null 573.1, CKA 1.000 (L0) → 0.9914 (L11). These are the paper's
   561 → 94, 573 and 1.000 → 0.991. They were recomputed from the stored files on CPU. No wrong input was fed to the
   model (the workflow forbids it).
2. **With correct inputs the final-layer ER is 2.4 times higher: 227, not 94.** Same 2,000 cells, same 1,500 genes,
   same code; only the gene order fed to the model changed. ER goes 561 (L0) → 328 (L1) → 227 (L11)
   [95% CI 225–230; jackknife over cells]. Two more disjoint 2,000-cell samples give 226 and 226. So most of the
   deployed drop to 94 came from the wrong input order.
3. **The fall in ER with depth is a property of the architecture, not of training.** Three random-initialised
   MaxToki-217M models, on the same cells, go 816 (L0) → 642–645 (L1) → 390–421 (L11). From L1 to L11 they lose
   **35–40%** of their ER. The trained model loses **31%** (30.8–31.3% across 3 samples). The 95% intervals do not
   overlap (trained at most 31.8%, random at least 32.5%). So depth alone, with random weights, compresses at
   least as much.
   - What training does change: ER is lower at every layer, starting at the embedding table (561 vs 816), and the
     first block cuts ER more (−41% vs −21%). Measured from L0, the trained model loses 59.5% by L11 and the random
     models 48–52%; that whole difference sits in L0 → L1.
   - The feature-shuffle null cannot separate the two cases. It gives near-full rank for every model (trained 646,
     random 809–812 at L11). So the paper's "6.1× gap rules out an architectural cause" does not follow.
   - One measure does separate them: local intrinsic dimension (TwoNN, also computed by the deployed code) falls
     from 102 to 5.3 in the trained model but stays at 107–120 in all random models. The paper does not quote it.
4. **Cross-sample CKA, on three truly disjoint samples, from L1:** trained **0.9997 (L1) → 0.9962 (L11)**
   [L11 95% CI 0.9959–0.9964 over genes; 0.9958–0.9966 over cells]. Deployed encoding: 0.9997 → 0.9914
   [0.9906–0.9922]. Random init (seed 0, same three samples): **0.9938 → 0.9647** [0.960–0.970]. So the trained
   model's per-gene vectors are clearly more stable across cell samples than a random model's.
5. **What these CKA values mean.** They measure how much a gene's average vector moves when it is averaged over
   a different random set of cells from the same tissue atlas. Halving the cells (1,000 vs 1,000) almost exactly
   doubles the distance from 1 (trained L11: 0.0038 → 0.0073). So nearly all of the gap from 1 is sampling noise in
   the averages. The scGPT numbers the paper compares with (0.979 → 0.779) compare different fine-tuned models,
   so the comparison is not like for like.
6. **All checks passed** (section 3). Every one of the 16,000 cells passed the encoding check before its forward
   pass. ER and CKA were re-derived by a second route; they agree to 3 decimals (ER) and within 4 × 10⁻⁶ (CKA).

---

## 1. Why a v3

- `checks/INPUT_ENCODING_AUDIT.md` found that the deployed Phase 0 (`scripts/phase0_extract.py`) and Phase 8
  (`scripts/phase8_stability.py`) fed `tokenize_cell(X)` with `X` = log1p(CP10k). MaxToki expects genes ranked by
  counts / gene median. For the 2,000 deployed cells (sample A) the old and correct orders agree at Spearman 0.843
  (UMI cells, n = 1,927; lowest 0.216). Only 60% of the first 200 genes are the same; 94% of the kept genes are the
  same. **None** of the 2,000 token sequences is the same under both encodings.
- Two design problems (revision plan E36, E37): the three Phase 8 samples shared 6, 8 and 8 cells, so they were
  not disjoint; and CKA at L0 is 1 by construction (L0 is the fixed token embedding). A feature-shuffle null
  cannot tell learned compression from architectural compression; that needs a random-weight model.
- No v2 re-run of spectral ER or CKA exists. The only v2 spectral output (`outputs/v2_crossmodel`) compares static
  embedding tables and does not use model inputs, so it is not affected.

## 2. What was run

| Part | Choice |
|---|---|
| Dataset | Tabula Sapiens immune (`tabula_sapiens_immune.h5ad`, 592,317 cells), as deployed |
| Input | `raw/X` integer counts → counts / row sum × 10,000 / Geneformer gene median → rank → first 2,046 genes (inputs_v3) |
| Sample A | the deployed Phase 0 cells: `default_rng(42).choice(592317, 2000)`; cell types match `phase0/cell_metadata.csv` 2,000/2,000 |
| Samples B, C | deployed Phase 8 draws (seeds 43, 44) with every cell already used by an earlier sample replaced by a new draw (`default_rng(20261002)` / `(20261003)`) from cells no deployed sample used. B: 6 cells replaced, C: 16. Pairwise overlap now 0. |
| Gene panel | the deployed 1,500 genes (`phase0/gene_features.csv`). Re-derived from the deployed rule: 1,500/1,500 match |
| Second panel (sensitivity) | same rule, but on log1p(CP10k) of raw counts (the deployed panel was picked on a double log): 1,426 of 1,500 genes shared |
| Context | 2,048 tokens, batch 1, float32, eager attention, MPS (as deployed) |
| Models | trained checkpoint; random init seeds 0, 1, 2 (`torch.manual_seed(s)`, `LlamaForCausalLM(config)`: weights normal(0, 0.02), norm weights 1; state-dict sha256 recorded and re-checked in every chunk) |
| Jobs | trained on A, B, C; random seeds 0, 1, 2 on A; random seed 0 also on B and C (CKA control) — 8 × 2,000 = 16,000 forward passes |
| Blocks | each sample split into 10 random blocks of 200 cells (`default_rng(20261005 + k)`), for cell-level intervals; Smart-seq2 cells also summed apart (UMI-only check) |
| Metric code | `_effective_rank`, `_svd_and_metrics`, `_two_nn`, `_participation_ratio`, `linear_cka` copied from the deployed scripts; AST-identical (check V1) |
| Intervals | jackknife (delete one group): over cells = 10 blocks of 200; over genes = 30 random groups of 50 genes. 95% CI = estimate ± t(0.975, g−1) × SE. Reasons in section 6.3 |
| Wall time | 10,134 s of MPS (2.8 h) for the 16,000 passes, in 30 job chunks inside calls of ≤ 7.5 min; plus about 1 h of CPU analysis |

## 3. Checks

| # | Check | Result |
|---|---|---|
| E1 | Encoding check before every forward pass (`tokenize_counts(check=True)` + `assert_encoding_batch` on every 50-cell batch; tokens must equal the tokens saved by `prepare`) | 16,000 / 16,000 cells in 320 batches passed |
| E2 | Count check on every input row (`raw/X` whole numbers) | 6,000 / 6,000 rows |
| V1 | Copied metric code is identical (AST) to `phase1_svd.py` and `phase8_stability.py` | 5 / 5 functions identical |
| V2 | Samples disjoint; A = deployed draw | overlaps 0 / 0 / 0; A identical |
| V3 | Independent re-tokenisation (own code, float64) of 120 random cells | 115 identical; 5 differ only inside exact ties |
| V4 | Bookkeeping: sum of per-cell sums = sum of per-gene block sums, per layer, all 8 jobs | max relative difference 3.9e-9 |
| V5 | L0 per-gene mean = the model's embedding row (does not depend on input) | max relative difference 1.6e-6 (all jobs); deployed file 1.6e-5 |
| V6a | ER by a second route (eigenvalues of the float64 covariance) | equals the SVD route to 3 decimals (deployed, trained A, random 0 A, all layers) |
| V6b | CKA by a second route (HSIC on centred gram matrices, float64) | equals `linear_cka` to ≤ 4e-6 (all layers, all three sets) |
| V7 | Two cells per model recomputed from scratch on MPS and on CPU; `output_hidden_states` vs `hooks_v2.ResidualEditor` captures at all 12 sites | captures identical (max abs diff 0.0); stored per-cell sums reproduced (≤ 7e-7 relative); CPU vs MPS ≤ 2.6e-6 relative |
| R1 | Deployed numbers recomputed by the same code | ER 560.8 → 93.6 (seed 43: 93.0, seed 44: 93.7); shuffle 573.1; CKA L11 0.9914 — all as in the paper |
| H1 | Model health (next-gene loss, nats per gene token) | trained, UMI cells 3.28 in A, B and C (V3_INPUTS_REPORT: 3.35 on other cells); Smart-seq2 cells 7.5–7.7; random init 10.16 (a uniform guess is 9.92) |
| M1 | Memory | free memory never below 3.8 GB (guard at 3 GB); RSS ≤ 4.5 GB; see section 8 for the footprint |

## 4. Effective rank

### 4.1 Per layer

Panel D, all cells, deployed code. Rounded to whole dimensions.

| Model / sample | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Deployed encoding, sample A (stored file) | 561 | 333 | 338 | 298 | 232 | 212 | 169 | 174 | 202 | 202 | 172 | **94** |
| v3 trained, sample A | 561 | 328 | 355 | 343 | 302 | 304 | 275 | 261 | 273 | 267 | 266 | **227** |
| v3 trained, sample B | 561 | 328 | 355 | 343 | 302 | 303 | 275 | 261 | 273 | 267 | 266 | 226 |
| v3 trained, sample C | 561 | 328 | 355 | 343 | 302 | 304 | 276 | 262 | 273 | 267 | 266 | 226 |
| v3 random init seed 0, sample A | 816 | 642 | 649 | 624 | 589 | 556 | 518 | 482 | 452 | 419 | 394 | 398 |
| v3 random init seed 1, sample A | 817 | 645 | 650 | 625 | 583 | 546 | 506 | 464 | 431 | 403 | 383 | 390 |
| v3 random init seed 2, sample A | 817 | 645 | 652 | 632 | 605 | 570 | 538 | 504 | 470 | 443 | 413 | 421 |
| v3 random init seed 0, sample B | 816 | 643 | 650 | 626 | 593 | 562 | 526 | 490 | 461 | 428 | 404 | 411 |
| v3 random init seed 0, sample C | 816 | 642 | 650 | 625 | 592 | 560 | 525 | 488 | 459 | 425 | 400 | 406 |

Uncertainty (jackknife SE over cells, dimensions): trained 0.1 at L1, 0.3–0.6 in the middle, 0.8–1.0 at L11.
Random init 0.2 at L1, rising to 4–6 at L11. The spread across disjoint samples agrees: trained L11 225.6–227.2
(SD 0.8), random seed 0 L11 397.6–410.6 (SD 6.6).

### 4.2 Compression summary

ER ratios with 95% jackknife CIs over cells. Spearman ρ = rank correlation of ER with layer index.

| Model / sample | ER L0 | ER L1 | ER L11 [95% CI] | L1/L11 [95% CI] | Share lost L1 → L11 [95% CI] | L0/L11 | ρ (L0–L11) | ρ (L1–L11) | Shuffle ER at L11 |
|---|---|---|---|---|---|---|---|---|---|
| Deployed encoding, sample A | 560.8 | 332.6 | 93.6 | 3.55 | 71.8% | 5.99 | −0.895 | −0.864 | 573.1 |
| v3 trained, A | 560.8 | 328.1 | 227.2 [224.8, 229.5] | 1.44 [1.43, 1.46] | 30.8% [30.0, 31.5] | 2.47 | −0.930 | −0.909 | 645.6 |
| v3 trained, B | 560.8 | 328.2 | 225.6 [223.9, 227.3] | 1.45 [1.44, 1.47] | 31.3% [30.7, 31.8] | 2.49 | −0.930 | −0.909 | 639.0 |
| v3 trained, C | 560.8 | 328.1 | 226.3 [224.3, 228.2] | 1.45 [1.44, 1.46] | 31.0% [30.4, 31.6] | 2.48 | −0.930 | −0.909 | 641.2 |
| Random seed 0, A | 815.9 | 642.1 | 397.6 [386.7, 408.6] | 1.61 [1.57, 1.66] | 38.1% [36.4, 39.8] | 2.05 | −0.986 | −0.982 | 810.3 |
| Random seed 1, A | 816.8 | 644.9 | 389.8 [376.0, 403.5] | 1.65 [1.60, 1.71] | 39.6% [37.4, 41.7] | 2.10 | −0.986 | −0.982 | 809.3 |
| Random seed 2, A | 816.8 | 644.6 | 421.4 [408.2, 434.6] | 1.53 [1.48, 1.58] | 34.6% [32.5, 36.7] | 1.94 | −0.986 | −0.982 | 812.1 |
| Random seed 0, B | 815.9 | 642.9 | 410.6 [400.9, 420.3] | 1.57 [1.53, 1.60] | 36.1% [34.7, 37.6] | 1.99 | −0.986 | −0.982 | 810.9 |
| Random seed 0, C | 815.9 | 642.4 | 405.7 [394.3, 417.1] | 1.58 [1.54, 1.63] | 36.8% [35.1, 38.6] | 2.01 | −0.986 | −0.982 | 810.1 |

(The deployed row has no cell blocks, so no CI. Its seed-43 and seed-44 files give L11 93.0 and 93.7.)

### 4.3 Other local and global measures (deployed code, same matrices)

| Measure | Deployed encoding, A | v3 trained, A / B / C | Random init, seeds 0–2 (A) and seed 0 (B, C) |
|---|---|---|---|
| TwoNN local dimension, L0 → L11 | 102 → 6.7 | 102 → 5.3 / 5.3 / 5.3 | 204–207 → 107–120 (L6: 99–113) |
| Participation ratio, L1 → L11 | 230 → 24 | 243 → 119 / 116 / 117 | 487–489 → 70–93 |
| Share of variance on the top direction, L1 → L11 | 2.5% → 17.3% | 1.8% → 4.0% | 0.5% → 7.9–10.1% |

So random-weight models concentrate variance on a few directions with depth (participation ratio falls more than
in the trained model, and one direction grows to ~10%). But their points stay locally high-dimensional (TwoNN
~100). The trained model is the opposite: its global spread falls less, but its genes sit on a locally
low-dimensional structure (TwoNN ~5). These are observations; I did not test why.

### 4.4 Sensitivity checks

| Change | Trained, A: ER L1 / L6 / L11 | Random seed 0, A: ER L1 / L6 / L11 |
|---|---|---|
| Main (panel D, all cells) | 328.1 / 275.2 / 227.2 | 642.1 / 518.4 / 397.6 |
| UMI cells only (drop 73 Smart-seq2 cells) | 328.0 / 274.2 / 227.4 | 641.6 / 514.9 / 392.5 |
| Panel S (picked on a single log of raw counts) | 332.4 / 278.3 / 228.5 | 641.6 / 504.8 / 379.2 |

Neither changes the picture. Seeds 1 and 2 behave the same way (L1/L11 1.53–1.73 across the three settings).

## 5. Learned or architectural?

The paper says the depth-wise compression is learned, because a feature-shuffle null gives ER 573 at L11.

- **The test was the wrong one.** Shuffling each column on its own destroys all structure in any matrix. It gives
  near-full rank for the trained model (646 at L11), for random-weight models (810), and at every layer
  (trained 780–791 at L1–L10). It cannot tell training from architecture.
- **The right test (same architecture, random weights, same cells) says architecture.** From L1 to L11, random
  models lose 35–40% of their ER; the trained model loses 31%. The random models compress more, and every random
  run's CI lies above every trained run's CI.
- **What is learned:** the level, not the slope. The trained model uses fewer dimensions at every layer (227 vs
  390–421 at L11), starting with the embedding table (561 vs 816). Its first block cuts ER by 41% (random 21%).
  The last step (L10 → L11, which applies the final norm) cuts ER by 15% in the trained model; in random ones ER
  rises by 1–2% there. The learned norm weights are one possible reason; not tested.
- **Answer:** the fall of effective rank with depth is a property of the architecture (with these inputs), not
  something training adds. The deployed 561 → 94 also overstated the fall: with correct inputs it is 561 → 227.

## 6. Cross-sample CKA

### 6.1 Three disjoint 2,000-cell samples (mean of the 3 pairs)

L0 is 1.0000 for every model by construction (the same embedding rows). Intervals: jackknife 95% CI over genes
(the rows CKA compares; 30 groups of 50) and over cells (delete one 200-cell block in all three samples).

| Layer | Deployed encoding (6–8 shared cells per pair) [genes] | v3 trained [genes] [cells] | v3 random init seed 0 [genes] [cells] |
|---|---|---|---|
| L1 | 0.9997 [0.9997, 0.9997] | 0.9997 [0.9997, 0.9998] [0.9997, 0.9998] | 0.9938 [0.9936, 0.9941] [0.9936, 0.9941] |
| L2 | 0.9997 [0.9996, 0.9997] | 0.9997 [0.9996, 0.9997] [0.9996, 0.9997] | 0.9880 [0.9871, 0.9889] [0.9871, 0.9889] |
| L3 | 0.9993 [0.9992, 0.9993] | 0.9993 [0.9993, 0.9994] [0.9992, 0.9994] | 0.9804 [0.9785, 0.9823] [0.9780, 0.9828] |
| L4 | 0.9992 [0.9992, 0.9992] | 0.9991 [0.9990, 0.9991] [0.9989, 0.9992] | 0.9741 [0.9714, 0.9769] [0.9702, 0.9781] |
| L5 | 0.9991 [0.9991, 0.9992] | 0.9988 [0.9987, 0.9989] [0.9987, 0.9990] | 0.9696 [0.9662, 0.9731] [0.9645, 0.9748] |
| L6 | 0.9985 [0.9984, 0.9986] | 0.9981 [0.9980, 0.9982] [0.9979, 0.9982] | 0.9673 [0.9631, 0.9715] [0.9615, 0.9732] |
| L7 | 0.9966 [0.9964, 0.9968] | 0.9972 [0.9970, 0.9975] [0.9969, 0.9976] | 0.9666 [0.9619, 0.9714] [0.9602, 0.9730] |
| L8 | 0.9950 [0.9946, 0.9955] | 0.9974 [0.9972, 0.9976] [0.9972, 0.9977] | 0.9654 [0.9603, 0.9706] [0.9585, 0.9723] |
| L9 | 0.9928 [0.9920, 0.9936] | 0.9978 [0.9976, 0.9980] [0.9976, 0.9980] | 0.9655 [0.9601, 0.9709] [0.9589, 0.9722] |
| L10 | 0.9931 [0.9923, 0.9939] | 0.9976 [0.9974, 0.9978] [0.9973, 0.9978] | 0.9655 [0.9600, 0.9710] [0.9589, 0.9722] |
| L11 | **0.9914** [0.9906, 0.9922] | **0.9962** [0.9959, 0.9964] [0.9958, 0.9966] | **0.9647** [0.9598, 0.9697] [0.9588, 0.9706] |

- The trained model drops by 0.0035 from L1 to L11 (deployed encoding: 0.0083; random init: 0.029).
- With correct inputs the deep layers are more stable than the deployed run showed (L11 0.9962 vs 0.9914; the
  intervals do not overlap).
- Pairwise values agree: trained L11 pairs 0.99605, 0.99612, 0.99642; random 0.96263, 0.96530, 0.96631.

### 6.2 Split halves inside one sample (1,000 vs 1,000 cells; blocks 0–4 vs 5–9)

This gives a CKA control for all three random seeds without more forward passes.

| Job | L1 | L6 | L11 [genes] [cells] |
|---|---|---|---|
| trained A | 0.9995 | 0.9963 | 0.9927 [0.9922, 0.9932] [0.9919, 0.9936] |
| trained B | 0.9995 | 0.9963 | 0.9930 [0.9925, 0.9935] [0.9917, 0.9943] |
| trained C | 0.9995 | 0.9961 | 0.9924 [0.9919, 0.9929] [0.9906, 0.9942] |
| random seed 0, A | 0.9882 | 0.9462 | 0.9414 [0.9345, 0.9483] [0.9315, 0.9513] |
| random seed 1, A | 0.9880 | 0.9475 | 0.9435 [0.9366, 0.9503] [0.9324, 0.9545] |
| random seed 2, A | 0.9881 | 0.9386 | 0.9300 [0.9224, 0.9376] [0.9152, 0.9449] |
| random seed 0, B | 0.9881 | 0.9467 | 0.9385 [0.9300, 0.9470] [0.9315, 0.9455] |
| random seed 0, C | 0.9879 | 0.9425 | 0.9341 [0.9258, 0.9425] [0.9174, 0.9509] |

- All three random seeds are far less stable than the trained model at every layer.
- Going from 2,000 to 1,000 cells per average raises 1 − CKA at L11 from 0.0038 to 0.0073 (trained, ×1.9) and
  from 0.035 to 0.059 (random seed 0, ×1.7). If the gap were pure sampling noise, it would double. So for the trained
  model almost all of the gap from 1 is noise from which cells were averaged.

### 6.3 Why jackknife intervals, not bootstrap

`v3_spectral.py` also ran percentile bootstraps (in `analysis/cka.json`). Both were shifted:
- Drawing genes with replacement copies genes into identical rows in both matrices. This pushes CKA up. At L11
  the trained estimate 0.9962 sat at the lower edge of the bootstrap interval [0.9962, 0.9967].
- Drawing 200-cell blocks with replacement leaves only ~63% distinct cells. The averages get noisier and CKA goes
  down: trained L11 bootstrap interval [0.9903, 0.9943], below the estimate; random [0.918, 0.948] vs 0.965.

The jackknife never copies a gene or a cell, and its intervals are centred on the estimates. Two cross-checks of
its width: the gene-bootstrap SD equals the gene-jackknife SE (trained L11: 0.00013 vs 0.00012), and the spread of
the three pairwise CKAs (SD 0.0002) matches the cell-jackknife SE (0.00017). The block-bootstrap SD is about 5 times
larger and was not used. For the same reason the ER cell bootstrap (`analyze --part erboot`) was not run.

## 7. What changed compared with the deployed run (and v2)

| Quantity | Paper / deployed | Deployed, recomputed here (same code) | v3 (correct inputs) |
|---|---|---|---|
| ER L0 → L11 | 561 → 94 | 560.8 → 93.6 | 560.8 → 227.2 [224.8, 229.5] (B: 225.6, C: 226.3) |
| Spearman ρ of ER with layer (L0–L11) | −0.895, p = 8 × 10⁻⁵ | −0.895 | −0.930, p = 1 × 10⁻⁵ |
| Feature-shuffle null at L11 | 573 (6.1× the L11 value) | 573.1 | 645.6 (2.8×); random init 809–812 |
| "Compression is learned" | yes (shuffle null) | — | **No**: random-weight models compress as much or more from L1 (35–40% vs 31%) |
| CKA L0 → L11 | 1.000 → 0.991 (samples share 6–8 cells) | 1.0000 → 0.9914 | L1 0.9997 → L11 0.9962 [0.9959, 0.9964], disjoint samples |
| CKA random-init control | none | — | L1 0.9938 → L11 0.9647 [0.960, 0.970] |

- The ER change at L11 (94 → 227) is on the same cells and genes with the same code. Only the input order changed.
- CKA changes little in the early layers and rises in the deep layers (L9–L11: 0.993 → 0.998).
- The disjointness fix itself is negligible (0.3–0.4% shared cells).
- v2: there is no v2 re-run of spectral ER or CKA. The v2 cross-model check (`outputs/v2_crossmodel`) uses static
  embedding tables and is unchanged.

## 8. What was not done, and caveats

- **No forward pass on the deployed (wrong) inputs.** The workflow forbids it. The deployed numbers were
  recomputed from the stored deployed embeddings. The deployed run did not record its torch / transformers
  versions. Its log shows the same weight-loading progress bar as transformers 5.x, and its layer layout (12
  states, L0 = embedding, L11 final-normed) matches this one (V5, V7). So I attribute 94 → 227 to the input
  order, but I could not run the old order through today's software stack to prove it.
- **Random init for CKA: one seed on three samples.** Seeds 1 and 2 were run on sample A only. Their split-half CKA
  (section 6.2) agrees with seed 0.
- **Other spectral results were not redone:** SV enrichments, STRING, TF-vs-target, B/T compression, GC–plasma
  angle, BATF/BCL6, the Phase 8 SV5–SV7 principal angles, and autoloop H3-G. The per-gene vectors needed for them
  are saved for samples A, B and C in the deployed layout (`extract/trained_*/layer_gene_embeddings_panelD.npy`).
- **Gene panel:** the deployed 1,500 genes were kept so the comparison is paired. They were picked on a double log.
  A panel picked on a single log of raw counts gives the same ER (section 4.4).
- **Smart-seq2 cells** (73, 80, 80 in A, B, C) were kept, as in the deployed run. Under the correct encoding the
  model predicts them near chance (loss 7.5–7.7 vs 3.28). Dropping them changes trained ER at L11 by 0.2.
- **Counts:** `raw/X` (before ambient-RNA removal), as the workflow rule says; `layers/decontXcounts` was not tried.
- **TwoNN** has no interval here beyond the spread across samples and seeds (trained 5.3 in all three samples;
  random 107–120). It is a noisy estimator in high dimension; read it as large vs small, not as exact values.
- **The scGPT comparison** in the paper (0.979 → 0.779) was not redone; it compares fine-tuned models, not cell
  samples.
- **Speed and memory.** MPS ran at 0.5–1.5 s per cell (deployed: 0.39 s; other jobs shared the machine). The
  process peak footprint (`/usr/bin/time -l`) was first measured during trained B: 9.8 GB, because the MPS cache had
  grown to 5.4 GB. Earlier chunks were not measured this way (their MPS cache peaks were 3.3–4.4 GB). I then emptied
  the MPS cache after every cell (memory only; values unchanged). After that, full chunks peaked at 7.7–8.9 GB, around
  the ~8 GB target, with RSS ≤ 4.5 GB. Free memory never fell below 3.8 GB.
- **Script versions.** `outputs/v3_spectral/code_versions/` holds the two versions recorded by the jobs:
  4da5124c… (trained A and random seeds 0–2 on A finished under it; trained B started under it) and 1c41677d… (the
  rest). Their extraction code differs only in that cache call. Edits before 4da5124c (during the first 450 cells of
  trained A) changed only the default job list and analysis code. `v3_spectral_intervals.py` gained only the `index`
  part after the jackknife parts ran.

## 9. Files

All under `projects/maxtoki/runs/spectral-geometry-217M/`:
- `scripts/v3_spectral.py` — stages `prepare`, `extract` (resumable), `analyze --part metrics|cka|cka_half|summary`,
  `verify --part code|samples|tokens|sums|l0|metrics2|forward`.
- `scripts/v3_spectral_intervals.py` — `er`, `cka`, `half` (jackknife), `tables`, `index`.
- `outputs/v3_spectral/run_config.json` — index: devices, versions, seeds, sample hashes, per-job wall time,
  code sha256.
- `outputs/v3_spectral/prepare/` — `cells_{A,B,C}.npz` (rows, tokens, blocks, assay), `cells.csv` (per-cell
  order agreement with the deployed tokens), `panel.json`, `run_config.json` (cell IDs, gene IDs, counts and token
  sha256, encoding check).
- `outputs/v3_spectral/extract/<job>/` — `acc_sums.npy` (block sums), `acc_counts.npy`, `percell_sums.npy`,
  `cells.csv` (per-cell loss), `layer_gene_embeddings_panelD.npy` (12 × 1,500 × 1,232), `gene_counts_panelD.npy`,
  `state.json`, `run_config.json` (model, init seed and hash, env, cells, genes, encoding check, chunk times).
  Jobs: `trained_A/B/C`, `rand0_A`, `rand1_A`, `rand2_A`, `rand0_B`, `rand0_C`.
- `outputs/v3_spectral/analysis/` — `per_layer_metrics.csv` (+ `_optional.csv`), `cka.json` (point estimates and
  bootstraps), `cka_half.json`, `intervals.json` (jackknife), `report_tables.md`, `summary.json`, `run_config.json`.
- `outputs/v3_spectral/verify/verify.json` — checks V1–V7.
- Re-run: from `projects/maxtoki`, `OMP_NUM_THREADS=4 .venv/bin/python runs/spectral-geometry-217M/scripts/v3_spectral.py prepare`,
  then `extract --max-minutes 7` until it prints `ALL_DONE True` (add `--jobs rand0_B,rand0_C` for the optional
  jobs), then the analyze, intervals and verify parts.

## Plain-words summary

The spectral run measured how many directions MaxToki uses to represent genes, layer by layer, and how much
that picture changes when you use different cells. It fed the model genes in the wrong order. I redid it with the
right order, on the same cells and genes, with the same measuring code.

With the right input, the last layer uses about 227 directions, not 94. The fall from the first layer is much
smaller than the paper says. A model with the same design but random, untrained weights shows the same kind of fall,
and a slightly larger one. So the fall with depth comes from the design, not from training. What training does is
make every layer use fewer directions, starting from the gene embedding table. The shuffle test the paper used
cannot tell these two apart.

The "stable across cell samples" result holds and is a bit stronger with the right input (0.996 at the last layer,
on truly separate cell samples). A random-weight model is clearly less stable (0.965). But these numbers mostly
measure how much averaging over a different set of cells moves each gene's vector, so they should not be compared
with the scGPT numbers, which compare different trained models.

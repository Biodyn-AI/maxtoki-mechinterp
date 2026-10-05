# V2 hooks report — hook library, unit tests, static audit, zero-edit control (items D0 + D7a)

Date: 2026-10-01. Model: MaxToki-217M (HF safetensors, float32, Apple MPS). transformers 5.5.4, torch 2.11.0.
All numbers below come from files written by the scripts named in section 9.

## 0. Results first

1. **The published steering table is a hook artefact.** When the deployed Experiment 3 hook is run with a
   feature edit of exactly zero (α = 1), it gives the same Δs as the published α = 5 table:

   | Layer | Published Δs (α = 5, mean of 3 features) | Deployed hook, zero edit (50 cells) [95% CI] | Share of published value |
   |---|---|---|---|
   | L0 | +0.0962 | **+0.0963** [+0.0857, +0.1061] | 100% |
   | L3 | +0.0178 | **+0.0177** [+0.0160, +0.0193] | 100% |
   | L6 | −0.0079 | **−0.0078** [−0.0090, −0.0066] | 99% |
   | L9 | +0.0141 | **+0.0142** [+0.0124, +0.0160] | 101% |
   | L11 | +0.0048 | **+0.0045** [+0.0030, +0.0059] | 93% of the 3-feature mean; 101% of the matched feature |

   CI = cell-level percentile bootstrap, 10,000 resamples, seed 20261001, n = 50 cells.
   The zero edit also gives the same top up-regulated genes (L0 KITLG/SOD3, L3 C1QTNF3-AMACR, L6 C1QB, L9 PAQR3,
   L11 FTL) and the same fractions of cells pushed "toward maturity" (0.98 / 1.00 / 0.02 / 1.00 / 0.78).
   With the fixed hooks, the zero edit gives Δs = 0 exactly at every layer.
2. **The layer convention is now settled by measurement.** `hidden_states[l]` is the input of block l for
   l = 0..10, and `hidden_states[11]` is the final-normed output of block 10 (= the input of `lm_head`).
   The max difference between the hook-captured tensors and `hidden_states` is 0.0 (3 cells, all 12 sites).
   Each SAE `layer_XX` fits site XX best: variance explained 0.696–0.938 at its own site versus −0.22 to 0.78 one layer off.
3. **New hook library** `setup/hooks_v2.py` edits the residual stream where each SAE reads it and computes the SAE
   code from the live tensor. All 5 required unit tests pass on 3 real K562 control cells (section 3).
4. **Static audit:** 9 deployed scripts have a hook or layer bug (section 4): 6 with a broken intervention hook and
   3 that read the wrong layer for their SAE. Every intervention result in the paper (steering, 4.97 M edges,
   2,980 "triplets", 2.1 M Stage-2 edges and the CRISPRi test built on them, 1.01× patching specificity) and the
   TF-specificity results (0/48 TFs, GATA1 feature 2610) depend on one of them.

---

## 1. What `hidden_states[l]` is, and what each SAE was trained on (item a)

### 1.1 From the source (transformers 5.5.4)
- `utils/output_capturing.py:108–113`: the capture hook appends `args[0]` of the FIRST decoder layer
  (the embeddings), then the output of every `LlamaDecoderLayer` (`models/llama/modeling_llama.py:348–349`).
- `models/llama/modeling_llama.py:405–421`: after the 11 blocks, `hidden_states = self.norm(hidden_states)`.
- `utils/output_capturing.py:201, 255–263`: `tie_last_hidden_states=True` by default, so the last captured entry
  (raw output of block 10) is REPLACED by `last_hidden_state` = the normed tensor.
- `modeling_llama.py:480–482`: `lm_head` is applied to `outputs.last_hidden_state[:, slice]` (the whole sequence
  with the default `logits_to_keep=0`).

So, for MaxToki-217M (11 blocks, 12 entries):

| index | tensor | where an edit must go |
|---|---|---|
| `hidden_states[0]` | token embeddings = input of block 0 | forward pre-hook on `model.model.layers[0]` |
| `hidden_states[l]`, l = 1..10 | output of block l−1 = input of block l | forward pre-hook on `model.model.layers[l]` |
| `hidden_states[11]` | `model.model.norm(output of block 10)` = input of `lm_head` | forward pre-hook on `model.lm_head` |

### 1.2 Measured (test T0, 3 cells, 2,048 tokens each)
- Pre-hook input of block l vs `hidden_states[l]`, l = 0..10, and `lm_head` input vs `hidden_states[11]`:
  max abs difference **0.0** in all 3 cells.
- `hidden_states[11]` vs `norm(output of block 10)`: max abs difference **0.0**.
- `hidden_states[11]` vs the raw output of block 10: max abs difference **748** (cell 0). The raw output has
  very large values at the `<bos>` position (max |h| ≈ 350 at L2, position 0) that the norm removes.

### 1.3 What the SAEs were trained on
- All 12 SAEs now on disk (`runs/sae-atlas-217M/outputs/phase1/layer_00..11/sae_final.pt`) were written on
  Apr 19 by `full_12layer_pipeline.py`. Its log shows "Extracting activations" and "Training SAE" for every layer
  (`outputs/full_12layer.log`). It trains SAE l on `hidden[layer_idx]` from `forward_with_hidden_states`
  (`full_12layer_pipeline.py:139–141`), i.e. `hidden_states[l]`: 500 K562 non-targeting cells, seed 42,
  1,019,996 token positions, all positions including `<bos>`/`<eos>`.
- The older Apr 17 SAEs for L0/L5/L11 (`phase0_extract_positions.py:103–107`, `phase1_train_saes.py`) also used
  `hidden[l]`. They were overwritten on Apr 19 (same path). Any result computed on Apr 17–18 with those SAEs
  (sae-atlas Phase 6 patching, Phase 8t "0/48") cannot be re-run with its original SAE.
- **Numeric check** (`v2_sae_site_check.py`, 5 K562 control cells, 10,240 positions): variance explained (VE) of
  each SAE at its own site and at the neighbouring sites.

| SAE | VE in results.json | VE at site l−1 | **VE at site l** | VE at site l+1 |
|---|---|---|---|---|
| L0 | 0.693 | – | **0.696** | 0.098 |
| L1 | 0.904 | 0.781 | **0.908** | 0.478 |
| L2 | 0.935 | −0.082 | **0.938** | 0.721 |
| L3 | 0.932 | 0.455 | **0.936** | 0.672 |
| L4 | 0.924 | 0.730 | **0.930** | 0.745 |
| L5 | 0.889 | 0.616 | **0.897** | 0.665 |
| L6 | 0.846 | 0.583 | **0.859** | 0.665 |
| L7 | 0.803 | 0.678 | **0.816** | 0.541 |
| L8 | 0.772 | −0.217 | **0.786** | 0.539 |
| L9 | 0.745 | 0.446 | **0.759** | 0.511 |
| L10 | 0.757 | −0.216 | **0.775** | 0.506 |
| L11 | 0.828 | 0.512 | **0.845** | – |

  Every SAE fits best at its own site, at about its training VE. The L11 SAE fits the post-norm `lm_head` input
  (0.845) and not the pre-norm block-10 input (0.512). The L5 SAE applied to `hidden_states[6]` (what the
  sae-atlas Phase 6 / 8t / A8 / Phase 11 code did) explains only 0.665 of the variance, versus 0.897 at its own site.

### 1.4 SAE convention (`setup/topk_sae.py`)
- Encode: `z = TopK_32(W_enc (x − μ) + b_enc)`, kept values clamped at 0, all others 0. μ is the training-set mean.
  There is no other pre-bias.
- Decode: `x̂ = W_dec z + μ`. Decoder columns are renormalised to unit L2 norm after every optimiser step.
- `encode` uses `scatter_(1, …)`, so it needs a 2-D input (positions × 1,232).
- Decode is linear, so the part of the residual stream carried by feature f at a position is `z_f · W_dec[:, f]`.

## 2. The hook library `setup/hooks_v2.py` (item b)

- **Sites.** `site_module(model, l)`: `model.model.layers[l]` for l = 0..10, `model.lm_head` for l = 11.
  All edits are forward PRE-hooks, so they change the input the SAE was trained on and nothing else.
- **Live code.** Inside the hook the SAE code z is computed from the live input (after all upstream edits).
  So edits at several layers stack: a later edit sees the effect of earlier ones.
- **Operations** (all deltas are ADDED to the live tensor; the SAE reconstruction error is kept):
  - `Ablate(l, features)`: delta = −Σ_f z_f · W_dec[:, f]. This is the same delta the deployed code computed
    (`decode(z with f zeroed) − decode(z)`), but applied at the right place and from the live input.
  - `Steer(l, f, alpha)` (default `mode="scale"`): delta = (α − 1) · z_f · W_dec[:, f]. This matches experiment3,
    which set z_f → α·z_f and patched `decode(z′) − decode(z)` (`experiment3_rerun.py:259–265`). α = 1 is an exact
    no-op. It only acts at positions where the feature is active.
    `mode="add"` adds α · W_dec[:, f] at every edited position (a fixed direction). This is NOT what experiment3 did.
  - `AddVector(l, v)` and `ZeroDelta(l)`: for tests.
  - Several edits at the same layer share one z computed from that layer's live input; their deltas are summed.
  - `positions=` restricts an edit to some tokens. The default edits every position, including `<bos>`/`<eos>`,
    as the deployed scripts did.
- **Read-outs.** `ResidualEditor(capture=[...])` stores the live input of each listed site, before (`"pre"`) and
  after (`"post"`) any edit at that site. `ed.codes(l)` gives SAE codes; `downstream_codes(...)` is a one-call helper.
- **Deliberate differences from the deployed code.** (a) The delta goes on the input of block l, not on the output
  of block l. (b) z comes from the live input, not from a cached clean pass. For a single edit, (b) changes nothing,
  because the live input then equals the clean input. (c) At L11 the edit is applied after the final norm, so the
  norm is applied once.
- **Warning written into the module docstring.** Do not read `output_hidden_states` under edits. PyTorch passes the
  EDITED arguments to forward hooks, so transformers records the post-edit tensor for index 0 but the pre-edit tensor
  for indices 1..10. Use the captures instead.
- **Utilities.** Memory guard (`check_memory`, free + inactive pages from `vm_stat`, because `psutil` is not in
  the venv), `load_k562_control_cells` (same cell pool as experiment2), `load_saes`, `env_info`, `sha256_file`.

## 3. Unit tests (item c) — `setup/test_hooks_v2.py`

Cells: 3 K562 non-targeting control cells from the experiment2 pool (Replogle concat h5ad rows 289300, 292833,
293501; seed 42; 2,048 tokens each). MPS, float32. Wall time 53–73 s per cell.

| Test | What it checks | Threshold | Result (worst of 3 cells) | Pass |
|---|---|---|---|---|
| Noise floor | two clean runs, logits | – | max abs diff 0.0 | – |
| T0 convention | captured site inputs vs `hidden_states`; `hidden_states[11]` = norm(block 10 output) | ≤ 1e-6 | 0.0 and 0.0 | yes |
| (i) zero-delta | `ZeroDelta(l)` at each of the 12 sites vs clean logits | < 1e-4 | **0.0** | yes |
| (i') α = 1 | `Steer(l, f, 1.0)` at L0/3/6/9/11, deployed feature and an active feature | < 1e-4 | 0.0 | yes |
| (ii) locality | ablate the most active feature at L0, L3, L5, L9, L10, L11: layers < l and pre-edit input at l unchanged; every layer l+1..11 and the logits change | unchanged ≤ 1e-6; changed > 1e-4 | unchanged: **0.0**; smallest downstream max abs change **0.271**; SAE codes change at every downstream layer (3,399–4,927 of 4,928 features) | yes |
| (ii') delta | observed post − pre at l vs −z_f·W_dec[:, f] | < 1e-4 | ≤ 1.9e-6 | yes |
| (iii) stacking | A = L0, B = L5, C = L9 (most active features): AB vs B, ABC vs C, BC vs C, ABC vs BC at the L11 input and logits | > 1e-4 | AB vs B ≥ **0.554**; ABC vs C ≥ **2.55** (L11 max abs) | yes |
| (iii') live code | B's delta inside AB vs B alone | – | differs by 0.11–0.29 (max abs) | – |
| (iii'') legacy contrast | deployed overwrite hooks: AB vs B and ABC vs C | – | **0.0 exactly** (bug reproduced) | – |
| (iv) never-active | ablate a feature with z = 0 at every position (L0 F1, L5 F5/F26, L11 F0) | < 1e-5 | 0.0 | yes |
| (v) L11, no double norm | logit change from an L11 edit equals `lm_head(delta)` exactly (a norm after the edit would break this) | < 1e-3 | ≤ 5.6e-5 (changes themselves are 0.8–4.1) | yes |
| (v') legacy contrast | deployed L11 hook with delta = 0 | – | mean abs logit change 0.71–0.73, max 18–21 | – |

Other numbers from the tests:
- Deployed hook with delta = 0 at L0 (block 0 deleted): mean abs logit change 3.11–3.20. At L5: 1.17–1.29.
- **Observation for later agents (not investigated further).** In cell 0 the most active L0 feature (F2232) fires at
  `<bos>`. Its edit is tiny (RMS 0.0006 per position), but it changes the L2 residual at `<bos>` by up to 39
  (clean max |h| there is 350), and the largest change at the median position is about 2.0. Edits at `<bos>`/`<eos>` can therefore have
  outsized effects. Max-abs statistics will be dominated by `<bos>`; prefer per-position or mean statistics, and
  consider `positions=` to exclude special tokens when the design allows it.

## 4. Static audit of every intervention script (item d)

I read every deployed script in the 8 run folders (new `v2_*` files from other agents were excluded).
Bug types: **block-offset** = reads `hidden_states[l]` (input of block l) and writes it as the output of block l,
which deletes block l. **overwrite** = the hook returns a fixed tensor from a clean pass instead of adding to the live
tensor, so a later hook erases earlier edits and a block's own computation is lost. **double-norm** = the
post-norm `hidden_states[11]` is written back before the final norm. **layer-mismatch** = SAE l applied to a tensor
from another layer. **recon-substitution** = the tensor is replaced by the SAE reconstruction, so the SAE's error
(≈ 10–35% of the variance) is injected as part of the "edit".

| Run / script | Lines | Hooks the model? | Edits which tensor / SAE trained on | Bug type(s) | Results that depend on it |
|---|---|---|---|---|---|
| exhaustive-mapping `experiment1_exhaustive.py` | 143–174 | forward hook on `layers[5]` | writes `hidden_states[5]` + delta as block-5 output / SAE L5 on `hidden_states[5]` | **block-offset**, overwrite (single hook). Read-out of L6 is recorded before the patch, so L6 shows no change (0 of 4,970,096 edges at L6 across 1,000 features) | 4.97 M edges, "uniformly dense", hub and annotation-bias analyses (TEX 988; `experiment1/exhaustive_summary.json`) |
| exhaustive-mapping `experiments_2_3.py` (v1) | Exp 2: 161–182; Exp 3: 372–396 | forward hooks on `layers[0,5,9]` / `layers[li]` | as above | Exp 2: **block-offset + overwrite** (AB ≡ B, ABC ≡ C). Exp 3 v1 crashed at `layers[11]` (IndexError in `experiments_2_3.log`). Switch-feature detection (300–315) reads `hidden[li]` with SAE li: correct | v1 triplets (pairwise 0.1654, three-way 0.1897) quoted in the summary; `switch_features.csv` (valid, read-only) |
| exhaustive-mapping `experiment2_rerun.py` | 205–236 | forward hooks on `layers[0,5,9]` | as above | **block-offset + overwrite** (legacy contrast in T3 reproduces AB ≡ B and ABC ≡ C exactly) | "2,980 triplets", zero synergy, three-way 0.190, pair mean 0.324 (TEX 988–989, 1125–1130) |
| exhaustive-mapping `experiment3_rerun.py` | 147–155, 250–282 | forward hook on `layers[min(li,10)]` | writes `hidden_states[li]` + delta as block-li output; L11: writes post-norm `hidden_states[11]` as block-10 output | **block-offset** (L0–L9), **double-norm** (L11), overwrite (single hook). Confirmed by the zero-edit control (section 5) | Abstract (TEX 234–236), glossary (395–398), pipeline table (990), steering section and figure (1348–1401): Δs table, 5×/20×, L6 inversion, top genes |
| exhaustive-mapping `aggregate_alpha2.py`, `stage3_synthesize.py` | – | no | consume Exp 1–3 outputs | inherit | α-saturation note, stage-3 summary |
| circuit-tracing `circuit_trace.py` | 176–221 | forward hook on `layers[src]` | writes `hidden_states[src]` + delta as block-src output / SAE src on `hidden_states[src]` | **block-offset**, overwrite (single hook). No edges at target = src+1 for any source layer (`circuit_edges.csv`) | 2,144,011 edges ("2.1 million", TEX 526, 976), inhibitory 88%, coherence 36% (`circuit_summary.json`) |
| circuit-tracing `remaining_phases.py`, `audit_a2_groupkfold_crispri.py`, `audit_a4_edge_density_chance_baseline.py` | reads `circuit_edges.csv` (46; 37; 45) | no | – | inherit block-offset | CRISPRi direction 53.48% / 53.42% (TEX 1091–1102, 1549–1550), PMI, knowledge graph, disease mapping, audit A2/A4 |
| sae-atlas `phase6_patching.py` | 121–146 | forward hook on `layers[5]` | replaces block-5 output (= `hidden_states[6]`) with the L5-SAE reconstruction of `hidden_states[6]` | hook position is consistent with the tensor read, but **layer-mismatch** (L5 SAE on site 6: VE 0.665) and **recon-substitution**. Mean abs logit change ≈ 1.4 for every feature. Ran Apr 18 with the since-overwritten Apr 17 L5 SAE | median specificity 1.01× vs Geneformer 2.36× (TEX 1105–1109) |
| sae-atlas `remaining_phases.py` | Phase 6: 168–197; 8t: 390–418; 11: 605–606 | Phase 6: forward hook | Phase 6 same design as above; its hook returns `(_p,) + output[1:]`, which I infer fails on a Tensor output; the log says Phase 6 was skipped. 8t: controls = `hidden[5]` (phase0 file), perturbed = `hidden[6]`, both with SAE L5. 11: `hidden[6]` with SAE L5 | 8t and 11: **layer-mismatch** (read-only, no intervention). 8t compares two different layers | 0/48 TFs, 4/100 detection (TEX 1109, 1305); atlas cell-type labels (TEX 966) |
| sae-atlas `audit_a8_rerun_phase8t_targeted.py` | 138–139, 200–201 | no (reads) | `hidden[6]` with SAE L5 for both groups | **layer-mismatch** (read-only) | GATA1 feature 2610, p = 0.030, Fig gata1 (TEX 1302–1330) |
| sae-atlas `phase9_multitissue.py` | 52, 77–78, 134, 238–239 | no (reads) | trains a pooled SAE on K562 `hidden[5]` + Tabula Sapiens `hidden[6]`; perturbed cells `hidden[6]` | **layer-mismatch** in training data (read-only) | multi-tissue control; I found no claim in the paper that uses it |
| sae-atlas `audit_a8_positive_tf_case_study.py`, `audit_a4_phase8t_chance_baseline.py` | – | no | consume 8t outputs | inherit layer-mismatch | A4/A8 audit numbers |
| sae-atlas `phase0_extract_positions.py`, `full_12layer_pipeline.py`, `phase1_train_saes.py`, `phases2_to_8.py`, `build_atlas_data.py`, `phases12_13_viz.py` | – | no | define the convention (SAE l on `hidden_states[l]`); `phases2_to_8.py` Phase 8 is a co-occurrence test on stored L5 activations | none | atlas, annotations |
| attention-grn `phase4_causal_ablation.py` + `setup/maxtoki_adapter.py:221–244` | 216–241 | pre-hook on `o_proj` zeroing head slices | per-head attention output before `o_proj` | none in the hook (zeroing composes; the tensor is a fresh copy). Design caveat: the read-out is attention weights, which a head mask cannot change at its own layer (the mask acts after the softmax). Ran with 300 cells (run_config), not 2,000 | head-ablation table; I found no claim in the paper that uses it |
| attention-grn `phase0_*.py`, `phase0b_value_weighted.py`, `phase6_cssi.py`, others | – | read-only (`o_proj` pre-hook capture in phase0b) | – | none | attention-GRN results (other items: D1) |
| manifold-discovery `phase9_head_attribution.py`, `phase4_export_operators.py`, `phase11_factor_ablation.py`, `phase1bc_*.py`, others | – | no | `hidden_states[l]` = input of block l with `o_proj` weights of block l (consistent); the "ablation" is inside the trained 10-d head, not the model | none. `audit_residual6_cell_level_benchmark_lite.py:57,92` reads `hidden[6]` but calls it "L5" (naming only; no SAE) | head attribution, 18.3% (TEX 1283–1290) — not a hook question |
| longevity `maxtoki_runtime.py`, `run_stage1/2/3*.py` | 184–194 | no | reads `hidden[5]`; its own SAE is trained and read on the same stored vectors | none | – |
| topology-141 (all), spectral-geometry (all) | e.g. topology `phase0_extract.py:265–267`, spectral `phase0_extract.py:233–236` | no | read `hidden_states[l]` only | none | – |

## 5. Zero-edit control with the deployed steering code (item e)

Script: `runs/exhaustive-mapping-217M/scripts/v2_zero_edit_control.py` (logic copied from `experiment3_rerun.py`;
the original is untouched).

**Set-up (checked before any number was used).**
- Same 200 Tabula Sapiens immune cells (seed 42 + 3333), same PC1 pseudotime, same 50 earliest-quartile cells:
  recomputed pseudotime vs the cached `state_signatures.npz` differs by 0.0, and the early and late masks are identical.
- Same cached signatures `g_early_logits` / `g_late_logits`, same Δs formula (logit space, cosine, minus clean baseline).
- Clean position-averaged logits recomputed today vs the cached April values: max abs difference 2.4e-5 (MPS is reproducible).
- Zero edit = the deployed code with α = 1. The script asserts the delta tensor has 0 non-zero entries (it passed for
  every cell and layer). The hook still replaces the output of block min(li, 10) with clean `hidden_states[li]`.
- **Reproduction check:** the same code at α = 5 on the first feature of each layer reproduces the published
  per-feature Δs to 6 decimals (L0 F40 0.095751, L3 F4071 0.017851, L6 F4138 −0.007919, L9 F1014 0.013909,
  L11 F3924 0.004435; absolute difference 0.0 for all 5).

**Results (n = 50 cells; 95% CI = cell-level percentile bootstrap, 10,000 resamples, seed 20261001).**

| Layer | Published layer mean (α = 5) | Deployed hook, α = 5, one feature (reproduced) | **Deployed hook, zero edit** | frac Δs > 0 (zero edit) | Feature part: α5 − zero [95% CI] | Fixed hook, zero edit | Fixed hook, α = 5, same feature (preview) |
|---|---|---|---|---|---|---|---|
| L0 | +0.096226 | +0.095751 | **+0.096306** [0.085665, 0.106098] | 0.98 | −0.00056 [−0.00072, −0.00039] | 0 | −0.00089 [−0.00106, −0.00075] |
| L3 | +0.017829 | +0.017851 | **+0.017741** [0.016042, 0.019323] | 1.00 | +0.00011 [−0.00001, +0.00021] | 0 | −0.00019 [−0.00040, −0.00005] |
| L6 | −0.007897 | −0.007919 | **−0.007803** [−0.009026, −0.006607] | 0.02 | −0.00012 [−0.00015, −0.00008] | 0 | −0.00019 [−0.00021, −0.00017] |
| L9 | +0.014076 | +0.013909 | **+0.014180** [0.012437, 0.016034] | 1.00 | −0.00027 [−0.00033, −0.00021] | 0 | −0.00024 [−0.00029, −0.00018] |
| L11 | +0.004817 | +0.004435 | **+0.004491** [0.003042, 0.005905] | 0.78 | −0.00006 [−0.00007, −0.00004] | 0 | −0.00005 [−0.00006, −0.00004] |

- The zero edit accounts for 98.5–102% of each reproduced per-feature value. The feature edit itself adds at most
  0.0006 in absolute value, less than 1% of the L0 effect.
- Mean abs logit change per logit under the deployed hook with zero edit: L0 2.95, L3 1.51, L6 0.98, L9 1.24,
  L11 1.33. Under the fixed hook with the α = 5 feature edit: 0.040, 0.042, 0.027, 0.030, 0.008. So the deployed
  hook's own disturbance is 35–170 times larger than the feature edit.
- Top up-regulated genes under the zero edit: L0 KITLG, SOD3, APOE; L3 C1QTNF3-AMACR; L6 C1QB, HBE1, CPA3;
  L9 PAQR3; L11 FTL, MT-CO3, MT-CO1. These are the genes the paper reads as biology (TEX 1376–1378).
- **Conclusion.** The published steering numbers, the 5× and 20× ratios, the L6 "inversion" and the gene lists
  measure what happens when block li is deleted (L0–L9) or when the final norm is applied twice (L11).
  They are not effects of steering a feature.
- The last column is a preview only: one feature per layer, no null over features. It must not be read as a
  steering result. Item D7 owns the real re-run (random-feature null, more features).

## 6. What changed versus the deployed result

- Steering (E15): the table is fully explained by the hook (section 5). With the fixed hook and the same features,
  Δs is 40–110 times smaller in size and negative at every layer (preview, one feature per layer, no null).
- Triplets (E13): the legacy contrast in test (iii) reproduces AB ≡ B and ABC ≡ C exactly (difference 0.0). With the
  fixed hooks, AB differs from B (L11 max abs ≥ 0.55) and ABC from C (≥ 2.55). The triplet numbers must be re-run (D5).
- Edge maps (E14): Exp 1 and Stage-2 circuit tracing delete the source block; they must be re-run (D2, D6). The
  CRISPRi directional test is built on the Stage-2 edges, so it inherits the bug.
- Patching specificity 1.01× (E16): confirmed wrong layer (L5 SAE on `hidden_states[6]`, VE 0.665) plus
  reconstruction substitution. Needs a re-run with `Ablate(5, [f])` at site 5 (D4).
- TF specificity 0/48 and GATA1 2610 (E11, E12): read-side layer mismatch confirmed in code (D3).

## 7. What I could NOT do, and limits

- I did not re-run any of the affected experiments (D2–D6); this item only builds and checks the tools.
- The unit tests use 3 K562 cells, the site check 5 cells. They test mechanics, which are deterministic; they are not
  estimates of any biological effect.
- The "fixed hook, α = 5" column uses one feature per layer and has no null. It is not a result.
- I did not run the sae-atlas `remaining_phases.py` Phase 6 hook, so "it would fail on a Tensor output" is inferred
  from the code and the log line "Phase 6 skipped".
- I did not check the attention-GRN head-ablation numbers, only the hook mechanics and the design caveat.
- `psutil` is not installed in the venv. The memory guard reads `vm_stat` (free + inactive pages, the same
  definition psutil uses on macOS). I did not install anything.
- Other MPS jobs from unrelated projects (Geneformer and scGPT runs) were running during all my runs. They do not
  change results (every check that could show non-determinism gave 0.0), but wall times (5–33 s per steering cell)
  are not clean timings. The memory guard stopped one chunk of the zero-edit run and one site-check start; both were
  re-run after memory recovered (recorded in the run_config files).
- Peak memory of the test run: 7.4 GB peak footprint, 2.7 GB resident (from `/usr/bin/time -l`).

## 8. How to use the library (for D2–D9)

```python
import sys; sys.path.insert(0, ".../projects/maxtoki/setup")
import hooks_v2 as H
xt = H.load_model("mps"); model = xt.model
saes = H.load_saes([0, 5, 9, 11], device="mps")
edits = [H.Ablate(0, [fa]), H.Ablate(5, [fb]), H.Ablate(9, [fc])]      # stacks correctly
logits, ed = H.run_with_edits(model, input_ids, saes, edits, capture=[11])
z11 = ed.codes(11, which="pre")                                          # (T, 4928) on CPU
logits, _ = H.run_with_edits(model, input_ids, saes, [H.Steer(3, f, 5.0)])   # experiment3 semantics
```
Rules: never read `out.hidden_states` under edits (use captures); SAE l belongs to site l; add new helpers at the end
of `hooks_v2.py`, never change existing functions.

## 9. Files

- Library: `projects/maxtoki/setup/hooks_v2.py`
- Unit tests: `projects/maxtoki/setup/test_hooks_v2.py` → `runs/exhaustive-mapping-217M/outputs/v2_hooks_tests/`
  (`summary.json`, `cell_*_row*.json`, `run_config.json`)
- Site check: `runs/sae-atlas-217M/scripts/v2_sae_site_check.py` → `runs/sae-atlas-217M/outputs/v2_sae_site_check/`
- Zero-edit control: `runs/exhaustive-mapping-217M/scripts/v2_zero_edit_control.py` →
  `runs/exhaustive-mapping-217M/outputs/v2_zero_edit_control/` (`summary.json`, `zero_edit_table.csv`,
  `cells/cell_*.json` + `*_vecs.npz`, `early_cells_tokens.npz`, `run_config.json`)
- This report: `runs/exhaustive-mapping-217M/V2_HOOKS_REPORT.md`

## Plain-words summary

The old intervention code put its edits one step too late in the model. That quietly deleted a whole layer of the
model every time, and at the last layer it applied the final normalisation twice. When several edits were combined,
the last one wiped out the others. I wrote a small library that puts each edit exactly where the feature
dictionary (SAE) was trained, and computes it from what the model is actually carrying at that moment, so edits
combine properly. Five tests on real cells pass: an empty edit changes nothing (difference 0.0), an edit only
changes later layers, combined edits really combine, editing a silent feature changes nothing, and the last-layer
edit is not normalised twice. Every SAE fits its own site best, which confirms the layer mapping. Reading all the
deployed scripts, 9 have a hook or layer bug, and every intervention result in the paper depends on one of them.
Finally, running the old steering code with an edit of exactly zero gives the published steering numbers
(+0.096, +0.018, −0.008, +0.014, +0.004 against +0.096, +0.018, −0.008, +0.014, +0.005 published), with the same
genes. So the published steering table measures the bug, not the features.

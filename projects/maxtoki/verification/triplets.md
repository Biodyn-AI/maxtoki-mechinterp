# KEY = triplets — Reviewer 2 on "2,980 tested feature triplets"

All paths are absolute. "Verified" means I read the file or computed the number myself.
"Inferred" means it follows from code or numbers but I did not run the model.
I ran no model forward pass and no GPU/MPS job.

Short names used below:
- RUN = `<REPO_ROOT>/projects/maxtoki/runs/exhaustive-mapping-217M`
- SUM = `<REPO_ROOT>/projects/maxtoki/summaries/exhaustive-mapping-217M-FINAL_SUMMARY.md`
- TEX = `<REPO_ROOT>/projects/maxtoki/paper-plos-one/main.tex`
- HF = `<REPO_ROOT>/projects/maxtoki/.venv/lib/python3.12/site-packages/transformers` (version 5.5.4)

---

## 0. Answer in one paragraph

The reviewer is right, and the real problem is bigger than the one they raised.
Only **4 triplets** were ablated in the reported (v2) run. There was also an earlier v1 run with 4 triplets, and those 4 all shared the same L0 and L5 feature.
**2,980 is not a triplet count.** It is the number of downstream L11 SAE features (out of 4,928) whose mean change under the three-way ablation was bigger than 0.01.
Worse, the ablation code has **two bugs**. Together they make the triplet result carry no information about the model:
1. **Overwrite bug.** Each hook replaces a layer's output with a fixed tensor taken from the clean run. So the deepest hook wipes out any upstream ablation. The combined conditions are then identical to single ones: AB = B, AC = C, BC = C, ABC = C. With ABC = C, "superadditive" cases are impossible by arithmetic, so "0 of 2,980" is guaranteed. The three-way ratio 0.190 is just eff_C / (eff_A + eff_B + eff_C). I checked this identity against the reported numbers. It holds to 4 decimals for all 4 triplets.
2. **Off-by-one bug.** `hidden_states[l]` is the *input* to block l. The code replaces the *output* of block l with `hidden_states[l] + delta`. That deletes the whole transformer block l. So every "single-feature ablation" is really "remove block 0 / 5 / 9, plus a tiny feature edit". This explains why results are identical across different features (std < 0.0003). The same bug is in Experiment 1 (4.97 M edges), in Experiment 3 (steering), and in the Stage-2 circuit-tracing run.

---

## 1. How many triplets were ablated (verified)

| Run | Script | Triplets | Distinct features | Cells | Output |
|---|---|---|---|---|---|
| v1 (Apr 20) | `RUN/scripts/experiments_2_3.py` lines 79–110, 137–235 | 4 | L0: F0 only; L5: F0 only; L9: F74, F81, F101, F108 | 50 | `RUN/outputs/experiment2/combinatorial_summary.json` |
| v2 "diversified" (Apr 23) | `RUN/scripts/experiment2_rerun.py` | 4 | L0: F4388, F3872, F2160, F1103; L5: F226, F1879, F3317, F3554; L9: F3852, F3956, F3456, F457 | 50 | `RUN/outputs/experiment2_v2/combinatorial_summary.json` |

- Conditions per triplet: A, B, C, AB, AC, BC, ABC (7 conditions) plus 1 clean pass. So each cell needs 8 forward passes per triplet (`experiment2_rerun.py:193–197`).
- Separate interventions: v2 has 4 × 7 = **28 ablation conditions**. v1 has only 19 distinct conditions, because A, B and AB were the same in all 4 triplets.
- The pipeline spec asked for 8 triplets, 200 cells, and a Cohen's d readout at every target (`<REPO_ROOT>/pipelines/sparse-autoencoders/03-exhaustive-mapping-and-steering.md` lines 346–358). The run used 4 triplets and 50 cells, and read out raw mean changes (no d).

### How the v2 triplets were chosen (verified from code and log)
- Each layer's features come with enrichment terms from `runs/sae-atlas-217M/outputs/phase2/layer_XX/significant_enrichments.csv`. Annotated features: L0 = 2,016, L5 = 3,340, L9 = 3,442 (`RUN/outputs/experiment2_rerun.log`).
- The code lists every (L0, L5, L9) triple that shares at least one term. That gave **171,083,392 candidate triplets** covering **997 distinct "primary" pathways**. The primary pathway is the shared term that comes first alphabetically (`experiment2_rerun.py:100–117`).
- The candidates are sorted by number of shared terms. Pathways are then visited from most candidates to fewest. For each pathway the code takes the first candidate whose L0, L5 and L9 features have not been used yet, and stops at 4 (`experiment2_rerun.py:119–142`).
- Result: 4 triplets, one each in Reactome Post-translational Protein Modification (R-HSA-597592, 8 shared terms), Gene Expression (Transcription) (R-HSA-74160, 12), Immune System (R-HSA-168256, 8), and Cell Cycle Checkpoints (R-HSA-69620, 23).
- So yes: this is "4 distinct L0/L5/L9 features × 4 Reactome pathways". It is a greedy, top-of-list choice, not a random or stratified sample. Inferred: the 4 pathways are the 4 biggest candidate groups, which are very broad Reactome terms.
- No cross-pathway triplet was tested (SUM, validation item 9).

### Selection in v1 (verified)
`find_shared_features` returns the first 4 hits of a nested loop (`experiments_2_3.py:79–95`). That locked onto L0_F0 × L5_F0, with only the L9 feature changing. SUM admits this ("Deviations", Experiment 2).

---

## 2. What "~2,980" is (verified)

`experiment2_rerun.py:273–278` (the same code is in `experiments_2_3.py:218–221`):
```python
abc_effect = np.abs(condition_effects["ABC"])
sum_singles = (np.abs(condition_effects["A"]) + np.abs(condition_effects["B"]) + np.abs(condition_effects["C"]))
n_superadditive = int((abc_effect > sum_singles * 1.1).sum())
n_targets_with_effect = int((abc_effect > 0.01).sum())
```
- `condition_effects[X]` is a vector with 4,928 entries, one per **L11 SAE feature**. Each entry is the mean over 50 cells of (per-cell mean over token positions of `z_ablated − z_clean`) (`experiment2_rerun.py:238–253`).
- 2,980 = the number of L11 SAE features with |mean change under ABC| > 0.01. That is 60.5% of 4,928.
- Values per triplet. v2: 2,980 / 2,980 / 2,979 / 2,977. v1: 2,980 / 2,976 / 2,980 / 2,980.
- So 2,980 counts **downstream measurements for one triplet**. It does not count independent interventions. The paper gets this wrong in two places:
  - TEX line 988–989 (pipeline table, likely the reviewer's "Table 3"): "combinatorial ablation of 2{,}980 feature triplets".
  - TEX lines 1125–1130 (likely the reviewer's "lines 245–249"): "returns zero such cases out of 2{,}980 tested. The three-way redundancy ratio is 0.190".
  - The same text is in `<REPO_ROOT>/projects/maxtoki/paper-biosystems/main.tex` lines 879 and 996–998.

## 3. How synergy and redundancy were defined (verified)

- **Superadditive (synergy)**, per target t: |e_ABC[t]| > 1.1 × (|e_A[t]| + |e_B[t]| + |e_C[t]|). Only targets with |e_ABC[t]| > 0.01 are counted. There is no test statistic, no variance, no null and no p-value. It is one point estimate compared with a 10% margin.
- The spec defines the test differently. It uses Cohen's d over 200 cells, includes a target if |d| > 0.5 in any condition, and calls a target superadditive if |d_ABC| > sum (no 10% margin). It also defines a Möbius interaction term I_ABC (spec lines 377–390, 418–428). The MaxToki run does **not** compute I_ABC.
- **Redundancy ratios** are ratios of target-averaged magnitudes: eff_X = mean over 4,928 targets of |e_X[t]|.
  - pairwise XY = eff_XY / (eff_X + eff_Y)
  - three-way = eff_ABC / (eff_A + eff_B + eff_C)
  - marginal C|AB = (eff_ABC − eff_AB) / eff_C
  - (`experiment2_rerun.py:255–271`)
- Reported values (v2, `experiment2_v2/combinatorial_summary.json`):

| Triplet | AB | AC | BC | pair mean | three-way | marginal C\|AB | superadditive / targets |
|---|---|---|---|---|---|---|---|
| T0 | 0.1652 | 0.2189 | 0.5861 | 0.3234 | 0.1896 | 0.2938 | 0 / 2,980 |
| T1 | 0.1652 | 0.2189 | 0.5861 | 0.3234 | 0.1896 | 0.2938 | 0 / 2,980 |
| T2 | 0.1653 | 0.2190 | 0.5861 | 0.3235 | 0.1897 | 0.2938 | 0 / 2,979 |
| T3 | 0.1656 | 0.2191 | 0.5858 | 0.3235 | 0.1897 | 0.2930 | 0 / 2,977 |
| summary | | | | 0.3235 (std 0.0) | 0.1896 (std 0.0001) | | 0.0% |

- In v1 the "pairwise_ratio" is **AB only**: 0.1654, three-way 0.1897. SUM's headline table reports "Pairwise 0.165" next to the paper's 0.74. But the paper's 0.74 is a mean over pairs, so these are different quantities. The "1.00 → 0.324 → 0.190" chain in SUM's validation item 7 uses the v2 pair mean.

## 4. Cells used (verified)

- **Experiment 2:** K562 **non-targeting-control** cells from Replogle CRISPRi. The file is `<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad`: 30,010,775,846 bytes, dense float32 X of shape 643,413 × 6,546, cell_line = k562 (`setup/dataset_loader.py:149–187`). The code draws 100 cells with seed 42 (`experiment2_rerun.py:162–179`). All 100 tokenized ("100 cells" in `RUN/outputs/experiments_2_3.log`). The first **50** were used for every condition. The same 50 cells appear in v1 and v2.
- Each cell is one **single-cell** sequence: BOS, then up to 2,046 genes ranked by median-normalized expression, then EOS (`max_len=2048`, `setup/maxtoki_adapter.py:112–156`). This is *not* MaxToki's multi-cell trajectory input.
- **Experiment 1:** 20 K562 control cells (seed 42+111).
- **Experiment 3:** 200 Tabula Sapiens immune cells (`<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad`, 19,773,999,714 bytes, seed 42+3333). The 50 cells in the earliest pseudotime quartile were steered. Pseudotime is PC1 of log1p-normalized counts.

---

## 5. CRITICAL: the ablation hooks are broken (two bugs)

### 5a. Off-by-one: the hook deletes a whole transformer block

- In transformers 5.5.4, `hidden_states[0]` is the input to block 0 (the embeddings). Each later entry is the output of the next block. The last entry is replaced by `norm(final output)`.
  - `HF/utils/output_capturing.py:108–109` appends `args[0]` of the first decoder layer.
  - `HF/utils/output_capturing.py:259–263` swaps the last entry for `last_hidden_state`.
  - `HF/models/llama/modeling_llama.py:348–349` records `LlamaDecoderLayer` outputs.
  - `HF/models/llama/modeling_llama.py:421` applies the final norm.
- The project's own adapter says the same (`setup/maxtoki_adapter.py:262–264`): "first is the post-embedding residual stream, then one per transformer block".
- The SAEs for "layer l" were trained on `hidden[l]` (`runs/sae-atlas-217M/scripts/phase0_extract_positions.py:107`, `full_12layer_pipeline.py:141`).
- The exhaustive-mapping code does this:
  ```python
  h_patched = clean_hs[al] + delta            # hidden_states[al] = INPUT of block al
  xt.model.model.layers[al].register_forward_hook(_hook)   # replaces OUTPUT of block al
  ```
  (`experiment2_rerun.py:224–230`; `experiments_2_3.py:173–177`; `experiment1_exhaustive.py:160–164`; `experiment3_rerun.py:266–273`)
- So the output of block al is set to its own input plus a small delta. **Block al's attention and MLP are skipped.**
- The earlier sae-atlas Phase 6 script used the correct index, `hidden_states[PROBE_LAYER + 1]` (`runs/sae-atlas-217M/scripts/phase6_patching.py:125–127`). So the convention was known in this project, and the exhaustive-mapping scripts broke it.

### 5b. Overwrite: the deepest hook erases upstream ablations

- Each hook returns a **fixed tensor computed from the clean run**. When conditions are combined, the later hook replaces the whole residual stream at its layer. So nothing from earlier hooks survives. That gives AB ≡ B, AC ≡ C, BC ≡ C and ABC ≡ C.
- **Numeric check (verified).** I solved for A:B:C from the reported AB and BC ratios, then predicted the other three numbers. Script: `<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/check_identity.py`.

| Triplet | A:B:C | AC obs / pred | three-way obs / pred | marg C\|AB obs / pred |
|---|---|---|---|---|
| T0 | 3.569 : 0.706 : 1 | 0.2189 / 0.2189 | 0.1896 / 0.1896 | 0.2938 / 0.2938 |
| T1 | 3.569 : 0.706 : 1 | 0.2189 / 0.2189 | 0.1896 / 0.1896 | 0.2938 / 0.2938 |
| T2 | 3.566 : 0.706 : 1 | 0.2190 / 0.2190 | 0.1897 / 0.1897 | 0.2938 / 0.2938 |
| T3 | 3.563 : 0.707 : 1 | 0.2191 / 0.2192 | 0.1897 / 0.1898 | 0.2930 / 0.2929 |

- Consequences:
  - Three-way ratio = eff_C / (eff_A + eff_B + eff_C).
  - Pairwise BC = eff_C / (eff_B + eff_C).
  - Per target, |e_ABC| = |e_C| ≤ |e_A| + |e_B| + |e_C|. So **superadditivity cannot happen**. "0 / 2,980" is fixed by the arithmetic, not by the model.
  - SUM's reading of AB 0.165 vs BC 0.586 ("the code is treating pairs distinctly") is wrong. The spread only reflects the sizes of the single effects: deleting block 0 moves the L11 SAE features 3.57× as much as deleting block 9, and deleting block 5 moves them 0.71× as much.

### 5c. Independent signs that the effect is the block deletion, not the feature

1. **Exp 1 has zero L6 edges (verified).** All 1,000 feature files in `RUN/outputs/experiment1/` report 0 edges at L6. There are 2,532,553 at L8 and 2,437,543 at L11 (total 4,970,096). Why: the transformers capture hook was registered before the patch hook, so it records block 5's *unpatched* output as `hidden_states[6]`. That confirms the indexing in 5a. The Stage-2 run shows the same pattern: `runs/circuit-tracing-217M/outputs/circuit_edges.csv` has no edges at target layer = source + 1 for any source layer (0, 3, 6 or 9).
2. **The "feature" barely matters.** Exp 1 edges per feature: mean 4,970.1, median 4,967, min 4,900, max 5,400, SD 25.2 over 1,000 features. (SUM says "range 0–5,400". The real minimum is 4,900.) In Exp 2, four triplets with completely different features agree to 4 decimals.
3. **Some features are almost never active (verified from `runs/sae-atlas-217M/outputs/phase2/layer_XX/feature_catalog.json`).** L9_F3852 (in T0) has activation frequency 0.000146, about 0.3 active tokens per 2,048-token cell. In most cells its delta is exactly zero. Yet T0 matches T1–T3. The L0 features have max activation only 0.14–0.25, with unit-norm decoder columns, while token embeddings have norm about 0.74 (median).
4. **Steering ignores α and the feature (section 6).**

### 5d. Extra problem at L11 steering
For li = 11 the hook sits on `layers[10]` and replaces its output with `hidden_states[11]`, which has **already been through the final RMSNorm**. The model then applies the norm again. The final-norm weight has mean 2.90 (SD 0.22, max 5.57), read from `setup/MaxToki-217M-HF/model.safetensors`. So a double norm changes the logits even when delta = 0. Inferred: this is why L11 has the largest |Δlogit| (1.07) with a tiny Δs.

### 5e. Did the audits catch it?
No. `audits/audit_a2_groupkfold_status.md:94–118` treats the across-triplet std < 0.0003 as a stability check ("DOES NOT APPLY (already addressed)"). `audits/audit-20260507.md:87` rates exhaustive-mapping "clean".

---

## 6. Steering results (verified from `RUN/outputs/experiment3/steering_summary.json` and `RUN/outputs/experiment3_rerun.log`)

Per-layer means at α = 5, with α = 2 for comparison (3 features per layer, 50 early cells each):

| Layer | Δs α=5 | Δs α=2 | α5/α2 | frac_mat α=5 (α=2) | frac_dir α=5 | mean abs of mean Δlogit α=5 | top up-gene (all 3 features) |
|---|---|---|---|---|---|---|---|
| L0 | +0.096226 | +0.096279 | 0.999 | 0.98 (0.98) | 0.34 | 0.4212 | SOD3 / SOD3 / KITLG |
| L3 | +0.017829 | +0.017750 | 1.004 | 1.00 (1.00) | 0.40 | 0.1178 | C1QTNF3-AMACR ×3 |
| L6 | −0.007897 | −0.007827 | 1.009 | 0.00 (0.02) | 0.26 | 0.0432 | C1QB ×3 |
| L9 | +0.014076 | +0.014151 | 0.995 | 1.00 (1.00) | 0.64 | 0.0380 | PAQR3 ×3 |
| L11 | +0.004817 | +0.004602 | 1.047 | 0.79 (0.79) | 0.67 | 1.0747 | FTL ×3 |

- Per-feature Δs at α=5:
  - L0: F40 +0.0958, F1657 +0.0962, F4797 +0.0968. F40 and F4797 correlate negatively with pseudotime and F1657 positively, yet all three give the same result.
  - L3: 0.0179 / 0.0179 / 0.0177
  - L6: −0.0079 ×3
  - L9: 0.0139 / 0.0141 / 0.0142
  - L11: 0.0044 / 0.0043 / 0.0057
- frac_dir is just the sign of a mean logit shift that does not depend on the feature, checked against each feature's correlation sign. That is why L0 gives 0.04 / 0.96 / 0.02.
- Paper ratios (TEX lines 1370–1374, figure caption 1381–1392):
  - L0/L3 = 0.096226 / 0.017829 = **5.40** (the "5×")
  - L0/L11 = 0.096226 / 0.004817 = **19.98** (the "20×")
  - L0/L9 = 6.84
  - The "L6 inversion": Δs = −0.0079, frac_mat = 0.00 at α=5.
- **What this means.** Δs is the same across features, is the same at α=2 and α=5 (within 0.5–5%), and gives the same top gene for all 3 features at a layer. That is what a hook deleting block li (and double-normalizing at L11) would produce. Inferred: the L0 "push", the 5×/20× ratios and the L6 inversion measure what happens when a block is removed, not the effect of steering a feature. SUM's own "α-scaling saturation" section (the "incidental finding") is this artifact.
- Other mismatches between SUM and the files:
  - SUM says cos(g_early, g_late) ≈ 0.999. The log and `state_signatures.npz` give **0.8810** (verified by recomputing).
  - SUM says Exp 3 took "~4 min". The file times run from 19:43 (signatures) to 20:05 (last feature JSON), about 22 min, which is about 0.44 s per forward pass over 3,000 passes (inferred from file mtimes).

### Permutation null over feature selection
- **None exists** (verified). No script in `RUN/scripts` mentions a null or random features, and no other steering script exists under `projects/maxtoki/runs`. The paper admits this (TEX 1373–1374, 1386).
- The cheapest decisive test, with the *current* code, is α = 1 (delta = 0) at each of the 5 layers. If Δs stays at +0.096 / −0.0079 and so on, the steering table is purely a hook artifact. Cost: 5 layers × 50 cells × 2 passes ≈ 500 passes ≈ **4–5 min**.
- A real null after the hook is fixed: steer 20 random alive features per layer with 1 steered pass per cell. The clean mean logits for all 200 cells are already cached in `RUN/outputs/experiment3/state_signatures.npz` (shape 200 × 20,275). Cost: 100 features × 50 cells × ~0.41 s ≈ **34 min per α value**.

---

## 7. Cost of running many more triplets on this machine

### Inputs on disk (all verified present)
- Model: `<REPO_ROOT>/projects/maxtoki/setup/MaxToki-217M-HF/model.safetensors` (867,797,760 bytes; 11 layers, hidden 1,232, vocab 20,275).
- SAEs: `<REPO_ROOT>/projects/maxtoki/runs/sae-atlas-217M/outputs/phase1/layer_00…layer_11/sae_final.pt` (48,597,489 bytes each; d_sae 4,928, k 32). Note: the L0 SAE has 2,430 dead features (49.3%). The L11 SAE has 115.
- Annotations: `runs/sae-atlas-217M/outputs/phase2/layer_XX/{significant_enrichments.csv, feature_catalog.json}` (present for L0, L5, L9).
- Cells: `replogle_concat.h5ad` (above; reading 50–100 rows is about 2.6 MB). Gene-median pkl and symbol→Ensembl pkl are in `.../crispri_validation/data/`. Token dictionary: `setup/token_dictionary.json`.
- Code: `setup/topk_sae.py`, `setup/maxtoki_adapter.py`, `setup/dataset_loader.py`. The venv has torch 2.11.0 and transformers 5.5.4. The checkpoint config was written by transformers 4.44.2. In 5.5.4 the decoder layers return a tensor, which is why the v1 hook `_p.to(output.device)` did not crash.

### Measured speed (from logs, MPS, float32, max_len 2048)
- Exp 2 v2: 4 triplets × 50 cells × 8 passes = 1,600 passes in **15.0 min**, so **0.56 s per pass**, or **3.75 min per triplet** (`RUN/outputs/experiment2_rerun.log`).
- Exp 1: 1,000 features × 20 cells × 2 passes = 40,000 passes in **331.2 min**, so 0.50 s per pass (`RUN/outputs/experiment1.log`).
- Logits-only pass: 200 in 82.2 s, so 0.41 s per pass (`RUN/outputs/experiment3_rerun.log`).
- SUM's "~40 min exp2" cannot be checked. v1 has no wall-clock line.

### Estimates (inferred from 0.56 s per pass, 7 ablated passes per triplet per cell, clean pass shared)
| Design | Passes | Time |
|---|---|---|
| 100 random triplets, 50 cells | 100×7×50 = 35,000 | ≈ 5.5 h |
| 500 random triplets, 50 cells | 175,000 | ≈ 27 h |
| 100 triplets, 20 cells | 14,000 | ≈ 2.2 h |
| 500 triplets, 20 cells | 70,000 | ≈ 11 h |
| **Full grid 8 × 8 × 8 features = 512 triplets, singles and pairs shared** (24 singles + 192 pairs + 512 triples = 728 conditions), 20 cells | 14,560 | **≈ 2.3 h** |
| same grid, 50 cells | 36,400 | ≈ 5.7 h |

- The grid is the best design. It gives 512 triplets for the cost of about 146 at 50 cells. It can be split by layer and by annotation, for example 4 annotated + 4 unannotated features per layer, spread over activation frequency.
- Possible savings (not measured): run the base model `xt.model.model` without `lm_head` (the vocab projection is about 12% of the FLOPs); batch 4–8 cells with padding on MPS.
- **Memory** (inferred, no RSS was logged): model 0.87 GB, 4 SAEs 0.19 GB, clean hidden states about 0.12 GB per cell, eager attention about 0.13 GB per layer (transient), CPU SAE encode about 40 MB. Expected peak about 3–4 GB, which fits in 14 GB. Run it alone, not next to other MPS jobs.

### The fix that must come first
Use a **forward pre-hook** on `layers[l]`, so the edit lands on `hidden_states[l]` (the input the SAE was trained on). Compute the delta from the **live** input, so hooks stack:
```python
def make_prehook(sae, mu, fi):             # sae and mu on the same device as the model
    def pre(module, args, kwargs):
        x = args[0]                          # (1, seq, 1232) = live hidden_states[l]
        z = sae.encode(x[0].float(), mu)     # (seq, 4928)
        d = -z[:, fi:fi+1] * sae.W_dec.weight[:, fi][None, :]   # zero feature fi
        return (x + d[None].to(x.dtype),) + tuple(args[1:]), kwargs
    return pre
h = xt.model.model.layers[l].register_forward_pre_hook(make_prehook(...), with_kwargs=True)
```
Required sanity checks before any result:
- (i) a delta = 0 hook gives logits equal to clean (max abs diff < 1e-5);
- (ii) AB ≠ B in general;
- (iii) edges appear at L(l+1).

Also compute the spec's I_ABC and a per-cell bootstrap CI, not the single 1.1× threshold. Most features fire on only 0.1–2% of tokens, so the ablation effects will be small and noisy. Report a noise floor, for example from triplets of dead or never-active features and from a cell-bootstrap.

---

## 8. What this means for the paper's claims

- TEX 988–989, 1125–1130 ("2,980 feature triplets", "zero such cases out of 2,980", "three-way redundancy 0.190"): **not supported**. There were 4 triplets, 2,980 is a target count, and "zero synergy" and 0.190 come from the code, not the model.
- TEX 984–990 "4.97 million edges" and SUM "uniformly dense", "feature-invariant ablation magnitudes": these come from deleting block 5 (inferred; supported by the zero L6 edges and SD 25 edges across 1,000 features).
- TEX 234–236 (abstract) and 1348–1392 (steering, 5×/20×, L6 inversion): likely come from deleting blocks, plus the double norm at L11 (inferred). The α = 1 control in section 6 would settle it in about 5 min.
- The Stage-2 circuit-tracing run (`runs/circuit-tracing-217M/scripts/circuit_trace.py:178–208`) has the same off-by-one and the same missing L(l+1) edges. I did not check which paper claims depend on it.

## 9. What I did NOT do or check
- I ran no forward pass, so the block-deletion reading is not confirmed by an experiment. It rests on the code, the transformers source, the exact ratio identity, the zero L6 edges, and the feature and α invariance.
- I did not check the public GitHub copy at commit 89da276 against the local SUM.
- I did not measure per-cell token lengths or the real memory peak.
- I did not check other runs (attention-GRN, topology-141, manifold-discovery) for hook bugs.
- I did not open `runs/sae-atlas-217M/scripts/phase6_patching.py` in depth. It applies the **layer-5 SAE to `hidden_states[6]`** (lines 36, 47, 127), which is a separate layer mismatch worth checking.

## Plain-words summary
Only 4 triplets were tested. "2,980" is the number of downstream features that moved, not a count of triplets. The ablation code has two bugs. First, each hook overwrites everything before it, so "A+B+C" is really just "C". That makes "zero synergy" certain and turns the 0.190 ratio into simple arithmetic on the single effects. Second, the hook sits one layer off, so it deletes a whole transformer block instead of removing one feature. That is why every feature, every triplet and every steering strength gives the same numbers. The same off-by-one affects the 4.97 M edges and the steering results. The fix is simple (a pre-hook that edits the live input). A test with 512 triplets (an 8×8×8 grid, 20 cells) would take about 2.3 h on this Mac. A 5-minute α = 1 test would show whether the steering table is an artifact.

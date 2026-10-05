## Shared code: input encoding and intervention hooks

Sections 02 to 05 use two shared modules. `projects/maxtoki/setup/inputs_v3.py` builds the model input from counts and checks it. `projects/maxtoki/setup/hooks_v2.py` edits the residual stream with sparse autoencoder (SAE) features. Section 01 (attention, RPE1 cells) has its own extraction path; its input check is described there.

### Input encoding

MaxToki uses the rank-value encoding of Geneformer. For one cell with counts c_g:

1. value(g) = c_g / (sum of counts in the cell) × 10,000 / median(g). Here median(g) is the Geneformer gc104M gene median that the MaxToki tokenizer uses.
2. Keep the genes in the MaxToki vocabulary with value > 0. Sort them from high to low value.
3. Keep the first max_len − 2 genes. Add `<bos>` (token 2) at the start and `<eos>` (token 3) at the end.

The per-cell factor 10,000 / (sum of counts) does not change the order. So any per-cell multiple of the counts gives the same tokens.

Where the counts come from:

| Data file | Stored `X` | Counts used |
|---|---|---|
| K562 (`replogle_concat.h5ad`), Adamson | log1p(CP10k) | round(expm1(X) / u), with u = the smallest non-zero expm1(X) in the cell |
| RPE1 (`ReplogleWeissman2022_rpe1.h5ad`) | integer counts | `X` |
| Tabula Sapiens, Krasnow lung | log1p values | `raw/X` (integer counts) |

For K562 and Adamson the loader first checks that every non-zero expm1(X) value is a whole-number multiple of u. The tolerance is max(10⁻³, 5 × 10⁻⁶ × multiple), and a multiple above 10⁵ fails. Dividing by u and rounding gives integer counts up to one factor per cell. Rounding also removes the float32 noise in `X` (relative error about 3 × 10⁻⁷). In 2 of 9,200 K562 cells, expm1(X) is already a whole number with u = 1. The function `counts_accept_unit_one` accepts these two cells and returns the same result as the main reader for every other cell.

### Per-cell encoding check

`check_encoding(counts, tokens, ...)` compares a token sequence with the counts of the same cell. It passes only if all of these hold:

1. `<bos>` is first and `<eos>` is last. There is no other special token and no repeated gene. Every token is a vocabulary gene with count > 0.
2. counts / median does not increase along the sequence. Values within 10⁻⁶ of each other (relative) count as tied and may come in either order.
3. The length is min(number of expressed vocabulary genes, max_len − 2), and no left-out gene has a larger value than a kept gene.
4. The counts look like counts: whole numbers, or whole-number multiples of the smallest value. Log values fail this test.
5. The sequence agrees with an independent stable sort, except at positions that hold tied values.

`assert_encoding_batch` runs this check on every cell of a batch and stops the run on any failure. Every analysis in sections 02 to 05 calls it before each forward pass.

Tests of the encoding (`projects/maxtoki/setup/test_inputs_v3.py`; outputs in `projects/maxtoki/checks/v3_inputs/`):

| Test | Cells | Result |
|---|---|---|
| Count test on the count source | 530 cells from 10 dataset × context-length groups | 530 / 530 pass |
| Tokens pass `check_encoding` | same 530 | 530 / 530 pass |
| Same tokens as the tokenizer run on integer counts | same 530 | 528 / 530 identical; the other 2 (RPE1) differ only inside exact ties |
| Negative control | 475 non-RPE1 cells | The stored log1p rows fail the count test in 475 / 475, and the tokens ranked from them (the input order of the deployed analysis) fail `check_encoding` in 475 / 475 |
| Model smoke test (MPS, float32) | 50 cells, context 1,024 to 4,096 | 50 / 50 forward passes give finite logits |

### Intervention hooks

**Where the edit goes.** MaxToki-217M has 11 decoder blocks. `output_hidden_states` returns 12 tensors. `hidden_states[0]` is the token embedding, which is the input of block 0. For l = 1 to 10, `hidden_states[l]` is the output of block l − 1, which is the input of block l. `hidden_states[11]` is the final RMSNorm of the output of block 10, which is the input of `lm_head`. The SAE of layer l is trained on `hidden_states[l]`. So the edit for layer l is a forward pre-hook on block l (l = 0 to 10) or on `lm_head` (l = 11). The edit changes exactly the tensor the SAE reads. At layer 11 the edit comes after the final norm, so the norm is applied once.

**What the edit is.** The SAE code z is computed inside the forward pass from the live input x at the hook. A delta is added to x:

- Ablate(l, F): delta = − Σ over f in F of z_f(x) · W_dec[:, f]. The SAE reconstruction error stays in place.
- Steer(l, f, α), scale mode: delta = (α − 1) · z_f(x) · W_dec[:, f]. α = 1 is an exact no-op.
- ZeroDelta(l): delta = 0. AddVector(l, v): delta = v. Both are for tests.

If several edits target one layer, z is computed once and their deltas are summed. Because z is computed from the live tensor, an edit at a later layer sees the effect of the edits at lower layers. So edits at several layers combine. By default every position is edited, including `<bos>` and `<eos>`. Read-outs use captures of the live input of each layer (before and after the edit at that layer), not `output_hidden_states`.

The analyses load the SAEs of section 02 by passing `sae_dir = projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae`. They load cells with `inputs_v3.load_k562_control_cells`, which picks the same K562 rows as `hooks_v2.load_k562_control_cells` but encodes them from counts.

**Hook unit tests** (`projects/maxtoki/setup/test_hooks_v2.py`; 3 K562 non-targeting control cells, rows 289300, 292833 and 293501; MPS, float32; results in `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v2_hooks_tests/summary.json`). Every "difference" or "change" below is the largest absolute difference over all positions and dimensions.

| Test | What is done | Pass rule | Result (worst of 3 cells) |
|---|---|---|---|
| Layer convention | Compare the pre-hook input of each block with `hidden_states[l]`; compare `hidden_states[11]` with the final norm of the block-10 output | Difference ≤ 10⁻⁶ (≤ 10⁻⁵ for the norm) | 0.0 and 0.0 |
| 1. A zero edit changes nothing | ZeroDelta at each of the 12 sites; Steer with α = 1 at 5 layers | Logit change < 10⁻⁴ | 0.0 |
| 2. An edit changes only later layers | Ablate the most active feature at layer 0, 3, 5, 9, 10 or 11 | Inputs of lower layers and the pre-edit input of the same layer change ≤ 10⁻⁶; every later layer and the logits change > 10⁻⁴; the observed delta equals −z_f · W_dec[:, f] within 10⁻⁴ | Lower layers: 0.0. Smallest later-layer change: 0.27 |
| 3. Combined edits differ from single edits | Ablate one feature at L0 (A), L5 (B) and L9 (C); compare AB with B, ABC with C, BC with C, ABC with BC | Each difference > 10⁻⁴ at the input of `lm_head` and in the logits | Smallest AB vs B: 0.55. Smallest ABC vs C: 2.55 |
| 4. A silent feature changes nothing | Ablate a feature that is inactive at every position, at L0, L5 and L11 | Delta exactly 0; logit change < 10⁻⁵ | 0.0 |
| Layer-11 path | Add a random vector at layer 11; ablate a feature at layer 11 | Logit change equals `lm_head`(delta) within 10⁻³ | 5.6 × 10⁻⁵ |

These unit tests used the module's default SAE files (the SAEs of the deployed atlas) and cells from `hooks_v2.load_k562_control_cells`, which ranks the stored log1p values. They test the hook mechanics, which do not depend on the SAE or on the input order. The same four properties were checked again inside the analyses, with the SAEs of section 02 and correctly encoded inputs:

| Property | Analysis | Result |
|---|---|---|
| 1. A zero edit changes nothing | Encoding tests: ZeroDelta at all 12 sites, one correctly encoded K562 cell | Logit change 0.0 |
| | Section 04: ZeroDelta at 4 source layers, 200 cells | Largest SAE-code change 0.0; 0 edges |
| | Section 05: ZeroDelta at L0, L5, L9 and all three together, 2 cells | Logit change 0.0; SAE-code change 0.0 |
| 2. An edit changes only later layers | Section 04: a forward pass that starts at block s from the stored clean input, against a full forward pass (2 cells × 8 features) | Difference 0.0; edges appear at the very next layer from all 4 source layers |
| | Section 05: same comparison for edits at L5 and L9 | Difference 0.0 |
| The edit equals the SAE term | Section 02: ablate all features at layer 5 | The input changes by exactly the SAE reconstruction (largest error 9.5 × 10⁻⁷ on values up to 352) |
| 3. Combined edits differ from single edits | Section 05: AB vs B for all 64 (A, B) pairs; ABC vs C for all 512 triplets | 0 of 64 and 0 of 512 equal; smallest largest-differences 0.020 and 0.019 |
| 4. A silent feature changes nothing | Section 05: 24 triplets that contain an exactly silent L0 or L9 feature | Three-way interaction term 0.0 exactly in every cell and target; 2,144 of 2,144 paired conditions with and without the silent feature are bit-identical |

### Files

- Code: `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`, `projects/maxtoki/setup/topk_sae.py` (SAE class), `projects/maxtoki/setup/maxtoki_adapter.py` (tokenizer and model loader).
- Tests: `projects/maxtoki/setup/test_inputs_v3.py`, `projects/maxtoki/setup/test_hooks_v2.py`.
- Test outputs: `projects/maxtoki/checks/v3_inputs/` (`summary.json`, `cpu_results.json`, `cells_cpu.csv`, `model/`), `projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v2_hooks_tests/` (`summary.json`, one JSON per cell).

> **Warning.** This file, [`projects/maxtoki/summaries/`](summaries/), [`projects/maxtoki/audits/`](audits/) and the run READMEs and `FINAL_SUMMARY.md` files under [`projects/maxtoki/runs/`](runs/) are the deployment's original, uncorrected record (April–May 2026). Many of their headline numbers are wrong.
> The 149 confirmed errors are listed in S1 Table: [`projects/maxtoki/paper-plos-one/supporting/S1_Table.csv`](paper-plos-one/supporting/S1_Table.csv). The working ledger is in [`projects/maxtoki/framework-eval/`](framework-eval/).
> The corrected analyses are described in S3 Appendix: [`projects/maxtoki/paper-plos-one/supporting/S3_Appendix.pdf`](paper-plos-one/supporting/S3_Appendix.pdf). Their outputs are the `v2_`, `v2b_` and `v3_` folders and the `V2_`, `V2B_` and `V3_` reports under the run folders.
> Paths are given from the repository root.

# MaxToki — target model for pipeline runs

First external target model for the `biomi_automation` interpretability pipelines. This folder holds run artefacts, configs, and notes for applying the pipelines in `../../pipelines/` to MaxToki.

## Model identity

- **Name:** MaxToki
- **Paper:** Gómez Ortega & Theodoris, "Temporal AI model predicts drivers of cell state trajectories across human aging", bioRxiv 2026 — [10.64898/2026.03.30.715396v1](https://www.biorxiv.org/content/10.64898/2026.03.30.715396v1)
- **Authors' lab:** Theodoris lab, Gladstone Institutes
- **HF model card:** [`theodoris-lab/MaxToki`](https://huggingface.co/theodoris-lab/MaxToki)
- **Code repo:** [`NVIDIA-Digital-Bio/MaxToki`](https://github.com/NVIDIA-Digital-Bio/MaxToki)
- **License:** Apache-2.0
- **Gated:** no (public, direct download)
- **Training data:** ~1 trillion gene tokens from 175M human single-cell transcriptomes spanning birth to 90+ years; two-stage training — (1) single-cell transcriptome generation, (2) multi-state trajectory modeling
- **Primary task framing:** autoregressive generation of past / intervening / future cell states along context-specific trajectories; in-context learning to unseen trajectories; in-silico perturbation

## Available checkpoints

Both variants are released in two formats on the same HF repo (total repo ≈ 21 GB):

| Variant | Format | Files | On-disk size |
|---|---|---|---|
| **MaxToki-217M** | `MaxToki-217M-HF/` (HF safetensors)       |  4 |  ~0.87 GB |
| **MaxToki-217M** | `MaxToki-217M-bionemo/` (BioNeMo distcp) |  8 |  ~0.90 GB |
| **MaxToki-1B**   | `MaxToki-1B-HF/` (HF safetensors)         |  3 |  ~4.20 GB |
| **MaxToki-1B**   | `MaxToki-1B-bionemo/` (BioNeMo distcp)    | 70 | ~14.70 GB |

For mechanistic-interpretability work we want the **HF safetensors** variants — they load cleanly with `transformers` and expose attention/hidden states via standard `output_attentions=True` / `output_hidden_states=True` hooks, which the pipelines depend on. The BioNeMo variant is sharded distcp and is intended for NeMo training/inference infra.

## Architecture (both variants use `LlamaForCausalLM`)

Confirmed from `config.json` on HF:

| Field | MaxToki-217M | MaxToki-1B |
|---|---|---|
| `model_type` | `llama` | `llama` |
| `architectures` | `LlamaForCausalLM` | `LlamaForCausalLM` |
| `hidden_size` | 1232 | 2304 |
| `intermediate_size` | 2464 | 4608 |
| `num_hidden_layers` | 11 | 20 |
| `num_attention_heads` | 8 | 16 |
| `num_key_value_heads` | 8 (MHA) | 8 (GQA, ratio 2:1) |
| `head_dim` | 154 (implied) | 144 (explicit) |
| `max_position_embeddings` | 4096 | 4096 |
| `rope_theta` | 10,000 | 500,000 |
| `rope_scaling` | none | `llama3` (factor 1.0, low/high 1.0/4.0) |
| `rms_norm_eps` | 1e-6 | 1e-5 |
| `vocab_size` | 20,275 | 20,275 |
| `tie_word_embeddings` | false | false |
| `hidden_act` | silu | silu |
| `torch_dtype` | float32 | float32 |
| `bos_token_id / eos_token_id / pad_token_id` | 2 / 3 / 0 | 2 / 3 / 0 |

The shared 20,275-token vocabulary strongly implies a shared gene-symbol tokenizer across both sizes — the pipelines that depend on gene-id ↔ vocab mappings only need to be wired up once.

## What this means for our pipelines

- **Attention extraction (attention-GRN, topology, manifold-discovery pipelines):** works out-of-the-box via `transformers` `output_attentions=True`. 217M has 11 × 8 = **88 attention heads**; 1B has 20 × 16 = **320 attention heads**. Per-layer head-level attention tensors are the primary input to these pipelines.
- **Residual-stream hooks (spectral-geometry, SAE mega-pipeline):** standard `output_hidden_states=True` gives 12 hidden-state tensors for 217M (embed + 11 layers) and 21 for 1B (embed + 20 layers). SAE training in the mega-pipeline targets per-layer residual streams, so layer budgets scale linearly with depth.
- **Context length:** 4096 tokens. Single-cell trajectory inputs are "sequences of cells" rather than "sequences of genes within one cell", so 4096 is a *trajectory-length* budget, not a gene-count budget — pipelines that assume the Geneformer convention of `genes_per_cell ≤ 2048` need to be reframed for MaxToki's multi-cell context.
- **Two-stage training implication:** the model was trained first on single cells, then expanded to multi-cell trajectories. Interpretability analyses should probably run in **both** modes — single-cell prompt (stage-1-style) and trajectory prompt (stage-2-style) — because attention/SAE features may differ systematically between the two regimes.

## Environment

- Python 3.12.9 in `projects/maxtoki/.venv/`
- Frozen package list at [`requirements.txt`](requirements.txt) (71 packages, generated 2026-05-07)
- Reproduce with: `python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`
- Hardware used across runs: MacBook Pro (Apple Silicon, 32 GB unified memory), MPS backend, float32

## Plan for running the pipelines against MaxToki

We'll start with the smaller **MaxToki-217M-HF** variant for fast iteration, then re-run the headline analyses on **MaxToki-1B-HF**. Proposed order (cheap → expensive):

1. **Sanity / setup** — download HF variants, clone `NVIDIA-Digital-Bio/MaxToki` into `repos/`, pin commit, verify attention+hidden-state hooks, identify tokenizer + gene-symbol mapping, confirm trajectory prompt format from NVIDIA code.
2. **`residual-stream-spectral-geometry`** — cheapest; needs only hidden states. Lets us validate the full hook path and produces a first look at MaxToki's representational geometry.
3. **`attention-grn-extraction-and-evaluation`** — next cheapest; reuses the attention extraction path from (1). Runs the 153-test evaluation battery.
4. **`topology-geometry-141-hypotheses`** — reuses gene-embedding extractions, adds Geneformer V2-316M + scGPT as cross-model references.
5. **`sparse-autoencoders/` mega-pipeline** — most expensive (per-layer SAE training). 217M has 11 layers, so one full SAE sweep is 11 trainings; 1B has 20.
6. **`manifold-discovery-extraction-compactification`** — most open-ended; uses all of the above as priors. Run last.

Each subfolder below will be created per pipeline as it's executed:

```
projects/maxtoki/
├── README.md                           ← this file
├── setup/                              ← download + smoke-test scripts, commit pin
├── runs/
│   ├── spectral-geometry-217M/
│   ├── attention-grn-217M/
│   ├── topology-141-217M/
│   ├── sae-217M/
│   └── manifold-discovery-217M/
└── notes/                              ← observations, surprises, deviations from paper-of-origin
```

## Status of pipeline runs against MaxToki-217M

| # | Pipeline | Status | Headline verdict |
|---|---|---|---|
| 1 | Setup / smoke-test | ✅ done | HF safetensors load cleanly, attention + hidden-state hooks work, tokenizer + gene-median resolved |
| 2 | `residual-stream-spectral-geometry` | ✅ done (Phases 0-9b; Phase 10 autoloop deferred) | **Mixed replication.** Learned depth-wise rank compression replicates (6× collapse, feature-shuffle null 6.1×). PPI/STRING gradient does NOT replicate (Δ700→900 = −0.16). TF-vs-target AUROC above chance at 5/12 layers but weaker than paper's scGPT. Cell-type marker clustering strong. **Phase 8 added 2026-05-03**: CKA across cell samples shows MaxToki is *much more stable* than scGPT (1.000→0.991 vs paper's 0.979→0.779). **Phase 9b added 2026-05-03**: MaxToki and Geneformer V2-316M share the identical 20,275-token tokenizer; cross-model cosine alignment is significant (Pearson 0.382, p=0; paper's null does NOT replicate); STRING PPI is convergent across architectures (MaxToki layer-0 z=5.62, Geneformer static SV3 z=5.03). See `summaries/spectral-geometry-217M-FINAL_SUMMARY.md`. |
| 3 | `attention-grn-extraction-and-evaluation` | ✅ done (all 4 runs: 217M K562 + RPE1 + Adamson + 1B K562) | **Clear negative.** Attention captures co-expression, not regulatory logic. Curveball null z-score ≈ 0 in every run; incremental ΔAUROC ≤ +0.002 over gene-only baseline; residualised AUROC collapses to chance. See `summaries/attention-grn-217M-FINAL_SUMMARY.md`. |
| 4 | `topology-geometry-141-hypotheses` | ✅ done (headline backbone + extension phases, 2026-05-03 / 2026-05-07) | **Clean negative across the headline tests.** Persistent homology 0–1/12 layers feature-shuffle-sig per domain (vs paper's 11–12/12); rewire null kills any signal (0/12). Manifold distance hierarchy collapses — paper's strongest metric (triangle-defect spectrum) is anti-predictive (mean Δ −0.055 vs paper's +0.026). H123 signed motif-community (the source paper's strongest single finding, 22/22 rows positive in scGPT) yields **0–1/12 layers positive null-gap per domain** on MaxToki. H141 strict max-null mean margin **−0.050** (4/36 tests pass; paper-immune was +0.012 robust). **Phase 4 cross-model CCA done 2026-05-07**: Phase 14 vs Geneformer V2-316M gives CCA r=0.78 (within autoregressive-Ensembl family); **Phase 4 vs scGPT gives CCA r=0.40 with pairwise Pearson ≈ 0** — cross-model alignment does NOT generalize across architectural families. Source paper's "Layer 1 most-robust" claim survives within the autoregressive-Ensembl family but breaks against scGPT (continuous-value rank-binned tokenization, attention-only architecture). **H139 sectional-anisotropy spot-check, H124→H138 annotation chain, and Phase 11 autoloop substitute** also added 2026-05-07; novel positive: H123 motif feature has cross-layer Pearson 0.60–0.68 (single static axis across layers). See `summaries/topology-141-217M-FINAL_SUMMARY.md`. |
| 5a | SAE mega-pipeline — **Stage 1 atlas** | ✅ done | SAEs trained at all 12 taps (L0–L11); ~4,928 features/layer; 50% annotated via GO/KEGG/Reactome/STRING/TRRUST; Phase-6 causal feature-patching specificity 1.01× (paper Geneformer 2.36×) — **does not replicate** at the annotated-feature subset; Phase 9 multi-tissue SAE does not rescue. See `summaries/sae-atlas-217M-FINAL_SUMMARY.md`. |
| 5b | SAE mega-pipeline — **Stage 2 circuit tracing** | ✅ done | Selective trace at L0/L3/L6/L9 produces 2.14M edges (vs paper 52K), mean \|d\|=1.84, 88% inhibitory dominance. Phase-11 CRISPRi directional accuracy **54.6%** (paper 56.4%) — both near-chance, confirming co-expression not causal regulation. See `summaries/circuit-tracing-217M-FINAL_SUMMARY.md`. |
| 5c | SAE mega-pipeline — **Stage 3 exhaustive mapping + steering** | ✅ done (2026-04-23/24) | See detailed findings below. `summaries/exhaustive-mapping-217M-FINAL_SUMMARY.md`. |
| 6 | `manifold-discovery-extraction-compactification` | ✅ done (Phases 0-11 + 7-external + 8-zeroshot + lung-nonhema neg-control + Phase 12 LITE; full Phase 12 cell-level Robust-V2 and Phase 13 H38 second-manifold deferred). Run 2026-05-03/04 via autonomous /loop driver. | **POSITIVE for representational claims; mixed for benchmark superiority.** H65 hematopoietic developmental manifold is recoverable (internal trust 0.811 / branch 0.370 vs null −0.001), externally generalisable (strict non-overlap external trust 0.896, ρ 0.882), zero-shot transferable (frozen head trust 0.827, ρ 0.899), and compressible 2,400× to a 7.7 KB hard-sparse operator that still passes all four gates. Top-1 head is **L10H6** (late layer — diverges from scGPT's L2H5). **But mechanistically distributed**: top-4 SVD factors explain only **18.3%** of pooled ablation impact (paper: 66.2%); core sufficiency test collapses branch from 0.973 to 0.123. Lung non-hematopoietic negative control fails 3/4 gates decisively (rand/donor/branch ≤ 0.013, branch −0.147). **Phase 12 LITE benchmark** (anchor-level, 9 splits, 5 endpoints): extracted head beats raw_log1p (BH-q 0.049 on branch+stage) but does NOT outperform PCA-10/SVD-10/frozen-MaxToki-avgpool — diverges from source paper's "beats all 8 baselines" claim, but at small-anchor scope without scVI/Palantir; cell-level full benchmark still pending. **Phase 13 LITE H38** (curated 7-cat signalling ruler): POSITIVE — trust 0.814, category-holdout 0.293; null fails all 4 gates decisively. **Phase 3a sweeps** (12 candidate rulers, 2026-05-05): two new POSITIVE manifolds discovered — **H95 effector_modality** (trust 0.800, 3-gate fallback) and **H103 B-cell maturation** (trust 0.860, cleanest sweep positive). 7 additional candidates show strong directional signal (null margins +0.5 to +0.9 on rand/donor) but trust subthreshold. **MaxToki-217M's residual encodes a multi-axis biological hub — at least 4 distinct quality-gate-passing geometries (H65 developmental, H38 signalling, H95 effector modality, H103 B-cell maturation) coexist in the same hidden space, three of which transfer cleanly to held-out external donors (H65 / H95 / H103). H103 also transfers zero-shot.** See `summaries/manifold-discovery-217M-FINAL_SUMMARY.md`. |

### Stage-3 headline findings (2026-04-23 / 2026-04-24)

Exhaustive circuit mapping + higher-order combinatorial ablation + trajectory-guided feature steering on MaxToki-217M's L5 residual stream. Six findings, in approximate order of novelty:

1. **Uniformly dense circuit, no heavy-tail split.** 1,000 traced features at L5 yield 4,970,096 significant edges (\|d\|>0.5, consistency>0.7) across downstream layers L6/L8/L11. **100% of features have >1,000 edges** (paper Geneformer: 1.8% hub class). MaxToki is not heavy-tailed.
2. **Annotation-bias finding replicates.** Top-20 hubs are 55% annotated vs overall 50% (Fisher's exact n.s.). Computational centrality is not predictive of annotation status — consistent with paper.
3. **Redundancy is deeper than Geneformer's and feature-invariant.** Diversified v2 triplets (4 distinct L0/L5/L9 features × 4 distinct Reactome pathways) give pairwise ratios AB=0.165 / AC=0.219 / BC=0.586 (mean 0.324) and three-way 0.190 (paper: 0.74 / 0.59). **Across-triplet std<0.0003** — at a given layer, any feature has the same aggregate downstream magnitude. Monotonic deepening 1.00→0.324→0.190 replicates qualitatively.
4. **Zero higher-order synergy replicates.** 0 of ~2,980 target features are superadditive in any of 4 diversified triplets. No conjunctive gates at third order.
5. **Trajectory steering: layer-dependent directionality is non-monotonic.** 15 top switch features (3 per layer at L0/L3/L6/L9/L11) amplified in early-pseudotime Tabula Sapiens immune cells. Using the spec-compliant `Δs = cos(z',late)−cos(z',early)−baseline` metric: L0 `frac_mat`=0.98, L3=1.00, L9=1.00, L11=0.79 (**toward maturity**) but L6=0.00 (**counter-differentiates**). Unlike Geneformer's monotone L0→L17, MaxToki has a mid-layer counter-differentiation basin — plausibly reflecting trajectory-based pretraining distributing temporal signal throughout the stack.
6. **Gene-level coherence under steering.** Top-upregulated genes are biologically sensible per layer: L0→SOD3/KITLG (progenitor/stemness), L6→C1QB (myeloid complement), L11→FTL (terminal erythroid/ferritin).

**Incidental diagnostic findings:**
- α-scaling saturation: α=2 and α=5 produce near-identical Δs and \|Δlogit\| at every layer (agreement ≤ 1%), suggesting single-feature steering saturates before α=2 — likely absorbed by the final RMSNorm + lm_head path.
- MaxToki-217M has 11 transformer blocks, so the SAE atlas's "layer 11" tap is the final residual (output of `layers[10]`). The steering harness required a hook-index clamp `min(li, n_layers−1)` so that steering at L11 attaches to `layers[10]` rather than the non-existent `layers[11]`.

### Cross-pipeline synthesis

Taken together, the completed pipelines on MaxToki-217M produce a coherent narrative. **Five independent pipelines reach the same negative verdict on fine-grained regulatory logic via methodologically distinct probes** (attention-grn curveball z≈0; circuit-tracing CRISPRi 54.6%; sae-atlas Phase 6 patching 1.01× and Phase 8 TF specificity 0%; topology-141 H123 0–1/12 layers; exhaustive-mapping zero higher-order synergy). This convergent negative is the project's strongest implicit positive control.

- **Attention-GRN (co-expression, not regulation)** and **Stage-2 circuit tracing CRISPRi (54.6% near-chance directional)** converge on a shared negative verdict: **MaxToki does not provide attention-head or SAE-feature signals that beat trivial gene-level baselines for predicting CRISPRi/CRISPRa perturbation targets** at a resolution useful for in-silico perturbation. (Note the task-relative framing: this rules out incremental value on perturbation-target prediction; it does not rule out that attention encodes sparse direct-target signal too small to move this integrated endpoint — the endpoint–object mismatch caveat applies to all SCFM evaluations against perturbation DE.)
- **Stage-3 trajectory steering** provides the first **affirmative positive** mechanistic-interpretability finding: MaxToki *does* represent a latent progenitor→mature axis that is causally amplifiable via activation addition at selected layers. This is a coarse cell-identity claim, not a regulatory-mechanism claim — consistent with, rather than contradicting, the negative verdicts above.
- **Spectral geometry** adds the structural frame: learned depth-wise rank compression replicates, but biological axis-to-ontology mappings (STRING confidence gradient, GC-plasma compression trajectory) don't — MaxToki's internal geometry is compressed-but-less-biologically-aligned than scGPT's.
- **Manifold-discovery + topology-141** triangulate the cross-model alignment claim. MaxToki ↔ Geneformer V2-316M alignment holds (CCA r=0.78, Pearson 0.382 p=0) — both autoregressive Ensembl-token models. **MaxToki ↔ scGPT alignment fails: CCA r=0.40, pairwise Pearson ≈ 0** (continuous-value rank-binned tokenization, attention-only architecture). **The "models agree on geometry" claim from the source-paper hierarchy is therefore valid only within architectural families — not as a universal cross-foundation-model property.**

**Cross-pipeline scope qualifier (load-bearing).** Wherever this README or any run summary speaks of "MaxToki agrees with other foundation models" or "cross-model agreement", read it as **within the autoregressive-Ensembl-tokenizer family**. The agreement breaks across architectural families (cf. topology-141 Phase 4 vs scGPT). Any paper drawing on this project should state this scope explicitly.

**Field-wide regulatory-specificity context (load-bearing).** The negative findings on regulatory logic (sae-atlas Phase 6 patching specificity 1.01×; sae-atlas Phase 8t TF specificity 0/48; topology-141 H123 0/12 layers under group-aware CV; circuit-tracing CRISPRi 54.6% near-chance directional; attention-grn ΔAUROC ≤ +0.002 over gene-only baseline) should be read against the **field-wide pattern**: no published single-cell foundation model — including the source-paper baselines (scGPT and Geneformer V2-316M) — has been shown to encode TF-target regulatory logic above ~10% specificity at per-feature level under proper controls. The MaxToki negatives are not a model-specific deficit but the **next data point in a field-wide pattern**. The audit at `audits/audit-20260507.md` (P3, P6, A4, A8) and the sae-atlas Phase 8t chance-baseline analysis confirm that several of these tests are null-saturated by construction (TRRUST too narrow, ≥2-overlap threshold under ChIP-seq too loose). Any paper drawing on this project should frame the SCFM-regulatory negative as a field-wide-pattern observation with a methodological-tightening recommendation, not a MaxToki-specific deficit. See `pipelines/audit-recurring-review-issues.md` P3 ("endpoint–object mismatch") for the recurring reviewer concern this addresses.

## Open questions to resolve during setup

- What does MaxToki's trajectory prompt actually look like at the token level? (need to read NVIDIA repo)
- Is the gene-symbol tokenizer the same one Geneformer uses, or custom? (vocab size 20,275 is close to but not identical to Geneformer's ≈20k)
- What are the stage-1 vs stage-2 input formats, and does the HF checkpoint expose both cleanly?
- For attention-based pipelines: how are positional embeddings handled for "cell tokens" vs "gene tokens" — is there a structural separator we need to respect when computing gene-level attention rollups?

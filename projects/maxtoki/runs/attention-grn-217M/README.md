# attention-grn-217M — MaxToki-217M, Replogle K562

First scoped run of `pipelines/attention-grn-extraction-and-evaluation.md`
against MaxToki-217M-HF. Phases 0a, 1, 2, 3, 4, 12 only.

## Run configuration

| Parameter | Value |
|---|---|
| Model | MaxToki-217M-HF (`../setup/MaxToki-217M-HF/`) |
| Model arch | `LlamaForCausalLM`, 11 L × 8 H × 1232 d, causal, vocab 20 275 |
| Device | MPS (Apple Silicon), float32, eager attention |
| Dataset | Replogle K562 non-targeting (subset of `biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad`) |
| `N_ctrl` | 2 000 |
| `N_hvg` | 1 500 (top-variance HVG ∩ MaxToki vocab) |
| `max_len` | 2 048 tokens (MaxToki context is 4 096; cells with >2 046 genes truncated) |
| Tokenizer | MaxToki Ensembl-ID vocab extracted from `MaxToki-217M-bionemo/context/io.json` (`../setup/token_dictionary.json`) |
| Gene medians | Reused from Geneformer `gene_median_dictionary_gc104M.pkl` (100 % overlap with MaxToki vocab) |
| HGNC → Ensembl | Geneformer `gene_name_id_dict_gc104M.pkl` |
| Primary layer | L8 (0-indexed; ≈73 % depth, matching scGPT L9/12 and Geneformer V2 L13/18 conventions) |
| Reference network | TRRUST v2 only |
| Seed | 42 |

## Phases run (vs full pipeline 0–12)

| Phase | Script | Scope |
|---|---|---|
| 0a — attention extraction | `scripts/phase0_extract.py` | full 2 000-cell K562 NT extraction |
| 0b — value-weighted edges | — | **skipped** |
| 1 — trivial baselines | `scripts/phase1_trivial_baselines.py` | full |
| 2 — incremental value | `scripts/phase2_incremental_value.py` | cross-pert + cross-gene + joint GroupKFold; logreg + GBDT |
| 3 — residualization + nulls + matching | `scripts/phase3_residualization.py` | OLS + GBDT 5-fold cross-fit; curveball null n=50 (paper uses 200); label-shuffle n=1000; propensity matching k=5 |
| 4 — causal ablation | `scripts/phase4_causal_ablation.py` | 6-condition (baseline, top-5 TRRUST, 3×random-5, entropy-matched-5). No 13-condition expansion, no orthogonal/MLP interventions. |
| 5-11 — cross-context, CSSI, biological characterisation, boundary conditions | — | **skipped** |
| 12 — verdict | `scripts/phase12_verdict.py` | pass/fail on 4 mandatory criteria from available evidence |

## Four mandatory criteria evaluated in Phase 12

1. **C1 — Attention beats trivial gene-level baselines** (variance, mean, 1-dropout) on per-perturbation AUROC with Wilcoxon p<0.05 after BH.
2. **C2 — Attention adds incremental value** over gene features under cross-perturbation GroupKFold (ΔAUROC > 0.005).
3. **C3 — Residualized attention retains signal** — OLS cross-fitted residual retains >50 % of above-chance TRRUST AUROC.
4. **C4 — Top TRRUST heads are causal** — ablating top-5 TRRUST-ranked heads drops TRRUST AUROC by >0.02 and more than the median random-5 ablation.

## How to run from scratch

```bash
cd projects/maxtoki
# 0. create venv + install
<HOMEBREW_PREFIX>/bin/python3.12 -m venv .venv
.venv/bin/pip install -q 'torch>=2.3' 'transformers>=4.44' scanpy \
   'anndata>=0.10' h5py 'scikit-learn>=1.3' statsmodels 'scipy>=1.11' \
   'pandas>=2.0' 'numpy<2' tqdm 'huggingface_hub>=0.24' safetensors lightgbm
# 1. download weights
mkdir -p setup/MaxToki-217M-HF
for f in config.json generation_config.json model.safetensors; do
  curl -L -o setup/MaxToki-217M-HF/$f \
    "https://huggingface.co/theodoris-lab/MaxToki/resolve/main/MaxToki-217M-HF/$f"
done
# 2. tokenizer + gene medians — already in setup/ and biodyn-nmi-paper
.venv/bin/python setup/maxtoki_adapter.py  # smoke test
# 3. run phases (Phase 0 is the slow one — ~50 min on M-series Mac)
.venv/bin/python runs/attention-grn-217M/scripts/phase0_extract.py
.venv/bin/python runs/attention-grn-217M/scripts/phase1_trivial_baselines.py
.venv/bin/python runs/attention-grn-217M/scripts/phase2_incremental_value.py
.venv/bin/python runs/attention-grn-217M/scripts/phase3_residualization.py
.venv/bin/python runs/attention-grn-217M/scripts/phase4_causal_ablation.py
.venv/bin/python runs/attention-grn-217M/scripts/phase12_verdict.py
cat runs/attention-grn-217M/outputs/run_report.md
```

## Outputs

All outputs land in `outputs/`:

- `phase0/` — attention tensors, Spearman baseline, gene features, HVG table, control cell metadata
- `phase1/` — per-perturbation AUROCs, per-layer AUROC profile, Wilcoxon table
- `phase2/` — (pert, target) dataset, incremental-value results, Δ-AUROC summary
- `phase3/` — residualization results, null-model z-scores, propensity matched dataset
- `phase4/` — per-head TRRUST ranking, per-condition ablation results
- `phase12_verdict.json`, `run_report.md` — final verdict + human-readable summary
- `phase0.log` — full tee of the Phase-0 run

## Known deviations from the pipeline spec

1. **Causal Llama attention instead of bidirectional BERT.** MaxToki is a decoder-only Llama. The attention matrix is lower-triangular — `attn[i, j]` is zero for `j > i`. This is structurally the same as scGPT (also a gene-token decoder) and the pipeline handles it. For a symmetric comparison against bidirectional Geneformer, one could symmetrize: `edge(A, B) = 0.5 (attn[A→B] + attn[B→A])`, but this run uses the raw (directed) form.
2. **Per-gene medians from Geneformer's `gene_median_dictionary_gc104M.pkl`** instead of MaxToki's own gene median file (the latter is not published on HuggingFace). Overlap is 100 % on gene IDs; numerical values should be within noise since both are trained on ~175 M cells using the same Ensembl vocabulary.
3. **Curveball null at n=50** (paper uses n=200). Scoped down for compute budget.
4. **No layer pre-specification dataset.** The paper pre-specifies L13 for Geneformer using an independent dataset (DLPFC brain). This run uses L8 as primary layer based on depth-ratio parity with scGPT L9/12 and Geneformer L13/18. The per-layer AUROC profile is reported in Phase 1 so the choice is auditable.
5. **Head-mask hook.** `LlamaForCausalLM` doesn't accept `head_mask`. We install a `forward_pre_hook` on each layer's `o_proj` that zeros the contiguous head-slice in the `o_proj` input, equivalent to masking the head's contribution to the residual stream.

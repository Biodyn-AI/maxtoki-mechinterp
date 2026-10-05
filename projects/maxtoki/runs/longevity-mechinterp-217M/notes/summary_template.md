# Summary template (filled in after Stage 1/2/3 complete)

This file is a working draft. Final summary lives in
`projects/maxtoki/summaries/longevity-mechinterp-217M-FINAL_SUMMARY.md`.

## Sections to fill once results land

- **Headline** (one-paragraph synthesis)
- **G1 verdict** (Stage 1 best BA / null p / vs HVG-PCA)
- **G2 verdict** (Stage 2 top-k pc1↔age, within-cell-type controls)
- **Stage 3 SAE descriptive** (recon_mse, top |donor_age_spearman|, robust-feature
  count under formal vs descriptive filter)
- **Cross-pipeline implications** (how this fits with prior MaxToki runs:
  spectral-geometry's age-correlation is unmeasured, attention-GRN gave a
  negative regulatory verdict, manifold-discovery gave a positive
  developmental-axis verdict, so an aging-axis presence/absence here is a
  natural follow-on)
- **Deferred / not applicable**:
  - Stages 4–11 (cross-model convergence, intervention, strict gate, donor-
    threshold sweep, composition-matched reruns) require AIDA phase 1 v1/v2
    plus ≥30 donors after balanced subsampling. Tabula Sapiens immune subset
    20k yields only 17 donors — below the n_donors ≥ 20 threshold for donor-
    permutation calibration in Stage 3, and far below the ≥50 donor threshold
    where the strict gate becomes meaningful.
  - Stage 12 (MethylGPT CpG-window extension) is not applicable: MaxToki is a
    transcriptomics model, not a methylation model.

## What "executing the pipeline" means here

This is a "porting prototype" run as prescribed in pipeline §11:

> Run the pipeline at small scale first: tabula_sapiens_immune_subset_20000.h5ad
> → Stages 1–5 only. If Stage 1 balanced accuracy is < 0.5 + 0.05 above
> permutation null, stop — the model may not encode age at all, and the rest
> of the pipeline is moot.

We added the MaxTokiContextualRuntime adapter, ran Stage 1 (probes + manifold
+ permutation null) and Stage 2 (top-k geometry + within-cell-type controls)
end-to-end on the only locally available age-labelled scRNA-seq cohort. Stage
3 SAE pilot ran in descriptive-only mode because n_donors (17) is below the
upstream donor-permutation calibration threshold (20).

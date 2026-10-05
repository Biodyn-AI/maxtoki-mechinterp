# Held-out task pool for validating the audit checklist

Date: 2026-10-01. Scope: every project under
`<REPO_ROOT>/projects/` except `maxtoki/`.
Nothing in the repository was changed. Only files under this scratch folder were written.

Abbreviations used below:
- `BT` = `<REPO_ROOT>/projects/biotensor`
- `GM` = `BT/codebase/route_genemanifold`
- `RS` = `BT/codebase/route_a/scgpt/runpod_scale`
- `PL` = `<REPO_ROOT>/projects/protein-lm-sae`
- `C2S` = `<REPO_ROOT>/projects/c2s-scale/route_genemanifold_c2s`
- `AC` = `<REPO_ROOT>/projects/atlas-comparison`
- `MEM` = `<CLAUDE_HOME>/projects/<SESSION_DIR>/memory`
  (the user's memory notes; they were the fastest index to the documented errors)

"Verified" means I opened the file and saw the line or the number. "Inferred" means it comes from a
write-up (memory note or results markdown) that I did not re-check against the raw output.

---

## 0. Main points

1. I found **about 40 documented errors** outside `maxtoki/`. **36** are listed with enough detail to use.
   **About 20** have both a before-fix and an after-fix state on disk (code, output, or both).
2. The best pool is in `biotensor` (route_genemanifold, route_a GRN benchmark, route_b, route_branchpoint,
   mechinterp), plus `protein-lm-sae` and `c2s-scale`. `atlas-comparison` and
   `parameter-decomposition-scfm` add a few. `aido-cell-100m` adds only caveats that were never fixed.
3. Many errors fall **outside the ten patterns**. They are plain code or data bugs: AUROC tie ranks,
   label span-filling, sign/handedness of an SVD angle, a broken circular statistic, output files
   overwriting each other, missing mean-centering. A held-out test should include these on purpose. They
   test whether the checklist misses what it was never built to catch.
4. **Contamination risk.** The checklist came from reviews of seven earlier papers: attention, spectral,
   topology (141 hypotheses), SAE, circuits, manifold, exhaustive (`maxtoki/paper-plos-one/main.tex:297-299`,
   `:672-679`). Candidates whose object is one of those papers are **not clean held-out items**:
   H123 (topology-141 paper), CSSI (attention paper, arXiv 2602.17532), and to a smaller degree the
   annotation-rate floor (it scores the published scGPT SAE atlas) and the T141 battery (a new
   re-implementation of the topology-141 pipeline). They are marked "contam." below.
5. The strongest items share four features. The error is real and has numbers. The buggy and the fixed
   state both exist. Inputs are small (under ~500 MB). A reviewer could find the error from code plus
   outputs plus summary, without outside knowledge.

---

## 1. Ranked shortlist (best first)

Rank reflects: ground truth clear with numbers; before and after both on disk; runs on CPU in under
5 minutes on small data; error findable from code + outputs + summary; no contamination; spread over
patterns.

| # | ID | project | error in one line | pattern | before/after on disk | CPU < 5 min? |
|---|---|---|---|---|---|---|
| 1 | BRANCH-NULL | biotensor route_branchpoint | null built so it cannot fail; p pinned at floor 1/41; tested a superseded statistic | P1 + P8 + P3 | both code and outputs | yes (reduce N_NULL) |
| 2 | DISULFID | protein-lm-sae | two-endpoint bond features span-filled over the whole loop | outside (label bug) | both label files + both summaries | yes (seconds) |
| 3 | NORMAN-CEIL | biotensor GRN benchmark | split-half ceiling shares the singles and control across halves, so shared error looks reproducible | P1 (inflated anchor) / P4 | buggy code on disk; fixed code on disk | likely (699 MB h5ad) |
| 4 | T141-TIES | biotensor mechinterp | AUROC ranks via argsort(argsort) give tied values distinct ranks | outside (metric bug); contam. (low-mod) | both outputs; buggy code = 1-line revert | yes with `--pilot` / fewer nulls |
| 5 | DEGREE | biotensor GRN benchmark | target degree (column mean of the model's own score matrix) never tested as a baseline; it beats the model | P2 | outputs + registered numbers; 3-D matcher script not found | yes (seconds) |
| 6 | LOO-ABS | biotensor route_genemanifold | small-fold ridge + abs(Spearman) scores pure noise high | P1 (no chance anchor) / outside | buggy code on disk; no fixed file found | yes (seconds) |
| 7 | BRANCH-RAWCOUNT | biotensor route_celltoken | raw counts fed to scGPT's binned value encoder; fake "compact manifold" | outside (input preprocessing) / P3 | both caches on disk | yes |
| 8 | COEX-SIZE | biotensor route_genemanifold / gene-context | co-expression null modules fixed at 400 genes vs ~1,880/991-gene poles | P1 + P10 | `.bak`, deprecated, and `_v2` scripts + both JSONs | probably (322 MB input) |
| 9 | RANK1-DIR | biotensor GRN benchmark | direction label is 62% set by a per-gene ratio; the 0.500 null was vacuous | P1 / P2 | both scripts + outputs | yes (375 MB bulk h5ad) |
| 10 | SPARSE-AUROC | biotensor GRN benchmark | per-TF AUROC of a sparse binary predictor read as "chance" with no ceiling | P6 (+P1) | outputs + corrected stat in BENCHMARK.md | yes |
| 11 | CIRC-CORR | c2s-scale | circular correlation invalid when one variable is near-uniform; one false gate pass | outside (statistic) / P1 | retracted JSONs + fixed code | yes (from stored JSON) |
| 12 | STEER-FLOOR | biotensor route_genemanifold | per-domain statistic bounded below at 0 for unreachable targets; mean CI excludes 0 only because of that | P1 (+P5) | stored matrices + fixed script | yes (seconds) |
| 13 | HOX-ANALOGY | biotensor route_genemanifold | uniform 1/20,271 chance used as baseline (~6,700x); loose null flatters the model | P1 | both scripts + null JSON | probably |
| 14 | PHASE-HAND | c2s-scale | SVD phase angle has arbitrary sign per dataset; identical biology read as r = -0.983 | outside (sign ambiguity) | both functions in `cc_phase.py` | yes |
| 15 | MATCH-LEAK | biotensor GRN benchmark | "co-expression-matched" negatives still leak co-expression (gate 0.5389) and never match abundance | P1 + P2 | pre-fix JSONs (Aug 1-2) + fixed code; old matcher only in docstring | yes on stored matrices; model runs no |
| 16 | ANNOT-FLOOR | biotensor route_b salvage | ontology annotation rate reported with no chance floor; untrained SAE scores the same | P1 (+outside: Fisher background) ; contam. (mod) | artifacts + scripts; old claims in markdown | yes per README (~30 s) |
| 17 | CTX-INTERACT | biotensor gene-context / route_genemanifold | interaction z = +10.4 from heteroscedastic noise, a contaminated null, and a 65-housekeeping-gene set | P1 + P7 | small inputs; pilot version edited in place | yes |
| 18 | ROUTEB-CENTER | biotensor route_b | latents computed without subtracting the stored mean; top genes = argmax over genes seen once | outside (preprocessing) + P5 | buggy script on disk + reconciliation JSON | maybe (needs scGPT activations) |
| 19 | FROZEN-EMB | biotensor GRN benchmark | frozen pretrained gene vectors "help" only through fewer free parameters; random vectors do the same | P6 / P2 | outputs of all control arms | uncertain (small transformers) |
| 20 | TIE-FILL | atlas-comparison | top-20 gene list filled by argsort ties in alphabetical order for sparse features | outside (code bug) | buggy code on disk; catalogues need download | plant: yes |
| 21 | OVERWRITE | biotensor route_a + route_b | output filename keyed on one parameter; later runs silently overwrite earlier ones | P10 | `.MISLABELED...bak` + regenerated JSON; `one_ruler.py` still collides | code reading only |
| 22 | WINNER-CURSE | protein-lm-sae | features ranked and scored on the same split; inflates SAE-over-neuron gap and fakes a layer-16 peak | P5 | leaky and honest JSONs | no (needs GB activations) unless subsampled |
| 23 | PROT-SPLIT | protein-lm-sae | SAE held-out split by residue, not protein | P4 (leakage) | shipped vs protein-split SAE results | no (retrain) |
| 24 | TWIN-COMPUTE | biotensor omnimodal | "matched-compute" twin matched with a knob (more steps) that itself hurts; fake t = +3.02 win | P2 (unfair control) | outputs only | no (training) |
| 25 | C2S-TOKENIZER | c2s-scale | model compared to expression with a different encoding and gene selection | P3 / P2 | both arms' JSONs | maybe (885 MB cache) |
| 26 | SCREEN-GENESET | biotensor route_genemanifold | six-way gene intersection keeps 6,307 near-housekeeping genes (K562 costs 9,015) + retracted null | P7 + P1 | stale and current screen JSONs | no (GB caches) |
| 27 | RAW-PROFILE-NULL | biotensor route_genemanifold | raw expression profile used as the co-expression baseline; factorised baseline 0.044 -> 0.720 | P2 | scripts + JSONs | no (62,849 cells) |
| 28 | CSSI | biotensor GRN benchmark | max over K strata with no random-strata control | P5; contam. (high) | log + script | no (model attention) |
| 29 | BESTLAYER | biotensor GRN benchmark | raw best-layer AUROC compared across models of different depth | P5 | `score_2d.py` + table | yes if per-layer outputs used |
| 30 | GIII-NULL | biotensor route_b | Gaussian-covariance null for interaction ignores shared per-token scale; a contrary sub-test never reported | P1 + P5 | JSON + markdown | yes (small npz) |
| 31 | L2H5-INSAMPLE | biotensor route_l2h5 | matched null scored in-sample leaked at 0.70-0.81; holdout gives ~0 | P5 | scripts + JSON | unknown |
| 32 | H123 | biotensor route_genemanifold (audit of external code) | feature `motif_present` fires only on positives by construction | outside (label leak); contam. (high) | audit + log only; `h123_deleak.py` missing | no |
| 33 | PARAM-SAMPLER | parameter-decomposition-scfm | test sampler was tissue-biased; controls not matched | P7 / P2 | immutable before/after run dirs | no (model + attacks) |
| 34 | SMALL-CODE-BUGS | biotensor GRN benchmark | stale variable across loop (`replicate_score.py`), self-knockdown filter never fired, one-tailed significance flag | outside | fixed in place; described in BENCHMARK.md | plant: yes |
| 35 | DOUBLE-LOG | biotensor geometry-sc-models | log1p applied to data already in log1p(CP10k) | outside (preprocessing) | buggy code on disk, flagged, not fixed | plant: yes |
| 36 | AIDO-CAVEATS | aido-cell-100m | correlation baseline built on 50 control cells (0.509 vs paper 0.703); layer chosen on the test data | P2 + P5 | documented, never fixed | no (69 GB) |

Lower-value items noted but not detailed: within-dataset reliability used as a denominator for a
between-dataset number (`BT/codebase/route_a/scgpt/BENCHMARK.md:2925`, P3/P4); Replogle K562 ceiling
0.09 under every perturbation negative (`MEM/perturbation-ruler-was-broken.md`, P6); atlas degree null
not reproducible because `argsort` is unstable (`AC/pipeline/scripts/degree_null.py:42`, P10); c2s
27B-vs-2B SAE comparison confounded by undertraining (`c2s-scale/runs/RESULTS_matched_control_27b.md`,
P2); GPL adapter dropped RoPE and the causal mask (`BENCHMARK.md:532-547`, outside, GPU only); model
steering magnitude quoted as a mean carried by 2 of 22 chromosomes (`GM/steer_units.py`, P5 — but
chromosome topics are out of scope for the loop by user directive, `MEM/loop-scope-no-chromosome.md`).

---

## 2. Detailed entries (top 21 in full, the rest briefly)

### 1. BRANCH-NULL — significance test for branch-point curvature (best candidate)

- **Path:** `BT/codebase/route_branchpoint/`
- **Error (verified in `FULL_SPACE_SIGNIFICANCE_RESULTS.md:12-27`, `:75-86`):**
  1. `p` pinned at the floor. `N_NULL=40` (`branchpoint_controls.py:43`), so every one of 16
     model x tissue cells reports p = 1/41 = 0.0244.
  2. The null cannot fail. `synth_linear_trajectory` builds a target that is linear by construction.
     The null curvature centres at -0.09 to -0.23, so `real > null_p95` fires even for negative real
     curvature (pancreas Geneformer -0.043 and STATE -0.081 are stored with `above_null: true`).
  3. It tested a superseded statistic (kNN curvature) while the reported number is the poly-kernel one.
  4. Second-order: the replacement "NULL B" repeated the same defect (p = 0.005 floor in all 16 cells).
- **Pattern:** P1 (null too weak, cannot fail), P8 (p at floor, no resolution), P3 (tests a statistic
  that is not the reported endpoint).
- **Before:** `branchpoint_controls.py`, `results/branchpoint_controls*.json` (5 files, 8.7-10.9 KB).
- **After:** `full_space_significance.py` (N_NULL=200, blocked permutation NULL A),
  `results/full_space_significance.json` (14 KB), `.log`. Headline changed: "human > mouse" withdrawn.
- **Inputs:** `BT/data/branchpoint/{scgpt,geneformer,state}_{setty,gut,lung,pancreas}.npz`,
  12-20 MB each (verified sizes).
- **Runnable:** yes. Gram matrix is eigendecomposed once per fold; one cell with N_NULL=40 should take
  well under 5 minutes (inferred, not timed).
- **Why good:** three separate errors, each findable from code; clear numbers; small data; not one of
  the seven source papers.

### 2. DISULFID — bond annotations span-filled

- **Path:** `PL/setup/dataset.py:250-260` (verified: `labels[s:e, col] = True` for every feature, no
  special case for two-endpoint features).
- **Error:** `FT DISULFID 3..42` means Cys3 bonds Cys42. The code marks residues 3-42. Same for CROSSLNK.
- **Numbers (verified from `PL/data/labels/concept_summary*.json`, full 20,000-protein corpus):**
  DISULFID 120,551 positive residues (prevalence 0.0185) before vs 7,889 (0.00121) after the fix;
  CROSSLNK 3,312 vs 739; CARBOHYD 3,489 vs 3,489 (unchanged — the built-in control).
  Memory note (inferred, 18,176-protein prefix): 114,648 positives with 6.96% cysteine vs 7,235 with
  100.00% cysteine.
- **Pattern:** outside checklist (label construction bug). Arguably P3 (the label is not the object
  of the claim: "inside a disulfide loop" vs "bonded cysteine").
- **Before/after:** `PL/data/labels/residue_labels.npz` (947 KB) and `residue_labels_bondfix.npz`
  (952 KB); `setup/align.py -> load_aligned(bond_fix=True)`; affected outputs `PL/runs/sweep/sweep_*.json`.
- **Runnable:** seconds. A reviewer only needs labels + sequences (`PL/data/corpus/`). A self-contained
  task: "per-concept residue prevalence and amino-acid composition" — the cysteine share gives it away.
- **Also in this project (entry 23):** SAE held-out split by residue position (`PL/setup/topk_sae.py:242`).

### 3. NORMAN-CEIL — interaction ceiling inflated by shared error

- **Path:** `RS/norman_interaction.py:84-95` (verified: only the double's cells are split; both halves
  subtract the same `DA[x] + DA[y]` and the same control mean).
- **Error:** shared error in the singles and control looks perfectly reproducible, so the split-half
  ceiling is inflated.
- **Numbers (from `BT/codebase/route_a/scgpt/BENCHMARK.md:2279-2296`, inferred):** split-half 0.6117 ->
  0.3835; ceiling 0.871 -> 0.745. The best model moves from 46.3% to 54.2% of the ceiling.
  About 37% of the apparent reliability was shared error.
- **Pattern:** P1 (the anchor itself is wrong), P4 (the two halves are not independent units).
- **Before/after:** buggy `norman_interaction.py`; honest ceiling in `RS/norman_mlp.py` (line 4 says
  singles and control are split independently). Outputs `BT/runs/grn_benchmark/norman_interaction.json`,
  `norman_mlp*.json`, `rel_norman.json`.
- **Inputs:** `<HOME>/biodyn-work/single_cell_mechinterp/data/perturb/norman/NormanWeissman2019_filtered.h5ad`
  (699 MB, outside `projects/`), `BT/runs/grn_benchmark/norman_prep.npz`.
- **Runnable:** likely under 5 min (sparse means over ~100k cells). Not timed.
- **Why good:** subtle, realistic, and the direction of the error helps the model (so a biased reviewer
  would not look for it).

### 4. T141-TIES — AUROC with argsort ranks (contam.: low-moderate)

- **Path:** `BT/mechinterp/t141_battery.py:62-97` (verified: fixed `auroc()` uses `rankdata`; the
  docstring records the old `argsort(argsort(s))`; `_selftest_auroc` checks 2/5/9/40/500-valued scores).
- **Numbers (verified in `BT/mechinterp/RESULTS.md:707-800`):** H54_ffl_closed 0.9457 (buggy) vs
  0.5072 (sklearn); H52_onesided_knn 0.9388 vs 0.5073; H52_reach_ratio_asym 0.9354 vs 0.5102; a
  continuous score H53_lid_diff 0.5185 vs 0.5185. Headline "0 of 62" did not change because the null
  bar and baselines inflated too.
- **Also in the same file header (lines 1-30):** five faults of the generated harness (dead
  feature-shuffle null that never re-ran the hypothesis; leak controls computed then ignored — a control
  at 0.8941 vs hypothesis 0.8940 still reported margin +0.3827; rewiring null scoring the wrong
  statistic; Fisher p tied to replicate count; matched negatives leaking). Plus three faults in the first
  rewrite (baseline = max over 60 hypotheses; gate tolerance 0.01 below the 0.017 noise floor;
  random-weights twin ignored). The generated harness itself was not found on disk.
- **Pattern:** outside (metric bug). The harness faults map to P1, P2, P5.
- **Before/after:** `t141_fixed.json` (pre-AUROC-fix, 1.07 MB) and `t141_full.json` (post-fix); logs
  `t141fix.log`, `t141full.log`. Buggy code = one-line revert.
- **Inputs:** `t141_inputs.npz`, 57 MB.
- **Runnable:** full run 9.2 min on 20 cores (RESULTS.md:862). `--pilot` and `--n-fs/--n-rw/--n-lab`
  flags allow a small run.
- **Note from my synthetic check:** the size and sign of the inflation depend on row order. With a
  stable sort and positives listed after negatives, a 2-valued no-signal score read 0.719 vs sklearn
  0.470; with NumPy's default sort the error was small. A planted version must keep label-correlated
  row order, as in a pair table.

### 5. DEGREE — the model's own column mean beats the model

- **Error (from `BENCHMARK.md:17-56` and `MEM/degree-confound-retraction.md`, inferred):**
  `deg(h) = sum_g S(g,h)` is the same for every TF, so it carries no TF-specific information. As a probe
  it scores 0.6416 vs ours314M 0.6297; paired delta -0.0119 [-0.0250, +0.0013]. Under a 3-D matched null
  every model falls to 0.50-0.53 and the ranking reverses (scGPT > ours314M > ours1.13B).
- **Pattern:** P2 (trivial baseline absent). Very clean P2 example.
- **Before:** registered 2-D numbers in `BENCHMARK.md` and `COMPARISON_TABLE.md`.
  **After:** degree control in `RS/graph_propagate.py`. The 3-D matcher (`matched_pairs3`) was **not
  found** in `RS/*.py`.
- **Inputs:** score matrices in `BT/runs/grn_benchmark/` (e.g. `cd34_circuit314M.npz` 87 MB,
  `matrices/`), edges `RS/dorothea_trrust_edges.tsv` (3.2 MB).
- **Runnable:** seconds (column means + per-TF AUROC).

### 6. LOO-ABS — noise scores high under small folds and abs(Spearman)

- **Path:** `GM/run_probe.py:48` (`KFold(min(k, n))`) and `:62-63` (`abs(float(r))`). Verified on disk.
  `:70` also wraps `_circ_corr` in `abs()`.
- **Error:** with tiny n, cross-validated ridge predicts close to the training mean, which is
  anti-correlated with the held-out value. `abs()` turns that into a strong "score".
- **My synthetic check (verified, `checks/synthetic_checks2.py`):** pure noise, 50 features, ridge
  alpha 1000: n=5 gives mean signed rho -0.984 and mean |rho| 0.984; n=9 gives -0.606 / 0.608; n=12
  gives -0.488 / 0.489. Memory note reports 0.74-0.89 for noise at n~10 (inferred).
- **Pattern:** P1 (no chance anchor for the metric) / outside (metric bug).
- **Before/after:** buggy code on disk; I found **no fixed version**. The fix is known (signed Spearman,
  forbid folds < 3). Planting/fixing is trivial.
- **Runnable:** seconds. Affected hypothesis example: heme_biosynth n=9 (inferred).

### 7. BRANCH-RAWCOUNT — raw counts into a binned encoder

- **Error (verified text, `BT/codebase/route_celltoken/RESULTS.md:103-108`):**
  `data/branchpoint/scgpt_setty.npz` fed raw counts to scGPT, whose value encoder expects 51 bins.
  Mean-pool PR 1.5, PC1 81% ("beautifully compact") vs binned PR 9.2, PC1 22%.
- **Pattern:** outside (input preprocessing); P3-like (the representation is not the model's).
- **Before/after:** `BT/data/branchpoint/scgpt_setty.npz` (12.2 MB) vs `scgptbinL*_setty.npz` (present).
- **Runnable:** seconds (participation ratio from PCA).

### 8. COEX-SIZE — null modules the wrong size

- **Path:** `GM/ctx_coexpr_null.py:164` (verified: `iaN, ibN = 400, 400   # DEPRECATED ...`),
  `GM/ctx_coexpr_null.py.bak` (Jul 21, same 400/400), `GM/ctx_coexpr_null_v2.py` (fixed).
  Documented in `BT/gene-context/docs/PROVENANCE.md`.
- **Error:** functional poles are ~1,883/991 genes; small null modules have different coherence and
  power. The re-run z (+8 sigma "beyond") did not match the stored +1.7 sigma. A script drift was part of it.
- **Numbers (inferred, `MEM/coexpr-null-size-match-trap.md`):** functional power 5.32; size-matched
  co-expression null mean 2.23, empirical p 0.050; random-axis null 0.43; anti-correlated blocks 1.24.
- **Pattern:** P1 (null not matched on set size) + P10 (untracked script drift).
- **Outputs:** `GM/results/ctx_coexpr_null.json` (1.9 KB), `ctx_coexpr_null_v2.json` (45 KB).
- **Inputs:** `GM/results/ctx_maxtoki_L04.npz` (322 MB) + gene maps. Probably under 5 min (not timed).

### 9. RANK1-DIR — a direction null that could not come out otherwise

- **Error (inferred, `MEM/rank1-nuisance-direction-null-vacuous.md`; `BENCHMARK.md` near line 2214):**
  label `|dE(B|KD A)| > |dE(A|KD B)|` equals `c(B) > c(A)` with `c = a/b`, one number per gene.
  c(x) alone predicts direction at 0.6190 [0.6188, 0.6192]. The registered "at chance" 0.4968 was
  below this trivial baseline, not at chance.
- **Pattern:** P1 (vacuous null) + P2 (per-gene scalar baseline).
- **Before/after:** `RS/asym_direction.py`; `RS/rank1_nuisance.py` -> `BT/runs/grn_benchmark/rank1_k562.json`.
- **Inputs:** `BT/runs/grn_benchmark/replogle/K562_gwps_normalized_bulk_01.h5ad` (375 MB),
  `K562_essential_raw_bulk_01.h5ad` (80 MB), `rpe1_normalized_bulk_01.h5ad` (95 MB).
- **Runnable:** yes on the 80-95 MB files; 29M pairs from a 7,623-gene matrix fit in memory.

### 10. SPARSE-AUROC — a negative with no ceiling

- **Error (verified text, `BENCHMARK.md:1207-1240`):** per-TF AUROC of a sparse binary predictor
  (DoRothEA) read as 0.5010 = "chance". The achievable AUROC given sparsity was 0.5070, and median
  coverage was 1 measured target per TF. Pooled statistic: DoRothEA A 1.37x [1.23, 1.50] enrichment,
  confidence tiers in the right order.
- **Pattern:** P6 (no positive control / ceiling behind a negative) + P1.
- **Files:** `RS/perturb_residual.py`, `BT/runs/grn_benchmark/perturb_diag_k562.json`,
  `db_perturb_k562.json`. Replogle bulk data as in entry 9.
- **Runnable:** yes.

### 11. CIRC-CORR — circular correlation breaks

- **Path:** `C2S/RESULTS_transfer_and_names.md:194-270` (verified headings); retracted outputs marked
  in place: `C2S/results/transfer_test.json` has a `_RETRACTED` key (verified), also
  `gene_phase_map.json`, `transfer_rpe1.json`. False gate pass: `C2S/results/manifold_steer_pt_L25.json`
  (gate 0.618 while R_diff 0.002, 90 deg error; inferred from the markdown).
- **Numbers (inferred):** K562 -> RPE1: circ_corr -0.814 while R_diff +0.789 and median error 23 deg
  vs an 86 deg floor.
- **My synthetic check (verified):** near-uniform truth (R 0.045) with a good but concentrated readout
  (R 0.878): circ_corr mean -0.006, 50% of draws negative, minimum -0.945, while R_diff 0.358.
- **Pattern:** outside (statistic invalid in this regime) / P1 (floor assumed at 90 deg, measured 86).
- **Runnable:** seconds from the stored `rows` in the steering JSONs.

### 12. STEER-FLOOR — a statistic bounded below

- **Path:** `GM/steer_local.py:254-303` (verified: the fix and a "FLOOR GUARD" print are in the code).
  `GM/results/steer_local.json` holds the matrices (verified key `matrices`).
- **Error (inferred, `MEM/genemanifold-review-lessons.md`):** `tgt = D[B,T_B] - mean_other(D[o,T_B])`
  is >= 0 by construction when `T_B` never receives cells; 49/96 domains could only add non-negative
  values. Mean +0.026 with CI excluding 0; on reachable targets the CI covers 0. It also faked a
  second finding (Spearman -0.222 with co-regulation).
- **Pattern:** P1 (no floor anchor) + P5 (the effect is carried by a selected subset).
- **Caveat:** this is a chromosome-steering analysis. The user told the autonomous loop to avoid
  chromosome hypotheses. That rule is about new work; a benchmark item is probably fine, but check.

### 13. HOX-ANALOGY — uniform chance as the baseline

- **Path:** `GM/hox_analogy.py:93` (verified: prints "top1 by chance ~ 1/pool_size ... vocab pool ~20k")
  vs `GM/hox_analogy_null.py`, `GM/results/hox_analogy_null.json` (12 KB).
- **Numbers (verified text, `GM/RESULTS.md:530-550`):** "~6,700x chance" vs honest ~12x (loose null)
  and ~3.9x (strict). Under the strict null esm2 wins (10.7x vs 4.0x), so the model loses.
- **Pattern:** P1 (weak null) + the null choice favours the model (P2-like: sequence baseline wins).
- **Inputs:** gene tables via `GM/gm_lib.py` caches (`BT/data/genemanifold/`); 258 quadruples. Probably fast.

### 14. PHASE-HAND — SVD angle with arbitrary handedness

- **Path:** `C2S/cc_phase.py:58` (`phase_angle`, buggy) and `:80` (`phase_angle_oriented`, fixed). Verified.
- **Numbers (verified text, `C2S/RESULTS_transfer_and_names.md:13-23`):** K562 vs RPE1 per-gene peak
  phase circ_corr -0.983 (median abs diff 63 deg) before; +0.983 (15 deg) after.
- **Pattern:** outside (sign/orientation ambiguity).
- **Inputs:** `c2s-scale/route_genemanifold_c2s/data/` (607 MB total). Runnable on CPU.

### 15. MATCH-LEAK — the matched null leaked two ways

- **Path:** `RS/unified_grn.py:284-332` (verified: fixed nearest-unused 2-D matcher; the old
  "first unused from k-MATCH_WIN upward" rule is described in the docstring, lines 302-308).
  `:441` still prints "matched is trivially ~0.5 by construction"; `:523` now flags `*** NULL LEAKING ***`.
- **Numbers (verified in docstring):** co-expression scored against its own matched negatives 0.5389
  (z +3.09) before, 0.5010 (z +0.08) after. Abundance never matched: occurrence alone 0.5496, above
  GPL FFN 0.5441 and our attention 0.5315.
- **Pattern:** P1 (null leaks) + P2 (trivial abundance baseline beats models).
- **Before outputs:** `RS/results_unified/unified_grn.json`, `uni.log` (Aug 1), `combined*.json`
  (Aug 1-2). **After:** later benchmark runs.
- **Runnable:** the matcher itself runs in seconds on stored matrices; full model scoring needs GPU.
  Best used as a planted item: swap in the old matcher on a synthetic pair table.

### 16. ANNOT-FLOOR — no chance floor for the annotation rate (contam.: moderate)

- **Path:** `BT/codebase/route_b/salvage_2026-08-29/README.md` (verified), scripts in `scripts/`
  (`atlas_floor.py`, `floor.py`, `topk_floor.py`), artifacts 149 MB in `artifacts/`.
- **Numbers (verified in README):** trained atlas 42.236% unconditional / 46.729% specificity;
  untrained norm-matched SAE 39.160% / 46.777%; random 20-gene draw 39.893%. Min-count sweep: excess
  +16.50 -> +1.42 -> -8.11. About 8.5 points of floor is a Fisher-background bug (12,592-gene universe
  vs 6,863 eligible genes).
- **Pattern:** P1 (no chance anchor) + outside (wrong enrichment background).
- **Also here (P10):** `route_b/annotate/one_ruler.py:222` writes `one_ruler_{statistic}_min{min_count}.json`,
  so later layer-5 runs overwrote the 8-row table (verified line; README section 3).
- **Runnable:** README says regeneration from artifacts takes ~30 s (not checked). Needs GO annotations.

### 17. CTX-INTERACT — interaction manufactured by noise

- **Path:** `GM/ctx_interaction.py` (header describes the decomposition); `GM/results/ctx_interaction.json`
  (2 KB). Inputs `BT/data/genemanifold/ctx_L00@*.npz`, `ctx_L08@*.npz` (0.9-5.7 MB each).
- **Error (inferred, `MEM/gene-polysemy-program.md`):** z = +10.4 "PROCEED". Null sat at rho 0.823
  (magnitude driven by per-gene norm); direction test sits at the arithmetic floor -1/(n_ctx-1) = -0.167;
  token counts differ a median 6.1x across contexts, so sampling noise lands in the interaction term;
  the balanced set is 65 housekeeping genes.
- **Pattern:** P1 + P7 (one biased gene set) + outside (heteroscedastic noise).
- **Caveat:** the script was corrected in place; the pilot version may not be on disk. Planting is easy.

### 18. ROUTEB-CENTER — missing centering + rare-gene lottery

- **Path:** `BT/codebase/route_b/manifolds/interpret_manifolds.py:48` loads `ck["mean"]`; `:59` encodes
  `X` without subtracting it (verified). `top_genes` ranks by mean |activation| with no minimum count
  over a 60,000-token subsample (verified).
- **Numbers (inferred, `MEM/route-b-annotation-scorer-broken.md`):** uncentered/60k 6.0% -> centered/60k
  2.7% -> centered/full 0.0% -> centered/full/min-count-50 2.7%. The two defects cancel. Arm ordering
  reversed (Spearman -0.58).
- **Files:** fixed `route_b/annotate/one_ruler.py`, `annotate/reconcile_scorer.py`,
  `route_b/results/scgpt_L11_composite_muon_scorer_reconciliation.json` (643 B).
- **Runnable:** needs scGPT L11 activations (mmap; size not checked) and a checkpoint from
  `route_b/runs/` (1.8 GB folder). Probably feasible with the 60k subsample.

### 19. FROZEN-EMB — pretraining "helps" by removing parameters

- **Numbers (inferred, `MEM/frozen-embedding-gain-is-regularisation.md`):** frozen real vectors 0.3957,
  real vectors on the wrong genes 0.4016, Gaussian noise 0.3942, learned 0.3323, zeros 0.1897
  (d128 L4, one 5-fold split). Top three within 0.0074; CI half-width ~0.06.
- **Files:** `RS/norman_transformer.py` (modes `random-frozen`, `shuffled-frozen`),
  `BT/runs/grn_benchmark/norman_tf_d128L4.json`, `emb_control.json`, `emb_zero.json`.
- **Pattern:** P6 / P2 (no random-frozen control).
- **Runnable:** uncertain; 115 pairs but a transformer per fold.

### 20. TIE-FILL — catalogue filled by alphabetical ties

- **Path:** `AC/pipeline/atlas_h100/common/pipeline.py:57-61` (verified:
  `np.argsort(G[:, f])[::-1][:top]` over an alphabetically sorted gene list; zeros tie).
- **Numbers (verified text, `AC/paper_draft/CLAIMS.md:382-387`):** filler reaches the top 10 in 18.1%
  (tGPT), 13.5% (C2S), 11.1% (Tahoe), 7.4% (UCE) of features; "hub genes" ERO1B (12,096 features),
  GNAO1 (8,541), GAS7 (3,855) are filler.
- **Pattern:** outside (code bug).
- **Runnable:** release catalogues are not on disk (`fetch_runs.sh` downloads them). A synthetic plant
  takes seconds. A related unfixed bug in `BT/codebase/route_b/manifolds/interpret_manifolds.py` does not
  have this failure (it stops at zeros).

### 21. OVERWRITE — outputs silently overwritten

- **Instance A (verified text, `BT/codebase/route_a/scgpt/GAUGE_AND_LNFREE_RESULTS.md:216-232`):**
  `07_full_fidelity.py` keyed its output on `ATTN_MODE` only; the sparse sweep overwrote the dense
  student's gate. Stored 0/5 (L11 0.781) was a sparse student's; re-run gives 4/5 (L11 0.9509).
  Kept as `07_full_fidelity_sqnorm.MISLABELED_sparse_run.json.bak`.
- **Instance B (verified):** `route_b/annotate/one_ruler.py:222` still collides (entry 16).
- **Pattern:** P10.
- **Runnable:** reviewer detects it by reading code + comparing log timestamps; nothing to run.

### Brief entries (22-36)

- **22 WINNER-CURSE (P5).** `PL/run/04c_honest_selection.py`; leaky `PL/runs/sweep/sweep_probe.json`
  vs honest `sweep_honest.json`, `sweep_candidate_matched.json`. Leaky gap peaks at layer 16; honest gap
  rises to +0.13 at L33; negative control VARIANT falls from ~0.12 to 0.000 (`PL/runs/sweep/RESULTS.md:169-195`,
  verified text). Needs `PL/data/activations/layer_XX_activations.npy` (87 GB folder) — subsample first.
- **23 PROT-SPLIT (P4).** `PL/setup/topk_sae.py:242` splits by position (verified). Fixed SAE
  `layer_16/sae_final_proteinsplit.pt`, `results_proteinsplit.json`. VE 0.7455 vs 0.7572; redundancy
  moved 0.0016. Small effect — good as a "real but minor" item.
- **24 TWIN-COMPUTE (P2).** `BT/omnimodal/RESULTS.md:478-523`: omni - twinT +0.0717 (t +3.02, 4/4) was
  the knob; true omni - twinB +0.0115 (t +0.51). Training needed.
- **25 C2S-TOKENIZER (P3/P2).** `C2S/encoding_matched_transfer.py`, `selection_matched_arm.py`,
  `results/encoding_matched_transfer.json`. Mismatched selection gave +0.0153 [+0.0022, +0.0283]; matched
  gives -0.0022 [-0.0144, +0.0105]. Cache 885 MB.
- **26 SCREEN-GENESET (P7 + P1).** `GM/screen.py`; `GM/results/screen_STALE_pre-gate-fix.json` vs
  `screen.json`; history in `GM/results/screen_history/`. Six-way intersection 6,307 genes; without
  K562 15,322; survival 0.2% (most tissue-restricted decile) vs 88.9% (broadest). Needs GB caches.
- **27 RAW-PROFILE-NULL (P2).** `GM/gm_lib.py` `basis("coexpr*")`; `GM/coocc_bestprobe.py`,
  `coocc_strongest.py`, JSONs in `GM/results/`. Chromosome decoding by the baseline 0.044 -> 0.720.
  Chromosome topic (see caveat in 12). Heavy.
- **28 CSSI (P5; contam. high).** `RS/cssi_attn_grn.py`, `RS/cssi_run.log`. Top-K F1 x1.22 with real
  strata vs x1.19 with random strata.
- **29 BESTLAYER (P5).** `RS/score_2d.py`. ours314M best layer 0.5919 vs E[best of 24 | null] 0.5291
  (excess +0.0628); scGPT 0.5503 vs 0.5247 (+0.0256). Across-layer mean would reverse the ranking.
- **30 GIII-NULL (P1 + P5).** `BT/codebase/route_b/GATES_RESULTS.md`, `results/gates_giii_split.json`,
  salvage README section 4: excess 10.53 over a Gaussian null is shared token scale; unreported sub-test
  ratio 4.329.
- **31 L2H5-INSAMPLE (P5).** `BT/codebase/route_l2h5/07_let_controls.py`, `.json`; RESULTS.md:106
  (in-sample null 0.70-0.81 vs holdout ~0.000).
- **32 H123 (outside, label leak; contam. high).** `GM/H123_LEAK_AUDIT.md`, `GM/h123_deleak.log`.
  motif fires on 470/735 positives and 0/2,205 negatives; motif alone 0.8197; de-leaked delta -0.0025.
  `h123_deleak.py` and `results/h123_deleak.json` are **not on disk**; original code (`run_iter0046_screen.py`,
  subproject_38) is outside `projects/` and was not found in `<HOME>/biodyn-work`.
- **33 PARAM-SAMPLER (P7/P2).** `parameter-decomposition-scfm/runs/g2-scgpt-layer0-corpus-20260723/`
  (tissue-biased) vs `...-balanced-eval-...` vs `...-balanced-matched-eval-...` (verified header).
  Needs scGPT + attacks.
- **34 SMALL-CODE-BUGS (outside).** `RS/replicate_score.py` stale `rev_c/rev_s` (BENCHMARK.md:1366);
  `structure_budget.part_c` self-knockdown filter never fired because Replogle ids are
  `<libid>_<SYMBOL>_<P1>_<ENSG>` (BENCHMARK.md:1071-1076); significance flag tested only `lo > 0`
  (BENCHMARK.md:2268). All fixed in place. Good as planted items.
- **35 DOUBLE-LOG (outside).** `BT/geometry-sc-models/experiments/03_what_shapes_geometry/curvature/build_rulers.py:91-92`
  (flagged in `BT/geometry-sc-models/docs/CLAIMS.md:217`; not fixed).
- **36 AIDO-CAVEATS (P2 + P5).** `aido-cell-100m/runs/attention-grn/run_report.md:315-330`, `:562-570`.
  Correlation baseline on 50 controls (AUROC 0.509 vs paper 0.703); L8 chosen by argmax over 18 layers on
  the same 35 perturbations. Never fixed. Same pipeline family as a source paper (attention).

---

## 3. Clean, small analyses for correct controls

These are the fixed versions of real analyses. Several pair with a buggy item above, which gives a
matched "same analysis, no error" control.

| # | analysis | path | why it is clean | size / CPU |
|---|---|---|---|---|
| C1 | branch-point curvature significance, NULL A | `BT/codebase/route_branchpoint/full_space_significance.py`, `results/full_space_significance.json` | blocked permutation null, N_NULL=200 (p floor 0.005), tests the reported statistic, reproduces stored value to 1e-13 | 12-20 MB npz; minutes with fewer nulls |
| C2 | bond-fixed residue labels | `PL/data/labels/residue_labels_bondfix.npz`, `PL/setup/align.py` (`bond_fix=True`) | DISULFID 100% cysteine; CARBOHYD unchanged control | seconds |
| C3 | T141 battery after the fix | `BT/mechinterp/t141_battery.py`, `t141_full.json` | tie-safe AUROC with self-test; Gate 0 model-free probes at chance (worst 0.0227 vs bar 0.0312); intersection-union reporting | `--pilot`; 57 MB input |
| C4 | Norman honest ceiling + CV ladder | `RS/norman_mlp.py`, `BT/runs/grn_benchmark/norman_mlp*.json` | every group split independently; 5-fold CV over pairs | 699 MB h5ad |
| C5 | size-matched co-expression null | `GM/ctx_coexpr_null_v2.py`, `results/ctx_coexpr_null_v2.json` | nulls size-matched, empirical p, three nulls cross-checked | 322 MB input |
| C6 | encoding- and selection-matched transfer | `C2S/encoding_matched_transfer.py`, `C2S/selection_matched_arm.py` | paired 5,000-draw bootstrap; parity result with CI | 885 MB cache |
| C7 | dataset reliability (split-half) | `RS/dataset_reliability.py` | Papalexi cell-split r = 0.3685, mismatched 0.2507, ceiling 0.607 (inferred from memory) | small h5ad in biodyn-work |
| C8 | matched null with gates | `RS/unified_grn.py` `matched_pairs` (fixed) + gate print | each confounder scored against its own matched negatives; flags leaks | seconds on stored matrices |
| C9 | cell-token geometry, binned inputs | `BT/codebase/route_celltoken/compare_celltoken.py` | correct binned scGPT inputs; common n=3000; aborts below 2000 cells | small |
| C10 | registered decomposition runs | `parameter-decomposition-scfm/tests/` (42/42 pass), `runs/g3-...-validation-selected-tsp7-final-20260724/` | preregistered gates, donor-blocked, immutable run dirs | code-level control; runs need GPU |

Warning for controls: "clean" here means no known error. Several were built after an adversarial
review. None was proven error-free.

---

## 4. Suggested benchmark design notes

1. **Pair items.** For entries 1, 2, 3, 4, 8, 14, 15, 16 the same analysis exists before and after the
   fix. Show one version per reviewer run. This controls for topic difficulty.
2. **Balance patterns.** The pool is rich in P1 and P2 and in outside-checklist bugs. It is thin in
   P9 (causal overreach) and P8 (stability). For P8, use seed-dependence from
   `parameter-decomposition-scfm/runs/README.md` (TSP21 seed 20260723 passes, seed 20260724 fails:
   top-k 0.014736 vs SVD 0.005208). For P9 there is no strong documented item; plant one (e.g. rephrase
   the steering summary in entry 12 as "the model uses X to compute Y").
3. **Keep outside-checklist items.** About 12 of the 36 are code/data bugs. They measure what the
   checklist misses, which is exactly the held-out question.
4. **Strip the evidence of the fix.** Many buggy files contain comments that describe the bug
   (e.g. `ctx_coexpr_null.py:164`, `unified_grn.py:302-308`, `t141_battery.py:66-70`,
   `steer_local.py:254-303`). Remove these comments, and any `_RETRACTED` keys, before giving an item
   to a reviewer.
5. **Exclude or flag contaminated items** (H123, CSSI; weaker: ANNOT-FLOOR, T141, AIDO).

---

## 5. What I checked myself

Script: `checks/synthetic_checks.py` and `checks/synthetic_checks2.py` in this folder (CPU, seconds).

| mechanism | result |
|---|---|
| argsort(argsort) AUROC, stable sort, positives listed last, no signal | 2 values: 0.7193 vs sklearn 0.4698; 9 values: 0.5610 vs 0.5053; continuous: equal (0.4986) |
| same with NumPy default sort | small errors (0.3929 vs 0.4698 for 2 values; 0.5007 vs 0.5053 for 9) |
| small-fold ridge + abs(Spearman) on noise | n=5: |rho| 0.984; n=9: 0.608; n=12: 0.489 (signed means negative) |
| circ_corr with near-uniform truth | mean -0.006, 50% negative, min -0.945, while R_diff 0.358 |

I also verified: the DISULFID/CROSSLNK/CARBOHYD counts in both concept summaries; the buggy lines in
`dataset.py`, `topk_sae.py`, `interpret_manifolds.py`, `run_probe.py`, `ctx_coexpr_null.py`,
`branchpoint_controls.py`, `pipeline.py` (atlas), `one_ruler.py`, `degree_null.py`, `hox_analogy.py`;
the fixed code in `t141_battery.py`, `unified_grn.py`, `steer_local.py`, `cc_phase.py`; file sizes of
all inputs quoted.

## 6. What I did NOT do or check

- I did not run any real analysis end to end. Runtimes marked "likely"/"probably" are guesses from
  input sizes.
- I did not re-derive the headline numbers of entries 3, 5, 8, 9, 12, 17, 18, 19, 22-31 from raw
  outputs. They come from project markdown or memory notes.
- I did not find: the 3-D degree matcher (`matched_pairs3`), `h123_deleak.py`, the original generated
  T141 harness, the pilot version of `ctx_interaction.py`, or a fixed version of `run_probe.py`.
- I did not open `maxtoki/` except to read which papers seeded the checklist.
- I did not search `<HOME>/biodyn-work` (outside scope) beyond two quick look-ups.

## Plain-words summary

There are many real, documented mistakes in the other projects. About 20 have both the wrong and the
corrected version saved. The best test items are small and quick: the branch-point significance test
whose null could never fail, the protein labels that marked whole loops instead of bonded cysteines,
the Norman ceiling that shared noise between its two halves, the AUROC function that mishandled ties,
and the GRN benchmark that never tried the model's own column mean as a baseline. Many mistakes are
plain code bugs that the ten-pattern checklist does not cover. Those are worth keeping, because they
show what the checklist misses. Two items come from the papers that built the checklist and should
not count as held-out.

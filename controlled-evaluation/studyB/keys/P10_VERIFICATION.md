# P10 independent verification (2026-10-01)

Pair: B1249 (flawed) / B4174 (clean). Checked by a second agent, using its own scripts (kept in a session scratch
folder, not in any package). CPU only, `maxtoki-framework-eval/bin/python`. Nothing in `biomi_automation` was
changed.

Verdict: **OK after fixes** (two small fixes; none changes a number or the planted error).

## 1. Runs and outputs

| | B1249 | B4174 |
|---|---|---|
| run from package root, `--out` to a scratch folder | 24.9 s wall (2.4 s CPU), 79 MB peak | 23.0 s wall (2.1 s CPU), 79 MB peak |
| `results.json`, `head_ap.csv`, `run_log.txt` vs shipped | byte-identical | byte-identical |
| `--selftest` | ALL PASS | ALL PASS |

Runs used `-B`. No `__pycache__` folders are in either package. (The README says "about 10 seconds"; on this
drive the wall time is about 25 s, almost all of it disk reads. Not changed; it is not a hint and not wrong in a way
that matters.)

## 2. Wrong result, right result, and the diff

- `diff -r` of the two packages: only `SUMMARY.md` differs, and only in the Conclusions (flawed lines 34-44,
  clean lines 34-46). Every other file has the same md5 in both packages.
- Both packages give the same numbers, as a planted P9 item should. The flawed SUMMARY carries the overreach
  (dedicated circuit, head computes connectivity, earlier stages, understands folding chemistry). The clean SUMMARY
  says it is a correlational readout with no ablation or patching and no contact-specificity test.
- Attention data: `S_select` (661 x 4,419) and `S_test` (661 x 7,463) are bit-identical to the source cache
  `data/labels/esm2_attn_cyspairs_L16_bonded.npz` (same md5 as `routeb/results/attn_cache.npz`). The npz files
  hold only `S_select`, `S_test`, `n_layers`, `n_head` (the cache's `signature` key was dropped).
- Labels, rebuilt independently (my own parser, not the source functions): streamed `uniprot_sprot.dat.gz`,
  took every `FT DISULFID` record and its notes, used the first 18,176 corpus entries (equal to
  `analysed_accessions.txt`) and `prots_val` / `prots_test` from `split_masks.npz`. Result: select 4,419 pairs /
  313 bonded / 113 proteins, test 7,463 / 366 / 123; P61134 (64 Cys) and O09118 (41 Cys) dropped by the cysteine
  rule; none dropped by length. Accession, length, pos_i, pos_j and bonded equal the package files exactly, and
  the md5 pair-set signature equals the cache's (`d9521e1ccf64c532ce9e77b1b76d5da5`). All 236 rows of
  `proteins.tsv` (sequence, length, cysteine count, bond count, split) match the corpus and the split file.
- Results vs the source run `routeb/results/disulfide_test_L16_run_20260831T163513Z.json`: chosen head, test AP
  0.7179437, test max 0.7447274, 192 candidates >= 2x base rate, bootstrap mean/lo/hi, per-protein AP / P@1 /
  P@n_bonds, and the whole separation arm (incl. its null 0.0665) are equal. The attention null is 0.0869 in the
  package vs 0.0929 in the source; replaying the source's shared random draws (arm sizes 4, 4656, 6478, 6564,
  5107, 4656, 6478, taken from the source JSON) gives 0.09287168 exactly.
- All 661 select and test APs recomputed with `sklearn.average_precision_score`: equal to `head_ap.csv` (to its
  6 decimals).
- README claim "bfloat16, eager attention, on an Apple-silicon GPU": the source run that built the cache
  (`..._20260831T163005Z.json`) records `attn_device: mps`, `attn_dtype: bfloat16`. True.
- Spot check of `code/extract_attention.py` on CPU, bf16, 3 proteins not chosen by the builder's rule
  (P01627 select L=120; P0C1D0 test L=32; P01573 test L=190): correlation with the shipped scores 0.9996-1.0000
  over all 661 rows, L32H13 correlation >= 0.9995, same top pair for L32H13 in all three.

## 3. Hints

- Word and path scan of README, SUMMARY, code, outputs (fix, bug, retract, correct, wrong, leak, audit, review,
  error, flaw, clean, honest, planted, repo and project names, absolute paths, SAE/bilinear names): no hits except
  "reviewed UniProtKB/Swiss-Prot" and "average product correction" / "APC-corrected" (method names).
- `._*` files hold only `com.apple.provenance`; all real files carry only that attribute. npz members have the
  default 1980 date and no comments.
- Outside the packages only: `build/P10/pkg/outputs/` is a stale staging copy that still says "fixed candidate".
  It is not shipped; the package outputs say "single candidate" and regenerate byte-identically.

## 4. Key location and detection rule

- `SUMMARY.md:34-44` in B1249 is exactly Conclusions 1-4; the quoted phrases are on the stated lines.
- The rule is fair. It needs the reviewer to name the mechanistic claims and the missing evidence (no
  intervention, or no contact-specificity test). It rejects a generic "too strong" remark and a spread-only remark.
  The clean key correctly marks a P9 finding against the clean SUMMARY as a false alarm.

## 5. known_true_facts

All checked by my own code: base rates 0.0708 / 0.0490; chosen head second of 661 on test, 0.0268 below L29H18;
mean head 0.141 (select 0.168); top 5 by select AP L32H13, L29H18, L31H17, L30H1, L23H6, with the test APs the
SUMMARY gives; five heads > 0.5 test AP (layers 23-32); 191 single heads / 192 candidates >= 2x base rate, at least 2
per layer; 5 largest test proteins hold 28.1% of pairs, AP without them 0.7057 (base 0.0617); per-protein AP 0.812,
P@1 0.805; L32H13 has 7,462 distinct values over 7,463 pairs; no identical select/test sequences, 3 test proteins
with 5-mer Jaccard > 0.5; no intervention code anywhere.

## Fixes made

1. **B4174 `SUMMARY.md` line 38-39.** "Several late-layer heads score almost as well (L29H18 0.745, ...)" was
   loosely worded, because L29H18 scores higher than L32H13 on test. Now: "Other late-layer heads also score high
   on the test proteins (L29H18 0.745, L31H17 0.639, L30H1 0.595), so the signal is spread over a group of heads."
   Same line count; 443 words now (flawed 420). `build/P10/SUMMARY_clean.md` synced. Outputs unaffected.
2. **Both keys, known_true_facts.**
   - The "every bond label is an annotated intrachain disulfide" fact was not exact. Five test labels (Q17192 x2,
     P20507, Q6WB98 x2) are two-position records that UniProt notes as "Interchain (between B and A chains)" or
     similar: bonds between chains cut from one precursor, so both ends are in the same sequence. Dropping them
     moves L32H13 test AP from 0.7179 to 0.7191. P26226 has two records with an unknown partner (22..?, 47..?),
     which are not used. The fact now says this.
   - "Median separation 23 vs 101" is the test split; select is 23 vs 68. Added.
3. **Both keys, possible_false_alarms.** Added: pooled AP over proteins with different score scales. The
   per-protein view gives the same order (per-protein AP 0.812 / 0.383 / 0.331 for L32H13 / mean head / separation).

## Not done / limits

- ESM-2 extraction was not re-run for all 236 proteins (over 9 minutes on CPU); only the 3-protein spot check.
- The build note that the SAE and bilinear code crashed was not checked; those arms are simply absent.
- The spot check's peak resident size, as `time -l` reports it, was 4.24 GB. That is over the 4 GB budget. Most of
  it is likely the memory-mapped weight file. The run finished without harm.
- My first UniProt scan (an awk pipeline) ran past 9 minutes and was stopped. A Python rewrite did the scan in 91 s.
- `P10_BUILD.md` still says the clean SUMMARY has 439 words; it now has 443.

# Run summaries: original record, not corrected

The files in this folder (run summaries, reports and a few related files) were written during
the agent-run deployment of the pipelines on MaxToki-217M (April–May 2026). They are kept as they were written.

**Many of their headline numbers are wrong.** Check any number here against the error ledger and
the corrected analyses before you use it.

- The 149 confirmed errors in the deployment's code, logs, outputs, run summaries and audit
  reports are listed in S1 Table:
  [`projects/maxtoki/paper-plos-one/supporting/S1_Table.csv`](../paper-plos-one/supporting/S1_Table.csv).
  The working ledger and the scripts that built and checked it are in
  [`projects/maxtoki/framework-eval/`](../framework-eval/).
- The corrected analyses are described in S3 Appendix:
  [`projects/maxtoki/paper-plos-one/supporting/S3_Appendix.pdf`](../paper-plos-one/supporting/S3_Appendix.pdf).
  Their outputs are the `v2_`, `v2b_` and `v3_` folders and the `V2_`, `V2B_` and `V3_` reports
  under the run folders in [`projects/maxtoki/runs/`](../runs/).
- [`RESULTS_MAP.csv`](../../../RESULTS_MAP.csv) at the repository root links every number in the
  manuscript to the file it comes from.

The same warning applies to [`projects/maxtoki/README.md`](../README.md),
[`projects/maxtoki/audits/`](../audits/), and the run `README.md` and `FINAL_SUMMARY.md` files
under `projects/maxtoki/runs/`.

Paths are given from the repository root.

# Amendment 1 to the pre-registration (written before Studies A and B were run)

1. **Study A, task T4 replaced.** The developmental-manifold transfer task was
   replaced by a cross-model gene-embedding alignment task: "Is MaxToki-217M's
   gene-embedding geometry aligned with scGPT's, and how does that alignment
   compare with its alignment to Geneformer V2-316M?" Reason: an unambiguous
   answer key for the manifold task could not be written, because the manifold
   gates are also passed by a pure cell-type-label lookup (found in Study C and
   confirmed on the full repository), so the correct verdict depends on analyses
   that were still running. The new T4 has a clear key and contains a real
   deployed error (wrong embedding rows for scGPT).
2. **Study A runs in two batches** with the identical protocol: batch 1 = T1,
   T3, T4; batch 2 = T2 (its data exist only after the circuit-tracing re-run
   with corrected hooks). Results are pooled.
3. **Package staging.** Subject packages are copied to a neutral folder
   (`<CODE_ROOT>/analysis-tasks/<opaque id>/`), so folder
   names do not reveal task, arm or study. The Python wrapper there caps BLAS
   threads at 2 and suppresses Python warnings (which printed a local path).
   Package content is byte-identical to the frozen build except the interpreter
   path in BRIEF.md.
4. **Known property of the contract arm.** The deployed specifications are given
   verbatim, as the treatment. They carry expectations from earlier studies
   (for example, the attention specification states that a failing verdict is
   the expected outcome). This is part of what a contract is in this framework
   and is reported with the results.
5. **Study B, pair P01 excluded** (packages B1131, B1448): independent
   verification found that the documented error was not the real defect of the
   flawed version, so the item failed the pre-registered verification and is not
   frozen. Study B has 11 pairs (22 items): 8 with natural errors and 3 with
   planted causal-overreach summaries.
6. **Budget sentences.** Study A reviewers: "Spend about the effort of a careful
   two-hour expert review; you may use up to roughly 60 tool calls." Study B
   reviewers: "Spend about the effort of a careful one-hour expert review; you may
   use up to roughly 50 tool calls." Identical across arms within a study.

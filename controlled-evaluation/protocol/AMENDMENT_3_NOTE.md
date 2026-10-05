# Note on Amendment 3 (written 2026-10-03, after the re-run)

Amendment 3 says that two Study A repair agents (T1, contract arm, run 2 after the checklist
review and run 5 after the generic review) printed the first lines of the stray script
`w6252/analysis_script.py`. A complete scan of all Study A transcripts made afterwards
(`tools/package_write_check.py`, and a direct search of the ten T1 contract-arm repair
transcripts) found that all ten repair agents of T1 in the contract arm (runs 1-5, after both
review styles) opened that script; most printed its first 30 to 50 lines. No executor or reviewer
of Study A opened it, and no agent of Studies B or C. All ten deliverables reached the correct
verdict, and leaving out the T1 contract-arm runs does not change any Study A result (for
example, checklist minus no review in pitfalls handled: +0.029, 95% bootstrap interval 0.007 to
0.051, over 35 instead of 40 executor runs). The paper reports the corrected count.

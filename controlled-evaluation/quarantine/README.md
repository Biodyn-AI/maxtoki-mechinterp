# Files moved out of the Study A task folders

`2026-10-02_stray_files_in_task_folders/` holds three files that agents of the first exploratory
Claude Haiku 4.5 run wrote into shared Study A task folders, although the prompts asked them not to
write files:

| File | Written by | Task folder |
|---|---|---|
| `w6252/analysis_script.py` | Haiku executors of T1, contract arm | T1, contract arm |
| `w6252/verify_analysis.py` | a Haiku agent of the same run, after it was resumed | T1, contract arm |
| `w3318/analysis_script.py` | a Haiku executor of T2, contract arm | T2, contract arm |

They were moved here (not deleted) on 2 October 2026, and every task folder was then checked to be
identical to its source copy, apart from the interpreter path in the briefs. The first Haiku run was
set aside and re-run with one folder per agent (`protocol/AMENDMENT_3.md`, `studyA_haiku2/`).
All ten Opus 5.5 repair agents of Study A for T1 in the contract arm opened `w6252/analysis_script.py`
before it was moved (most printed its first 30 to 50 lines); see `protocol/AMENDMENT_3_NOTE.md`. This is
reported in the paper's Methods.

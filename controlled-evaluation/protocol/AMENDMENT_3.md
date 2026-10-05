# Amendment 3 (exploratory Haiku arm re-run with one folder per agent; written before the re-run)

Written 2026-10-02, before any agent of the re-run started. No result of the
re-run had been seen.

## Why

A scan of the first exploratory Haiku 4.5 run (Amendment 2) found that its
agents did not keep their work separate:

- 48 Haiku subject agents wrote temporary files to the shared `<TMP>` folder
  under common names (for example `<TMP>/analysis_script.py`), so concurrent
  agents could read or overwrite each other's scripts.
- Two Haiku executors wrote `analysis_script.py` into the shared task
  folders of T1 (contract) and T2 (contract), and Haiku reviewers of other
  runs then read and ran it. A resumed Haiku agent later wrote
  `verify_analysis.py` into the T1 (contract) folder.

The prompts asked the agents not to write files and to read only inside the
task folder; the Haiku agents did not follow this. The first Haiku run is
therefore set aside. Its outputs are kept unchanged on disk but are not
analysed.

The same scan of the pre-registered Opus 5.5 studies (A, B and C) found no
agent that read a file written by another agent, with one exception: two
Study A repair agents (T1, contract, run 2 after the checklist review and
run 5 after the generic review) printed the first 30 to 50 lines of the
stray Haiku script in the T1 (contract) folder, which they took for part of
the package. Graders of T1 (contract) deliverables also opened it. This is
reported in the paper. The stray files were moved to
`quarantine/2026-10-02_stray_files_in_task_folders/`, and all Study A and
Study B task folders were checked to be identical to their sources
(apart from the interpreter path in the Study A briefs, which was set when
the folders were built).

## What changes in the re-run

1. **One folder per agent.** Every agent (executor, reviewer, repairer and
   grader) gets its own folder with an opaque name under
   `analysis-tasks/`. The folder mirrors its task package: real
   sub-folders, and one symbolic link per package file. A file an agent
   writes therefore stays in its own folder. The package files are checked
   against their source copies after the run.
2. **Prompt wording.** In place of "do not write files" and "read only
   inside <package>", every prompt says that the folder is the agent's own,
   that it may write files only inside it, and that it must not read or
   write anywhere else (naming `<TMP>` and scratch folders). Everything else
   in the prompts is unchanged.
3. **Labels.** Agents are labelled `AI:` instead of `AH:`.

Unchanged: subject model (Claude Haiku 4.5), tasks, packages, answer keys,
review and repair instructions, budget sentences, five replicates per cell,
Opus 5.5 graders (two per deliverable, a third on disagreement), review-point
grading, and the analysis code. The arm stays exploratory and is reported
separately from the pre-registered Study A.

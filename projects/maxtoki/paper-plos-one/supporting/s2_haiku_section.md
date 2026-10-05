This arm repeats Study A with Claude Haiku 4.5 as executor, reviewer and repairer. Tasks, packages, answer keys, review and repair instructions, budget sentences, five runs per cell, and the Opus 5.5 graders are the same as in Study A. It was added after the first Study A task (T2) had shown that every Opus 5.5 deliverable reached the correct verdict (Amendment 2). Each agent worked in its own folder of links to the task files, and the prompts said that files could be written only inside that folder (Amendment 3). The arm is exploratory and does not enter the pre-registered contrasts.

**Isolation.** After the run we scanned the tool calls of all 188 subject agents (40 executors, 80 reviewers, 68 repairers). The shared task folders were unchanged afterwards. Many agents still wrote temporary files to the shared `<TMP>` folder (51 of 188). We therefore checked, for every read of a `<TMP>` file, which agent had last written it before that moment: in all 112 such events it was the reading agent itself. Five reviewers opened the folder of the executor whose deliverable they were reviewing, because the deliverable named it. No subject agent read the answer keys or the project repository.

**Missing deliverables.** Twelve reviews (7 checklist, 5 generic) did not return a valid structured output after five attempts; their repaired deliverables do not exist and are counted as missing. Graders agreed on the verdict of all 108 graded deliverables.

**Correct verdicts** (deliverables reaching the key's verdict / deliverables graded):

| Task | Paper, no review | Paper, generic | Paper, checklist | Contract, no review | Contract, generic | Contract, checklist |
|---|---|---|---|---|---|---|
| T1 | 0/5 | 0/5 | 1/5 | 3/5 | 3/4 | 3/4 |
| T2 | 4/5 | 4/5 | 3/3 | 5/5 | 4/4 | 4/5 |
| T3 | 2/5 | 3/4 | 1/4 | 3/5 | 4/5 | 2/3 |
| T4 | 4/5 | 4/5 | 4/5 | 4/5 | 2/3 | 3/4 |
| All | 10/20 | 11/19 | 9/17 | 15/20 | 13/16 | 12/16 |

**Share of the key's pitfalls handled** (mean over deliverables; yes = 1, partly = 0.5, no = 0):

| Task | Paper, no review | Paper, generic | Paper, checklist | Contract, no review | Contract, generic | Contract, checklist |
|---|---|---|---|---|---|---|
| T1 | 0.10 | 0.19 | 0.22 | 0.32 | 0.44 | 0.45 |
| T2 | 0.33 | 0.28 | 0.46 | 0.45 | 0.42 | 0.57 |
| T3 | 0.20 | 0.30 | 0.26 | 0.27 | 0.24 | 0.28 |
| T4 | 0.58 | 0.60 | 0.58 | 0.40 | 0.44 | 0.54 |
| All | 0.30 | 0.35 | 0.38 | 0.36 | 0.37 | 0.48 |

For comparison, Opus 5.5 executors handled 0.96 (paper arm) and 0.93 (contract arm) of the pitfalls without review.

**Contrasts** (the same tests as the pre-registered Study A contrasts; Holm adjustment across the four):

| Contrast | Result |
|---|---|
| Contract vs paper, no review | 15/20 vs 10/20; difference +0.25 (Newcombe 95% interval −0.05 to +0.49); Fisher p = 0.19; Mantel–Haenszel p = 0.15; Holm p = 0.76 |
| Checklist vs generic (paired, 28 runs with both) | 18 vs 19 correct; 2 runs right only with the checklist, 3 right only with the generic review; exact McNemar p = 1.0 |
| Checklist vs none (paired, 33 runs) | 21 vs 21 correct; 3 and 3 discordant; p = 1.0 |
| Generic vs none (paired, 35 runs) | 24 vs 22 correct; 2 and 0 discordant; p = 0.50 |

**Secondary outcomes** (paired differences in the share of pitfalls handled, with 95% bootstrap intervals over executor runs pooled over tasks, and in brackets the pre-registered version that resamples runs within each task): checklist vs none +0.092 (0.049 to 0.136) [0.050 to 0.133]; checklist vs generic +0.064 (0.013 to 0.117) [0.021 to 0.104]; generic vs none +0.017 (−0.022 to +0.053) [−0.017 to +0.049]; contract vs paper without review +0.059 (−0.048 to +0.164) [−0.010 to +0.130]. Materially false statements per deliverable: checklist vs none +0.61 (−0.06 to +1.27); generic vs none −0.07 (−0.71 to +0.57). Key numbers within tolerance: checklist vs none +0.083 (0.008 to 0.174) [0.028 to 0.143].

**Repair.** After generic reviews, repair turned 2 wrong verdicts right and none wrong (22 stayed right, 11 stayed wrong). After checklist reviews, it turned 3 wrong verdicts right and 3 right verdicts wrong (18 stayed right, 9 stayed wrong).

**Review points.** Generic reviews raised 4.4 points each (155 in 35 reviews), of which graders with the key judged 50% invalid; checklist reviews raised 8.2 each (269 in 33 reviews), 56% invalid.

**Effort.** Median tool calls and wall time per agent: executor 16 calls, 8 minutes; generic review 25 calls, 6 minutes; checklist review 19 calls, 5 minutes; repair 10–12 calls, 4–6 minutes.

Files: `controlled-evaluation/studyA_haiku2/results/` (workflow output, `analysis.json`, `deliverables.csv`, `reviews.csv`, `costs.csv`, `isolation_check.json`, `tmp_race_check.json`).

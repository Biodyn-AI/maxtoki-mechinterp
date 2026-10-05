export const meta = {
  name: 'studyC-blind-reaudit',
  description: 'Study C: blind re-audit of the pre-audit MaxToki snapshot by clean Opus agents under three review arms, then blinded grading against the frozen error ledger',
  phases: [
    { title: 'Review', detail: '3 arms x 3 replicates, clean read-only Opus agents' },
    { title: 'Grade', detail: 'two blinded graders per review + adjudication' },
    { title: 'Verify', detail: 'classify findings that match no ledger item' },
  ],
}

const WS = '<EVAL_ROOT>'
const SNAP = WS + '/studyC/snapshot/maxtoki'
const PY = WS + '/bin/python'
const KEY = WS + '/protocol/STUDYC_KEY.csv'

const CONTEXT = `The folder ${SNAP} contains a research project: an agent-run mechanistic-interpretability study of MaxToki-217M, a single-cell foundation model (an autoregressive transformer over ranked gene tokens). It holds pipeline method specifications (pipelines/), per-run scripts, logs, configs and outputs (runs/<pipeline>-217M/), setup code (setup/), small reference data (reference_data/) and results summaries (summaries/ and runs/*/FINAL_SUMMARY.md, README.md, EXTENDED_FINDINGS.md). Files larger than 20 MB and the model weights are not included; listing files record their names, sizes and shapes. You cannot run the model. You can read every file and run Python on the files present, using the interpreter ${PY} (numpy, pandas, scipy, scikit-learn, torch CPU). Run Python inline (python - <<'EOF' ... EOF); do not write files.
Read only inside ${SNAP}` 

const BUDGET = `Work as a careful expert would in about one working day. You may use up to roughly 150 tool calls; stop when you have covered the project. Return your result as the structured output; do not write any files.`

const ARMS = {
  deployed: `TASK: Audit this project using the audit specification at ${WS}/protocol/CHECKLIST_DEPLOYED.md (you may also read that one file). Follow its methodology (section 6, including the workflow in 6.11) against the project in ${SNAP}. Where the specification says to write a report file, return the report content in the structured output instead.`,
  checklist: `TASK: Audit this project using the checklist at ${WS}/protocol/CHECKLIST_GENERIC.md (you may also read that one file). Apply all ten patterns to every pipeline run in ${SNAP}.`,
  generic: `TASK: Review this project critically, as an expert reviewer would before publication. Find errors and weaknesses that affect the correctness of its results or its claims: check the code, the numbers and the claims against each other, for every pipeline run in ${SNAP}.`,
}

const OUTPUT = `OUTPUT: list every problem you found that affects the correctness of a result or a claim (code bugs, wrong statistics, missing controls, wrong descriptions, unsupported claims, reproducibility gaps). One finding per distinct problem. For each: location (file path relative to the project folder, with line numbers where possible), the pipeline it concerns, a short category label, severity (critical = a headline result or claim cannot stand; major = it stands only with material qualification; minor), a description of what is wrong, the evidence you checked (numbers, code lines, computations), and what should be done.`

const REVIEW_SCHEMA = { type: 'object', properties: {
  summary: { type: 'string' },
  findings: { type: 'array', items: { type: 'object', properties: {
    id: { type: 'string' }, location: { type: 'string' }, pipeline: { type: 'string' }, category: { type: 'string' },
    severity: { type: 'string', enum: ['critical', 'major', 'minor'] }, description: { type: 'string' }, evidence: { type: 'string' }, recommended_action: { type: 'string' } },
    required: ['id', 'location', 'pipeline', 'category', 'severity', 'description', 'evidence', 'recommended_action'] } } },
  required: ['summary', 'findings'] }

const GRADE_SCHEMA = { type: 'object', properties: {
  items: { type: 'array', items: { type: 'object', properties: {
    ledger_id: { type: 'string' }, detected: { type: 'string', enum: ['yes', 'partial', 'no'] }, finding_ids: { type: 'array', items: { type: 'string' } }, note: { type: 'string' } },
    required: ['ledger_id', 'detected', 'finding_ids', 'note'] } },
  findings: { type: 'array', items: { type: 'object', properties: {
    finding_id: { type: 'string' }, matched_ledger_ids: { type: 'array', items: { type: 'string' } } }, required: ['finding_id', 'matched_ledger_ids'] } } },
  required: ['items', 'findings'] }

const VERIFY_SCHEMA = { type: 'object', properties: {
  verdicts: { type: 'array', items: { type: 'object', properties: {
    finding_id: { type: 'string' }, verdict: { type: 'string', enum: ['real_error', 'real_but_trivial', 'false_alarm', 'cannot_decide'] }, severity_if_real: { type: 'string', enum: ['critical', 'major', 'minor', 'n/a'] }, reason: { type: 'string' } },
    required: ['finding_id', 'verdict', 'severity_if_real', 'reason'] } } },
  required: ['verdicts'] }

function gradePrompt(review, g) {
  return `You are grading one audit of a research project against a fixed answer key. Do not try to guess how the audit was produced.
ANSWER KEY: the CSV file ${KEY} lists ${'68'} known errors (one row each: id, short_name, pipeline, error_type, severity, where_it_lives, deterministic_evidence, notes). Read it fully. You may also read files in ${SNAP} to understand locations.
AUDIT FINDINGS (JSON):\n${JSON.stringify(review.findings)}
For EVERY key row, decide whether the audit detected that error: 'yes' = at least one finding identifies the same problem (same nature and essentially the same location or result) clearly enough that acting on it would lead to the fix; 'partial' = a finding points at the same result or code and raises a related concern, but misses or misstates the actual problem; 'no' otherwise. List the matching finding ids. Be strict and consistent; a generic remark ("add confidence intervals everywhere") counts as 'partial' at most for a specific error, and only if it would plausibly lead to finding it.
Then, for EVERY finding, list the key rows it matches (empty list if none).
Grader ${g}. Read only the key file and inside ${SNAP}. Return the structured output; do not write files.`
}

function adjudicatePrompt(review, a, b) {
  return `Two graders scored the same audit against an answer key (${KEY}). Resolve their disagreements. Read the key and the audit findings below; for each key row where the two graders differ in 'detected', decide the correct value using the same rules: 'yes' = a finding identifies the same problem clearly enough that acting on it would lead to the fix; 'partial' = points at the same result or code with a related but wrong or incomplete concern; 'no' otherwise. For rows where they agree, keep their value. Also reconcile the finding-to-key matches (union where both are defensible; drop a match only if clearly wrong).
AUDIT FINDINGS:\n${JSON.stringify(review.findings)}
GRADER A:\n${JSON.stringify(a)}
GRADER B:\n${JSON.stringify(b)}
Read only the key file and inside ${SNAP}. Return the full resolved grading (all key rows, all findings) as the structured output.`
}

function verifyPrompt(review, unmatched) {
  return `Below are findings from an audit of the research project in ${SNAP} that do not match any entry in a list of known errors. For each, decide by checking the project files (and running Python inline with ${PY} if useful) whether it is: 'real_error' (a genuine problem that affects a result or claim), 'real_but_trivial' (true but with no effect on any result or claim), 'false_alarm' (the claimed problem does not exist, or the code/number is actually correct), or 'cannot_decide' (cannot be checked with the files present). Give the severity if real (critical/major/minor) and a short reason with evidence. Be fair in both directions.
FINDINGS:\n${JSON.stringify(unmatched)}
Read only inside ${SNAP}. Do not write files. Return the structured output.`
}

const runs = []
for (const arm of ['deployed', 'checklist', 'generic']) for (let r = 1; r <= 3; r++) runs.push({ arm, r })

const results = await pipeline(runs,
  (x) => agent(CONTEXT + '\n\n' + ARMS[x.arm] + '\n\n' + OUTPUT + '\n\n' + BUDGET + `\n(Run ${x.r}.)`, { agentType: 'Explore', model: 'opus', schema: REVIEW_SCHEMA, label: `C:${x.arm}:${x.r}`, phase: 'Review' }),
  async (review, x) => {
    if (!review) return null
    const [a, b] = await parallel([1, 2].map(g => () => agent(gradePrompt(review, g), { agentType: 'Explore', model: 'opus', schema: GRADE_SCHEMA, label: `C-grade:${x.arm}:${x.r}:${g}`, phase: 'Grade' })))
    const final = await agent(adjudicatePrompt(review, a, b), { agentType: 'Explore', model: 'opus', schema: GRADE_SCHEMA, label: `C-adjudicate:${x.arm}:${x.r}`, phase: 'Grade' })
    return { review, a, b, final }
  },
  async (g, x) => {
    if (!g || !g.final) return g
    const matched = new Set(g.final.findings.filter(f => f.matched_ledger_ids && f.matched_ledger_ids.length).map(f => f.finding_id))
    const unmatched = g.review.findings.filter(f => !matched.has(f.id))
    let ver = { verdicts: [] }
    if (unmatched.length) ver = await agent(verifyPrompt(g.review, unmatched), { agentType: 'Explore', model: 'opus', schema: VERIFY_SCHEMA, label: `C-verify:${x.arm}:${x.r}`, phase: 'Verify' })
    return { arm: x.arm, rep: x.r, ...g, verify: ver }
  })
return results

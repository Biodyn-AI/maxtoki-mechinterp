export const meta = {
  name: 'studyB-heldout-benchmark',
  description: 'Study B: checklist vs generic review on 22 held-out flawed/clean analysis packages, 3 replicates, blinded grading with adjudication and code verification',
  phases: [
    { title: 'Review', detail: 'clean read-only Opus reviewers' },
    { title: 'Grade', detail: 'two graders, third on disagreement, verifier for disputed findings' },
  ],
}
const WS = '<EVAL_ROOT>'
const N = '<CODE_ROOT>/analysis-tasks'
const PY = N + '/bin/python'
const CHECKLIST = WS + '/protocol/CHECKLIST_GENERIC.md'
const ITEMS = args.items  // [{id,...}]
const BUDGET = 'Spend about the effort of a careful one-hour expert review; you may use up to roughly 50 tool calls.'
const ARM = {
  generic: 'Review the analysis critically, as an expert reviewer would before publication: check the code, the outputs and the claims in the summary against each other and against the data.',
  checklist: `Review the analysis using the checklist at ${CHECKLIST} (you may also read that one file). Apply all ten patterns to the analysis.`,
}
const REVIEW_SCHEMA = { type: 'object', properties: { summary: { type: 'string' }, findings: { type: 'array', items: { type: 'object', properties: {
  id: { type: 'string' }, location: { type: 'string' }, severity: { type: 'string', enum: ['critical', 'major', 'minor'] }, description: { type: 'string' }, evidence: { type: 'string' }, recommended_action: { type: 'string' } },
  required: ['id', 'location', 'severity', 'description', 'evidence', 'recommended_action'] } } }, required: ['summary', 'findings'] }
const GRADE_SCHEMA = { type: 'object', properties: {
  error_detected: { type: ['boolean', 'null'] }, detecting_finding_ids: { type: 'array', items: { type: 'string' } }, detection_reason: { type: 'string' },
  findings: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, classification: { type: 'string', enum: ['documented_error', 'other_real_problem', 'false_alarm', 'minor_or_style', 'disputed'] }, load_bearing: { type: 'boolean' }, reason: { type: 'string' } }, required: ['id', 'classification', 'load_bearing', 'reason'] } } },
  required: ['error_detected', 'detecting_finding_ids', 'detection_reason', 'findings'] }
const VER_SCHEMA = { type: 'object', properties: { verdicts: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, verdict: { type: 'string', enum: ['real_problem', 'false_alarm', 'cannot_decide'] }, reason: { type: 'string' } }, required: ['id', 'verdict', 'reason'] } } }, required: ['verdicts'] }

function reviewPrompt(id, arm) {
  return `The folder ${N}/${id} contains a data analysis: a README, code, data, outputs and a results summary (SUMMARY.md). ${ARM[arm]} Read only inside ${N}/${id}${arm === 'checklist' ? ' and the checklist file' : ''}. You may run the code or your own checks with ${PY} (run Python inline; do not write files; if the code writes outputs, do not run it in a way that changes the folder). List every problem that affects the correctness of the analysis's results or of the claims in its summary. For each: location (file and line), severity (critical = a headline result or claim cannot stand; major; minor), description, evidence, recommended action. If you find no problem, return an empty list. ${BUDGET} Return the structured output.`
}
function gradePrompt(id, rv, g) {
  return `Grade one review of the analysis in ${N}/${id} against its answer key ${WS}/studyB/keys/${id}.json (read the key fully; it says whether the analysis contains a documented error, gives a detection rule, known true facts and possible false alarms). You may read the package and run checks with ${PY} (inline; do not write files). Do not try to guess how the review was produced.\n(1) If the key has a documented error: does at least one finding detect it under the key's detection rule? List the finding ids. If the key has no documented error, set error_detected to null.\n(2) Classify EVERY finding: 'documented_error' (it is the key's error), 'other_real_problem' (a genuine problem not in the key, that you can confirm), 'false_alarm' (it claims a problem that does not exist, e.g. contradicted by the key's known true facts or by the data), 'minor_or_style' (true but trivial), or 'disputed' (you cannot decide without deeper checking). Say whether the finding is load-bearing (claims a headline result or claim is wrong). Grader ${g}. Return the structured output.\n\nREVIEW FINDINGS (JSON):\n${JSON.stringify(rv.findings)}`
}
function same(a, b) {
  if (!a || !b) return false
  if (a.error_detected !== b.error_detected) return false
  const fa = (a.findings.filter(f => f.classification === 'false_alarm' && f.load_bearing).length)
  const fb = (b.findings.filter(f => f.classification === 'false_alarm' && f.load_bearing).length)
  return fa === fb
}
const runs = []
for (const it of ITEMS) for (const arm of ['generic', 'checklist']) for (let r = 1; r <= 3; r++) runs.push({ id: it.id, arm, rep: r })

const res = await pipeline(runs,
  x => agent(reviewPrompt(x.id, x.arm) + `\n(Run ${x.rep}.)`, { agentType: 'Explore', model: 'opus', schema: REVIEW_SCHEMA, label: `B:${x.id}:${x.arm}:${x.rep}:review`, phase: 'Review' }),
  async (rv, x) => {
    if (!rv) return null
    const lab = `B:${x.id}:${x.arm}:${x.rep}`
    const [a, b] = await parallel([1, 2].map(g => () => agent(gradePrompt(x.id, rv, g), { agentType: 'Explore', model: 'opus', schema: GRADE_SCHEMA, label: `${lab}:grade${g}`, phase: 'Grade' })))
    let c = null
    if (!same(a, b)) c = await agent(gradePrompt(x.id, rv, 3), { agentType: 'Explore', model: 'opus', schema: GRADE_SCHEMA, label: `${lab}:grade3`, phase: 'Grade' })
    const disputedIds = new Set()
    for (const g of [a, b, c].filter(Boolean)) for (const f of g.findings) if (f.classification === 'disputed') disputedIds.add(f.id)
    let ver = null
    const disputed = rv.findings.filter(f => disputedIds.has(f.id))
    if (disputed.length) ver = await agent(`Check the findings below about the analysis in ${N}/${x.id} by running code and reading files (${PY}, inline, do not write files; read only inside the package). Use the answer key ${WS}/studyB/keys/${x.id}.json as background. For each finding decide: 'real_problem' (it affects a result or a claim), 'false_alarm', or 'cannot_decide', with evidence.\n\nFINDINGS (JSON):\n${JSON.stringify(disputed)}`, { agentType: 'Explore', model: 'opus', schema: VER_SCHEMA, label: `${lab}:verify`, phase: 'Grade' })
    return { ...x, review: rv, grade_a: a, grade_b: b, grade_c: c, verify: ver }
  })
log(`done ${res.filter(Boolean).length}/${runs.length}`)
return res

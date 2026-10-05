export const meta = {
  name: 'studyA-factorial',
  description: 'Study A: contract vs paper executors, then none/generic/checklist review with repair, blinded grading against frozen keys (clean read-only Opus agents)',
  phases: [
    { title: 'Execute', detail: 'executor agents per task x arm x replicate' },
    { title: 'Review', detail: 'generic and checklist reviews of each output' },
    { title: 'Repair', detail: 'identical repair step after each review' },
    { title: 'Grade', detail: 'two blinded graders per final deliverable, third on disagreement' },
  ],
}

const WS = '<EVAL_ROOT>'
const PY = '<CODE_ROOT>/analysis-tasks/bin/python'
const CHECKLIST = WS + '/protocol/CHECKLIST_GENERIC.md'
const TASKS = args.tasks            // [{task, paper_pkg, contract_pkg, key}]
const REPS = args.reps || 5

const EST = { type: 'object', properties: { name: { type: 'string' }, value: { type: ['number', 'string', 'null'] }, ci_low: { type: ['number', 'null'] }, ci_high: { type: ['number', 'null'] }, method_of_uncertainty: { type: 'string' } }, required: ['name', 'value', 'ci_low', 'ci_high', 'method_of_uncertainty'] }
const EXEC_SCHEMA = { type: 'object', properties: {
  report: { type: 'string' },
  results: { type: 'object', properties: { verdict: { type: 'string' }, estimates: { type: 'array', items: EST }, analysis_script: { type: 'string' } }, required: ['verdict', 'estimates', 'analysis_script'] } },
  required: ['report', 'results'] }
const REVIEW_SCHEMA = { type: 'object', properties: { summary: { type: 'string' }, points: { type: 'array', items: { type: 'object', properties: {
  id: { type: 'string' }, location: { type: 'string' }, severity: { type: 'string', enum: ['critical', 'major', 'minor'] }, description: { type: 'string' }, evidence: { type: 'string' }, recommended_action: { type: 'string' } },
  required: ['id', 'location', 'severity', 'description', 'evidence', 'recommended_action'] } } }, required: ['summary', 'points'] }
const GRADE_SCHEMA = { type: 'object', properties: {
  verdict_given: { type: 'string' }, verdict_correct: { type: 'boolean' }, verdict_reason: { type: 'string' },
  traps: { type: 'array', items: { type: 'object', properties: { trap_id: { type: 'string' }, handled: { type: 'string', enum: ['yes', 'partly', 'no', 'not_applicable'] }, reason: { type: 'string' } }, required: ['trap_id', 'handled', 'reason'] } },
  false_statements: { type: 'array', items: { type: 'object', properties: { statement: { type: 'string' }, why_false: { type: 'string' } }, required: ['statement', 'why_false'] } },
  key_numbers: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, reported_value: { type: 'string' }, within_tolerance: { type: ['boolean', 'null'] } }, required: ['name', 'reported_value', 'within_tolerance'] } } },
  required: ['verdict_given', 'verdict_correct', 'verdict_reason', 'traps', 'false_statements', 'key_numbers'] }
const RGRADE_SCHEMA = { type: 'object', properties: { points: { type: 'array', items: { type: 'object', properties: {
  id: { type: 'string' }, judgement: { type: 'string', enum: ['valid', 'invalid', 'unclear'] }, reason: { type: 'string' } }, required: ['id', 'judgement', 'reason'] } } }, required: ['points'] }

const BUDGET = 'Spend about the effort of a careful two-hour expert review; you may use up to roughly 60 tool calls.'
const ARM = {
  generic: 'Review the deliverable critically, as an expert reviewer would before publication: check the code, the numbers and the claims against each other and against the data.',
  checklist: `Review the deliverable using the checklist at ${CHECKLIST} (you may also read that one file). Apply all ten patterns to the deliverable.`,
}

function execPrompt(pkg) {
  return `Your task is described in the file ${pkg}/BRIEF.md. Read it and carry it out fully. Read only inside ${pkg}. Run Python inline (python - <<'EOF' ... EOF) with the interpreter named in the brief; do not write files. When you are done, return the deliverable described in the brief as the structured output: 'report' (the Markdown report) and 'results' (verdict, estimates, analysis_script).`
}
function reviewPrompt(pkg, arm, d) {
  return `The folder ${pkg} contains an analysis task (BRIEF.md, data/, methods/ and possibly contract/). An analyst carried out the task and produced the deliverable below. ${ARM[arm]} Read only inside ${pkg}${arm === 'checklist' ? ' and the checklist file' : ''}; you may run Python inline with ${PY} to check numbers or code (do not write files). List every problem that affects the correctness of the deliverable's results, verdict or claims. For each point give: location (report section or script line), severity (critical = the verdict or a headline result cannot stand; major; minor), description, evidence, recommended action. If you find no problem, return an empty list. ${BUDGET} Return the structured output.\n\nDELIVERABLE — report:\n${d.report}\n\nDELIVERABLE — results (JSON):\n${JSON.stringify(d.results)}`
}
function repairPrompt(pkg, d, rv) {
  return `The folder ${pkg} contains an analysis task (read BRIEF.md). An analyst produced the deliverable below, and a second analyst raised the points listed after it. Revise the deliverable: check each point against the data (run Python inline with ${PY}; do not write files), fix what is right, keep what is right in the original, and re-run analyses as needed. The revised report must stand on its own: do not mention the points, the second analyst, or that this is a revision. Return the full revised deliverable as structured output in the same format: 'report' and 'results' (verdict, estimates, complete analysis_script). Read only inside ${pkg}.\n\nORIGINAL DELIVERABLE — report:\n${d.report}\n\nORIGINAL DELIVERABLE — results (JSON):\n${JSON.stringify(d.results)}\n\nPOINTS RAISED (JSON):\n${JSON.stringify(rv.points)}`
}
function redact(s) {
  return String(s).replace(/\b(specifications?|specs?|contracts?|checklists?|audit(?:s|or|ors|ed|ing)?|review(?:s|er|ers|ed|ing)?|repair(?:s|ed|ing)?|revised|revision|second analyst)\b/gi, '[REDACTED]').replace(/\bSPEC\.md\b/g, '[REDACTED]').replace(/\bP(10|[1-9])\b/g, '[REDACTED]')
}
function gradePrompt(pkg, key, d, g) {
  const red = { report: redact(d.report), results: JSON.parse(redact(JSON.stringify(d.results))) }
  return `You are grading one deliverable for an analysis task against a fixed answer key. The task brief is ${pkg}/BRIEF.md (read it; you may read the package data and run Python inline with ${PY}, without writing files). The answer key is ${key} (read it fully: key_verdict, key_conclusion, acceptable_alternatives, key_numbers, traps, false_statement_checks). Some words in the deliverable are replaced by [REDACTED]; ignore that and do not try to guess how the deliverable was produced.\nGrade: (1) verdict: is the deliverable's verdict correct according to the key? An acceptable alternative counts as correct only under the conditions the key states. (2) every trap in the key: handled yes / partly / no / not_applicable, with a one-line reason based on the report AND the analysis script. (3) materially false statements in the report (use the key's false_statement_checks and your own checking; count a statement as false only if the data or the key clearly contradict it). (4) every key number for which the deliverable reports a matching quantity: is it within the key's tolerance? Grader ${g}. Return the structured output.\n\nDELIVERABLE — report:\n${red.report}\n\nDELIVERABLE — results (JSON):\n${JSON.stringify(red.results)}`
}
function rgradePrompt(pkg, key, d, rv) {
  return `Below are review points raised about a deliverable for the analysis task in ${pkg} (brief: ${pkg}/BRIEF.md). The answer key for the task is ${key}. For each point decide: 'valid' (a real problem in the deliverable that matters for its results, verdict or claims), 'invalid' (the point is wrong, or the deliverable already handles it, or it does not matter), or 'unclear'. You may read the package and run Python inline with ${PY} (do not write files). Give a one-line reason for each. Return the structured output.\n\nDELIVERABLE — report:\n${d.report}\n\nDELIVERABLE — results (JSON):\n${JSON.stringify(d.results)}\n\nREVIEW POINTS (JSON):\n${JSON.stringify(rv.points)}`
}

async function grade(pkg, key, d, lab) {
  if (!d) return null
  const [a, b] = await parallel([1, 2].map(g => () => agent(gradePrompt(pkg, key, d, g), { agentType: 'Explore', model: 'opus', schema: GRADE_SCHEMA, label: `${lab}:grade${g}`, phase: 'Grade' })))
  let c = null
  if (!a || !b || a.verdict_correct !== b.verdict_correct) c = await agent(gradePrompt(pkg, key, d, 3), { agentType: 'Explore', model: 'opus', schema: GRADE_SCHEMA, label: `${lab}:grade3`, phase: 'Grade' })
  return { a, b, c }
}

const runs = []
for (const t of TASKS) for (const arm of ['paper', 'contract']) for (let r = 1; r <= REPS; r++) runs.push({ task: t.task, arm, rep: r, pkg: arm === 'paper' ? t.paper_pkg : t.contract_pkg, key: t.key })

const results = await pipeline(runs,
  x => agent(execPrompt(x.pkg) + `\n(Run ${x.rep}.)`, { agentType: 'Explore', model: 'opus', schema: EXEC_SCHEMA, label: `A:${x.task}:${x.arm}:${x.rep}:exec`, phase: 'Execute' }),
  async (d0, x) => {
    if (!d0) return null
    const lab = `A:${x.task}:${x.arm}:${x.rep}`
    const [rg, rc] = await parallel(['generic', 'checklist'].map(arm => () => agent(reviewPrompt(x.pkg, arm, d0), { agentType: 'Explore', model: 'opus', schema: REVIEW_SCHEMA, label: `${lab}:review-${arm}`, phase: 'Review' })))
    const [dg, dc] = await parallel([rg, rc].map((rv, i) => () => rv ? agent(repairPrompt(x.pkg, d0, rv), { agentType: 'Explore', model: 'opus', schema: EXEC_SCHEMA, label: `${lab}:repair-${i ? 'checklist' : 'generic'}`, phase: 'Repair' }) : Promise.resolve(null)))
    const [g0, gg, gc, rgg, rgc] = await parallel([
      () => grade(x.pkg, x.key, d0, `${lab}:none`),
      () => grade(x.pkg, x.key, dg, `${lab}:generic`),
      () => grade(x.pkg, x.key, dc, `${lab}:checklist`),
      () => rg ? agent(rgradePrompt(x.pkg, x.key, d0, rg), { agentType: 'Explore', model: 'opus', schema: RGRADE_SCHEMA, label: `${lab}:rgrade-generic`, phase: 'Grade' }) : Promise.resolve(null),
      () => rc ? agent(rgradePrompt(x.pkg, x.key, d0, rc), { agentType: 'Explore', model: 'opus', schema: RGRADE_SCHEMA, label: `${lab}:rgrade-checklist`, phase: 'Grade' }) : Promise.resolve(null),
    ])
    return { ...x, exec: d0, review_generic: rg, review_checklist: rc, repaired_generic: dg, repaired_checklist: dc, grade_none: g0, grade_generic: gg, grade_checklist: gc, rgrade_generic: rgg, rgrade_checklist: rgc }
  })
log(`done: ${results.filter(Boolean).length} of ${runs.length} executor runs completed`)
return results

export const meta = {
  name: 'build-studyA-studyB-packages',
  description: 'Build and independently verify Study A task packages (T1, T3, T4) and the Study B held-out flawed/clean analysis pairs',
  phases: [
    { title: 'Select', detail: 'choose Study B analysis pairs' },
    { title: 'Build', detail: 'build packages + answer keys' },
    { title: 'Verify', detail: 'independent verification of each package and key' },
  ],
}

const ROOT = '<REPO_ROOT>'
const MX = ROOT + '/projects/maxtoki'
const INV = MX + '/verification'
const WS = '<EVAL_ROOT>'
const PY = WS + '/bin/python'

const COMMON = `You are helping build materials for a pre-registered controlled experiment on agent-run analyses (protocol: ${WS}/protocol/PREREGISTRATION.md - read sections 2-4). Experiment subjects are clean Claude agents that will see ONLY the package folder you build (read-only; they run Python inline with ${PY}; CPU only). Answer keys go OUTSIDE the packages and must never be copied into them.
RULES: Read ${ROOT}/CLAUDE.md for writing conventions (plain English). Never modify or delete files in ${ROOT}; copy what you need. Python: ${PY}. CPU only, no GPU/MPS, memory < 4 GB, each command < 9 minutes, no background processes, no 'sleep'. Packages must contain NO hints of the answer: no comments or names like fixed/buggy/v2/retracted/correct/wrong/leak, no outcome statements in READMEs, no paths that reveal the repo layout beyond the package itself (rewrite paths to be package-relative), no mention of audits, reviews, errors, or this experiment. Final answer <= 400 words with paths and a list of anything you could not do.`

// ---------------- Study A ----------------
const A_TASKS = [
  { id: 'T1', prompt: `STUDY A task T1 (attention vs TRRUST, MaxToki-217M, RPE1). Source data: ${MX}/runs/attention-grn-217M/outputs/v2_eval/task_data_rpe1/ (read its README) and the report ${MX}/runs/attention-grn-217M/V2_EVAL_REPORT.md (for the reference answer; never copy it into the package). Method spec for the contract arm: ${ROOT}/pipelines/attention-grn-extraction-and-evaluation.md (copy verbatim as contract/SPEC.md). Source-method paper: ${ROOT}/references/2602.17532_Kendiukhov_SCFM_interpretability.pdf (copy as methods/source_method_paper.pdf).
Scientific question for the brief (neutral wording): "Do MaxToki-217M's attention maps encode regulator-to-target relationships? Use the TRRUST curated network as ground truth." Verdict options: supported / not supported / inconclusive.` },
  { id: 'T3', prompt: `STUDY A task T3 (TF specificity of SAE features, MaxToki-217M, K562). Source data: ${MX}/runs/sae-atlas-217M/outputs/v2_tf_specificity/task_data/ (read its README) and report ${MX}/runs/sae-atlas-217M/V2_TF_SPECIFICITY_REPORT.md (reference answer only). Spec: ${ROOT}/pipelines/sparse-autoencoders/01-sae-atlas.md (copy as contract/SPEC.md). Paper: ${ROOT}/references/2603.02952_Kendiukhov_SAE_atlas.pdf.
Question: "Are the sparse-autoencoder features that respond to a transcription factor's knockdown specific to that factor's target genes?" Ground truth resources to include: TRRUST and DoRothEA (with confidence levels and ChIP-seq flags) restricted to the gene universe. Verdict options: supported / not supported / inconclusive. Include data for all TFs in the task data (not only GATA1); the question is about TFs in general, and the brief may say that GATA1 is a well-studied example the analyst may wish to look at.` },
  { id: 'T4', prompt: `STUDY A task T4 (cross-model gene-embedding alignment: MaxToki-217M vs scGPT, with Geneformer V2-316M as a same-tokenizer reference). Source: ${MX}/runs/topology-141-217M/V2_CROSSMODEL_REPORT.md (reference answer only; read it fully, including its Verification notes), the deployed scripts ${MX}/runs/topology-141-217M/scripts/phase4_scgpt_cross_model.py and phase14_cross_model_cca.py and their inputs. Build task data with: the MaxToki-217M input gene-embedding table (with its token dictionary Ensembl->row), the Geneformer V2-316M gene-embedding table (with its dictionary), the scGPT whole-human gene-embedding table AS DISTRIBUTED with its own vocabulary file (symbol->row; include the checkpoint's vocab.json exactly), an Ensembl<->symbol mapping table, and the three tissue-domain gene panels used in the deployment. Do NOT pre-join tables; the analyst must map genes to rows themselves. Keep the package under ~600 MB; if a table is too large, subset to genes present in any panel plus a random 5,000-gene background (document it neutrally). Decide which deployed spec governs this analysis (topology-141 or spectral-geometry; check which spec describes cross-model CCA/alignment) and copy THAT spec as contract/SPEC.md; source paper accordingly (${ROOT}/references/2602.22289_Kendiukhov_topology_141_hypotheses.pdf or 2602.22247_Kendiukhov_spectral_geometry.pdf).
Question: "Is MaxToki-217M's gene-embedding geometry aligned with scGPT's, and how does that alignment compare with its alignment to Geneformer V2-316M (which shares MaxToki's gene vocabulary)?" Verdict options: aligned with scGPT about as well as with Geneformer / aligned with scGPT but clearly less than with Geneformer / not aligned with scGPT beyond chance / inconclusive.` },
]

const A_BUILD = (t) => `${COMMON}

${t.prompt}

BUILD two package folders that are identical except for contract/:
  ${WS}/studyA/tasks/${t.id}-paper/   and   ${WS}/studyA/tasks/${t.id}-contract/
Each contains: BRIEF.md, data/ (task data, package-relative paths, outcome-free READMEs), methods/source_method_paper.pdf. The -contract folder also has contract/SPEC.md. 
BRIEF.md (identical in both folders except ONE sentence, which in the contract version reads exactly: "A method specification for this kind of analysis is provided in contract/SPEC.md; follow it." and in the paper version is absent) must contain: (1) the question and verdict options; (2) short neutral background on the model and data (what each file is, shapes, units, how it was produced, at the level a competent analyst needs); (3) the deliverable: a report (Markdown) with sections Question, Data, Methods, Results (every estimate with its uncertainty and the unit of resampling), Verdict (one of the options + 2-4 sentences of justification), Limitations; plus a results object with fields: verdict (one of the options), estimates (list of {name, value, ci_low, ci_high, method_of_uncertainty}), and analysis_script (a complete, self-contained Python script that, run from the package root with ${PY}, recomputes every reported number from data/ and prints them as JSON); (4) limits: CPU only; each command must finish within 9 minutes; aim to finish within about 2 hours of work; do not write files (run Python inline); read only inside the package folder. Do NOT mention confounds, baselines, nulls, controls, traps or what a good analysis should include beyond what any brief would ask (estimates with uncertainty).
ANSWER KEY at ${WS}/studyA/keys/${t.id}/ (outside packages): key.json with: key_verdict (one of the options), key_conclusion (2-4 sentences), acceptable_alternatives (verdicts that should count as correct, if any, with reason; default none), key_numbers (list of {name, value, tolerance, definition} for quantities an analyst would likely compute), traps (list of 3-6 {id, description, what correct handling looks like, what mishandling looks like}) - the traps must be the genuine methodological pitfalls of this task, including those the original deployment fell into, false_statement_checks (list of tempting but false statements); plus reference.py that computes every key number from the PACKAGE data (run it, save its output as reference_output.json) and NOTES.md explaining the reference. The key must agree with the corrected analysis in the source report; if your own computation disagrees with that report, investigate and document why.`

const A_VERIFY = (t, built) => `${COMMON}

You are independently verifying a Study A task package and its answer key built by another agent (task ${t.id}). Builder summary:\n---\n${built}\n---\nTask definition:\n---\n${t.prompt}\n---
Check: (1) the two folders differ ONLY by contract/SPEC.md and the one BRIEF sentence (diff them); (2) no hints or outcome statements anywhere in the packages (grep for leak words: fixed, bug, v2, retract, correct, wrong, leak, confound, baseline, null, chance, artefact, artifact, expected, should find, GATA1 result, 0.6, AUROC values from the report, etc.; read every README); (3) data files load and match their descriptions; (4) re-derive the key numbers with YOUR OWN code from the package data (do not reuse reference.py) and compare; (5) the key verdict follows from the evidence and agrees with the corrected source report; (6) traps are real and their handling descriptions are fair. Fix problems directly (in the packages or keys) and log every change in ${WS}/studyA/keys/${t.id}/VERIFICATION.md. Return verdict OK / OK after fixes / PROBLEMS with details.`

// ---------------- Study B ----------------
const SELECT_SCHEMA = { type: 'object', properties: { pairs: { type: 'array', items: { type: 'object', properties: {
  pair_id: { type: 'string' }, source_project: { type: 'string' }, analysis: { type: 'string' }, error: { type: 'string' },
  error_kind: { type: 'string', enum: ['natural', 'planted'] }, in_checklist: { type: 'boolean' }, pattern: { type: 'string' },
  flawed_source: { type: 'string' }, clean_source: { type: 'string' }, build_notes: { type: 'string' } },
  required: ['pair_id', 'source_project', 'analysis', 'error', 'error_kind', 'in_checklist', 'pattern', 'flawed_source', 'clean_source', 'build_notes'] } } }, required: ['pairs'] }

phase('Select')
const sel = await agent(`${COMMON}

STUDY B selection. Read ${INV}/heldout_tasks.md fully (and its checks/ folder). Choose exactly 12 analysis PAIRS for a held-out audit benchmark. Each pair = one analysis in a FLAWED version (containing exactly one documented error) and a CLEAN version (identical except that the error is fixed). Requirements:
- At least 9 pairs with NATURAL errors (real historical errors from projects in ${ROOT}/projects/ other than maxtoki), where both versions exist on disk or the clean one can be made by applying the documented fix.
- 2-3 PLANTED pairs for causal/mechanistic overreach (pattern P9), built from a clean analysis by rewriting only its results summary to overclaim (flawed) vs state it accurately (clean).
- Balance: about half the errors inside the ten checklist patterns (P1 weak null/no chance anchor; P2 no trivial baseline; P3 endpoint-object mismatch; P4 wrong unit/pseudo-replication/leaky CV; P5 selection inflation; P6 no positive control; P7 one-condition generalisation; P8 no CI/stability; P9 causal overreach; P10 reproducibility), half outside (pure code/data bugs: tie handling, label filling, sign flips, overwritten outputs, leakage through code).
- Diverse domains/models (SCFMs other than MaxToki, protein language models, text-LLM cell models, GRN benchmarks, perturbation models).
- Each version must run end-to-end on CPU in < 5 minutes on small data (< 200 MB per package).
- AVOID analyses from the seven earlier papers whose reviews built the checklist (the 141-hypotheses/H123 topology work, the CSSI/attention paper, and the other scGPT/Geneformer papers listed in the report as contaminated).
Return the 12 pairs (pair_id P01..P12) with exact source paths for both versions and build notes.`, { label: 'B:select', phase: 'Select', schema: SELECT_SCHEMA })

const B_BUILD = (p) => `${COMMON}

STUDY B pair ${p.pair_id}: ${JSON.stringify(p)}
Build TWO self-contained packages for this analysis: a FLAWED version (with exactly the documented error) and a CLEAN version (identical except the error is fixed). Give each an opaque id: write them to ${WS}/studyB/packages/<OPAQUE_ID>/ where OPAQUE_ID is 'B' followed by 4 random digits (generate two different ones; they must not reveal the pair or the status). Each package: README.md (neutral description of the analysis, its data and how to run it), code/ (the analysis code, runnable from the package root with ${PY}; strip comments that reveal or describe the error or its fix), data/ (inputs, small), outputs/ (the outputs produced by running the code - re-run it to make sure they match), SUMMARY.md (the results summary as an analyst would write it: for the flawed version, the conclusions the original analyst drew from the flawed result; for the clean version, the conclusions from the correct result, written in the same style and length). For planted P9 pairs, only SUMMARY.md differs.
Answer key at ${WS}/studyB/keys/<OPAQUE_ID>.json: {opaque_id, pair_id, status: 'flawed'|'clean', twin_id, source_project, analysis, error: {nature, location (file:line in the package), wrong_result, correct_result, pattern, in_checklist} (null for clean), detection_rule (what a finding must say to count as detecting it), known_true_facts (statements about the package that are correct, so that graders can reject false alarms), possible_false_alarms (plausible-looking concerns that are not real errors here)}. Also record the build in ${WS}/studyB/keys/${p.pair_id}_BUILD.md (sources, what you changed, how you verified flawed->wrong and clean->right numbers). If the pair turns out infeasible, say so clearly and stop.`

const B_VERIFY = (p, built) => `${COMMON}

Independently verify Study B pair ${p.pair_id} built by another agent. Pair definition: ${JSON.stringify(p)}. Builder summary:\n---\n${built}\n---
Find its two packages via ${WS}/studyB/keys/${p.pair_id}_BUILD.md and the keys. Check: (1) both packages run from their roots on CPU in < 5 min and reproduce their outputs/; (2) the flawed version reproduces the documented wrong result and the clean version the correct one, and the two differ ONLY at the error (diff them); (3) no hints anywhere (comments, names, READMEs, SUMMARY wording, file dates are fine); (4) the key's error location is correct and the detection rule is fair; (5) known_true_facts are true. Fix problems directly and log changes in ${WS}/studyB/keys/${p.pair_id}_VERIFICATION.md. Return OK / OK after fixes / PROBLEMS / INFEASIBLE with details.`

phase('Build')
const [aRes, bRes] = await parallel([
  () => pipeline(A_TASKS,
    t => agent(A_BUILD(t), { label: 'A-build:' + t.id, phase: 'Build' }),
    (built, t) => agent(A_VERIFY(t, built), { label: 'A-verify:' + t.id, phase: 'Verify' }).then(v => ({ id: t.id, built, verify: v }))),
  () => sel && sel.pairs ? pipeline(sel.pairs,
    p => agent(B_BUILD(p), { label: 'B-build:' + p.pair_id, phase: 'Build' }),
    (built, p) => agent(B_VERIFY(p, built), { label: 'B-verify:' + p.pair_id, phase: 'Verify' }).then(v => ({ pair: p, built, verify: v }))) : Promise.resolve([]),
])
return { studyA: aRes, studyB_selection: sel, studyB: bRes }

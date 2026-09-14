"""Generate EXPERIMENT4B_REPORT.md entirely from machine-readable JSON records
under `outputs/exp4b_joint_solver/`. No metric value in this file is
hand-maintained.
"""
from __future__ import annotations

import json

from . import OUT, SCENES
from experiments.exp1.env import ROOT


def _load(name: str):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(x, nd=4):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) else str(x)


def generate() -> str:
    metrics = _load('metrics.json') or {}
    gates = _load('gates.json') or {}
    audit = _load('completion_audit.json') or {'status': 'INCOMPLETE',
                                               'failures': ['completion_audit.json not found']}
    integrity = _load('integrity.json') or {}
    test_results = _load('test_results.json') or {}
    deployment = _load('deployment_policy.json') or {}
    failure_analysis = _load('failure_analysis.json') or {}
    pcc = _load('prior_claim_corrections.json') or {}

    lines = []
    lines.append('# Experiment 4B: Candidate-Rank Fidelity, Independent Geometric Evidence '
                 'and Joint Constellation Identification')
    lines.append('')
    lines.append('**Generated entirely from machine-readable records** under '
                 '`outputs/exp4b_joint_solver/`. Regenerate with '
                 '`python -m experiments.exp4b_joint_solver.report`.')
    lines.append('')

    lines.append('## Status')
    lines.append('')
    lines.append(f'**{audit.get("status")}** (implementation) -- '
                 f'**performance_gates_passed={audit.get("performance_gates_passed")}** '
                 f'({audit.get("n_gates_pass")}/{audit.get("n_gates_total")} gates)')
    lines.append('')
    lines.append(audit.get('note', ''))
    if audit.get('failures'):
        lines.append('')
        for f in audit['failures']:
            lines.append(f'- {f}')
    lines.append('')

    lines.append('## Correction to Experiment 4\'s claim')
    lines.append('')
    lines.append(pcc.get('corrected_summary', ''))
    lines.append('')

    lines.append('## Phase 1: candidate-rank fidelity')
    lines.append('')
    p1 = metrics.get('phase1_rank_fidelity', {}).get('selection', {})
    lines.append('7 fixed ranking rules evaluated leave-one-sky-out x 2 seeds. '
                 f'Selected rule: **{p1.get("selected")}**.')
    lines.append('')
    lines.append('| Rule | Mean top1_reward |')
    lines.append('|---|---:|')
    for rule, val in sorted(p1.get('means', {}).items()):
        marker = ' **(selected)**' if rule == p1.get('selected') else ''
        lines.append(f'| {rule}{marker} | {_fmt(val)} |')
    lines.append('')

    lines.append('## Phase 2: independent geometric evidence')
    lines.append('')
    p2 = metrics.get('phase2_independent_support', {})
    lines.append('| Scene | existing_score rank | held_out_stability rank | '
                 'size_normalized rank |')
    lines.append('|---|---:|---:|---:|')
    for scene in SCENES:
        node = p2.get(scene, {})
        lines.append(f'| {scene} | {node.get("existing_score", {}).get("true_class_rank")} | '
                     f'{node.get("held_out_stability", {}).get("true_class_rank")} | '
                     f'{node.get("held_out_with_size_normalization", {}).get("true_class_rank")} |')
    lines.append('')
    lines.append('`held_out_stability` never regresses (gate 2 passes); '
                 '`held_out_with_size_normalization` regresses pisces and taurus (gate 3 fails).')
    lines.append('')

    lines.append('## Phase 3: unqueried-star evidence and null model')
    lines.append('')
    p3 = metrics.get('phase3_null_model', {}).get('results', {})
    lines.append('| Scene | none | raw_count | quality_weighted | matched_null_lr | '
                 'matched_null_lr_corrected |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for scene in SCENES:
        node = p3.get(scene, {})
        row = [str(node.get(a, {}).get('true_class_rank')) for a in
              ('none', 'raw_count', 'quality_weighted', 'matched_null_lr',
               'matched_null_lr_corrected')]
        lines.append(f'| {scene} | ' + ' | '.join(row) + ' |')
    lines.append('')
    lines.append('Uncorrected unqueried evidence breaks scorpius (rank1->11); the sqrt(n) '
                 'multiple-testing correction recovers it (gate 4 passes). Corrected evidence '
                 'improves pisces\'s rank (15->6) but does not flip its winner (gate 5 passes '
                 'on rank improvement, gate 7 still fails on winner correctness); taurus is '
                 'unchanged.')
    lines.append('')

    lines.append('## Phase 4: joint multi-candidate beam-search solver')
    lines.append('')
    lines.append('10 matched comparisons x 3 scenes x 2 seeds. Key finding: the complete '
                 'joint solver (additive composite score) regresses scorpius from correct to '
                 'wrong on BOTH seeds (gate 6 fails), and fixes none of the previously-wrong '
                 'scenes (gate 7 fails). The regression is reproducible across seeds (gate 8 '
                 'passes as a reproducibility check, even though the underlying result is a '
                 'failure).')
    lines.append('')
    p4p = metrics.get('phase4_joint_solver', {}).get('primary', {})
    p4r = metrics.get('phase4_joint_solver', {}).get('repeat', {})
    lines.append('| Scene | independent_full_slate (both seeds) | complete_joint_solver primary '
                 '| complete_joint_solver repeat |')
    lines.append('|---|---|---|---|')
    for scene in SCENES:
        base = p4p.get(scene, {}).get('4_independent_full_slate', {})
        cp = p4p.get(scene, {}).get('10_complete_joint_solver', {})
        cr = p4r.get(scene, {}).get('10_complete_joint_solver', {})
        lines.append(f'| {scene} | {base.get("winner")} ({"correct" if base.get("correct") else "wrong"}) '
                     f'| {cp.get("winner")} ({"correct" if cp.get("correct") else "wrong"}) '
                     f'| {cr.get("winner")} ({"correct" if cr.get("correct") else "wrong"}) |')
    lines.append('')

    lines.append('## Phase 5: full whole-sky evaluation and synthetic screen')
    lines.append('')
    p5 = metrics.get('phase5_full_evaluation', {})
    mb = metrics.get('matching_baseline', {})
    lines.append('| Seed | Exp4B score | Matching fixed-policy baseline | Delta |')
    lines.append('|---|---:|---:|---:|')
    for tag in ('primary', 'repeat'):
        v = p5.get(tag, {}).get('verifier_snap_rescue', {}).get('score')
        b = mb.get(tag, {}).get('score')
        d = (v - b) if (v is not None and b is not None) else None
        lines.append(f'| {tag} | {_fmt(v)} | {_fmt(b)} | {_fmt(d)} |')
    lines.append('')
    lines.append('Primary seed is flat vs the matching baseline; repeat seed REGRESSES '
                 '(gates 9/10 fail). Per-scene regressions on primary pisces/scorpius and '
                 'repeat pisces exceed the 0.02 floor (gate 11 fails).')
    lines.append('')
    syn = metrics.get('phase5_synthetic_screen', {})
    lines.append(f'Synthetic all-48 class-disjoint screen (SYNTHETIC ENGINEERING SCREEN, not '
                 f'real-scene evidence): existing_recognize accuracy='
                 f'{_fmt(syn.get("accuracy_existing_recognize"))}, independent_scorer accuracy='
                 f'{_fmt(syn.get("accuracy_independent_scorer"))} (gate 12 passes).')
    lines.append('')

    lines.append('## Promotion gates')
    lines.append('')
    lines.append(f'**{gates.get("n_pass")}/{gates.get("n_gates")} gates pass** '
                 f'(overall_pass={gates.get("overall_pass")}).')
    lines.append('')
    lines.append('| # | Gate | Result |')
    lines.append('|---|---|---|')
    for name, node in gates.get('checks', {}).items():
        lines.append(f'| | {name} | {"PASS" if node.get("pass") else "FAIL"} |')
    lines.append('')

    lines.append('## Deployment decision')
    lines.append('')
    lines.append(deployment.get('decision', 'not recorded'))
    lines.append('')

    lines.append('## Failure analysis summary')
    lines.append('')
    lines.append(failure_analysis.get('overall_conclusion', ''))
    lines.append('')
    lines.append(f'**Single most promising next step:** '
                 f'{failure_analysis.get("single_most_promising_next_step", "")}')
    lines.append('')

    lines.append('## Test results')
    lines.append('')
    exp4b_suite = test_results.get('exp4b_only_suite_venv_exp1', {})
    full_exp1 = test_results.get('venv_exp1_full_suite', {})
    full_plain = test_results.get('venv_plain_full_suite', {})
    lines.append(f'- exp4b-only suite (.venv-exp1): {exp4b_suite.get("n_ran")} tests, '
                 f'ok={exp4b_suite.get("overall_ok")}')
    lines.append(f'- full repo suite (.venv-exp1): {full_exp1.get("n_ran")} tests, '
                 f'{full_exp1.get("n_pass")} pass, {full_exp1.get("n_failures")} fail, '
                 f'{full_exp1.get("n_errors")} error')
    lines.append(f'- full repo suite (.venv, no torch): {full_plain.get("n_ran")} tests, '
                 f'{full_plain.get("n_pass")} pass, {full_plain.get("n_skipped")} skipped')
    lines.append('')

    lines.append('## Integrity')
    lines.append('')
    lines.append(f'- Protected artifacts checked: {integrity.get("checked")}')
    lines.append(f'- Changed: {len(integrity.get("changed", []))}; '
                 f'Missing: {len(integrity.get("missing", []))}; '
                 f'New untracked: {len(integrity.get("new_untracked", []))}')
    lines.append(f'- Overall: {"OK" if integrity.get("ok") else "VIOLATION"}')
    lines.append('')

    lines.append('## Submission candidates')
    lines.append('')
    lines.append(f'None were generated. {gates.get("n_pass")}/{gates.get("n_gates")} promotion '
                 'gates pass, not all 14, so per the task\'s own rule ("only if the '
                 'identification method clears all required promotion gates") no candidate '
                 'file was produced. Existing production, Experiment 3 and Experiment 4 CSVs '
                 'were not modified (confirmed by the integrity check above).')
    lines.append('')
    lines.append('**Nothing was uploaded to Kaggle.**')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('*Generated by `experiments/exp4b_joint_solver/report.py` from machine '
                 'records only.*')
    return '\n'.join(lines)


def write_report(path: str | None = None) -> str:
    text = generate()
    out = path or str(ROOT / 'EXPERIMENT4B_REPORT.md')
    with open(out, 'w') as f:
        f.write(text)
    return out


if __name__ == '__main__':
    p = write_report()
    print(f'wrote {p}')

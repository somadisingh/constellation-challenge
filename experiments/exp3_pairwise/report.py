"""Generate EXPERIMENT3_REPORT.md ENTIRELY from machine-readable JSON records
(repair task §7). No metric in this file is hand-maintained; every number is
read from `outputs/exp3_pairwise/{record,metrics,gates,integrity,test_results,
completion_audit,deployment_policy,submission_validation,failure_analysis}.json`.

If `completion_audit.json` says INCOMPLETE, the report says INCOMPLETE. There is
no code path that lets this module print "complete" while a required field is
missing, pending, or contradictory.
"""
from __future__ import annotations

import json

from . import OUT
from experiments.exp1.env import ROOT


def _load(name: str) -> dict | None:
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(x, nd=4):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) else str(x)


def generate() -> str:
    record = _load('record.json') or {}
    metrics = _load('metrics.json') or {}
    gates = _load('gates.json') or {}
    integrity = _load('integrity.json') or {}
    test_results = _load('test_results.json') or {}
    audit = _load('completion_audit.json') or {'status': 'INCOMPLETE',
                                               'failures': ['completion_audit.json not found']}
    deployment = _load('deployment_policy.json') or {}
    submission_validation = _load('submission_validation.json') or {}
    failure_analysis = _load('failure_analysis.json') or {}
    corrections = _load('corrections.json') or {'corrections': []}

    status = audit.get('status', 'INCOMPLETE')
    primary_stage = metrics.get('primary_stage', 'unknown')
    primary = metrics.get('primary', {})
    repeat = metrics.get('repeat', {})
    pb = metrics.get('primary_baseline', {})
    rb = metrics.get('repeat_baseline', {})

    lines = []
    lines.append('# Experiment 3: Pairwise Verifier with Frozen HardNet Fusion')
    lines.append('')
    lines.append('**This report is generated entirely from machine-readable records**')
    lines.append('under `outputs/exp3_pairwise/`. No number below is hand-maintained;')
    lines.append('regenerate with `python -m experiments.exp3_pairwise.report`.')
    lines.append('')
    lines.append('## Status')
    lines.append('')
    lines.append(f'**{status}**' + (f' ({len(audit.get("failures", []))} failing check(s))'
                                     if status != 'COMPLETE' else ''))
    if audit.get('failures'):
        lines.append('')
        lines.append('Failing checks (from `completion_audit.json`):')
        for f in audit['failures']:
            lines.append(f'- {f}')
    lines.append('')

    lines.append('## Primary selected system')
    lines.append('')
    lines.append(f'- Integration stage: `{primary_stage}`')
    lines.append(f'- Rationale: {metrics.get("stage_selection_rationale", "")}')
    lines.append(f'- Primary seed: {metrics.get("primary_seed")}; repeat seed: '
                 f'{metrics.get("repeat_seed")}')
    lines.append('')

    lines.append('## Primary component metrics (out-of-fold, seed '
                 f'{metrics.get("primary_seed")})')
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|---|---:|')
    for k in ('presence', 'localization', 'recovery', 'identification', 'score'):
        lines.append(f'| {k} | {_fmt(primary.get(k))} |')
    lines.append('')

    lines.append('## Matching baseline and delta (Experiment 2 PRIMARY)')
    lines.append('')
    lines.append(f'- Baseline score: {_fmt(pb.get("score"))} '
                 f'(source: `{pb.get("source_file")}`, seed tag: `{pb.get("seed_tag")}`)')
    lines.append(f'- Delta: {pb.get("delta"):+.6f}' if isinstance(pb.get('delta'), float)
                else f'- Delta: {pb.get("delta")}')
    lines.append('')
    lines.append('Per-scene deltas (primary):')
    lines.append('')
    lines.append('| Scene | Experiment score | Baseline score | Delta |')
    lines.append('|---|---:|---:|---:|')
    for scene, d in metrics.get('primary_per_scene_vs_baseline', {}).items():
        lines.append(f'| {scene} | {_fmt(d.get("experiment_score"))} | '
                     f'{_fmt(d.get("baseline_score"))} | '
                     f'{d.get("delta"):+.4f} |' if isinstance(d.get('delta'), float)
                     else f'| {scene} | {_fmt(d.get("experiment_score"))} | '
                          f'{_fmt(d.get("baseline_score"))} | {d.get("delta")} |')
    lines.append('')

    lines.append('## Repeat component metrics (out-of-fold, seed '
                 f'{metrics.get("repeat_seed")})')
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|---|---:|')
    for k in ('presence', 'localization', 'recovery', 'identification', 'score'):
        lines.append(f'| {k} | {_fmt(repeat.get(k))} |')
    lines.append('')

    lines.append('## Matching repeat baseline and delta (Experiment 2 REPEAT)')
    lines.append('')
    lines.append(f'- Baseline score: {_fmt(rb.get("score"))} '
                 f'(source: `{rb.get("source_file")}`, seed tag: `{rb.get("seed_tag")}`)')
    lines.append(f'- Delta: {rb.get("delta"):+.6f}' if isinstance(rb.get('delta'), float)
                else f'- Delta: {rb.get("delta")}')
    lines.append('')
    lines.append('Per-scene deltas (repeat):')
    lines.append('')
    lines.append('| Scene | Experiment score | Baseline score | Delta |')
    lines.append('|---|---:|---:|---:|')
    for scene, d in metrics.get('repeat_per_scene_vs_baseline', {}).items():
        delta = d.get('delta')
        delta_str = f'{delta:+.4f}' if isinstance(delta, float) else str(delta)
        lines.append(f'| {scene} | {_fmt(d.get("experiment_score"))} | '
                     f'{_fmt(d.get("baseline_score"))} | {delta_str} |')
    lines.append('')

    lines.append('## All four integration stages (both seeds)')
    lines.append('')
    all_stages = metrics.get('all_stages', {})
    for tag in ('primary', 'repeat'):
        lines.append(f'### {tag.capitalize()} (seed '
                     f'{metrics.get(f"{tag}_seed")})')
        lines.append('')
        lines.append('| Stage | Score | Presence | Localization | Recovery | Identification |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        for stage, m in all_stages.get(tag, {}).items():
            lines.append(f'| {stage} | {_fmt(m.get("score"))} | {_fmt(m.get("presence"))} '
                         f'| {_fmt(m.get("localization"))} | {_fmt(m.get("recovery"))} '
                         f'| {_fmt(m.get("identification"))} |')
        lines.append('')

    lines.append('## Gate result')
    lines.append('')
    lines.append(f'Overall: **{"PASS" if gates.get("pass") else "FAIL"}**')
    lines.append('')
    lines.append('| Gate | Pass |')
    lines.append('|---|---|')
    for name, node in gates.get('checks', {}).items():
        lines.append(f'| {name} | {"PASS" if node.get("pass") else "FAIL"} |')
    lines.append('')

    lines.append('## Test result')
    lines.append('')
    exp3_suite = test_results.get('exp3_suite', {})
    prod_suite = test_results.get('production_suite', {})
    lines.append(f'- Exp3 suite (`{exp3_suite.get("command")}`): '
                 f'{exp3_suite.get("passed")} passed, {exp3_suite.get("failures")} failed, '
                 f'{exp3_suite.get("errors")} errors, {exp3_suite.get("skipped")} skipped '
                 f'(of {exp3_suite.get("tests_run")} total)')
    lines.append(f'- Production suite (`{prod_suite.get("command")}`): '
                 f'{prod_suite.get("passed")} passed, {prod_suite.get("failures")} failed, '
                 f'{prod_suite.get("errors")} errors, {prod_suite.get("skipped")} skipped '
                 f'(of {prod_suite.get("tests_run")} total)')
    lines.append('')
    lines.append('A skipped test is never counted as a pass. A test import error is a failure.')
    lines.append('')

    lines.append('## Integrity result')
    lines.append('')
    lines.append(f'- Protected artifacts checked: {integrity.get("checked")}')
    lines.append(f'- Changed: {len(integrity.get("changed", []))}')
    lines.append(f'- Missing: {len(integrity.get("missing", []))}')
    lines.append(f'- New untracked: {len(integrity.get("new_untracked", []))}')
    lines.append(f'- Overall: {"OK" if integrity.get("ok") else "VIOLATION"}')
    lines.append('')

    lines.append('## Deployment readiness')
    lines.append('')
    if deployment:
        lines.append(f'- Policy type: `{deployment.get("policy_type")}`')
        lines.append(f'- Selected architecture: arm `{deployment.get("selected_arm")}`')
        lines.append(f'- Ensemble members: {len(deployment.get("ensemble_members", []))}')
        lines.append(f'- Offset head used: {deployment.get("offset_head_used")}')
        lines.append(f'- Geometry stage: `{deployment.get("geometry_stage")}`')
    else:
        lines.append('- deployment_policy.json not yet generated')
    if submission_validation:
        lines.append(f'- Submission candidate valid: {submission_validation.get("valid")}')
        lines.append(f'- Real queries: {submission_validation.get("real_queries")}')
        lines.append(f'- Reported present: {submission_validation.get("reported_present")}')
    lines.append('')
    lines.append('**Nothing in this experiment was uploaded to Kaggle.** '
                 '`validation_inference.py` writes `submission_candidate.csv` only; '
                 'the existing production `outputs/joint_submission/submission.csv` '
                 'was never overwritten.')
    lines.append('')

    lines.append('## Corrections applied during this repair')
    lines.append('')
    lines.append('Every correction below was identified against the PRIOR '
                 '(superseded) implementation and report, and is recorded with its '
                 'cause and the files changed to fix it. The superseded artifacts '
                 'are preserved under `outputs/exp3_pairwise/superseded_20260913/`.')
    lines.append('')
    for c in corrections.get('corrections', []):
        lines.append(f'### {c["id"]}: {c["title"]}')
        lines.append('')
        lines.append(f'**Defect:** {c["defect"]}')
        lines.append('')
        lines.append(f'**Fix:** {c["fix"]}')
        lines.append('')
        if 'impact' in c:
            lines.append(f'**Impact:** {c["impact"]}')
            lines.append('')

    lines.append('## Failure analysis summary')
    lines.append('')
    for seed, doc in failure_analysis.items():
        lines.append(f'### Seed {seed}')
        lines.append('')
        lines.append('| Fold | Present F1 | Absent F1 | Figure loc | Off-figure loc | '
                     'Brier | Pool-missing rate |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|')
        for fold, a in doc.get('per_fold', {}).items():
            lines.append(f'| {fold} | {_fmt(a.get("present_class_f1"))} | '
                         f'{_fmt(a.get("absent_class_f1"))} | '
                         f'{_fmt(a.get("figure_localization"))} | '
                         f'{_fmt(a.get("offfigure_localization"))} | '
                         f'{_fmt(a.get("calibration_brier_score"))} | '
                         f'{_fmt(a.get("pool_missing_rate"))} |')
        lines.append('')

    lines.append('## Reproduction commands (every one of these actually runs)')
    lines.append('')
    lines.append('```bash')
    lines.append('# Baseline verification')
    lines.append('.venv-exp1/bin/python -m experiments.exp3_pairwise.baseline_verification')
    lines.append('')
    lines.append('# Overfit gate (fold-specific HardNet, corrected)')
    lines.append('OMP_NUM_THREADS=1 .venv-exp1/bin/python -c "')
    lines.append('from experiments.exp3_pairwise import overfit_check')
    lines.append("report = overfit_check.run(fold='pisces', arm='F', device='cpu')")
    lines.append('"')
    lines.append('')
    lines.append('# Training matrix, one seed')
    lines.append('OMP_NUM_THREADS=1 .venv-exp1/bin/python -c "')
    lines.append('from experiments.exp3_pairwise.matrix import run_all')
    lines.append("run_all(device='mps', seed=31004)")
    lines.append('"')
    lines.append('')
    lines.append('# Held-out evaluation')
    lines.append('.venv-exp1/bin/python -c "')
    lines.append('from experiments.exp3_pairwise.held_out_runner import run_all_folds')
    lines.append("run_all_folds(seed=31004, device='mps')")
    lines.append('"')
    lines.append('')
    lines.append('# Finalize (metrics, gates, integrity, tests, all machine records)')
    lines.append('.venv-exp1/bin/python -c "')
    lines.append('from experiments.exp3_pairwise.finalize import run')
    lines.append("run(device='mps')")
    lines.append('"')
    lines.append('')
    lines.append('# Completion audit (the ONLY authority for COMPLETE status)')
    lines.append('.venv-exp1/bin/python -m experiments.exp3_pairwise.completion_audit')
    lines.append('')
    lines.append('# Deployment policy + validation inference + submission candidate')
    lines.append('.venv-exp1/bin/python -c "')
    lines.append('from experiments.exp3_pairwise.deployment import build_deployment_policy')
    lines.append('build_deployment_policy()')
    lines.append('"')
    lines.append('.venv-exp1/bin/python -m experiments.exp3_pairwise.validation_inference')
    lines.append('')
    lines.append('# Regenerate this report')
    lines.append('.venv-exp1/bin/python -m experiments.exp3_pairwise.report')
    lines.append('```')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('*Generated by `experiments/exp3_pairwise/report.py` from machine '
                 'records only. See `outputs/exp3_pairwise/corrections.json` for the '
                 'full list of defects found and fixed during this repair.*')

    return '\n'.join(lines)


def write_report(path: str | None = None) -> str:
    text = generate()
    out = path or str(ROOT / 'EXPERIMENT3_REPORT.md')
    with open(out, 'w') as f:
        f.write(text)
    return out


if __name__ == '__main__':
    p = write_report()
    print(f'wrote {p}')

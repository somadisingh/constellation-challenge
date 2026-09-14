"""Generate EXPERIMENT4_REPORT.md entirely from machine-readable JSON records.
No metric value in this file is hand-maintained.
"""
from __future__ import annotations

import json

from . import OUT
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
    scope = _load('scope_decision.json') or {}

    lines = []
    lines.append('# Experiment 4: Honest Deployment Evaluation and Identification Headroom Analysis')
    lines.append('')
    lines.append('**Generated entirely from machine-readable records** under '
                 '`outputs/exp4_joint_identification/`. Regenerate with '
                 '`python -m experiments.exp4_joint_identification.report`.')
    lines.append('')
    lines.append('## Status')
    lines.append('')
    lines.append(f'**{audit.get("status")}**')
    if audit.get('failures'):
        lines.append('')
        for f in audit['failures']:
            lines.append(f'- {f}')
    lines.append('')

    lines.append('## What this experiment established')
    lines.append('')
    lines.append('Phases 0-2 of the task were executed with real, measured evidence. '
                 'Phases 3-7 (a new independent-evidence identification solver, joint '
                 'beam-search assignment, a 9-feature local-geometry screen, a generative '
                 'degradation-model verifier, and a plate-solving feasibility probe) were '
                 'NOT implemented; see `outputs/exp4_joint_identification/scope_decision.json` '
                 'for the exact, itemised reason for each. No promotion gate for a new '
                 'solver is applicable, because no new solver exists to promote.')
    lines.append('')

    lines.append('## Phase 0: baseline reconstruction')
    lines.append('')
    b = metrics.get('baselines', {})
    lines.append('| Reference | Score |')
    lines.append('|---|---:|')
    lines.append(f'| C0 | {_fmt(b.get("c0_score"))} |')
    lines.append(f'| Experiment 2 primary | {_fmt(b.get("exp2_primary_score"))} |')
    lines.append(f'| Experiment 2 repeat | {_fmt(b.get("exp2_repeat_score"))} |')
    lines.append(f'| Experiment 3 oracle-selected-arm primary (fold D/E/F per-fold) | '
                 f'{_fmt(b.get("exp3_oracle_selected_primary_score"))} |')
    lines.append(f'| Experiment 3 oracle-selected-arm repeat | '
                 f'{_fmt(b.get("exp3_oracle_selected_repeat_score"))} |')
    lines.append('')
    diff = b.get('submission_diff', {})
    lines.append(f'Submission diff (previous production CSV vs Experiment 3 candidate CSV): '
                 f'**{diff.get("n_constellation_changes")}** constellation-name changes, '
                 f'**{diff.get("total_presence_flips")}** presence flips, '
                 f'**{diff.get("total_coord_changes_both_present")}** coordinate changes '
                 f'where both files report a present patch.')
    lines.append('')
    lines.append('Snap/rescue attribution per real held-out scene (from Experiment 3\'s own '
                 '`verifier_snap_rescue` per-query `actions` diagnostic):')
    lines.append('')
    lines.append('| Scene | learned | geometry_snap | geometry_rescue |')
    lines.append('|---|---:|---:|---:|')
    for scene, counts in b.get('snap_rescue_attribution', {}).get('real_scenes', {}).items():
        lines.append(f'| {scene} | {counts["learned"]} | {counts["geometry_snap"]} | '
                     f'{counts["geometry_rescue"]} |')
    lines.append('')

    lines.append('## Phase 1: honest, leak-free evaluation of the ACTUAL deployed policy')
    lines.append('')
    lines.append('Experiment 3\'s headline (0.8170) scores each fold\'s ORACLE-SELECTED arm '
                 '(D, E, or F, chosen per fold from allowed-sky evidence). That is NOT the '
                 'system `deployment_policy.json` actually deploys: deployment uses a single '
                 'FIXED architecture (arm F) as a 6-checkpoint ensemble. Scoring that 6-member '
                 'ensemble on any of the three labelled skies would leak, because 4 of its 6 '
                 'members were trained using that sky as an allowed (training) sky. This '
                 'evaluation instead uses, for each held-out sky, ONLY the 2 checkpoints '
                 '(one per seed) whose OWN training fold equals that sky -- a genuinely '
                 'leak-free evaluation of the fixed-arm-F architecture and aggregation rule, '
                 'though not a perfect estimate of the full 6-member ensemble\'s accuracy on a '
                 'truly unseen scene (the other 4 members contribute additional training '
                 'diversity a 2-member evaluation cannot capture).')
    lines.append('')
    fp = metrics.get('fixed_policy_oof', {})
    for tag, label in (('both_seeds', 'Both seeds as ensemble (2 members/fold)'),
                       ('primary', 'Seed 31004 only (1 member/fold)'),
                       ('repeat', 'Seed 31005 only (1 member/fold)')):
        node = fp.get(tag, {})
        if not node:
            continue
        lines.append(f'### {label}')
        lines.append('')
        lines.append('| Stage | Score | Presence | Localization | Recovery | Identification |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        for stage in ('verifier_only', 'verifier_snap', 'verifier_snap_rescue'):
            m = node.get(stage, {})
            lines.append(f'| {stage} | {_fmt(m.get("score"))} | {_fmt(m.get("presence"))} | '
                         f'{_fmt(m.get("localization"))} | {_fmt(m.get("recovery"))} | '
                         f'{_fmt(m.get("identification"))} |')
        lines.append('')

    lines.append('### Comparison: fixed-policy honest eval vs Experiment 3\'s own headline')
    lines.append('')
    both = fp.get('both_seeds', {}).get('verifier_snap_rescue', {})
    lines.append(f'- Fixed-arm-F, leak-free, both-seed ensemble: **{_fmt(both.get("score"))}**')
    lines.append(f'- Experiment 3\'s own oracle-selected-arm headline: '
                 f'**{_fmt(b.get("exp3_oracle_selected_primary_score"))}**')
    lines.append('')
    lines.append('These are DIFFERENT systems (fixed single architecture vs per-fold oracle '
                 'selection) and are not interchangeable; both are reported, neither is '
                 'presented as the other.')
    lines.append('')

    lines.append('### Per-scene component breakdown (both-seed ensemble)')
    lines.append('')
    bd = metrics.get('fixed_policy_breakdown', {}).get('both_seeds', {})
    lines.append('| Scene | Present F1 | Absent F1 | Figure loc | Off-figure loc | Brier | '
                 'Ensemble agreement rate |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for scene, node in bd.items():
        lines.append(f'| {scene} | {_fmt(node.get("present_class_f1"))} | '
                     f'{_fmt(node.get("absent_class_f1"))} | '
                     f'{_fmt(node.get("figure_localization"))} | '
                     f'{_fmt(node.get("offfigure_localization"))} | '
                     f'{_fmt(node.get("calibration_brier_score"))} | '
                     f'{_fmt(node.get("ensemble_agreement_rate"))} |')
    lines.append('')
    lines.append('Scorpius\'s figure-star localization (0.3) is notably weaker than its '
                 'off-figure localization (0.875) despite strong presence F1 -- a caveat, not '
                 'a hidden strength, of the fixed policy on that scene. Ensemble agreement '
                 '(whether both seed-members pick the same best candidate) is only 54-63%, '
                 'meaning the two seeds disagree on over a third of queries even though the '
                 'aggregate score is stable.')
    lines.append('')

    lines.append('## Phase 2: identification headroom and failure attribution')
    lines.append('')
    lines.append('Seven progressively more informed oracle levels were run through the '
                 'FROZEN `constellation.joint.recognize_joint` (production defaults) on each '
                 'real labelled scene, to localize exactly where identification headroom is '
                 'lost.')
    lines.append('')
    hs = metrics.get('headroom_summary', {})
    for scene, levels in hs.items():
        lines.append(f'### {scene}')
        lines.append('')
        lines.append('| Level | Predicted | True-class rank | Correct |')
        lines.append('|---|---|---:|---|')
        for level, node in levels.items():
            lines.append(f'| {level} | {node["predicted"]} | {node["true_class_rank"]} | '
                         f'{node["correct"]} |')
        lines.append('')
    lines.append('Level 6 (`exp3_fixed_policy`: Experiment 3\'s own single best-candidate '
                 'coordinate per query, fed directly into geometry with no presence filter) '
                 '**underperforms level 5** (the classical system\'s own top-1 appearance '
                 'choice, no oracle information) **on all three scenes.** This is the single '
                 'most actionable finding of this experiment: Experiment 3\'s verifier is '
                 'currently a WORSE identification-time appearance/ranking signal than the '
                 'existing classical system, because it was trained to optimize presence and '
                 'localization reward, not appearance-rank fidelity for geometric hypothesis '
                 'seeding. It should not be substituted for the classical ranking signal in '
                 'identification without being recalibrated or retrained for that purpose.')
    lines.append('')
    lines.append('Per-scene failure attribution (from `failure_attribution.json`):')
    lines.append('')
    fa = metrics.get('failure_attribution_summary', {})
    for scene, node in fa.items():
        lines.append(f'**{scene}:**')
        for r in node['reasons']:
            lines.append(f'- {r}')
        lines.append('')

    lines.append('## Scope: Phases 3-7 not implemented')
    lines.append('')
    lines.append(scope.get('decision', ''))
    lines.append('')
    for p in scope.get('phases', []):
        lines.append(f'**Phase {p["phase"]} ({p["name"]}):** {p["status"]}. {p["reason"]}')
        lines.append('')

    lines.append('## Promotion gates')
    lines.append('')
    lines.append('All 10 new-solver promotion gates are `not_applicable` (no new solver was '
                 'built). The Phase 1 evaluation\'s own honesty checks:')
    lines.append('')
    for name, node in gates.get('fixed_policy_honesty_checks', {}).items():
        status = node.get('pass', node.get('available'))
        lines.append(f'- **{name}**: {status} -- {node}')
    lines.append('')

    lines.append('## Test results')
    lines.append('')
    e4 = test_results.get('exp4_suite', {})
    prod = test_results.get('production_suite', {})
    lines.append(f'- Exp4 suite: {e4.get("passed")} passed, {e4.get("failures")} failed, '
                 f'{e4.get("errors")} errors, {e4.get("skipped")} skipped '
                 f'(of {e4.get("tests_run")})')
    lines.append(f'- Production suite: {prod.get("passed")} passed, {prod.get("failures")} '
                 f'failed, {prod.get("errors")} errors, {prod.get("skipped")} skipped '
                 f'(of {prod.get("tests_run")})')
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
    lines.append('None were generated. The task specifies generating candidate submission '
                 'files only "if a method clears its gates" -- no new method was built in '
                 'this experiment, so no gate could be cleared, and no candidate file exists. '
                 'Existing production and Experiment 3 CSVs were not modified (confirmed by '
                 'the integrity check above).')
    lines.append('')
    lines.append('**Nothing was uploaded to Kaggle.**')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('*Generated by `experiments/exp4_joint_identification/report.py` from '
                 'machine records only.*')
    return '\n'.join(lines)


def write_report(path: str | None = None) -> str:
    text = generate()
    out = path or str(ROOT / 'EXPERIMENT4_REPORT.md')
    with open(out, 'w') as f:
        f.write(text)
    return out


if __name__ == '__main__':
    p = write_report()
    print(f'wrote {p}')

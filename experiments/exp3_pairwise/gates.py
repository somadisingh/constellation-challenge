"""Promotion gates against Experiment 2 (task §15).

The selected pairwise system passes only if ALL ten checks hold. Gates are never
weakened after seeing results.
"""
from __future__ import annotations

from . import SCENES

E2_PRIMARY_SCORE = 0.7619585865218471
E2_REPEAT_SCORE = 0.7512746188820096
MAX_SCENE_REGRESSION = 0.04


def evaluate_gates(exp3_primary: dict, exp3_repeat: dict, e2_primary_metrics: dict,
                   e2_repeat_metrics: dict, integrity_ok: bool,
                   completed_folds: list, both_seeds_evaluated: bool) -> dict:
    """`exp3_primary`/`exp3_repeat` are the SELECTED stage's mean metrics dict
    (out-of-fold across all three folds) for the best-performing integration
    stage. `e2_primary_metrics`/`e2_repeat_metrics` are Exp2's `snap_and_rescue_relocated`
    per-scene metrics for the SAME scenes, used for the per-scene regression check.
    """
    checks = {}
    checks['1_primary_exceeds_e2_by_0.01'] = {
        'value': exp3_primary['score'] - E2_PRIMARY_SCORE,
        'pass': exp3_primary['score'] >= E2_PRIMARY_SCORE + 0.01}
    checks['2_repeat_exceeds_e2_by_0.005'] = {
        'value': exp3_repeat['score'] - E2_REPEAT_SCORE,
        'pass': exp3_repeat['score'] >= E2_REPEAT_SCORE + 0.005}
    checks['3_presence_not_below_e2'] = {
        'exp3': exp3_primary['presence'], 'e2': e2_primary_metrics['mean']['presence'],
        'pass': exp3_primary['presence'] >= e2_primary_metrics['mean']['presence']}
    checks['4_localization_improves_0.01'] = {
        'value': exp3_primary['localization'] - e2_primary_metrics['mean']['localization'],
        'pass': (exp3_primary['localization']
                >= e2_primary_metrics['mean']['localization'] + 0.01)}
    checks['5_recovery_drop_at_most_0.02'] = {
        'value': e2_primary_metrics['mean']['recovery'] - exp3_primary['recovery'],
        'pass': (exp3_primary['recovery']
                >= e2_primary_metrics['mean']['recovery'] - 0.02)}
    checks['6_identification_unchanged'] = {
        'exp3': exp3_primary['identification'],
        'e2': e2_primary_metrics['mean']['identification'],
        'pass': abs(exp3_primary['identification']
                   - e2_primary_metrics['mean']['identification']) < 1e-9}
    scene_reg = {}
    for s in SCENES:
        d = exp3_primary.get('per_scene', {}).get(s, {}).get('score')
        e = e2_primary_metrics['scenes'][s]['score']
        scene_reg[s] = (None if d is None else d - e)
    checks['7_no_scene_regression_0.04'] = {
        'per_scene_delta': scene_reg,
        'pass': all(v is None or v >= -MAX_SCENE_REGRESSION for v in scene_reg.values())}
    checks['8_integrity_ok'] = {'pass': bool(integrity_ok)}
    checks['9_all_folds_complete'] = {
        'completed': completed_folds, 'expected': list(SCENES),
        'pass': set(completed_folds) == set(SCENES)}
    checks['10_both_seeds_evaluated'] = {'pass': bool(both_seeds_evaluated)}

    overall = all(c['pass'] for c in checks.values())
    return {'checks': checks, 'pass': overall,
           'e2_primary_reference': E2_PRIMARY_SCORE, 'e2_repeat_reference': E2_REPEAT_SCORE}

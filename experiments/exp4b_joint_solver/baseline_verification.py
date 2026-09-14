"""Phase 0 (continued): reproduce every number the task's "Confirmed starting
evidence" section asserts, before using any of them.
"""
from __future__ import annotations

import json

import numpy as np

from . import ROOT, SCENES
from experiments.exp1.env import write_json


def verify_fixed_arm_f_oof() -> dict:
    doc = json.load(open(ROOT / 'outputs' / 'exp4_joint_identification' /
                        'fixed_policy_oof_both_seeds.json'))
    per_scene = {}
    for fold, fdata in doc.items():
        held = fdata['held_out']
        per_scene[held] = fdata['stages']['verifier_snap_rescue']['held_out_metrics']
    keys = ('presence', 'localization', 'recovery', 'identification', 'score')
    mean = {k: float(np.mean([per_scene[s][k] for s in per_scene])) for k in keys}
    expected = {'score': 0.8208, 'presence': 0.885, 'localization': 0.748,
               'recovery': 1.000, 'identification': 0.667}
    ok = all(abs(mean[k] - expected[k]) < 0.001 for k in expected)
    return {'expected_approx': expected, 'actual': mean, 'ok': ok}


def verify_headroom_ranks() -> dict:
    doc = json.load(open(ROOT / 'outputs' / 'exp4_joint_identification' /
                        'headroom_oracles.json'))
    expected = {
        'pisces':   {'full_slate_rank': 7,  'oracle_best_candidate_correct': True,
                     'exp3_rank': 24, 'classical_rank': 5},
        'scorpius': {'full_slate_rank': 1,  'oracle_best_candidate_correct': True,
                     'exp3_rank': 5,  'classical_rank': 12},
        'taurus':   {'full_slate_rank': 16, 'oracle_best_candidate_correct': False,
                     'exp3_rank': 13, 'classical_rank': 11},
    }
    actual = {}
    ok = True
    for scene in SCENES:
        levels = doc[scene]['levels']
        a = {
            'full_slate_rank': levels['7_full_slate']['diagnostics_summary']['true_class_rank'],
            'oracle_best_candidate_correct': levels['4_oracle_best_candidate']['diagnostics_summary']['correct_class_wins'],
            'exp3_rank': levels['6_exp3_fixed_policy']['diagnostics_summary']['true_class_rank'],
            'classical_rank': levels['5_c0_rank1']['diagnostics_summary']['true_class_rank'],
        }
        actual[scene] = a
        if a != expected[scene]:
            ok = False
    return {'expected': expected, 'actual': actual, 'ok': ok}


def verify_taurus_pool_missing() -> dict:
    doc = json.load(open(ROOT / 'outputs' / 'exp4_joint_identification' /
                        'failure_attribution.json'))
    taurus = doc['taurus']
    ok = (taurus['retrieval_failure'] and taurus['insufficient_figure_coverage']
         and taurus['winner_clutter_signature'] is not None)
    return {'ok': ok, 'actual': taurus}


def run(say=print) -> dict:
    say('=== Phase 0: verifying task-prompt starting evidence ===')
    fp = verify_fixed_arm_f_oof()
    say(f"fixed-arm-F leak-free OOF: {'OK' if fp['ok'] else 'MISMATCH'} {fp['actual']}")
    hr = verify_headroom_ranks()
    say(f"headroom ranks: {'OK' if hr['ok'] else 'MISMATCH'} {hr['actual']}")
    tp = verify_taurus_pool_missing()
    say(f"taurus pool-missing/clutter attribution: {'OK' if tp['ok'] else 'MISMATCH'}")

    all_ok = fp['ok'] and hr['ok'] and tp['ok']
    say('ALL TASK-PROMPT STARTING EVIDENCE VERIFIED.' if all_ok
       else 'STARTING EVIDENCE VERIFICATION FAILED.')
    return {'ok': all_ok, 'fixed_arm_f_oof': fp, 'headroom_ranks': hr,
           'taurus_pool_missing': tp}


if __name__ == '__main__':
    from . import OUT
    result = run()
    write_json(OUT / 'baseline_verification.json', result)
    raise SystemExit(0 if result['ok'] else 1)

"""Phase 0: reproduce and record every starting number Experiment 4C's own
task prompt asserts, before using any of them. Fails immediately (returns
`ok=False`) if a required baseline does not reproduce exactly from the real,
already-written JSON records -- no number here is retyped by hand.
"""
from __future__ import annotations

import json

import numpy as np

from . import OUT, ROOT, SCENES
from experiments.exp1.env import sha256_file, write_json


def verify_exp3_fixed_policy_oof() -> dict:
    """Exp3's corrected fixed-policy OOF (reused verbatim from Exp4/Exp4B's own
    verification of the same file)."""
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
    return {'expected_approx': expected, 'actual': mean, 'per_scene': per_scene, 'ok': ok}


def verify_exp4b_rank_fidelity() -> dict:
    doc = json.load(open(ROOT / 'outputs' / 'exp4b_joint_solver' / 'rank_fidelity_ablation.json'))
    means = doc['selection']['means']
    expected = {'1_classical_rank': 0.7863247863247863, '2_exp3_rank': 0.816951566951567,
               '5_linear_score_fusion': 0.8266856600189932}
    ok = all(abs(means[k] - v) < 1e-6 for k, v in expected.items())
    return {'expected': expected, 'actual': {k: means[k] for k in expected},
           'selected_rule': doc['selection']['selected'], 'ok': ok}


def verify_exp4b_independent_support() -> dict:
    doc = json.load(open(ROOT / 'outputs' / 'exp4b_joint_solver' / 'independent_support.json'))
    expected = {
        'pisces':   {'existing_score': 15, 'held_out_stability': 8},
        'scorpius': {'existing_score': 1,  'held_out_stability': 1},
        'taurus':   {'existing_score': 18, 'held_out_stability': 11},
    }
    actual = {s: {k: doc[s][k]['true_class_rank'] for k in ('existing_score', 'held_out_stability')}
             for s in SCENES}
    ok = actual == expected
    return {'expected': expected, 'actual': actual, 'ok': ok}


def verify_exp4b_unqueried_star() -> dict:
    doc = json.load(open(ROOT / 'outputs' / 'exp4b_joint_solver' / 'null_model.json'))
    r = doc['results']
    expected = {
        'pisces':   {'none': 15, 'raw_count': 4, 'matched_null_lr_corrected': 6},
        'scorpius': {'none': 1,  'raw_count': 11, 'matched_null_lr_corrected': 1},
        'taurus':   {'none': 18, 'raw_count': 15, 'matched_null_lr_corrected': 18},
    }
    actual = {s: {k: r[s][k]['true_class_rank'] for k in
                 ('none', 'raw_count', 'matched_null_lr_corrected')} for s in SCENES}
    ok = actual == expected
    return {'expected': expected, 'actual': actual, 'ok': ok}


def verify_exp4b_joint_solver_failures() -> dict:
    out = {}
    ok = True
    for tag in ('primary', 'repeat'):
        doc = json.load(open(ROOT / 'outputs' / 'exp4b_joint_solver' / f'joint_solver_{tag}.json'))
        scorpius_correct = doc['scorpius']['10_complete_joint_solver']['correct']
        out[tag] = {'scorpius_complete_joint_solver_correct': scorpius_correct}
        if scorpius_correct is not False:
            ok = False
    return {'expected': {'primary': {'scorpius_complete_joint_solver_correct': False},
                        'repeat': {'scorpius_complete_joint_solver_correct': False}},
           'actual': out, 'ok': ok}


def verify_exp4b_gate_correction() -> dict:
    doc = json.load(open(OUT / 'prior_gate_corrections.json'))
    expected = {'n_positive': 7, 'n_failed': 6, 'n_diagnostic_only': 1}
    actual = {k: doc['recount'][k] for k in expected}
    return {'expected': expected, 'actual': actual, 'ok': actual == expected}


def verify_current_production_predictions() -> dict:
    """Current production/Exp3 constellation predictions per scene, read from
    the frozen classical `outputs/joint_train/{scene}.json` records (the same
    source Exp4B's `joint_solver_comparisons.py` uses for its own production
    baseline)."""
    from constellation.contracts import read_truth
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    out = {}
    for scene in SCENES:
        doc = json.load(open(ROOT / 'outputs' / 'joint_train' / f'{scene}.json'))
        out[scene] = {'predicted': doc['constellation'],
                     'true': truth[scene].constellation,
                     'correct': doc['constellation'] == truth[scene].constellation}
    return out


def verify_exp3_candidate_csv_hash() -> dict:
    path = ROOT / 'outputs' / 'exp3_pairwise' / 'submission_candidate.csv'
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path),
           'exists': path.exists()}


def run(say=print) -> dict:
    say('=== Phase 0: verifying Exp4C task-prompt starting evidence ===')
    exp3 = verify_exp3_fixed_policy_oof()
    say(f"Exp3 fixed-policy OOF: {'OK' if exp3['ok'] else 'MISMATCH'} {exp3['actual']}")
    rf = verify_exp4b_rank_fidelity()
    say(f"Exp4B rank fidelity: {'OK' if rf['ok'] else 'MISMATCH'} selected={rf['selected_rule']}")
    isup = verify_exp4b_independent_support()
    say(f"Exp4B independent support: {'OK' if isup['ok'] else 'MISMATCH'} {isup['actual']}")
    us = verify_exp4b_unqueried_star()
    say(f"Exp4B unqueried-star: {'OK' if us['ok'] else 'MISMATCH'} {us['actual']}")
    js = verify_exp4b_joint_solver_failures()
    say(f"Exp4B joint-solver failures: {'OK' if js['ok'] else 'MISMATCH'} {js['actual']}")
    gc = verify_exp4b_gate_correction()
    say(f"Exp4B gate correction (this experiment's Task #1): "
       f"{'OK' if gc['ok'] else 'MISMATCH'} {gc['actual']}")
    prod = verify_current_production_predictions()
    say(f"Current production predictions: {prod}")
    csv_hash = verify_exp3_candidate_csv_hash()
    say(f"Exp3 candidate CSV: {csv_hash}")

    all_ok = exp3['ok'] and rf['ok'] and isup['ok'] and us['ok'] and js['ok'] and gc['ok']
    say('ALL REQUIRED BASELINES VERIFIED.' if all_ok else 'BASELINE VERIFICATION FAILED.')
    return {
        'ok': all_ok,
        'exp3_fixed_policy_oof': exp3,
        'exp4b_rank_fidelity': rf,
        'exp4b_independent_support': isup,
        'exp4b_unqueried_star': us,
        'exp4b_joint_solver_failures': js,
        'exp4b_gate_correction': gc,
        'current_production_predictions': prod,
        'exp3_candidate_csv': csv_hash,
    }


if __name__ == '__main__':
    result = run()
    write_json(OUT / 'baseline_verification.json', result)
    raise SystemExit(0 if result['ok'] else 1)

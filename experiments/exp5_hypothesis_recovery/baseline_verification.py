"""Phase 0: reproduce every starting number Experiment 5's own task prompt
asserts about Experiment 4C, before using any of them as motivation.
"""
from __future__ import annotations

import json

from . import OUT, ROOT, SCENES
from experiments.exp1.env import sha256_file, write_json


def verify_exp4c_hypothesis_counts() -> dict:
    expected = {'pisces': 1, 'scorpius': 9, 'taurus': 0}
    actual = {}
    ok = True
    for scene in SCENES:
        doc = json.load(open(ROOT / 'outputs' / 'exp4c_calibrated_fusion' /
                            f'hypotheses_{scene}_s31004.json'))
        n_pos = sum(1 for r in doc['records'] if r['placement_correct'])
        actual[scene] = n_pos
        if n_pos != expected[scene]:
            ok = False
    return {'expected': expected, 'actual': actual, 'ok': ok}


def verify_exp4c_gates() -> dict:
    doc = json.load(open(ROOT / 'outputs' / 'exp4c_calibrated_fusion' / 'gates.json'))
    ok = doc['n_pass'] == 12 and doc['n_gates'] == 20 and doc['overall_pass'] is False
    return {'expected': {'n_pass': 12, 'n_gates': 20, 'overall_pass': False},
           'actual': {'n_pass': doc['n_pass'], 'n_gates': doc['n_gates'],
                     'overall_pass': doc['overall_pass']}, 'ok': ok}


def verify_exp4c_raw_winner() -> dict:
    """The complete calibrated model's raw (unguarded) winner on all three
    held-out skies is claimed to be 'eridanus' -- verify against the real
    per-seed OOF record."""
    ok = True
    actual = {}
    for tag, seed in (('primary', 31004), ('repeat', 31005)):
        doc = json.load(open(ROOT / 'outputs' / 'exp4c_calibrated_fusion' /
                            f'oof_{tag}.json'))
        winners = {s: doc['per_scene'][s]['raw']['winner'] for s in SCENES}
        actual[tag] = winners
        if not all(w == 'eridanus' for w in winners.values()):
            ok = False
    return {'expected': 'eridanus for every scene, both seeds', 'actual': actual, 'ok': ok}


def verify_exp4c_completion_audit() -> dict:
    doc = json.load(open(ROOT / 'outputs' / 'exp4c_calibrated_fusion' / 'completion_audit.json'))
    ok = (doc['status'] == 'COMPLETE' and doc['implementation_complete'] is True
         and doc['performance_gates_passed'] is False)
    return {'expected': {'status': 'COMPLETE', 'implementation_complete': True,
                        'performance_gates_passed': False},
           'actual': {'status': doc['status'],
                     'implementation_complete': doc['implementation_complete'],
                     'performance_gates_passed': doc['performance_gates_passed']},
           'ok': ok}


def verify_current_production_predictions() -> dict:
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
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path), 'exists': path.exists()}


def run(say=print) -> dict:
    say('=== Phase 0: verifying Experiment 5 task-prompt starting evidence ===')
    hc = verify_exp4c_hypothesis_counts()
    say(f"Exp4C hypothesis counts: {'OK' if hc['ok'] else 'MISMATCH'} {hc['actual']}")
    g = verify_exp4c_gates()
    say(f"Exp4C gates: {'OK' if g['ok'] else 'MISMATCH'} {g['actual']}")
    w = verify_exp4c_raw_winner()
    say(f"Exp4C raw winner=eridanus: {'OK' if w['ok'] else 'MISMATCH'} {w['actual']}")
    c = verify_exp4c_completion_audit()
    say(f"Exp4C completion audit: {'OK' if c['ok'] else 'MISMATCH'} {c['actual']}")
    prod = verify_current_production_predictions()
    say(f"Current production predictions: {prod}")
    csv_hash = verify_exp3_candidate_csv_hash()
    say(f"Exp3 candidate CSV: {csv_hash}")

    all_ok = hc['ok'] and g['ok'] and w['ok'] and c['ok']
    say('ALL REQUIRED BASELINES VERIFIED.' if all_ok else 'BASELINE VERIFICATION FAILED.')
    return {
        'ok': all_ok,
        'exp4c_hypothesis_counts': hc,
        'exp4c_gates': g,
        'exp4c_raw_winner': w,
        'exp4c_completion_audit': c,
        'current_production_predictions': prod,
        'exp3_candidate_csv': csv_hash,
    }


if __name__ == '__main__':
    result = run()
    write_json(OUT / 'baseline_verification.json', result)
    raise SystemExit(0 if result['ok'] else 1)

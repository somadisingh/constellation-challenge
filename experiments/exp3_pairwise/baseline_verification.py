"""Reproduce and verify the frozen C0/Exp1B/Exp2 baseline numbers (repair task §3.6).

This module was referenced by EXPERIMENT3_REPORT.md's Appendix B but did not
exist; `outputs/exp3_pairwise/baseline_verification.log` was produced by an
ad-hoc script that was never committed. This is the reconstructed, committed
version, producing byte-comparable output.

Run:
    python -m experiments.exp3_pairwise.baseline_verification
"""
from __future__ import annotations

import json

from experiments.exp1.env import ROOT


def _score(m: dict) -> dict:
    return {'presence': m['presence'], 'localization': m['localization'],
           'recovery': m['recovery'], 'identification': m['identification'],
           'score': m['score']}


def verify_c0() -> dict:
    doc = json.load(open(ROOT / 'outputs' / 'joint_train' / 'metrics.json'))
    m = doc['mean']
    expected = {'presence': 0.717315544749591, 'localization': 0.6495726495726496,
               'recovery': 0.8777777777777778, 'identification': 0.6666666666666666,
               'score': 0.7286878605463721}
    actual = _score(m)
    ok = all(abs(actual[k] - expected[k]) < 1e-9 for k in expected)
    return {'expected': expected, 'actual': actual, 'ok': ok}


def verify_exp1b() -> dict:
    frozen = json.load(open(ROOT / 'outputs' / 'exp1b' / 'selection_frozen.json'))
    metrics = json.load(open(ROOT / 'outputs' / 'exp1b' / 'metrics.json'))
    m = metrics['oof']['SELECTED']['mean']
    expected = {'presence': 0.9064134829015783, 'localization': 0.7241215574548908,
               'recovery': 0.6, 'identification': 0.6666666666666666,
               'score': 0.7214276822163727}
    actual = _score(m)
    ok = all(abs(actual[k] - expected[k]) < 1e-9 for k in expected)
    gate_pass = actual['score'] >= expected['score'] + 0.01   # Exp1B's own gain gate
    return {'expected': expected, 'actual': actual, 'ok': ok,
           'gate_pass': gate_pass, 'selection_frozen': frozen}


def verify_exp2(seed_tag: str) -> dict:
    doc = json.load(open(ROOT / 'outputs' / 'exp2_geometry' / 'record.json'))
    node = doc[seed_tag]['rules']['snap_and_rescue_relocated']['metrics']['mean']
    expected = ({'presence': 0.865213263466306, 'localization': 0.6866096866096866,
                'recovery': 0.8333333333333334, 'identification': 0.6666666666666666,
                'score': 0.7619585865218471} if seed_tag == 'primary' else
               {'presence': 0.873094676857573, 'localization': 0.6927825261158596,
                'recovery': 0.7777777777777778, 'identification': 0.6666666666666666,
                'score': 0.7512746188820096})
    actual = _score(node)
    ok = all(abs(actual[k] - expected[k]) < 1e-9 for k in expected)
    return {'expected': expected, 'actual': actual, 'ok': ok}


def run(say=print) -> dict:
    c0 = verify_c0()
    say(f"C0 CONFIRMED: {c0['actual']}" if c0['ok'] else f"C0 MISMATCH: {c0}")
    exp1b = verify_exp1b()
    say(f"Exp1B B/SELECTED CONFIRMED: {exp1b['actual']}" if exp1b['ok']
       else f"Exp1B MISMATCH: {exp1b}")
    say(f"Exp1B gate pass: {exp1b['gate_pass']}")
    exp2p = verify_exp2('primary')
    say(f"Exp2 PRIMARY CONFIRMED: {exp2p['actual']}" if exp2p['ok']
       else f"Exp2 PRIMARY MISMATCH: {exp2p}")
    exp2r = verify_exp2('repeat')
    say(f"Exp2 REPEAT CONFIRMED: {exp2r['actual']}" if exp2r['ok']
       else f"Exp2 REPEAT MISMATCH: {exp2r}")
    all_ok = c0['ok'] and exp1b['ok'] and exp2p['ok'] and exp2r['ok']
    say('ALL TASK-PROMPT BASELINE NUMBERS VERIFIED EXACTLY.' if all_ok
       else 'BASELINE VERIFICATION FAILED -- DO NOT PROCEED.')
    return {'ok': all_ok, 'c0': c0, 'exp1b': exp1b, 'exp2_primary': exp2p,
           'exp2_repeat': exp2r}


if __name__ == '__main__':
    result = run()
    raise SystemExit(0 if result['ok'] else 1)

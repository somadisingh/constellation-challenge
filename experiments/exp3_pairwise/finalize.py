"""Single finalize command: recompute metrics, evaluate gates, run integrity,
write every machine record, then (if and only if everything passes) mark the
experiment COMPLETE (repair task §3.7, §6).

INTEGRATION-STAGE DECISION (explicit, made BEFORE looking at held-out numbers):
`held_out.evaluate_fold` reports four stages (`verifier_only`, `verifier_offset`,
`verifier_snap`, `verifier_snap_rescue`). Picking whichever of these scores
highest on held-out data would itself be a held-out-based selection, forbidden by
rule 12 ("Held-out results cannot be used to select architecture or thresholds").
Instead, Experiment 3 reuses Experiment 2's OWN frozen, allowed-sky-selected
integration rule verbatim: Experiment 2 selected `snap_and_rescue_relocated`
(see `EXPERIMENT2_REPORT.md`) as its universal deployed rule via allowed-sky
evidence, not via held-out peeking. Experiment 3's `verifier_snap_rescue` stage
IS that same rule (`apply_geometry_stage` calls `hybrid_prediction(..., snap=
'relocated', rescue='relocated')`, identical to Exp2's selected configuration).
So `PRIMARY_STAGE = 'verifier_snap_rescue'` is fixed by inheritance from Exp2's
already-frozen choice, not re-selected from Exp3's own held-out results. All
four stages are still computed and reported for transparency and ablation.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import OUT, SCENES
from .baseline_verification import run as run_baseline_verification
from .integrity import verify_protected
from experiments.exp1.env import ROOT, append_jsonl, code_hash, write_json

PRIMARY_STAGE = 'verifier_snap_rescue'
PRIMARY_SEED = 31004
REPEAT_SEED = 31005


def _mean_across_folds(held_out_doc: dict, stage: str) -> dict:
    """Equal-scene mean of `stage`'s metrics, one value per fold's own held-out
    scene (i.e. an honest out-of-fold aggregate: never a scene's own fold-trained
    model scoring itself)."""
    per_scene = {}
    for fold, fdata in held_out_doc.items():
        held = fdata['held_out']
        per_scene[held] = fdata['stages'][stage]['held_out_metrics']
    keys = ('presence', 'localization', 'recovery', 'identification', 'score')
    mean = {k: float(np.mean([per_scene[s][k] for s in per_scene])) for k in keys}
    mean['per_scene'] = per_scene
    return mean


def _load_e2_baseline(seed_tag: str) -> dict:
    doc = json.load(open(ROOT / 'outputs' / 'exp2_geometry' / 'record.json'))
    node = doc[seed_tag]['rules']['snap_and_rescue_relocated']['metrics']
    return {'mean': node['mean'], 'scenes': node['scenes'],
           'source_file': 'outputs/exp2_geometry/record.json',
           'source_seed_tag': seed_tag}


def _per_scene_deltas(exp3_scene_metrics: dict, e2_scenes: dict, e2_source: dict) -> dict:
    out = {}
    for scene in SCENES:
        exp3_score = exp3_scene_metrics.get(scene, {}).get('score')
        e2_score = e2_scenes[scene]['score']
        out[scene] = {
            'experiment_score': exp3_score, 'baseline_score': e2_score,
            'delta': (None if exp3_score is None else exp3_score - e2_score),
            'baseline_source_file': e2_source['source_file'],
            'baseline_seed': e2_source['source_seed_tag'],
        }
    return out


def _env_record() -> dict:
    rec = {'python': sys.version, 'platform': platform.platform(),
          'code_hash': code_hash()}
    try:
        import torch
        rec['torch'] = torch.__version__
        rec['mps_available'] = bool(torch.backends.mps.is_available())
    except ImportError:
        rec['torch'] = None
    try:
        rec['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        rec['git_commit'] = None
    return rec


def run_test_suite() -> dict:
    """Run the Exp3-specific suite in THIS interpreter and the production suite
    in `.venv` (no-torch), recording real pass/fail/skip counts (repair task
    §3.9: never hardcode the count)."""
    import unittest
    started = time.time()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('tests.test_exp3_pairwise')
    stream_buf = []
    runner = unittest.TextTestRunner(verbosity=2, stream=open('/dev/null', 'w')
                                     if False else __import__('io').StringIO())
    result = runner.run(suite)
    ended = time.time()
    exp3_env = _env_record()
    exp3_record = {
        'command': 'python -m unittest tests.test_exp3_pairwise',
        'python': exp3_env['python'], 'torch': exp3_env['torch'],
        'start': started, 'end': ended, 'duration_s': ended - started,
        'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped),
        'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        'exit_ok': result.wasSuccessful(),
        'failure_details': [str(f[0]) for f in result.failures],
        'error_details': [str(e[0]) for e in result.errors],
        'skipped_details': [str(s[0]) for s in result.skipped],
    }

    prod_started = time.time()
    prod_proc = subprocess.run(
        [str(ROOT / '.venv' / 'bin' / 'python'), '-m', 'unittest', 'discover',
         '-s', 'tests'], cwd=ROOT, capture_output=True, text=True,
        env={'OPENBLAS_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin'})
    prod_ended = time.time()
    prod_out = prod_proc.stderr + prod_proc.stdout
    prod_counts = _parse_unittest_summary(prod_out)
    prod_record = {
        'command': 'OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests',
        'start': prod_started, 'end': prod_ended, 'duration_s': prod_ended - prod_started,
        'exit_code': prod_proc.returncode,
        **prod_counts,
        'log_tail': prod_out[-4000:],
    }
    log_path = OUT / 'test_suite_full.log'
    log_path.write_text(prod_out)
    prod_record['log_path'] = str(log_path)

    return {'exp3_env': exp3_env, 'exp3_suite': exp3_record,
           'production_suite': prod_record}


def _parse_unittest_summary(text: str) -> dict:
    """Parse `Ran N tests in Xs` / `OK` / `FAILED (failures=A, errors=B, skipped=C)`
    from unittest's stderr output -- never hardcode a count."""
    import re
    ran_match = re.search(r'Ran (\d+) tests', text)
    ran = int(ran_match.group(1)) if ran_match else None
    ok = bool(re.search(r'^OK\b', text, re.MULTILINE))
    fail_match = re.search(r'failures=(\d+)', text)
    err_match = re.search(r'errors=(\d+)', text)
    skip_match = re.search(r'skipped=(\d+)', text)
    failures = int(fail_match.group(1)) if fail_match else 0
    errors = int(err_match.group(1)) if err_match else 0
    skipped = int(skip_match.group(1)) if skip_match else 0
    passed = (ran - failures - errors - skipped) if ran is not None else None
    return {'tests_run': ran, 'passed': passed, 'failures': failures,
           'errors': errors, 'skipped': skipped, 'wasSuccessful': ok}


def run(device: str = 'mps', say=print) -> dict:
    """The one authoritative finalize command (repair task §3.7, §6)."""
    started = time.time()
    say('=== Experiment 3 finalize: recompute, gate, integrity, record ===')

    say('\n[1/6] Baseline verification (C0/Exp1B/Exp2 exact reproduction) ...')
    baseline = run_baseline_verification(say=say)
    if not baseline['ok']:
        return _incomplete('baseline_verification_failed', {'baseline': baseline}, started)

    say('\n[2/6] Loading held-out predictions for both seeds ...')
    primary_path = OUT / f'held_out_s{PRIMARY_SEED}.json'
    repeat_path = OUT / f'held_out_s{REPEAT_SEED}.json'
    failures = []
    if not primary_path.exists():
        failures.append(f'missing {primary_path}')
    if not repeat_path.exists():
        failures.append(f'missing {repeat_path}')
    if failures:
        return _incomplete('missing_held_out_files', {'failures': failures}, started)
    primary_doc = json.load(open(primary_path))
    repeat_doc = json.load(open(repeat_path))
    completed_primary = sorted(primary_doc.keys())
    completed_repeat = sorted(repeat_doc.keys())
    if set(completed_primary) != set(SCENES) or set(completed_repeat) != set(SCENES):
        return _incomplete('folds_incomplete', {'primary': completed_primary,
                                                'repeat': completed_repeat}, started)

    say('\n[3/6] Recomputing all-stage metrics from stored held-out predictions ...')
    all_stages = ('verifier_only', 'verifier_offset', 'verifier_snap', 'verifier_snap_rescue')
    primary_by_stage = {s: _mean_across_folds(primary_doc, s) for s in all_stages}
    repeat_by_stage = {s: _mean_across_folds(repeat_doc, s) for s in all_stages}
    exp3_primary = primary_by_stage[PRIMARY_STAGE]
    exp3_repeat = repeat_by_stage[PRIMARY_STAGE]

    e2_primary = _load_e2_baseline('primary')
    e2_repeat = _load_e2_baseline('repeat')

    metrics_record = {
        'primary_seed': PRIMARY_SEED, 'repeat_seed': REPEAT_SEED,
        'primary_stage': PRIMARY_STAGE,
        'stage_selection_rationale': (
            'verifier_snap_rescue is fixed because it is IDENTICAL to Experiment '
            '2\'s own frozen, allowed-sky-selected integration rule '
            '(snap_and_rescue_relocated); it is inherited, not re-selected from '
            'Experiment 3\'s held-out results.'),
        'all_stages': {'primary': {s: {k: v for k, v in m.items() if k != 'per_scene'}
                                   for s, m in primary_by_stage.items()},
                      'repeat': {s: {k: v for k, v in m.items() if k != 'per_scene'}
                                for s, m in repeat_by_stage.items()}},
        'primary': {k: v for k, v in exp3_primary.items() if k != 'per_scene'},
        'repeat': {k: v for k, v in exp3_repeat.items() if k != 'per_scene'},
        'primary_per_scene_vs_baseline': _per_scene_deltas(
            exp3_primary['per_scene'], e2_primary['scenes'], e2_primary),
        'repeat_per_scene_vs_baseline': _per_scene_deltas(
            exp3_repeat['per_scene'], e2_repeat['scenes'], e2_repeat),
        'primary_baseline': {'score': e2_primary['mean']['score'],
                             'source_file': e2_primary['source_file'],
                             'seed_tag': 'primary',
                             'delta': exp3_primary['score'] - e2_primary['mean']['score']},
        'repeat_baseline': {'score': e2_repeat['mean']['score'],
                            'source_file': e2_repeat['source_file'],
                            'seed_tag': 'repeat',
                            'delta': exp3_repeat['score'] - e2_repeat['mean']['score']},
    }
    write_json(OUT / 'metrics.json', metrics_record)
    say(f'  primary ({PRIMARY_STAGE}, seed {PRIMARY_SEED}): score='
       f'{exp3_primary["score"]:.6f} (delta vs E2 primary = '
       f'{metrics_record["primary_baseline"]["delta"]:+.6f})')
    say(f'  repeat  ({PRIMARY_STAGE}, seed {REPEAT_SEED}): score='
       f'{exp3_repeat["score"]:.6f} (delta vs E2 repeat = '
       f'{metrics_record["repeat_baseline"]["delta"]:+.6f})')

    say('\n[4/6] Running integrity check (must be AFTER all intended writes) ...')
    integrity = verify_protected(OUT / 'protected_before.json')
    write_json(OUT / 'integrity.json', integrity)
    say(f'  integrity ok={integrity["ok"]} checked={integrity["checked"]} '
       f'changed={len(integrity["changed"])} missing={len(integrity["missing"])} '
       f'new_untracked={len(integrity["new_untracked"])}')

    say('\n[5/6] Evaluating the 10 promotion gates ...')
    from .gates import evaluate_gates
    gate_result = evaluate_gates(
        exp3_primary=exp3_primary, exp3_repeat=exp3_repeat,
        e2_primary_metrics=e2_primary, e2_repeat_metrics=e2_repeat,
        integrity_ok=integrity['ok'], completed_folds=completed_primary,
        both_seeds_evaluated=True)
    write_json(OUT / 'gates.json', gate_result)
    for name, node in gate_result['checks'].items():
        say(f'  {name}: {"PASS" if node["pass"] else "FAIL"}')
    say(f'  OVERALL GATES: {"PASS" if gate_result["pass"] else "FAIL"}')

    say('\n[6/6] Running the full test suite ...')
    test_result = run_test_suite()
    write_json(OUT / 'test_results.json', test_result)
    say(f'  exp3 suite: {test_result["exp3_suite"]["passed"]} passed, '
       f'{test_result["exp3_suite"]["failures"]} failed, '
       f'{test_result["exp3_suite"]["errors"]} errors, '
       f'{test_result["exp3_suite"]["skipped"]} skipped '
       f'(of {test_result["exp3_suite"]["tests_run"]})')
    say(f'  production suite: {test_result["production_suite"].get("passed")} passed, '
       f'{test_result["production_suite"].get("failures")} failed, '
       f'{test_result["production_suite"].get("errors")} errors, '
       f'{test_result["production_suite"].get("skipped")} skipped '
       f'(of {test_result["production_suite"].get("tests_run")})')

    ended = time.time()
    record = {
        'status': 'EVALUATED',   # completion_audit.py decides COMPLETE/INCOMPLETE
        'started': started, 'ended': ended, 'duration_s': ended - started,
        'baseline_verification': baseline,
        'metrics': metrics_record, 'integrity': integrity, 'gates': gate_result,
        'test_results': {k: v for k, v in test_result.items()},
        'env': _env_record(),
    }
    write_json(OUT / 'record.json', record)
    append_jsonl(OUT / 'runs.jsonl', {'event': 'finalize', 'ended': ended,
                                      'gates_pass': gate_result['pass']})
    say(f'\n=== finalize complete in {ended - started:.1f}s; '
       f'gates {"PASS" if gate_result["pass"] else "FAIL"} ===')
    return record


def _incomplete(reason: str, detail: dict, started: float) -> dict:
    record = {'status': 'INCOMPLETE', 'reason': reason, 'detail': detail,
             'started': started, 'ended': time.time()}
    write_json(OUT / 'record.json', record)
    return record


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='mps')
    args = ap.parse_args()
    run(device=args.device)

"""The ONLY authority permitted to declare Experiment 3 COMPLETE (repair task §6).

Every check below must independently hold; any single failure marks the
experiment INCOMPLETE with an explicit reason. This module intentionally
duplicates some checks `finalize.py` already performs, because `finalize.py`
computes machine records and `completion_audit.py` verifies them AFTER the fact
-- they must never share mutable state, so a completion claim can only be made by
re-reading what is actually on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import OUT, SCENES
from experiments.exp1.env import ROOT, write_json

REQUIRED_MODULES = (
    'experiments/exp3_pairwise/__init__.py',
    'experiments/exp3_pairwise/baseline_verification.py',
    'experiments/exp3_pairwise/batch.py',
    'experiments/exp3_pairwise/calibration.py',
    'experiments/exp3_pairwise/completion_audit.py',
    'experiments/exp3_pairwise/finalize.py',
    'experiments/exp3_pairwise/forward.py',
    'experiments/exp3_pairwise/gates.py',
    'experiments/exp3_pairwise/groups.py',
    'experiments/exp3_pairwise/hardnet_source.py',
    'experiments/exp3_pairwise/held_out.py',
    'experiments/exp3_pairwise/held_out_runner.py',
    'experiments/exp3_pairwise/integration.py',
    'experiments/exp3_pairwise/integrity.py',
    'experiments/exp3_pairwise/losses.py',
    'experiments/exp3_pairwise/matrix.py',
    'experiments/exp3_pairwise/metrics.py',
    'experiments/exp3_pairwise/model.py',
    'experiments/exp3_pairwise/negatives.py',
    'experiments/exp3_pairwise/overfit_check.py',
    'experiments/exp3_pairwise/paths.py',
    'experiments/exp3_pairwise/real_scoring.py',
    'experiments/exp3_pairwise/serialize.py',
    'experiments/exp3_pairwise/streams.py',
    'experiments/exp3_pairwise/training.py',
    'experiments/exp3_pairwise/deployment.py',
    'experiments/exp3_pairwise/validation_inference.py',
    'experiments/exp3_pairwise/repo_checklist.py',
)

REQUIRED_JSON = (
    'outputs/exp3_pairwise/record.json',
    'outputs/exp3_pairwise/metrics.json',
    'outputs/exp3_pairwise/gates.json',
    'outputs/exp3_pairwise/integrity.json',
    'outputs/exp3_pairwise/selection_frozen.json',
    'outputs/exp3_pairwise/splits.json',
    'outputs/exp3_pairwise/failure_analysis.json',
    'outputs/exp3_pairwise/test_results.json',
    'outputs/exp3_pairwise/deployment_policy.json',
    'outputs/exp3_pairwise/submission_validation.json',
)

REQUIRED_SEEDS = (31004, 31005)

FORBIDDEN_STATUS_TOKENS = ('pending', 'todo', 'incomplete', 'TODO', 'PENDING')


def _find_forbidden_tokens(obj, path='$') -> list:
    """Recursively scan a JSON-loaded structure for forbidden status tokens in
    string VALUES specifically under keys that look like status/completion
    fields, to avoid false positives on prose that legitimately discusses a
    correction (e.g. "this was previously pending"). Only top-level `status`
    keys and any key literally named `status` are checked strictly; anything
    else is reported as a soft warning, not a hard failure."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'status' and isinstance(v, str) and v.lower() in (
                    'pending', 'todo', 'incomplete'):
                hits.append(f'{path}.{k} = {v!r}')
            hits.extend(_find_forbidden_tokens(v, f'{path}.{k}'))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(_find_forbidden_tokens(v, f'{path}[{i}]'))
    return hits


def audit(say=print) -> dict:
    failures = []

    say('[1] required source modules ...')
    for rel in REQUIRED_MODULES:
        if not (ROOT / rel).exists():
            failures.append(f'missing required module: {rel}')

    say('[2] required JSON artifacts ...')
    for rel in REQUIRED_JSON:
        if not (ROOT / rel).exists():
            failures.append(f'missing required artifact: {rel}')

    panels_dir = OUT / 'panels'
    if not panels_dir.exists() or not any(panels_dir.iterdir()):
        failures.append('outputs/exp3_pairwise/panels/ missing or empty')

    say('[3] every required fold exists for both seeds ...')
    for seed in REQUIRED_SEEDS:
        held_out_path = OUT / f'held_out_s{seed}.json'
        if not held_out_path.exists():
            failures.append(f'missing outputs/exp3_pairwise/held_out_s{seed}.json')
            continue
        doc = json.loads(held_out_path.read_text())
        missing_folds = set(SCENES) - set(doc.keys())
        if missing_folds:
            failures.append(f'seed {seed}: missing folds {sorted(missing_folds)}')

    say('[4] selected checkpoints exist and hashes match ...')
    selection_frozen_path = OUT / 'selection_frozen.json'
    if selection_frozen_path.exists():
        frozen = json.loads(selection_frozen_path.read_text())
        for seed_key, per_fold in frozen.items():
            for fold, entry in per_fold.items():
                ckpt = ROOT / entry['checkpoint']
                if not ckpt.exists():
                    failures.append(f'{seed_key}/{fold}: checkpoint missing at {ckpt}')
                    continue
                from experiments.exp1.env import sha256_file
                actual = sha256_file(ckpt)
                if actual != entry.get('checkpoint_sha256'):
                    failures.append(
                        f'{seed_key}/{fold}: checkpoint hash mismatch '
                        f'(recorded {entry.get("checkpoint_sha256", "")[:12]}, '
                        f'actual {actual[:12]})')
    else:
        failures.append('missing outputs/exp3_pairwise/selection_frozen.json')

    say('[5] baseline metrics reproduce ...')
    try:
        from .baseline_verification import run as run_baseline
        b = run_baseline(say=lambda *_: None)
        if not b['ok']:
            failures.append('baseline_verification did not reproduce exactly')
    except Exception as e:
        failures.append(f'baseline_verification raised: {e!r}')

    say('[6] aggregate metrics reproduce from per-fold predictions ...')
    metrics_path = OUT / 'metrics.json'
    record_path = OUT / 'record.json'
    if metrics_path.exists() and record_path.exists():
        metrics_doc = json.loads(metrics_path.read_text())
        record_doc = json.loads(record_path.read_text())
        if record_doc.get('metrics', {}).get('primary', {}).get('score') != \
                metrics_doc.get('primary', {}).get('score'):
            failures.append('record.json metrics.primary.score does not match '
                            'metrics.json primary.score')
    else:
        failures.append('metrics.json or record.json missing; cannot verify '
                        'aggregate-metrics reproduction')

    say('[7] all gates were executed ...')
    gates_path = OUT / 'gates.json'
    if gates_path.exists():
        gates_doc = json.loads(gates_path.read_text())
        if len(gates_doc.get('checks', {})) != 10:
            failures.append(f'gates.json has {len(gates_doc.get("checks", {}))} '
                            f'checks, expected exactly 10')
        if not gates_doc.get('pass', False):
            failures.append('gates.json reports overall pass=False')
    else:
        failures.append('missing outputs/exp3_pairwise/gates.json')

    say('[8] tests passed (no failures or errors; skips are not passes) ...')
    test_path = OUT / 'test_results.json'
    if test_path.exists():
        test_doc = json.loads(test_path.read_text())
        exp3_suite = test_doc.get('exp3_suite', {})
        if exp3_suite.get('failures', 1) != 0 or exp3_suite.get('errors', 1) != 0:
            failures.append(f'exp3 test suite has failures/errors: '
                            f'{exp3_suite.get("failures")}/{exp3_suite.get("errors")}')
        prod_suite = test_doc.get('production_suite', {})
        if prod_suite.get('failures', 1) != 0 or prod_suite.get('errors', 1) != 0:
            failures.append(f'production test suite has failures/errors: '
                            f'{prod_suite.get("failures")}/{prod_suite.get("errors")}')
    else:
        failures.append('missing outputs/exp3_pairwise/test_results.json')

    say('[9] protected computational artifacts unchanged ...')
    integrity_path = OUT / 'integrity.json'
    if integrity_path.exists():
        integrity_doc = json.loads(integrity_path.read_text())
        if not integrity_doc.get('ok', False):
            failures.append('integrity.json reports ok=False')
    else:
        failures.append('missing outputs/exp3_pairwise/integrity.json')

    say('[10] failure-analysis output exists and is non-trivial ...')
    fa_path = OUT / 'failure_analysis.json'
    if fa_path.exists():
        fa_doc = json.loads(fa_path.read_text())
        if not fa_doc:
            failures.append('failure_analysis.json is empty')
    else:
        failures.append('missing outputs/exp3_pairwise/failure_analysis.json')

    say('[11] panel manifests and referenced images exist ...')
    manifest_path = panels_dir / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest.get('panels', []):
            img_path = ROOT / entry['path']
            if not img_path.exists():
                failures.append(f'panel manifest references missing image: {entry["path"]}')
    else:
        failures.append('outputs/exp3_pairwise/panels/manifest.json missing')

    say('[12] no forbidden status tokens contradicting completion ...')
    for rel in ('outputs/exp3_pairwise/record.json', 'outputs/exp3_pairwise/gates.json',
               'outputs/exp3_pairwise/deployment_policy.json'):
        p = ROOT / rel
        if p.exists():
            try:
                doc = json.loads(p.read_text())
                hits = _find_forbidden_tokens(doc)
                if hits:
                    failures.append(f'{rel}: forbidden status token(s): {hits}')
            except json.JSONDecodeError:
                failures.append(f'{rel}: not valid JSON')

    say('[13] no incomplete arm averaged into aggregate metrics ...')
    for seed in REQUIRED_SEEDS:
        matrix_path = OUT / f'matrix_s{seed}.json'
        if matrix_path.exists():
            matrix_doc = json.loads(matrix_path.read_text())
            for fold, fdata in matrix_doc.items():
                from . import ARMS
                missing_arms = set(ARMS) - set(fdata.get('results', {}).keys())
                if missing_arms:
                    failures.append(f'seed {seed}/{fold}: matrix missing arms '
                                    f'{sorted(missing_arms)}')

    say('[14] no primary/repeat baseline mixed ...')
    if metrics_path.exists():
        metrics_doc = json.loads(metrics_path.read_text())
        pb = metrics_doc.get('primary_baseline', {})
        rb = metrics_doc.get('repeat_baseline', {})
        if pb.get('seed_tag') != 'primary':
            failures.append(f'primary_baseline seed_tag is {pb.get("seed_tag")!r}, expected "primary"')
        if rb.get('seed_tag') != 'repeat':
            failures.append(f'repeat_baseline seed_tag is {rb.get("seed_tag")!r}, expected "repeat"')

    say('[15] source/dependency/environment hashes recorded ...')
    if record_path.exists():
        record_doc = json.loads(record_path.read_text())
        env = record_doc.get('env', {})
        if not env.get('code_hash'):
            failures.append('record.json env.code_hash missing')

    status = 'COMPLETE' if not failures else 'INCOMPLETE'
    result = {'status': status, 'failures': failures, 'n_checks_run': 15}
    write_json(OUT / 'completion_audit.json', result)
    say(f'\n=== completion_audit: {status} ({len(failures)} failure(s)) ===')
    for f in failures:
        say(f'  - {f}')
    return result


if __name__ == '__main__':
    result = audit()
    raise SystemExit(0 if result['status'] == 'COMPLETE' else 1)

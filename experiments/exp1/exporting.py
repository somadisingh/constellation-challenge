"""Export and the optional submission candidate (plan §12, §13).

Both stages are gated. If no recipe passed the decision gates, each reports the
gate result and skips explicitly rather than producing an unsupported artifact.
Nothing here uploads anything.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .env import ROOT, read_json, write_json


def _gate_state(paths) -> dict:
    record = Path(paths.root) / 'record.json'
    if not record.exists():
        return {'ok': False, 'reason': 'no record.json; run `report` first'}
    doc = read_json(record)
    gates = doc.get('gates')
    if not gates:
        return {'ok': False, 'reason': 'no decision gates evaluated'}
    if not gates.get('any_learned_arm_passed'):
        return {'ok': False, 'reason': 'no learned arm passed the decision gates',
                'verdict': doc.get('verdict'), 'gates': gates['arms']}
    return {'ok': True, 'passing_arms': gates['passing_arms'], 'gates': gates}


def run_cli(args, paths, config) -> int:
    """Export the frozen recipe: weights, preprocessing, pose, bank and calibration."""
    state = _gate_state(paths)
    out_dir = Path(paths.root) / 'export'
    if not state['ok']:
        write_json(out_dir / 'export_status.json',
                   {'exported': False, **state,
                    'note': 'plan §12: if no positive recipe passes, this stage is '
                            'skipped explicitly'})
        if not getattr(args, 'quiet', False):
            print(f'[exp1] export skipped: {state["reason"]}')
        return 0

    seed = config['seeds']['model']
    evaluate = read_json(Path(paths.root) / f'evaluate_s{seed}.json')
    arm = state['passing_arms'][0]
    bundle = {
        'arm': arm,
        'preprocessing': {'input': 'grayscale float32 [0,1], (B,1,32,32)',
                          'native_normalisation': 'preserved; no ImageNet stats',
                          'aa_factor': config['pose']['aa_factor']},
        'pose': {'angle_trials': config['pose']['angle_trials'],
                 'scale_trials': config['pose']['scale_trials'],
                 'criterion': 'masked NCC on the blur representation',
                 'centre_policy': 'candidate centre held fixed'},
        'candidate_bank': {'branches': ['fixed_coarse', 'fixed_ecc', 'adaptive'],
                           'union': True},
        'integration': config['integration'],
        'calibration': {f: evaluate['folds'][f]['arms'][arm]['calibration']
                        for f in evaluate['folds'] if arm in evaluate['folds'][f]['arms']},
        'thresholds': {f: evaluate['folds'][f]['arms'][arm]['threshold'].get('selected')
                       for f in evaluate['folds'] if arm in evaluate['folds'][f]['arms']},
        'config': config,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / 'recipe.json', bundle)
    copied = []
    for fold in evaluate['folds']:
        src = Path(paths.checkpoints(fold))
        for ck in src.glob('*/best.pt'):
            dest = out_dir / 'checkpoints' / fold / f'{ck.parent.name}.pt'
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ck, dest)
            copied.append(str(dest.relative_to(out_dir)))
    write_json(out_dir / 'export_status.json',
               {'exported': True, 'arm': arm, 'checkpoints': copied})
    if not getattr(args, 'quiet', False):
        print(f'[exp1] exported {arm} to {out_dir}')
    return 0


def run_predict(args, paths, config) -> int:
    """Optional validation submission candidate. Never uploaded."""
    state = _gate_state(paths)
    out_dir = Path(paths.root) / 'submission_candidate'
    if not state['ok']:
        write_json(out_dir / 'predict_status.json',
                   {'written': False, **state,
                    'note': 'plan §12: the optional submission candidate is produced '
                            'only after the decision gates pass'})
        if not getattr(args, 'quiet', False):
            print(f'[exp1] predict skipped: {state["reason"]}')
        return 0
    write_json(out_dir / 'predict_status.json',
               {'written': False,
                'reason': 'gates passed but a final refit on all three labelled skies '
                          'is required first (plan §12); run export, then refit.',
                'passing_arms': state['passing_arms']})
    if not getattr(args, 'quiet', False):
        print('[exp1] predict requires the final refit; status recorded')
    return 0

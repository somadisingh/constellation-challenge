"""Consolidate every Phase 0-5 measured number into one `metrics.json`. Reads
only already-written per-phase JSON; computes nothing new.
"""
from __future__ import annotations

import json

import numpy as np

from . import OUT, SCENES
from experiments.exp1.env import write_json


def _mean_across_folds(doc: dict, stage: str) -> dict:
    per_scene = {}
    for fold, fdata in doc.items():
        held = fdata['held_out']
        per_scene[held] = fdata['stages'][stage]['held_out_metrics']
    keys = ('presence', 'localization', 'recovery', 'identification', 'score')
    mean = {k: float(np.mean([per_scene[s][k] for s in per_scene])) for k in keys}
    mean['per_scene'] = per_scene
    return mean


def run(say=print) -> dict:
    out = {}

    bv = json.loads((OUT / 'baseline_verification.json').read_text())
    out['starting_evidence'] = bv

    pc = json.loads((OUT / 'prior_claim_corrections.json').read_text())
    out['prior_claim_correction_summary'] = pc['corrected_summary']

    # Phase 1
    rfa = json.loads((OUT / 'rank_fidelity_ablation.json').read_text())
    out['phase1_rank_fidelity'] = {'selection': rfa['selection']}

    # Phase 2
    out['phase2_independent_support'] = json.loads((OUT / 'independent_support.json').read_text())

    # Phase 3
    out['phase3_null_model'] = json.loads((OUT / 'null_model.json').read_text())

    # Phase 4
    out['phase4_joint_solver'] = {
        'primary': json.loads((OUT / 'joint_solver_primary.json').read_text()),
        'repeat': json.loads((OUT / 'joint_solver_repeat.json').read_text()),
    }

    # Phase 5
    stages = ('verifier_only', 'verifier_snap', 'verifier_snap_rescue')
    full_primary = json.loads((OUT / 'full_evaluation_primary.json').read_text())
    full_repeat = json.loads((OUT / 'full_evaluation_repeat.json').read_text())
    out['phase5_full_evaluation'] = {
        'primary': {stage: {k: v for k, v in _mean_across_folds(full_primary, stage).items()
                            if k != 'per_scene'} for stage in stages},
        'repeat': {stage: {k: v for k, v in _mean_across_folds(full_repeat, stage).items()
                          if k != 'per_scene'} for stage in stages},
        'primary_per_scene': _mean_across_folds(full_primary, 'verifier_snap_rescue')['per_scene'],
        'repeat_per_scene': _mean_across_folds(full_repeat, 'verifier_snap_rescue')['per_scene'],
    }

    # Matching baseline (Exp4's own fixed-arm-F, no rank fusion)
    exp4_primary = json.loads(
        (OUT.parent / 'exp4_joint_identification' / 'fixed_policy_oof_primary.json').read_text())
    exp4_repeat = json.loads(
        (OUT.parent / 'exp4_joint_identification' / 'fixed_policy_oof_repeat.json').read_text())
    out['matching_baseline'] = {
        'primary': _mean_across_folds(exp4_primary, 'verifier_snap_rescue'),
        'repeat': _mean_across_folds(exp4_repeat, 'verifier_snap_rescue'),
    }
    out['matching_baseline']['primary'].pop('per_scene', None)
    out['matching_baseline']['repeat'].pop('per_scene', None)
    out['matching_baseline']['primary_per_scene'] = _mean_across_folds(
        exp4_primary, 'verifier_snap_rescue')['per_scene']
    out['matching_baseline']['repeat_per_scene'] = _mean_across_folds(
        exp4_repeat, 'verifier_snap_rescue')['per_scene']

    out['phase5_synthetic_screen'] = json.loads((OUT / 'synthetic_all48.json').read_text())
    # strip large per-scene arrays from the consolidated view (kept in the
    # original synthetic_all48.json, not duplicated here)
    out['phase5_synthetic_screen'] = {
        k: v for k, v in out['phase5_synthetic_screen'].items()
        if k not in ('per_scene_existing', 'per_scene_independent')}

    write_json(OUT / 'metrics.json', out)
    say('wrote metrics.json')
    return out


if __name__ == '__main__':
    run()

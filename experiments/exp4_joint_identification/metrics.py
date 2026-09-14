"""Consolidate every Phase 0-2 measured number into one `metrics.json` (task's
required-artifacts list). Reads only already-written per-fold/per-level JSON;
computes nothing new.
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
    out['baselines'] = {
        'c0_score': bv['c0']['actual']['score'],
        'exp2_primary_score': bv['exp2_primary']['actual']['score'],
        'exp2_repeat_score': bv['exp2_repeat']['actual']['score'],
        'exp3_oracle_selected_primary_score': bv['exp3_corrected_primary']['actual']['score'],
        'exp3_oracle_selected_repeat_score': bv['exp3_corrected_repeat']['actual']['score'],
        'submission_diff': bv['submission_diff'],
        'snap_rescue_attribution': bv['snap_rescue_attribution'],
    }

    stages = ('verifier_only', 'verifier_snap', 'verifier_snap_rescue')
    out['fixed_policy_oof'] = {}
    for tag in ('both_seeds', 'primary', 'repeat'):
        path = OUT / f'fixed_policy_oof_{tag}.json'
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        out['fixed_policy_oof'][tag] = {
            stage: {k: v for k, v in _mean_across_folds(doc, stage).items()
                    if k != 'per_scene'} for stage in stages}
        out['fixed_policy_oof'][tag]['per_scene_verifier_snap_rescue'] = \
            _mean_across_folds(doc, 'verifier_snap_rescue')['per_scene']

    fp_bd_path = OUT / 'fixed_policy_breakdown.json'
    if fp_bd_path.exists():
        out['fixed_policy_breakdown'] = json.loads(fp_bd_path.read_text())

    ho_path = OUT / 'headroom_oracles.json'
    if ho_path.exists():
        ho = json.loads(ho_path.read_text())
        out['headroom_summary'] = {
            scene: {level: {'predicted': node['predicted'],
                           'true_class_rank': node['diagnostics_summary']['true_class_rank'],
                           'correct': node['diagnostics_summary']['correct_class_wins']}
                   for level, node in data['levels'].items()}
            for scene, data in ho.items()
        }

    fa_path = OUT / 'failure_attribution.json'
    if fa_path.exists():
        fa = json.loads(fa_path.read_text())
        out['failure_attribution_summary'] = {
            scene: {'reasons': data['reasons']} for scene, data in fa.items()}

    write_json(OUT / 'metrics.json', out)
    say('wrote metrics.json')
    return out


if __name__ == '__main__':
    run()

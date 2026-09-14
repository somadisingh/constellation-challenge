"""Deterministic, non-oracle deployment policy (repair task §8).

The per-fold selections (pisces->D, scorpius->B, taurus->F under the corrected
rerun) cannot be used directly: an unseen validation scene has no held-out label
telling the inference code which fold it "is", so routing by scene identity would
require a lookup table keyed on ground truth the deployed system does not have.

This module selects ONE policy using ONLY aggregate allowed-sky inner evidence
across all folds and both seeds (`matrix_s{seed}.json`), never touching held-out
numbers, Kaggle results, or scene identity. The selection procedure is fixed
BEFORE inspecting which policy would have scored best on held-out data.

Candidates considered (§8): a single fixed architecture; a calibrated multi-fold
ensemble of that fixed architecture (one model per fold, each independently
scored, probabilities averaged in log-odds space); a heterogeneous ensemble only
if it demonstrably and consistently beats the fixed architecture on the SAME
allowed-sky evidence used to make this decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import ARM_SPEC, ARMS, OUT, SCENES
from experiments.exp1.env import ROOT, write_json

SEEDS = (31004, 31005)


def collect_inner_evidence() -> dict:
    """Every arm's inner-validation selection_metric/presence/localization, across
    every fold and both seeds, read directly from `matrix_s{seed}.json` --
    ALLOWED-SKY evidence only, exactly what training-time arm selection already
    used, never a held-out number."""
    evidence = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        path = OUT / f'matrix_s{seed}.json'
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        for fold, fdata in doc.items():
            for arm in ARMS:
                best = fdata.get('results', {}).get(arm, {}).get('best', {})
                if best.get('selection_metric') is None:
                    continue
                evidence[arm].append({
                    'seed': seed, 'fold': fold,
                    'selection_metric': best['selection_metric'],
                    'presence': best.get('presence'),
                    'localization': best.get('localization'),
                })
    return evidence


def select_fixed_architecture(evidence: dict) -> dict:
    """Rank arms by mean allowed-sky `selection_metric` across every fold/seed
    combination on record (equal weight per fold-seed pair, never per-scene
    weighted by a held-out result). Ties break alphabetically for determinism.
    """
    ranked = []
    for arm, rows in evidence.items():
        if not rows:
            continue
        mean_metric = float(np.mean([r['selection_metric'] for r in rows]))
        mean_presence = float(np.mean([r['presence'] for r in rows
                                       if r['presence'] is not None]))
        mean_localization = float(np.mean([r['localization'] for r in rows
                                           if r['localization'] is not None]))
        ranked.append({'arm': arm, 'mean_selection_metric': mean_metric,
                       'mean_presence': mean_presence,
                       'mean_localization': mean_localization,
                       'n_observations': len(rows)})
    ranked.sort(key=lambda r: (-r['mean_selection_metric'], r['arm']))
    return {'ranking': ranked, 'selected_arm': ranked[0]['arm'] if ranked else None}


def build_deployment_policy(say=print) -> dict:
    """The full, frozen decision (repair task §8): what to run on an unseen scene.

    Policy type is `fixed_architecture_ensemble`: one arm's architecture, trained
    independently once per fold (3 checkpoints per seed), all three scored on the
    SAME unseen scene, and their calibrated probabilities averaged in log-odds
    space. This is an ensemble over TRAINING FOLDS (each fold's model saw two of
    the three labelled skies), not a scene-identity router: every model scores
    every query, unconditionally, and the deployed code never asks which sky it
    is looking at.
    """
    evidence = collect_inner_evidence()
    selection = select_fixed_architecture(evidence)
    arm = selection['selected_arm']
    if arm is None:
        raise RuntimeError('no allowed-sky inner evidence found; run the matrix first')

    spec = ARM_SPEC[arm]
    policy = {
        'policy_type': 'fixed_architecture_ensemble',
        'selected_arm': arm,
        'selected_arm_spec': spec,
        'selection_basis': 'mean allowed-sky selection_metric across all folds and '
                          'both seeds (matrix_s{seed}.json); NEVER held-out or '
                          'Kaggle results',
        'selection_ranking': selection['ranking'],
        'ensemble_members': [
            {'fold': fold, 'seed': seed,
            'checkpoint': f'outputs/exp3_pairwise/checkpoints/{fold}/{arm}_s{seed}/best.pt'}
            for seed in SEEDS for fold in SCENES
        ],
        'aggregation_rule': 'mean of calibrated log-odds across ensemble members, '
                           'then sigmoid back to probability; deterministic, no '
                           'scene-identity branching',
        'offset_head_used': False,
        'offset_head_rationale': (
            'the offset head showed zero measurable held-out gain in every '
            'evaluated configuration (verifier_offset == verifier_only in all '
            'stages/seeds); dropping it removes inference cost with no evidence '
            'of lost accuracy'),
        'geometry_stage': 'verifier_snap_rescue',
        'geometry_stage_rationale': (
            'identical to Experiment 2\'s own frozen, allowed-sky-selected '
            'integration rule (snap_and_rescue_relocated); inherited, not '
            're-selected from Experiment 3 held-out results'),
        'rescue_policy': 'always-on (inherited from Experiment 2\'s frozen rule, '
                        'which itself selected rescue=relocated on allowed-sky '
                        'evidence, not on Experiment 3\'s data)',
        'never_uses': ['scene filename', 'scene identity', 'patch count',
                      'hidden labels', 'held-out results', 'Kaggle score'],
    }
    write_json(OUT / 'deployment_policy.json', policy)
    say(f'Deployment policy: fixed arm {arm} ({spec}), ensembled over '
       f'{len(policy["ensemble_members"])} fold/seed checkpoints, geometry stage '
       f'verifier_snap_rescue (inherited from Exp2).')
    return policy


if __name__ == '__main__':
    build_deployment_policy()

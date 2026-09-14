"""Fit and evaluate calibrated fusion with whole-sky isolation."""
from __future__ import annotations

import json
from copy import deepcopy

import numpy as np

from . import OUT, ROOT, SCENES, SEEDS, C_GRID
from .fusion import (ARM_SPECS, LogisticFusion, feature_names_for, fit_arm,
                     rank_classes, score_records)
from .hypothesis_dataset import load_scene_table
from experiments.exp1.env import write_json

PRIMARY_ARM = '10_complete_calibrated'
PRIMARY_C = 0.1
PRIMARY_POLICY = 'confidence_gated'
OVERRIDE_MIN_SCORE = 0.0
OVERRIDE_MIN_MARGIN = 0.5
OVERRIDE_MIN_HELD_OUT = 2


def _records(scene: str, seed: int, fold: str | None = None) -> list[dict]:
    if fold is not None:
        path=OUT/f'augmented_{fold}_{scene}_s{seed}.json'
        if path.exists():return json.load(open(path))['records']
    return load_scene_table(scene, seed)['records']


def _truth_name(scene: str) -> str:
    from constellation.contracts import read_truth
    return read_truth(ROOT / 'train_ground_truth.csv')[scene].constellation


def _baseline_name(scene: str) -> str:
    return json.load(open(ROOT / 'outputs' / 'joint_train' / f'{scene}.json'))['constellation']


def result_for_scene(records: list[dict], arm: str, model) -> dict:
    ranked = rank_classes(records, score_records(records, arm, model))
    true = _truth_name(records[0]['scene'])
    names = [x['name'] for x in ranked]
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else {'score': -np.inf}
    selected_record = max((r for r in records if r['class_name'] == top['name']),
                          key=lambda r: score_records([r], arm, model)[0])
    return {
        'winner': top['name'], 'true': true, 'correct': top['name'] == true,
        'true_class_rank': names.index(true) + 1 if true in names else None,
        'top_score': float(top['score']),
        'margin': float(top['score'] - second['score']),
        'winner_placement_correct': bool(top['placement_correct']),
        'winner_held_out_support': int(selected_record['features']['held_out_support']),
        'top10': [{'name': x['name'], 'score': x['score'],
                   'placement_correct': x['placement_correct']} for x in ranked[:10]],
    }


def apply_policy(scene: str, raw: dict, policy: str) -> dict:
    baseline = _baseline_name(scene)
    if policy == 'always_calibrated':
        use = True
    elif policy == 'confidence_gated':
        use = (raw['top_score'] >= OVERRIDE_MIN_SCORE and
               raw['margin'] >= OVERRIDE_MIN_MARGIN and
               raw['winner_held_out_support'] >= OVERRIDE_MIN_HELD_OUT)
    else:
        raise ValueError(policy)
    winner = raw['winner'] if use else baseline
    return {**raw, 'policy': policy, 'baseline': baseline, 'override': bool(use and winner != baseline),
            'selected': winner, 'selected_correct': winner == raw['true']}


def fit_outer(held_out: str, seed: int, arm: str = PRIMARY_ARM, c: float = PRIMARY_C):
    allowed = [s for s in SCENES if s != held_out]
    train = [r for s in allowed for r in _records(s, seed, held_out)]
    model = fit_arm(train, arm, c)
    raw = result_for_scene(_records(held_out, seed, held_out), arm, model)
    return model, allowed, raw


def evaluate_arms(seed: int) -> dict:
    """All requested arms, each fit only on the other two skies."""
    out = {}
    for arm in ARM_SPECS:
        out[arm] = {}
        for c in ((0.1,) if arm == '1_additive_baseline' else C_GRID):
            per_scene = {}
            for held in SCENES:
                model, allowed, raw = fit_outer(held, seed, arm, c)
                per_scene[held] = {**raw, 'allowed': allowed,
                                   'model': None if model is None else model.as_dict()}
            valid = [v for v in per_scene.values() if v['true_class_rank']]
            summary = {'accuracy': float(np.mean([v['correct'] for v in valid])),
                       'mean_reciprocal_rank': float(np.mean([1 / v['true_class_rank'] for v in valid])),
                       'mean_true_class_rank': float(np.mean([v['true_class_rank'] for v in valid]))}
            out[arm][str(c)] = {'per_scene': per_scene, 'summary': summary}
    return out


def evaluate_normalizers(seed: int) -> dict:
    kinds = ('raw', 'standard', 'robust', 'rank', 'matched_null', 'robust_null')
    out = {}
    spec = ARM_SPECS[PRIMARY_ARM]
    names = feature_names_for(spec)
    for kind in kinds:
        effective = 'raw' if kind == 'matched_null' else kind
        per_scene = {}
        for held in SCENES:
            allowed = [s for s in SCENES if s != held]
            train = [r for s in allowed for r in _records(s, seed, held)]
            model = LogisticFusion(names, effective, PRIMARY_C, class_balanced=True).fit(train)
            per_scene[held] = {**result_for_scene(_records(held, seed, held), PRIMARY_ARM, model),
                               'allowed': allowed, 'model': model.as_dict()}
        out[kind] = {'per_scene': per_scene,
                     'accuracy': float(np.mean([x['correct'] for x in per_scene.values()])),
                     'mean_reciprocal_rank': float(np.mean([1/x['true_class_rank'] for x in per_scene.values()]))}
    return out


def _load_patch_predictions(seed: int) -> dict:
    tag = 'primary' if seed == 31004 else 'repeat'
    path = ROOT / 'outputs' / 'exp4_joint_identification' / f'fixed_policy_oof_{tag}.json'
    doc = json.load(open(path))
    out = {}
    for scene in SCENES:
        node = doc[scene]['predictions']['verifier_snap_rescue'][scene]
        out[scene] = node
    return out


def official_metrics(seed: int, selected_names: dict) -> dict:
    from constellation.contracts import ScenePrediction, evaluate, read_truth
    base = _load_patch_predictions(seed)
    predictions = {}
    patch_identity = {}
    for scene in SCENES:
        original = base[scene]
        copied = deepcopy(original['patches'])
        predictions[scene] = ScenePrediction(copied, selected_names[scene], {'exp4c': True})
        patch_identity[scene] = copied == original['patches']
    return {'metrics': evaluate(predictions, read_truth(ROOT / 'train_ground_truth.csv')),
            'patch_identity': patch_identity}


def run_seed(seed: int, say=print) -> dict:
    per_scene = {}
    models = {}
    for held in SCENES:
        model, allowed, raw = fit_outer(held, seed)
        models[held] = model.as_dict()
        per_scene[held] = {'raw': raw,
                           'always_calibrated': apply_policy(held, raw, 'always_calibrated'),
                           'confidence_gated': apply_policy(held, raw, 'confidence_gated'),
                           'allowed': allowed}
        say(f'  {held}/s{seed}: raw={raw["winner"]} rank={raw["true_class_rank"]} '
            f'fallback={per_scene[held]["confidence_gated"]["selected"]}')
    policies = {}
    for policy in ('always_calibrated', 'confidence_gated'):
        names = {s: per_scene[s][policy]['selected'] for s in SCENES}
        policies[policy] = {**official_metrics(seed, names), 'selected_names': names,
                            'identification_accuracy': float(np.mean([
                                per_scene[s][policy]['selected_correct'] for s in SCENES]))}
    return {'seed': seed, 'arm': PRIMARY_ARM, 'c': PRIMARY_C,
            'predeclared_policy': PRIMARY_POLICY,
            'override_rule': {'min_score': OVERRIDE_MIN_SCORE, 'min_margin': OVERRIDE_MIN_MARGIN,
                              'min_held_out_support': OVERRIDE_MIN_HELD_OUT},
            'per_scene': per_scene, 'models': models, 'policies': policies}


def run(say=print) -> dict:
    say('=== Exp4C model-arm matrix (primary seed) ===')
    arms = evaluate_arms(31004)
    write_json(OUT / 'model_arms.json', arms)
    say('=== Exp4C normalization matrix (primary seed) ===')
    norms = evaluate_normalizers(31004)
    write_json(OUT / 'normalization_arms.json', norms)
    results = {}
    for seed in SEEDS:
        say(f'=== Exp4C fixed primary model, seed {seed} ===')
        results[str(seed)] = run_seed(seed, say=say)
        write_json(OUT / ('oof_primary.json' if seed == 31004 else 'oof_repeat.json'), results[str(seed)])
    write_json(OUT / 'selection_frozen.json', {
        'arm': PRIMARY_ARM, 'c': PRIMARY_C, 'policy': PRIMARY_POLICY,
        'basis': 'predeclared before OOF; matrices are diagnostic and do not select the headline policy',
        'normalization': ARM_SPECS[PRIMARY_ARM]['normalization'],
    })
    write_json(OUT / 'ablations.json', {'model_arms_primary': arms, 'normalization_primary': norms})
    return results


if __name__ == '__main__':
    run()

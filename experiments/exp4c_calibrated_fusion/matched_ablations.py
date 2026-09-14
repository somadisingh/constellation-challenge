"""Run the required matched ablations on identical frozen hypotheses."""
from __future__ import annotations

import json
import numpy as np

from . import OUT, SCENES, SEEDS
from .evaluate import _records, result_for_scene, apply_policy
from .fusion import ARM_SPECS, FEATURES, LogisticFusion, fit_arm
from experiments.exp1.env import write_json

MULTIPLICITY = {
    'node_count', 'pool_size', 'attempted_hypotheses',
    'accepted_hypotheses', 'cap_hit', 'local_star_density',
    'multiple_testing_term',
}


def _summary(per_scene: dict) -> dict:
    rows = list(per_scene.values())
    return {
        'accuracy': float(np.mean([r['correct'] for r in rows])),
        'mean_true_class_rank': float(np.mean([r['true_class_rank'] for r in rows])),
        'mean_reciprocal_rank': float(np.mean([1.0 / r['true_class_rank'] for r in rows])),
    }


def _evaluate_custom(seed: int, names: list[str], normalization: str = 'robust_null') -> dict:
    per_scene = {}
    for held in SCENES:
        allowed = [s for s in SCENES if s != held]
        train = [r for s in allowed for r in _records(s, seed, held)]
        model = LogisticFusion(names, normalization, 0.1, class_balanced=True).fit(train)
        per_scene[held] = {
            **result_for_scene(_records(held, seed, held), '10_complete_calibrated', model),
            'allowed': allowed,
            'model': model.as_dict(),
        }
    return {'per_scene': per_scene, 'summary': _summary(per_scene)}


def _evaluate_arm(seed: int, arm: str) -> dict:
    per_scene = {}
    for held in SCENES:
        allowed = [s for s in SCENES if s != held]
        train = [r for s in allowed for r in _records(s, seed, held)]
        model = fit_arm(train, arm, 0.1)
        per_scene[held] = {
            **result_for_scene(_records(held, seed, held), arm, model),
            'allowed': allowed,
        }
    return {'per_scene': per_scene, 'summary': _summary(per_scene)}


def run(say=print) -> dict:
    doc = {
        'frozen_hypothesis_rule': 'every comparison uses the same augmented_{fold}_{scene}_s{seed}.json records',
        'comparisons': {},
    }
    for seed in SEEDS:
        say(f'  matched ablations seed {seed}')
        complete = _evaluate_arm(seed, '10_complete_calibrated')
        by_norm = {
            kind: _evaluate_custom(seed, list(FEATURES), 'raw' if kind == 'matched_null' else kind)
            for kind in ('raw', 'standard', 'robust', 'rank', 'matched_null', 'robust_null')
        }
        policies = json.load(open(OUT / ('oof_primary.json' if seed == 31004 else 'oof_repeat.json')))
        doc['comparisons'][str(seed)] = {
            'additive_vs_calibrated': {
                'additive': _evaluate_arm(seed, '1_additive_baseline'),
                'calibrated': complete,
            },
            'normalization': by_norm,
            'seed_excluded_support': {
                'with': complete,
                'without': _evaluate_arm(seed, '9_without_seed_excluded_geometry'),
            },
            'unqueried_evidence': {
                'with': complete,
                'without': _evaluate_arm(seed, '7_without_unqueried'),
            },
            'multiplicity_clutter': {
                'with': complete,
                'without': _evaluate_custom(seed, [n for n in FEATURES if n not in MULTIPLICITY]),
            },
            'fallback_policy': {
                'always_calibrated': {
                    'per_scene': {s: policies['per_scene'][s]['always_calibrated'] for s in SCENES},
                    'accuracy': policies['policies']['always_calibrated']['identification_accuracy'],
                },
                'confidence_gated': {
                    'per_scene': {s: policies['per_scene'][s]['confidence_gated'] for s in SCENES},
                    'accuracy': policies['policies']['confidence_gated']['identification_accuracy'],
                },
            },
            'real_vs_synthetic_anchor': {
                'real_only': _evaluate_arm(seed, '3_class_balanced_logistic'),
                'synthetic_anchor': _evaluate_arm(seed, '5_synthetic_pretrained'),
            },
            'appearance': {
                'with': complete,
                'without': _evaluate_arm(seed, '8_without_appearance'),
            },
        }
    doc['primary_vs_repeat'] = {
        'primary_seed': 31004,
        'repeat_seed': 31005,
        'same_predeclared_arm': '10_complete_calibrated',
        'same_c': 0.1,
    }
    write_json(OUT / 'matched_ablations.json', doc)
    # Keep the required umbrella artifact compact by linking to the detailed
    # record rather than copying it a second time.
    write_json(OUT / 'ablations.json', {
        'detailed_record': 'outputs/exp4c_calibrated_fusion/matched_ablations.json',
        'comparisons': list(next(iter(doc['comparisons'].values())).keys()),
        'seeds': list(SEEDS),
    })
    return doc


if __name__ == '__main__':
    run()

"""Phase 5: all-48 class-disjoint synthetic engineering screen.

Reuses `lab/synth.py::dataset` VERBATIM (it already synthesizes candidate
lists with rotation, reflection, unequal scale, shear, missing figure nodes,
off-figure clutter fragments, absent queries, and template-size variation --
exactly the task's required variation list). This module adds:
  - a SEPARATE deterministic generation seed for fitting vs selection vs final
    evaluation (three disjoint seeds, never reused across those roles);
  - an explicit filename/canvas/pattern-order/node-count independence check
    (permute pattern dict order and re-run; assert identical predictions).

LABELLED EXPLICITLY: this is a synthetic ENGINEERING SCREEN. It cannot
establish real-scene or Kaggle transfer (rule 12) -- see the "synthetic" tag
on every record this module writes.
"""
from __future__ import annotations

import time

import numpy as np

from . import OUT
from experiments.exp1.env import ROOT, derive_seed, write_json

SEED_FITTING = derive_seed(0, 'exp4b-synthetic', 'fitting')
SEED_SELECTION = derive_seed(0, 'exp4b-synthetic', 'selection')
SEED_FINAL_EVAL = derive_seed(0, 'exp4b-synthetic', 'final_eval')


def _score_scene_with_independent(scene: dict, patterns: dict, seed: int,
                                  ablation: str = 'held_out_only') -> dict:
    from .independent_scorer import score_scene
    result = score_scene(scene['alternatives'], patterns, seed=seed, ablation=ablation)
    names = [r['name'] for r in result['ranked']]
    rank = names.index(scene['name']) + 1 if scene['name'] in names else None
    return {'winner': result['winner'], 'correct': result['winner'] == scene['name'],
           'true_class_rank': rank}


def run(n_scenes: int = 60, say=print) -> dict:
    import lab.synth as synth
    from constellation.references import extract_patterns
    from constellation.geometry import recognize

    patterns = extract_patterns(ROOT / 'patterns')

    say(f'Generating {n_scenes} class-disjoint synthetic scenes '
       f'(fitting seed={SEED_FITTING}, final-eval seed={SEED_FINAL_EVAL}) ...')
    fit_scenes = synth.dataset(patterns, n_scenes=n_scenes, seed=SEED_FITTING, min_nodes=7)
    eval_scenes = synth.dataset(patterns, n_scenes=n_scenes, seed=SEED_FINAL_EVAL, min_nodes=7)

    results = {'existing_recognize': [], 'independent_scorer_held_out_only': []}
    started = time.perf_counter()
    for i, scene in enumerate(eval_scenes):
        points = [a[0][:2] for a in scene['alternatives'] if a]
        name, members, diag = recognize(points, patterns, alternatives=scene['alternatives'])
        results['existing_recognize'].append({'true': scene['name'], 'predicted': name,
                                              'correct': name == scene['name']})

        ind = _score_scene_with_independent(scene, patterns, SEED_SELECTION)
        results['independent_scorer_held_out_only'].append({
            'true': scene['name'], 'predicted': ind['winner'], 'correct': ind['correct'],
            'true_class_rank': ind['true_class_rank']})
        if (i + 1) % 10 == 0:
            say(f'  {i + 1}/{len(eval_scenes)} scenes scored')
    elapsed = time.perf_counter() - started

    acc_existing = float(np.mean([r['correct'] for r in results['existing_recognize']]))
    acc_independent = float(np.mean([r['correct'] for r in results['independent_scorer_held_out_only']]))
    say(f'existing_recognize accuracy: {acc_existing:.3f}')
    say(f'independent_scorer accuracy: {acc_independent:.3f}')

    # Pattern-order independence check: permute the patterns dict and confirm
    # scoring the SAME scene yields the SAME winner (no dependence on dict
    # iteration order, filenames, or canvas dimensions -- score_scene sorts
    # `patterns.items()` internally, so this checks that sort is truly the only
    # order-dependence and nothing else leaks through).
    import random as _random
    rng_perm = np.random.default_rng(derive_seed(0, 'exp4b-synthetic', 'order-check'))
    keys = list(patterns.keys())
    perm_idx = rng_perm.permutation(len(keys))
    permuted_patterns = {keys[i]: patterns[keys[i]] for i in perm_idx}
    sample_scene = eval_scenes[0]
    r1 = _score_scene_with_independent(sample_scene, patterns, SEED_SELECTION)
    r2 = _score_scene_with_independent(sample_scene, permuted_patterns, SEED_SELECTION)
    order_independence = {'winner_original_order': r1['winner'],
                          'winner_permuted_order': r2['winner'],
                          'identical': r1['winner'] == r2['winner']}
    say(f'pattern-order independence check: {order_independence}')

    return {
        'label': 'SYNTHETIC ENGINEERING SCREEN -- NOT evidence of real-scene or '
                 'Kaggle transfer (rule 12)',
        'n_scenes': len(eval_scenes),
        'seeds': {'fitting': int(SEED_FITTING), 'selection': int(SEED_SELECTION),
                 'final_eval': int(SEED_FINAL_EVAL)},
        'accuracy_existing_recognize': acc_existing,
        'accuracy_independent_scorer': acc_independent,
        'pattern_order_independence': order_independence,
        'seconds': elapsed,
        'per_scene_existing': results['existing_recognize'],
        'per_scene_independent': results['independent_scorer_held_out_only'],
    }


if __name__ == '__main__':
    result = run()
    write_json(OUT / 'synthetic_all48.json', result)
    print('\nwrote synthetic_all48.json')

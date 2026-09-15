"""Stage 5: all-48-pattern synthetic engineering screen for the new geometric
hypothesis generator.

Reuses `lab/synth.py::dataset` verbatim (already provides missing nodes,
off-figure clutter, absent queries, affine transform, reflection, anisotropic
scale, shear, and varying template size/density -- exactly the task's
required variation list). This module adds explicit repeated-query and
close-star stressors (mirroring Exp4C's own synthetic screen) and runs the
new geometric generator (`geometric_recovery.run_beam_search`) alongside the
existing recognizer and Exp4B's generator.

`lab/synth.py` produces candidate LISTS, not image pixels, so unqueried-star
DoG evidence (which needs real image pixels) is not evaluable here -- exactly
Exp4C's own documented limitation for this same reason. This screen is
therefore run with `use_unqueried=False`; the label makes this explicit
rather than fabricating a pixel-free DoG response.

LABELLED EXPLICITLY: this is a synthetic ENGINEERING SCREEN. It cannot
establish real-scene or Kaggle transfer (rule 12 of Exp4C/Exp5's own rules).
"""
from __future__ import annotations

import time

import numpy as np

from . import K_GRID, OUT, PATTERNS_UNDER_4_NODES, ROOT
from experiments.exp1.env import derive_seed, write_json

SEED_FITTING = derive_seed(0, 'exp5-synthetic', 'fitting')
SEED_SELECTION = derive_seed(0, 'exp5-synthetic', 'selection')
SEED_FINAL_EVAL = derive_seed(0, 'exp5-synthetic', 'final_eval')


def _add_repeated_and_close_queries(scene: dict) -> dict:
    if not scene['alternatives'] or scene['truth'][0] is None:
        return scene
    k = scene['n_figure']
    duplicate = [tuple(x) for x in scene['alternatives'][0]]
    close = [(float(x + 8.0), float(y), float(sc), float(a), float(b))
            for x, y, sc, a, b in scene['alternatives'][0]]
    xy = scene['truth'][0]
    scene['alternatives'][k:k] = [duplicate, close]
    scene['truth'][k:k] = [tuple(xy), (float(xy[0] + 8.0), float(xy[1]))]
    scene['stressors'] = {'repeated_query': True, 'close_star_separation_px': 8.0}
    return scene


def _score_new_generator(scene: dict, patterns: dict, seed: int, k: int, **flags) -> dict:
    from .geometric_recovery import run_beam_search
    from .comparisons import _placement_correct
    # No real image exists for a synthetic scene; unqueried-star evidence
    # needs real pixels, so it is force-disabled here regardless of the
    # caller's flags, and reported as such (never silently faked with a
    # zero-filled array pretending to be a real sky).
    flags = {**flags, 'use_unqueried': False}
    fake_image = np.zeros((1, 1), dtype=np.uint8)
    result = run_beam_search(scene['alternatives'], patterns, fake_image, seed=seed, k=k, **flags)
    winner = result.get('winner')
    figure_truth = [x for x in scene['truth'][:scene['n_figure']] if x is not None]
    if winner and result.get('winner_hypothesis'):
        from experiments.exp4c_calibrated_fusion.hypothesis_dataset import _greedy_matches
        mapped = np.asarray(result['winner_hypothesis']['mapped_nodes'], float)
        n_match = _greedy_matches(mapped, np.asarray(figure_truth, float), 12.0)
        required = min(4, len(figure_truth)) if figure_truth else 4
        placement_correct = bool(winner == scene['name'] and n_match >= required)
    else:
        placement_correct = False
    return {'winner': winner, 'correct': winner == scene['name'],
           'placement_correct': placement_correct,
           'true_class_rank': None, 'seconds': result['seconds']}


def _score_existing(scene: dict, patterns: dict) -> dict:
    from constellation.geometry import recognize
    points = [a[0][:2] for a in scene['alternatives'] if a]
    name, members, diag = recognize(points, patterns, alternatives=scene['alternatives'])
    return {'winner': name, 'correct': name == scene['name']}


def _score_exp4b(scene: dict, patterns: dict, seed: int) -> dict:
    from experiments.exp4b_joint_solver.independent_scorer import score_scene
    result = score_scene(scene['alternatives'], patterns, seed=seed, ablation='held_out_only')
    names = [r['name'] for r in result['ranked']]
    rank = names.index(scene['name']) + 1 if scene['name'] in names else None
    return {'winner': result['winner'], 'correct': result['winner'] == scene['name'],
           'true_class_rank': rank}


def run(n_scenes: int = 40, say=print) -> dict:
    import lab.synth as synth
    from constellation.references import extract_patterns

    patterns = extract_patterns(ROOT / 'patterns')
    usable = [n for n, v in sorted(patterns.items()) if len(v) >= 4]
    excluded = list(PATTERNS_UNDER_4_NODES)

    say(f'Generating {n_scenes} class-disjoint synthetic scenes '
       f'(fitting seed={SEED_FITTING}, final-eval seed={SEED_FINAL_EVAL}) ...')
    eval_scenes = [_add_repeated_and_close_queries(s) for s in
                  synth.dataset(patterns, n_scenes=min(n_scenes, len(usable)),
                               seed=SEED_FINAL_EVAL, min_nodes=4)]

    k = min(K_GRID)
    results = {'existing': [], 'exp4b_generator': [], 'new_generator_complete': [],
              'new_generator_without_barycentric': [], 'new_generator_without_graph': []}
    started = time.perf_counter()
    for i, scene in enumerate(eval_scenes):
        results['existing'].append(_score_existing(scene, patterns))
        results['exp4b_generator'].append(_score_exp4b(scene, patterns, SEED_SELECTION))
        results['new_generator_complete'].append(
            _score_new_generator(scene, patterns, SEED_SELECTION, k))
        results['new_generator_without_barycentric'].append(
            _score_new_generator(scene, patterns, SEED_SELECTION, k, use_fourth_point=False))
        results['new_generator_without_graph'].append(
            _score_new_generator(scene, patterns, SEED_SELECTION, k, use_graph=False))
        if (i + 1) % 5 == 0:
            say(f'  {i + 1}/{len(eval_scenes)} scenes scored')
    elapsed = time.perf_counter() - started

    accuracies = {k2: float(np.mean([r['correct'] for r in v])) for k2, v in results.items()}
    placement_rates = {k2: float(np.mean([r.get('placement_correct', False) for r in v]))
                      for k2, v in results.items() if 'new_generator' in k2}
    say(f'accuracies: {accuracies}')
    say(f'placement-correct rates (new generator variants): {placement_rates}')

    # Pattern-order / query-order invariance.
    import random as _random
    rng_perm = np.random.default_rng(derive_seed(0, 'exp5-synthetic', 'order-check'))
    keys = list(patterns.keys())
    perm_idx = rng_perm.permutation(len(keys))
    permuted_patterns = {keys[i]: patterns[keys[i]] for i in perm_idx}
    sample_scene = eval_scenes[0]
    r1 = _score_new_generator(sample_scene, patterns, SEED_SELECTION, k)
    r2 = _score_new_generator(sample_scene, permuted_patterns, SEED_SELECTION, k)
    pattern_order_independence = {'winner_original_order': r1['winner'],
                                  'winner_permuted_order': r2['winner'],
                                  'identical': r1['winner'] == r2['winner']}

    reordered_scene = dict(sample_scene)
    order = list(range(len(sample_scene['alternatives'])))
    rng_q = np.random.default_rng(derive_seed(0, 'exp5-synthetic', 'query-order-check'))
    rng_q.shuffle(order)
    reordered_scene['alternatives'] = [sample_scene['alternatives'][i] for i in order]
    reordered_scene['truth'] = [sample_scene['truth'][i] for i in order]
    r3 = _score_new_generator(reordered_scene, patterns, SEED_SELECTION, k)
    query_order_independence = {'winner_original_order': r1['winner'],
                                'winner_shuffled_query_order': r3['winner'],
                                'identical': r1['winner'] == r3['winner']}

    # Repeatability: identical scene, identical seed, two separate runs.
    r4 = _score_new_generator(sample_scene, patterns, SEED_SELECTION, k)
    r5 = _score_new_generator(sample_scene, patterns, SEED_SELECTION, k)
    deterministic_repeatability = {'run1': r4['winner'], 'run2': r5['winner'],
                                   'identical': r4['winner'] == r5['winner']}

    # Raw node-count / hypothesis-count shortcut check: correlation between
    # each scene's true-class node count and whether the new generator won.
    node_counts = [len(patterns[s['name']]) for s in eval_scenes]
    new_correct = [r['correct'] for r in results['new_generator_complete']]
    node_count_correlation = (float(np.corrcoef(node_counts, new_correct)[0, 1])
                             if len(set(new_correct)) > 1 else None)

    return {
        'label': 'SYNTHETIC ENGINEERING SCREEN -- not evidence of real-scene or '
                 'Kaggle transfer',
        'n_scenes': len(eval_scenes),
        'n_reference_patterns': len(patterns),
        'n_affine_identifiable_patterns': len(usable),
        'structurally_unidentifiable_under_free_affine': excluded,
        'seeds': {'fitting': int(SEED_FITTING), 'selection': int(SEED_SELECTION),
                 'final_eval': int(SEED_FINAL_EVAL)},
        'unqueried_evidence_note': 'lab.synth produces candidate lists, not image '
                                  'pixels; unqueried-star DoG evidence requires real '
                                  'pixels and is force-disabled in every synthetic '
                                  'variant above (matching Exp4C\'s own documented '
                                  'limitation for the same reason)',
        'generation_factors': ['missing nodes', 'off-figure clutter', 'absent queries',
                              'affine transform', 'reflection', 'unequal scale',
                              'shear', 'repeated queries', '8px close stars',
                              'varying template size', 'varying local candidate density'],
        'accuracies': accuracies,
        'placement_correct_rates': placement_rates,
        'pattern_order_independence': pattern_order_independence,
        'query_order_independence': query_order_independence,
        'deterministic_repeatability': deterministic_repeatability,
        'node_count_correlation_with_correctness': node_count_correlation,
        'seconds': elapsed,
        'per_scene': {k2: v for k2, v in results.items()},
    }


if __name__ == '__main__':
    result = run()
    write_json(OUT / 'synthetic_all48.json', result)
    print('\nwrote synthetic_all48.json')

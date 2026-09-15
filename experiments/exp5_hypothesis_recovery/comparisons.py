"""Combined runner for Stage 2's "required matched ablations" and Stage 4's
"required comparisons" -- both lists overlap substantially (both ask for
with/without fourth-point, graph, held-out-support, unqueried evidence,
multiplicity correction, and primary-vs-repeat seed), so this module computes
every beam-search variant ONCE per scene/seed and reports it under both
framings, per the task's own token/compute-efficiency instruction.

All comparisons use IDENTICAL stored inputs (same classical alternatives,
same image, same seed) -- only ONE factor changes between adjacent
comparisons, exactly matching the task's ablation discipline.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import OUT, ROOT, SCENES, SEEDS
from experiments.exp1.env import write_json

TOLERANCE = 18.0
PER_CLASS_BUDGET = 300


def _true_constellation(scene: str) -> str:
    from constellation.contracts import read_truth
    return read_truth(ROOT / 'train_ground_truth.csv')[scene].constellation


def _placement_correct(scene: str, class_name: str, mapped_nodes) -> dict:
    """Reused verbatim from Experiment 4C -- the exact, already-frozen
    correct-placement rule (>=4 greedily-matched true figure stars within
    12px, or all of them if fewer than 4 exist, AND the class name correct).
    """
    from experiments.exp4c_calibrated_fusion.hypothesis_dataset import placement_label
    return placement_label(scene, class_name, mapped_nodes)


def _production_result(scene: str) -> dict:
    doc = json.load(open(ROOT / 'outputs' / 'joint_train' / f'{scene}.json'))
    return {'winner': doc['constellation'], 'correct': doc['constellation'] == _true_constellation(scene)}


def _exp4b_result(scene: str, patterns: dict, seed: int) -> dict:
    from experiments.exp4b_joint_solver.independent_scorer import score_scene
    from experiments.exp4b_joint_solver.independent_support import _classical_alternatives
    alternatives = _classical_alternatives(scene)
    true_name = _true_constellation(scene)
    result = score_scene(alternatives, patterns, seed=seed, ablation='held_out_only')
    names = [r['name'] for r in result['ranked']]
    rank = names.index(true_name) + 1 if true_name in names else None
    return {'winner': result['winner'], 'correct': result['winner'] == true_name, 'true_class_rank': rank}


def _classical_alternatives_k(scene: str, k: int) -> list:
    import numpy as np
    from experiments.exp1.data import query_records
    from experiments.exp1.evaluation import _aligned_set, load_real_aligned
    from experiments.exp1.stages import Paths

    node = load_real_aligned(Paths(ROOT / 'outputs' / 'exp1'), scene)
    records = query_records(scene)
    alternatives = []
    for r in records:
        aligned = _aligned_set(node, r['index'])
        valid = np.asarray(aligned.admissible) & ~np.asarray(aligned.low_info)
        if not valid.any():
            alternatives.append([])
            continue
        order = np.argsort(-np.where(valid, aligned.ncc, -np.inf))[:k]
        cands = [(float(aligned.xy[i, 0]), float(aligned.xy[i, 1]),
                  float(aligned.ncc[i]), float(aligned.poses[i, 0]), float(aligned.poses[i, 1]))
                 for i in order if valid[i]]
        alternatives.append(cands)
    return alternatives


def _exp3_alternatives_k(scene: str, seed: int, k: int) -> list | None:
    """Exp3 rank-1-per-fold candidates, widened to the top-k by classical NCC
    (Exp3's fixed-arm-F policy ranks presence/best-candidate, not a full
    per-candidate-logit slate at this granularity, so this uses the same
    classical bank restricted to whatever the corresponding real
    fixed_policy rows file reports -- if absent, reports unavailable rather
    than fabricating a slate)."""
    rows_path = ROOT / 'outputs' / 'exp4_joint_identification' / 'folds' / f'fixed_policy_{scene}_both_seeds_rows.json'
    if not rows_path.exists():
        return None
    return _classical_alternatives_k(scene, k)   # Exp3's own logits are per-fold-checkpoint;
    # a fair like-for-like "Exp3 candidates" bank at hypothesis-generation
    # granularity is the SAME classical positions with Exp3's presence
    # decision layered on identification-only elsewhere -- recorded here as
    # the classical bank (comparison 3 already covers classical; comparison
    # 4 is reported identically and explicitly noted as such, since Exp3
    # does not supply an independent per-candidate coordinate different from
    # the classical bank's own candidate set at this stage).


def run_variant(scene: str, patterns: dict, seed: int, k: int, image, **flags) -> dict:
    from experiments.exp5_hypothesis_recovery.geometric_recovery import run_beam_search
    alternatives = _classical_alternatives_k(scene, k)
    true_name = _true_constellation(scene)
    result = run_beam_search(alternatives, patterns, image, seed=seed, k=k,
                             tolerance=TOLERANCE, per_class_budget=PER_CLASS_BUDGET, **flags)
    winner = result.get('winner')
    label = (_placement_correct(scene, winner, result['winner_hypothesis']['mapped_nodes'])
            if winner and result.get('winner_hypothesis') else
            {'class_correct': False, 'placement_correct': False,
             'n_figure_matches_12px': 0, 'n_figure_truth': None, 'required_figure_matches': None})
    return {
        'winner': winner, 'class_correct': label['class_correct'],
        'placement_correct': label['placement_correct'],
        'n_figure_matches_12px': label['n_figure_matches_12px'],
        'required_figure_matches': label['required_figure_matches'],
        'held_out_support': (result['winner_hypothesis']['held_out_support']
                             if result.get('winner_hypothesis') else None),
        'expansions': result['expansions'], 'pruned': result['pruned'],
        'cap_hits': result['cap_hits'], 'seconds': result['seconds'],
        'peak_memory_bytes': result['peak_memory_bytes'], 'beam_width': result['beam_width'],
        'k': k,
    }


def run_scene(scene: str, patterns: dict, seed: int, image, say=print) -> dict:
    out = {}
    out['1_production'] = _production_result(scene)
    say(f'  [1] production: {out["1_production"]}')
    out['2_exp4b_hypothesis_generator'] = _exp4b_result(scene, patterns, seed)
    say(f'  [2] exp4b generator: {out["2_exp4b_hypothesis_generator"]}')

    from . import K_GRID
    out['3_new_generator_classical_candidates'] = run_variant(scene, patterns, seed, min(K_GRID), image)
    say(f'  [3] new generator (classical, k={out["3_new_generator_classical_candidates"]["k"]}): '
       f'{out["3_new_generator_classical_candidates"]["winner"]} '
       f'placement_correct={out["3_new_generator_classical_candidates"]["placement_correct"]}')

    # comparison 4 (Exp3 candidates): reported identically to comparison 3 with
    # an explicit note (see `_exp3_alternatives_k`'s docstring) rather than a
    # fabricated independent slate.
    out['4_new_generator_exp3_candidates'] = dict(out['3_new_generator_classical_candidates'])
    out['4_new_generator_exp3_candidates']['note'] = (
        'Exp3 does not supply an independent per-candidate coordinate set distinct '
        'from the classical bank at hypothesis-generation granularity (its own '
        'signal is a per-query best-candidate + presence decision, already covered '
        'as an identification-only policy elsewhere) -- reported identically to '
        'comparison 3 rather than fabricating a different candidate slate.')

    out['5_new_generator_full_multi_candidate_bank'] = run_variant(scene, patterns, seed, max(K_GRID), image)
    say(f'  [5] full multi-candidate bank (k={max(K_GRID)}): '
       f'{out["5_new_generator_full_multi_candidate_bank"]["winner"]} '
       f'placement_correct={out["5_new_generator_full_multi_candidate_bank"]["placement_correct"]}')

    out['6_without_barycentric_validation'] = run_variant(scene, patterns, seed, min(K_GRID), image,
                                                          use_fourth_point=False)
    say(f'  [6] without barycentric validation: {out["6_without_barycentric_validation"]["winner"]}')

    out['7_without_graph_evidence'] = run_variant(scene, patterns, seed, min(K_GRID), image, use_graph=False)
    say(f'  [7] without graph evidence: {out["7_without_graph_evidence"]["winner"]}')

    out['8_without_held_out_support'] = run_variant(scene, patterns, seed, min(K_GRID), image,
                                                    use_held_out_support=False)
    say(f'  [8] without held-out support: {out["8_without_held_out_support"]["winner"]}')

    out['9_without_unqueried_evidence'] = run_variant(scene, patterns, seed, min(K_GRID), image,
                                                      use_unqueried=False)
    say(f'  [9] without unqueried evidence: {out["9_without_unqueried_evidence"]["winner"]}')

    out['10_without_multiplicity_correction'] = run_variant(scene, patterns, seed, min(K_GRID), image,
                                                            use_unqueried=True,
                                                            use_multiplicity_correction=False)
    say(f'  [10] without multiplicity correction: {out["10_without_multiplicity_correction"]["winner"]}')

    # "complete" variant with every mechanism enabled, at the primary K.
    out['11_complete_new_generator'] = run_variant(scene, patterns, seed, min(K_GRID), image)
    say(f'  [11] complete new generator: {out["11_complete_new_generator"]["winner"]} '
       f'placement_correct={out["11_complete_new_generator"]["placement_correct"]}')

    return out


def run(say=print) -> dict:
    from constellation.references import extract_patterns
    from experiments.exp1.data import load_scene
    patterns = extract_patterns(ROOT / 'patterns')
    out = {}
    for scene in SCENES:
        image = load_scene(scene).image
        out[scene] = {}
        for seed in SEEDS:
            say(f'\n===== comparisons: {scene} / seed {seed} =====')
            out[scene][str(seed)] = run_scene(scene, patterns, seed, image, say=say)
    return out


if __name__ == '__main__':
    result = run()
    write_json(OUT / 'geometry_ablations.json', result)
    print('\nwrote geometry_ablations.json')

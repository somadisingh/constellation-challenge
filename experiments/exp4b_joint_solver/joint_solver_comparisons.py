"""Phase 4 runner: the 10 required matched comparisons, changing exactly one
factor between adjacent comparisons, on all 3 real labelled scenes.

  1. existing production recognizer (frozen `outputs/joint_train/{scene}.json`)
  2. existing `recognize_joint` (re-run directly, classical alternatives)
  3. new independent scorer, classical rank-1 only (single point per query)
  4. new independent scorer, full classical slates
  5. joint solver, classical evidence, no unqueried, no appearance
  6. joint solver, classical evidence + appearance
  7. joint solver, + appearance, WITHOUT unqueried-star evidence (baseline for 8)
  8. joint solver, + appearance, WITH unqueried-star evidence
  9. joint solver, + appearance + unqueried, WITHOUT multiple-testing correction
  10. complete joint solver (appearance + unqueried + correction)

NOTE ON A KNOWN LIMITATION SURFACED BY THIS RUN: the joint solver's additive
composite (appearance_score + held_out_geometric_support + unqueried_evidence)
mixes terms on very different natural scales (support is an integer count of
matched pairs, typically 1-10; the corrected z-score-sum evidence can exceed
10 even after the sqrt(n) correction). This can let a hypothesis with very
weak geometric support (1-2 matched pairs) outscore a much better-supported
one purely on unqueried-node evidence. This is reported as measured, not
patched with an undisclosed reweighting -- see EXPERIMENT4B_REPORT.md's
limitations section.
"""
from __future__ import annotations

import time

import numpy as np

from . import OUT, SCENES
from experiments.exp1.data import load_scene, query_records
from experiments.exp1.env import ROOT, write_json


def _true_constellation(scene: str) -> str:
    from constellation.contracts import read_truth
    return read_truth(ROOT / 'train_ground_truth.csv')[scene].constellation


def _production_result(scene: str) -> dict:
    import json
    doc = json.load(open(ROOT / 'outputs' / 'joint_train' / f'{scene}.json'))
    return {'winner': doc['constellation'], 'correct': doc['constellation'] == _true_constellation(scene)}


def _recognize_joint_result(scene: str, patterns: dict) -> dict:
    from constellation.joint import recognize_joint
    from .independent_support import _classical_alternatives
    alternatives = _classical_alternatives(scene)
    name, chosen, diag = recognize_joint(alternatives, patterns, diag_top=48)
    hyps = diag.get('hypotheses', [])
    ranked = sorted(hyps, key=lambda h: (-h.get('score', -1e9), h.get('name', '')))
    names = [h.get('name') for h in ranked]
    true_name = _true_constellation(scene)
    rank = names.index(true_name) + 1 if true_name in names else None
    return {'winner': name, 'correct': name == true_name, 'true_class_rank': rank}


def _independent_scorer_result(scene: str, patterns: dict, seed: int, rank1_only: bool) -> dict:
    from .independent_scorer import score_scene
    from .independent_support import _classical_alternatives
    from experiments.exp1.evaluation import _aligned_set, load_real_aligned
    from experiments.exp1.stages import Paths

    if rank1_only:
        node = load_real_aligned(Paths(ROOT / 'outputs' / 'exp1'), scene)
        records = query_records(scene)
        alternatives = []
        for r in records:
            aligned = _aligned_set(node, r['index'])
            valid = aligned.admissible & ~aligned.low_info
            if not valid.any():
                alternatives.append([])
                continue
            best = int(np.argmax(np.where(valid, aligned.ncc, -np.inf)))
            alternatives.append([(float(aligned.xy[best, 0]), float(aligned.xy[best, 1]),
                                float(aligned.ncc[best]), float(aligned.poses[best, 0]),
                                float(aligned.poses[best, 1]))])
    else:
        alternatives = _classical_alternatives(scene)

    result = score_scene(alternatives, patterns, seed=seed, ablation='held_out_only')
    true_name = _true_constellation(scene)
    names = [r['name'] for r in result['ranked']]
    rank = names.index(true_name) + 1 if true_name in names else None
    return {'winner': result['winner'], 'correct': result['winner'] == true_name,
           'true_class_rank': rank}


def _joint_solver_result(scene: str, patterns: dict, seed: int, **kwargs) -> dict:
    from .joint_solver import run_beam_search
    from .independent_support import _classical_alternatives
    alternatives = _classical_alternatives(scene)
    image = load_scene(scene).image
    result = run_beam_search(alternatives, patterns, image, seed=seed, **kwargs)
    true_name = _true_constellation(scene)
    return {'winner': result['winner'], 'correct': result['winner'] == true_name,
           'expansions': result['expansions'], 'pruned': result['pruned'],
           'cap_hits': result['cap_hits'], 'seconds': result['seconds'],
           'peak_memory_bytes': result['peak_memory_bytes'],
           'beam_width': result['beam_width']}


def run_scene(scene: str, patterns: dict, seed: int, say=print) -> dict:
    out = {}
    out['1_production_recognizer'] = _production_result(scene)
    say(f'  [1] production: {out["1_production_recognizer"]}')
    out['2_recognize_joint'] = _recognize_joint_result(scene, patterns)
    say(f'  [2] recognize_joint: {out["2_recognize_joint"]}')
    out['3_independent_rank1_only'] = _independent_scorer_result(scene, patterns, seed, True)
    say(f'  [3] independent (rank1 only): {out["3_independent_rank1_only"]}')
    out['4_independent_full_slate'] = _independent_scorer_result(scene, patterns, seed, False)
    say(f'  [4] independent (full slate): {out["4_independent_full_slate"]}')

    out['5_joint_classical_only'] = _joint_solver_result(
        scene, patterns, seed, use_appearance=False, use_unqueried=False,
        use_multiple_testing_correction=False)
    say(f'  [5] joint (classical geometry only): {out["5_joint_classical_only"]}')

    out['6_joint_plus_appearance'] = _joint_solver_result(
        scene, patterns, seed, use_appearance=True, use_unqueried=False,
        use_multiple_testing_correction=False)
    say(f'  [6] joint (+ appearance): {out["6_joint_plus_appearance"]}')

    out['7_joint_appearance_no_unqueried'] = out['6_joint_plus_appearance']

    out['8_joint_appearance_plus_unqueried'] = _joint_solver_result(
        scene, patterns, seed, use_appearance=True, use_unqueried=True,
        use_multiple_testing_correction=False)
    say(f'  [8] joint (+ appearance + unqueried, uncorrected): '
       f'{out["8_joint_appearance_plus_unqueried"]}')

    out['9_joint_no_correction'] = out['8_joint_appearance_plus_unqueried']

    out['10_complete_joint_solver'] = _joint_solver_result(
        scene, patterns, seed, use_appearance=True, use_unqueried=True,
        use_multiple_testing_correction=True)
    say(f'  [10] complete joint solver: {out["10_complete_joint_solver"]}')

    return out


def run(seed: int = 31004, say=print) -> dict:
    from constellation.references import extract_patterns
    patterns = extract_patterns(ROOT / 'patterns')
    out = {}
    for scene in SCENES:
        say(f'\n===== joint solver comparisons: {scene} =====')
        out[scene] = run_scene(scene, patterns, seed, say=say)
    return out


if __name__ == '__main__':
    result = run()
    write_json(OUT / 'joint_solver_primary.json', result)
    print('\nwrote joint_solver_primary.json')

"""Phase 2 runner: matched ablations of the independent scorer across all 3
real labelled scenes, using the classical top-1 candidate slate (matching
`headroom_oracles.py`'s level-5 input) as the alternatives source, so results
are directly comparable to Experiment 4's own headroom numbers.

Ablations 5 ("held_out_appearance") and 6 ("all above with template-size
normalization") from the task spec:
  - `held_out_appearance` is NOT separately implemented: no appearance score
    is currently plumbed into `independent_scorer.py`'s hypothesis scoring
    (it only uses geometric support). This is recorded explicitly as
    `not_implemented`, not silently skipped or faked.
  - template-size normalization IS implemented, reusing `constellation.joint.
    decorrelate_size` verbatim (the existing, already-tested size-bias
    correction), applied as a further pass over the held_out_only ablation.
"""
from __future__ import annotations

import time

import numpy as np

from . import OUT, SCENES
from experiments.exp1.env import ROOT, write_json


def _classical_alternatives(scene: str) -> list:
    from experiments.exp1.data import query_records
    from experiments.exp1.evaluation import _aligned_set, load_real_aligned
    from experiments.exp1.stages import Paths

    node = load_real_aligned(Paths(ROOT / 'outputs' / 'exp1'), scene)
    records = query_records(scene)
    alternatives = []
    for r in records:
        aligned = _aligned_set(node, r['index'])
        valid = aligned.admissible & ~aligned.low_info
        if not valid.any():
            alternatives.append([])
            continue
        order = np.argsort(-np.where(valid, aligned.ncc, -np.inf))
        cands = [(float(aligned.xy[i, 0]), float(aligned.xy[i, 1]),
                  float(aligned.ncc[i]) if valid[i] else -1e9,
                  float(aligned.poses[i, 0]), float(aligned.poses[i, 1]))
                 for i in order if valid[i]]
        alternatives.append(cands)
    return alternatives


def _true_constellation(scene: str) -> str:
    from constellation.contracts import read_truth
    return read_truth(ROOT / 'train_ground_truth.csv')[scene].constellation


def _rank_of(ranked: list, name: str) -> int | None:
    names = [r['name'] for r in ranked]
    return (names.index(name) + 1) if name in names else None


def run_scene(scene: str, patterns: dict, seed: int, say=print) -> dict:
    from .independent_scorer import score_scene
    from constellation.joint import decorrelate_size

    alternatives = _classical_alternatives(scene)
    true_name = _true_constellation(scene)
    out = {}
    for ablation in ('existing_score', 'seed_removed', 'held_out_only', 'held_out_stability'):
        started = time.perf_counter()
        result = score_scene(alternatives, patterns, seed=seed, ablation=ablation)
        elapsed = time.perf_counter() - started
        rank = _rank_of(result['ranked'], true_name)
        out[ablation] = {
            'winner': result['winner'], 'winner_score': result['winner_score'],
            'true_class_rank': rank, 'correct': result['winner'] == true_name,
            'top5': [r['name'] for r in result['ranked'][:5]],
            'seconds': elapsed,
        }
        say(f'  [{scene}/{ablation}] winner={result["winner"]} true_rank={rank} '
           f'correct={result["winner"] == true_name} ({elapsed:.1f}s)')

    # held_out_appearance: not implemented (see module docstring)
    out['held_out_appearance'] = {'status': 'not_implemented',
                                  'reason': 'No appearance score is plumbed into '
                                          'independent_scorer.py; this ablation would '
                                          'require adding a candidate-appearance term '
                                          '(e.g. mean classical NCC of held-out-matched '
                                          'pairs) to hyp_score, which was not built in '
                                          'this pass.'}

    # template-size normalization on top of held_out_only
    started = time.perf_counter()
    result = score_scene(alternatives, patterns, seed=seed, ablation='held_out_only')
    hyps_for_decorrelate = []
    for name, node in result['hypotheses_by_class'].items():
        if node['hypotheses']:
            best = max(node['hypotheses'], key=lambda h: h['held_out_support'])
            hyps_for_decorrelate.append({'name': name, 'score': best['held_out_support'],
                                        'support': best['total_support'],
                                        'nodes_count': len(best['mapped_nodes'])})
        else:
            hyps_for_decorrelate.append({'name': name, 'score': 0, 'support': 0,
                                        'nodes_count': 0})
    decorrelate_size(hyps_for_decorrelate, min_support=4)
    hyps_for_decorrelate.sort(key=lambda h: (-h['score'], h['name']))
    elapsed = time.perf_counter() - started
    rank_decorr = _rank_of(hyps_for_decorrelate, true_name)
    out['held_out_with_size_normalization'] = {
        'winner': hyps_for_decorrelate[0]['name'] if hyps_for_decorrelate else None,
        'true_class_rank': rank_decorr,
        'correct': (hyps_for_decorrelate[0]['name'] == true_name) if hyps_for_decorrelate else False,
        'top5': [h['name'] for h in hyps_for_decorrelate[:5]],
        'seconds': elapsed,
    }
    say(f'  [{scene}/held_out_with_size_normalization] winner='
       f'{out["held_out_with_size_normalization"]["winner"]} true_rank={rank_decorr}')
    return out


def run(seed: int = 31004, say=print) -> dict:
    from constellation.references import extract_patterns
    patterns = extract_patterns(ROOT / 'patterns')
    out = {}
    for scene in SCENES:
        say(f'\n===== independent support: {scene} =====')
        out[scene] = run_scene(scene, patterns, seed, say=say)
    return out


if __name__ == '__main__':
    result = run()
    write_json(OUT / 'independent_support.json', result)
    print('\nwrote independent_support.json')

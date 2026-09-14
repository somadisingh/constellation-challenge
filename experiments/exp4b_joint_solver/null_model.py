"""Phase 3 runner: 5 matched ablations of unqueried-star evidence, combined
with the Phase 2 held-out-support score, evaluated on all 3 real scenes.

The combined hypothesis score for ablation X is:
    held_out_support (Phase 2's own best ablation) + unqueried_evidence_score(X)
so every ablation differs by EXACTLY one factor (which unqueried-evidence
term is added), holding the geometric term fixed, per the task's "change only
one factor between matched comparisons" rule.
"""
from __future__ import annotations

import time

import numpy as np

from . import OUT, SCENES
from experiments.exp1.data import load_scene
from experiments.exp1.env import ROOT, write_json

ABLATIONS = ('none', 'raw_count', 'quality_weighted', 'matched_null_lr',
            'matched_null_lr_corrected')


def _true_constellation(scene: str) -> str:
    from constellation.contracts import read_truth
    return read_truth(ROOT / 'train_ground_truth.csv')[scene].constellation


def run_scene(scene: str, patterns: dict, seed: int, say=print) -> dict:
    from .independent_scorer import score_scene
    from .unqueried_star_evidence import score_unqueried_nodes
    from .independent_support import _classical_alternatives

    image = load_scene(scene).image
    alternatives = _classical_alternatives(scene)
    true_name = _true_constellation(scene)

    geo = score_scene(alternatives, patterns, seed=seed, ablation='held_out_only')
    out = {}
    for ablation in ABLATIONS:
        started = time.perf_counter()
        combined = []
        for name, node in geo['hypotheses_by_class'].items():
            if not node['hypotheses']:
                combined.append({'name': name, 'score': -1e9})
                continue
            best = max(node['hypotheses'], key=lambda h: h['held_out_support'])
            mapped = np.array(best['mapped_nodes'], float)
            ev = score_unqueried_nodes(image, mapped, best['total_pairs'],
                                       np.array(geo.get('pool_size', 0)), seed,
                                       f'{scene}:{name}', ablation=ablation)
            combined.append({'name': name, 'score': best['held_out_support'] + ev['evidence_score'],
                            'geometric_support': best['held_out_support'],
                            'unqueried_evidence': ev['evidence_score'],
                            'n_unqueried_scored': ev['n_scored']})
        combined.sort(key=lambda h: (-h['score'], h['name']))
        elapsed = time.perf_counter() - started
        rank = next((i + 1 for i, h in enumerate(combined) if h['name'] == true_name), None)
        winner = combined[0]['name'] if combined else None
        out[ablation] = {'winner': winner, 'true_class_rank': rank,
                         'correct': winner == true_name,
                         'top5': [h['name'] for h in combined[:5]],
                         'seconds': elapsed}
        say(f'  [{scene}/{ablation}] winner={winner} true_rank={rank} '
           f'correct={winner == true_name} ({elapsed:.1f}s)')
    return out


def run(seed: int = 31004, say=print) -> dict:
    from constellation.references import extract_patterns
    patterns = extract_patterns(ROOT / 'patterns')
    out = {}
    for scene in SCENES:
        say(f'\n===== unqueried-star / null model: {scene} =====')
        out[scene] = run_scene(scene, patterns, seed, say=say)
    return out


if __name__ == '__main__':
    result = run()
    write_json(OUT / 'unqueried_star_evidence.json', result)
    write_json(OUT / 'null_model.json', {
        'ablations': list(ABLATIONS),
        'multiple_testing_correction': (
            'matched_null_lr_corrected divides the summed z-score evidence by '
            'sqrt(n_unqueried_nodes_searched), so a reference offering more '
            'unqueried nodes to search does not get an automatic score advantage '
            'purely from having more chances to find a clutter peak -- this is the '
            'exact large-template bias (documented in lab/LEDGER.md as a +0.660 '
            'wrong-class score-vs-node-count correlation) this correction targets.'),
        'results': result,
    })
    print('\nwrote unqueried_star_evidence.json and null_model.json')

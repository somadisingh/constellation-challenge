"""Mechanism diagnostics on the three repeatedly used development skies."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from constellation.contracts import read_truth
from experiments.exp1.env import sha256_file
from experiments.exp5c_affine_recovery import OUT, SCENES, TRAIN_GROUND_TRUTH, DEFAULT_CONFIG
from experiments.exp5c_affine_recovery.confidence import gate_decisions
from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from experiments.exp5c_affine_recovery.solver import alternatives_from_prediction_json, solve
from experiments.exp5c_affine_recovery.synthetic import atomic_json


SOURCE = OUT / 'reconstructed_train'
EVALUATION_ROLES = {
    'pisces': 'development',
    'scorpius': 'development',
    'taurus': 'adversarial_diagnostic_only',
}


def _greedy_matches(mapped, figure, tolerance=12.0):
    if not len(mapped) or not len(figure): return 0
    d = np.linalg.norm(mapped[:, None, :] - figure[None, :, :], axis=2)
    cand = np.argwhere(d < tolerance)
    order = np.argsort(d[cand[:, 0], cand[:, 1]], kind='stable')
    a, b = set(), set()
    for z in order:
        i, j = map(int, cand[z])
        if i not in a and j not in b: a.add(i); b.add(j)
    return len(a)


def run(policy_path: Path | None = None, say=print) -> dict:
    index = get_or_create_index()
    truth = read_truth(TRAIN_GROUND_TRUTH)
    policy_path = policy_path or OUT / 'deployment_policy.json'
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {
        'model': {}, 'primary_gate': DEFAULT_CONFIG['primary_gate']}
    rows = {}
    for scene in SCENES:
        source = SOURCE / f'{scene}.json'
        doc = json.loads(source.read_text())
        alternatives = alternatives_from_prediction_json(doc)
        result = solve(alternatives, index, scene_shape=(3000, 3000), seed=42001)
        target = truth[scene]
        figure_rows = [(i, p) for i, p in enumerate(target.patches)
                       if p is not None and int(p[2]) == 1]
        figure = np.asarray([p[:2] for _, p in figure_rows], float)
        candidate_hits = []
        for qi, p in figure_rows:
            hit = any(np.linalg.norm(np.asarray(c[:2]) - np.asarray(p[:2])) <= 12
                      for c in alternatives[qi][:5])
            candidate_hits.append(bool(hit))
        names = [h['name'] for h in result['ranked']]
        true_h = next((h for h in result['ranked'] if h['name'] == target.constellation), None)
        placement_matches = (_greedy_matches(np.asarray(true_h['mapped_nodes']), figure)
                             if true_h else 0)
        decisions = gate_decisions(result, policy.get('model'), policy)
        accepted = decisions['primary']
        final_name = result['winner'] if accepted else doc['constellation']
        trace = {
            'correct_candidate_exists': all(candidate_hits),
            'correct_candidate_hits_top5': sum(candidate_hits),
            'correct_candidate_total': len(candidate_hits),
            'correct_quad_can_form': sum(candidate_hits) >= 4,
            'correct_descriptor_retrieved': true_h is not None,
            'transform_fit_valid': bool(true_h),
            'correct_hypothesis_survives_budget': placement_matches >= min(4, len(figure)),
            'correct_placement_matches_12px': placement_matches,
            'held_out_support_found': None if not true_h else true_h['held_out_support'],
            'graph_support_found': None if not true_h else true_h['graph']['supported_edges'],
            'confidence_gate_accepts': accepted,
            'final_hybrid_decision': final_name,
        }
        rows[scene] = {
            'scene': scene, 'true_name': target.constellation,
            'evaluation_role': EVALUATION_ROLES[scene],
            'existing_name': doc['constellation'], 'affine_winner': result['winner'],
            'true_rank': names.index(target.constellation) + 1 if target.constellation in names else None,
            'final_name': final_name, 'accepted': accepted,
            'correct_final': final_name == target.constellation,
            'decisions': decisions, 'trace': trace,
            'winner': result['ranked'][0] if result['ranked'] else None,
            'runtime_seconds': result['runtime_seconds'],
            'peak_rss_platform_units': result['peak_rss_platform_units'],
            'proposals_evaluated': result['proposals_evaluated'],
            'candidate_source_sha256': sha256_file(source),
            'stream_summary': result['streams'],
        }
        say(f"{scene}: {doc['constellation']} -> {result['winner']} accepted={accepted} final={final_name}")
    promotion_scenes = [scene for scene in SCENES
                        if EVALUATION_ROLES[scene] == 'development']
    out = {'status': 'COMPLETE', 'scope': 'development_only_not_unbiased',
           'config': DEFAULT_CONFIG, 'scenes': rows,
           'promotion_development_scenes': promotion_scenes,
           'excluded_from_validation_and_promotion': ['taurus'],
           'no_regressions': all(rows[scene]['correct_final']
                                 for scene in promotion_scenes)}
    atomic_json(OUT / 'development_results.json', out)
    atomic_json(OUT / 'proposal_recall.json', {
        'status': 'COMPLETE', 'development': {s: r['trace'] for s, r in rows.items()},
        'note': 'Development traces use truth only for post-inference mechanism attribution.'})
    return out


if __name__ == '__main__':
    run()

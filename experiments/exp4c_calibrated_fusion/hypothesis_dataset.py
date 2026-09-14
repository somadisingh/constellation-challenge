"""Build the frozen hypothesis-level table used by every Exp4C arm.

Geometry is generated once from Exp4B's classical slates.  Exp4C only adds
features and labels; model arms never regenerate or select a different pool.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import OUT, ROOT, SCENES, SEEDS, PLACEMENT_CORRECT_PX
from experiments.exp1.env import write_json

FEATURES = (
    'classical_appearance', 'exp3_appearance', 'exp3_probability',
    'exp3_disagreement', 'rank_fusion_appearance', 'total_support', 'seed_support',
    'held_out_support', 'support_fraction', 'mean_residual',
    'stability_mean_shift_px', 'unique_sources', 'coverage',
    'unmatched_nodes', 'corrected_unqueried', 'unqueried_raw_count',
    'n_unqueried', 'n_unqueried_scored', 'node_count', 'pool_size',
    'attempted_hypotheses', 'accepted_hypotheses', 'cap_hit',
    'local_star_density', 'multiple_testing_term',
)

# Machine-readable provenance for every implemented feature.  Several fields
# requested by the experiment are absent from the frozen Exp4B hypothesis
# cache; those are recorded explicitly below rather than fabricated.
FEATURE_SOURCES = {
    'classical_appearance': 'Exp4B frozen classical candidate scores averaged over total_pairs',
    'exp3_appearance': 'Exp3 arm-F fold/seed candidate pair logits, attached by appearance_augmentation.py',
    'exp3_probability': 'sigmoid of the attached Exp3 aggregate logit',
    'exp3_disagreement': 'standard deviation across matching Exp3 candidate logits',
    'rank_fusion_appearance': 'within-record z-normalized classical plus Exp3 evidence',
    'total_support': 'Exp4B independent_scorer hypothesis total_support',
    'seed_support': 'Exp4B independent_scorer hypothesis seed_support',
    'held_out_support': 'Exp4B seed-excluded hypothesis held_out_support',
    'support_fraction': 'total_support / reference-node count',
    'mean_residual': 'Exp4B hypothesis mean_residual',
    'stability_mean_shift_px': 'Exp4B transform perturbation stability statistic',
    'unique_sources': 'unique pool indices in Exp4B total_pairs',
    'coverage': 'total_support / reference-node count',
    'unmatched_nodes': 'reference-node count minus uniquely assigned reference nodes',
    'corrected_unqueried': 'Exp4B matched_null_lr_corrected evidence_score recomputed on frozen hypothesis',
    'unqueried_raw_count': 'Exp4B raw_count evidence_score recomputed on frozen hypothesis',
    'n_unqueried': 'Exp4B unqueried-star evaluator n_unqueried',
    'n_unqueried_scored': 'Exp4B unqueried-star evaluator n_scored',
    'node_count': 'number of mapped reference nodes',
    'pool_size': 'size of frozen Exp4B candidate pool',
    'attempted_hypotheses': 'Exp4B per-class attempted hypothesis count',
    'accepted_hypotheses': 'Exp4B per-class accepted hypothesis count',
    'cap_hit': 'Exp4B per-class hypothesis cap indicator',
    'local_star_density': 'candidate pool size / 3000^2',
    'multiple_testing_term': 'log1p(attempted_hypotheses)',
}


def _truth():
    from constellation.contracts import read_truth
    return read_truth(ROOT / 'train_ground_truth.csv')


def _pool_for_scene(scene: str):
    from constellation.geometry import consolidate
    from constellation.joint import build_pool
    from constellation.slate import slate_from_candidates
    from experiments.exp4b_joint_solver.independent_support import _classical_alternatives

    alternatives = _classical_alternatives(scene)
    slates = [slate_from_candidates(x) for x in alternatives]
    anchors = np.array([s.seed_xy() for s in slates if len(s)]).reshape(-1, 2)
    _, groups = consolidate(anchors)
    live = [i for i, s in enumerate(slates) if len(s)]
    pool, tags, scores, ranks = build_pool([slates[i] for i in live], groups)
    return alternatives, pool, tags, scores, ranks


def _greedy_matches(mapped: np.ndarray, truth_xy: np.ndarray, radius: float) -> int:
    if not len(mapped) or not len(truth_xy):
        return 0
    d = np.linalg.norm(mapped[:, None, :] - truth_xy[None, :, :], axis=2)
    used_a, used_b, count = set(), set(), 0
    for flat in np.argsort(d, axis=None, kind='stable'):
        i, j = np.unravel_index(flat, d.shape)
        if d[i, j] > radius:
            break
        if i not in used_a and j not in used_b:
            used_a.add(int(i)); used_b.add(int(j)); count += 1
    return count


def placement_label(scene: str, class_name: str, mapped_nodes: list) -> dict:
    truth = _truth()[scene]
    fig = np.array([p[:2] for p in truth.patches if p is not None and p[2] == 1], float)
    mapped = np.asarray(mapped_nodes, float).reshape(-1, 2)
    n_match = _greedy_matches(mapped, fig, PLACEMENT_CORRECT_PX)
    # Four independently recovered issued figure stars are sufficient to
    # validate an affine placement: three determine it and the fourth is
    # independent evidence.  Requiring a fraction of every issued node would
    # incorrectly reject a geometrically correct partial placement, which is
    # explicitly allowed by the competition.
    required = min(4, len(fig)) if len(fig) else 4
    class_correct = class_name == truth.constellation
    placement_correct = bool(class_correct and n_match >= required)
    return {'class_correct': class_correct, 'placement_correct': placement_correct,
            'n_figure_matches_12px': n_match, 'n_figure_truth': int(len(fig)),
            'required_figure_matches': required}


def _hypothesis_features(scene: str, class_name: str, hyp: dict, node: dict,
                         image: np.ndarray, pool: np.ndarray,
                         pool_scores: np.ndarray, seed: int, pool_size: int) -> dict:
    from experiments.exp4b_joint_solver.unqueried_star_evidence import score_unqueried_nodes

    pair_idx = [j for _, j in hyp['total_pairs'] if 0 <= j < len(pool_scores)]
    appearance = float(np.mean(pool_scores[pair_idx])) if pair_idx else 0.0
    corrected = score_unqueried_nodes(
        image, np.asarray(hyp['mapped_nodes'], float), hyp['total_pairs'], pool,
        seed, f'exp4c:{scene}:{class_name}:{hyp["seed_triple"]}',
        ablation='matched_null_lr_corrected')
    raw = score_unqueried_nodes(
        image, np.asarray(hyp['mapped_nodes'], float), hyp['total_pairs'], pool,
        seed, f'exp4c:{scene}:{class_name}:{hyp["seed_triple"]}',
        ablation='raw_count')
    node_count = len(hyp['mapped_nodes'])
    total = int(hyp['total_support'])
    held = int(hyp['held_out_support'])
    residual = hyp.get('mean_residual')
    stability = hyp.get('stability_mean_shift_px')
    values = {
        'classical_appearance': appearance,
        # Fold-specific Exp3 values are attached later by
        # appearance_augmentation.py; geometry is generated only once.
        'exp3_appearance': None, 'exp3_probability': None,
        'exp3_disagreement': None, 'rank_fusion_appearance': None,
        'total_support': total,
        'seed_support': int(hyp['seed_support']),
        'held_out_support': held,
        'support_fraction': total / max(node_count, 1),
        'mean_residual': float(residual) if residual is not None else None,
        'stability_mean_shift_px': float(stability) if stability is not None else None,
        'unique_sources': len({int(j) for _, j in hyp['total_pairs']}),
        'coverage': total / max(node_count, 1),
        'unmatched_nodes': node_count - len({int(i) for i, _ in hyp['total_pairs']}),
        'corrected_unqueried': float(corrected['evidence_score']),
        'unqueried_raw_count': float(raw['evidence_score']),
        'n_unqueried': int(corrected['n_unqueried']),
        'n_unqueried_scored': int(corrected['n_scored']),
        'node_count': node_count,
        'pool_size': pool_size,
        'attempted_hypotheses': int(node['attempted']),
        'accepted_hypotheses': int(node['accepted']),
        'cap_hit': int(bool(node['cap_hit'])),
        'local_star_density': pool_size / float(3000 * 3000),
        'multiple_testing_term': float(np.log1p(max(node['attempted'], 0))),
    }
    missing = {k: values[k] is None or not np.isfinite(values[k]) for k in FEATURES}
    return {'features': values, 'missing': missing,
            'unqueried_details': {'corrected': corrected, 'raw': raw}}


def build_scene(scene: str, seed: int, say=print) -> dict:
    from constellation.references import extract_patterns
    from experiments.exp1.data import load_scene
    from experiments.exp4b_joint_solver.independent_scorer import score_scene

    patterns = extract_patterns(ROOT / 'patterns')
    alternatives, pool, tags, pool_scores, pool_ranks = _pool_for_scene(scene)
    geo = score_scene(alternatives, patterns, seed=seed, ablation='held_out_only')
    image = load_scene(scene).image
    records = []
    for class_name, node in sorted(geo['hypotheses_by_class'].items()):
        for rank, hyp in enumerate(node['hypotheses']):
            f = _hypothesis_features(scene, class_name, hyp, node, image, pool,
                                     pool_scores, seed, len(pool))
            label = placement_label(scene, class_name, hyp['mapped_nodes'])
            records.append({
                'scene': scene, 'seed': seed, 'class_name': class_name,
                'hypothesis_rank_within_class': rank,
                'seed_triple': hyp['seed_triple'], 'matrix': hyp['matrix'],
                'mapped_nodes': hyp['mapped_nodes'], 'total_pairs': hyp['total_pairs'],
                **label, **f,
            })
    counts = {
        'n_records': len(records),
        'n_positive': sum(r['placement_correct'] for r in records),
        'n_true_class_wrong_placement': sum(r['class_correct'] and not r['placement_correct'] for r in records),
        'n_wrong_class': sum(not r['class_correct'] for r in records),
    }
    say(f'  {scene}/s{seed}: {counts}')
    return {'scene': scene, 'seed': seed, 'feature_names': list(FEATURES),
            'counts': counts, 'records': records,
            'pool_size': int(len(pool)), 'pool_xy': pool.tolist(),
            'pool_classical_scores': pool_scores.tolist(),
            'n_groups': int(len(set(tags.tolist())))}


def run(say=print) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {'placement_definition': {
        'radius_px': PLACEMENT_CORRECT_PX,
        'required': 'four issued figure stars (or every star when fewer than four); three fit and the fourth independently validates',
        'class_and_placement_required': True,
    }, 'feature_names': list(FEATURES), 'files': {}}
    for seed in SEEDS:
        for scene in SCENES:
            doc = build_scene(scene, seed, say=say)
            path = OUT / f'hypotheses_{scene}_s{seed}.json'
            write_json(path, doc)
            manifest['files'][f'{scene}:s{seed}'] = str(path.relative_to(ROOT))
    write_json(OUT / 'hypothesis_dataset.json', manifest)
    write_json(OUT / 'feature_schema.json', {
        'features': list(FEATURES),
        'feature_sources': FEATURE_SOURCES,
        'missing_policy': 'training-fold median plus an explicit missing indicator per feature',
        'unavailable_exp4b_features': [
            'number/fraction of high-confidence assignments',
            'candidate-rank summary and next-candidate gap at hypothesis level',
            'median and maximum assigned-node residual',
            'number of independently agreeing transforms',
            'duplicate-source conflict count beyond the one-to-one assignment',
            'source-quality-weighted unqueried support',
            'mean matched-null response and observed-minus-null response',
            'existing explicit Exp4B clutter penalty',
            'ECC convergence', 'HardNet scalar distance', 'centre-annulus score',
        ],
        'unavailable_policy': 'explicitly unavailable from frozen inputs; excluded rather than approximated after seeing held-out results',
    })
    return manifest


def load_scene_table(scene: str, seed: int) -> dict:
    return json.load(open(OUT / f'hypotheses_{scene}_s{seed}.json'))


if __name__ == '__main__':
    run()

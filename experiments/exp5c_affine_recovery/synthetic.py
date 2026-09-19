"""Resume-safe all-pattern synthetic engineering screens."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from experiments.exp5c_affine_recovery import OUT, DEFAULT_CONFIG
from experiments.exp5c_affine_recovery.confidence import (
    feature_vector, fit_logistic, gate_decisions, selective_metrics, calibration_bins,
)
from experiments.exp5c_affine_recovery.hashing import compute_quad_descriptors
from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from experiments.exp5c_affine_recovery.solver import ARMS, build_scene_quads, solve


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def _affine_case(name: str, nodes: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    norm = (nodes - nodes.mean(0)) / np.maximum(np.ptp(nodes, axis=0), 1e-6)
    theta = rng.uniform(0, 2 * np.pi); c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    shape = np.array([[rng.uniform(850, 1550), rng.uniform(-.22, .22) * 900],
                      [0, rng.uniform(700, 1450)]])
    matrix = rotation @ shape
    if rng.random() < .5: matrix[0] *= -1
    placed = norm @ matrix.T
    placed -= placed.mean(0)
    extent = np.ptp(placed, axis=0)
    if float(extent.max()) > 2550:
        placed *= 2550 / float(extent.max())
        extent = np.ptp(placed, axis=0)
    placed += rng.uniform(140 + extent / 2, 2860 - extent / 2)
    placed += rng.normal(0, rng.uniform(2.5, 6.0), placed.shape)

    n_issue = min(len(nodes), max(4, int(round(len(nodes) * rng.uniform(.55, .85)))))
    issued = sorted(map(int, rng.choice(len(nodes), n_issue, replace=False)))
    alternatives, truth_map = [], []
    for node in issued:
        rank = int(rng.choice([0, 1, 2, 3, 4, -1], p=[.43, .18, .14, .10, .08, .07]))
        base = rng.uniform(0, 3000, (5, 2))
        # Nearly coincident distractor stresses duplicate consolidation.
        if rng.random() < .25: base[1] = base[0] + rng.normal(0, 1.2, 2)
        scores = np.sort(rng.uniform(.58, .94, 5))[::-1]
        if rank >= 0:
            base[rank] = placed[node] + rng.normal(0, 1.0, 2)
        alternatives.append([(float(p[0]), float(p[1]), float(scores[i]), 0., 1.)
                             for i, p in enumerate(base)])
        truth_map.append({'node': node, 'correct_rank': rank})
    # Off-figure clutter queries and absent-query false positives.
    for _ in range(int(rng.integers(8, 19))):
        pts = rng.uniform(0, 3000, (5, 2)); scores = np.sort(rng.uniform(.45, .91, 5))[::-1]
        alternatives.append([(float(p[0]), float(p[1]), float(scores[i]), 0., 1.)
                             for i, p in enumerate(pts)])
        truth_map.append({'node': None, 'correct_rank': -1})
    for _ in range(int(rng.integers(5, 12))):
        pts = rng.uniform(0, 3000, (5, 2)); scores = np.sort(rng.uniform(.35, .82, 5))[::-1]
        alternatives.append([(float(p[0]), float(p[1]), float(scores[i]), 0., 1.)
                             for i, p in enumerate(pts)])
        truth_map.append({'node': None, 'correct_rank': -1})
    return {'name': name, 'alternatives': alternatives, 'truth_map': truth_map,
            'placed_nodes': placed.tolist(), 'seed': int(seed)}


def _placement_correct(result: dict, case: dict, tolerance: float = 18.0) -> bool:
    if result.get('winner') != case['name'] or not result.get('ranked'):
        return False
    h = result['ranked'][0]
    mapped = np.asarray(h['mapped_nodes'], float)
    placed = np.asarray(case['placed_nodes'], float)
    d = np.linalg.norm(mapped - placed, axis=1)
    return int(np.sum(d <= tolerance)) >= min(4, len(placed))


def proposal_trace(case: dict, pattern_index, arm: str, cfg: dict, seed: int) -> dict:
    sq = build_scene_quads(case['alternatives'], arm, cfg['max_scene_quads'], seed)
    available = [(qi, row['node'], row['correct_rank']) for qi, row in enumerate(case['truth_map'])
                 if row['node'] is not None and 0 <= row['correct_rank'] < 5]
    point_lookup = {(int(q), int(r)): i for i, (q, r) in enumerate(zip(sq.point_queries, sq.point_ranks))}
    correct_quads = []
    for four in __import__('itertools').combinations(available, 4):
        try: scene_ids = tuple(point_lookup[(q, r)] for q, _, r in four)
        except KeyError: continue
        key = tuple(sorted(scene_ids))
        if any(tuple(sorted(map(int, row))) == key for row in sq.ids):
            node_ids = [node for _, node, _ in four]
            _, sd = compute_quad_descriptors(sq.points, [scene_ids])
            graph = pattern_index.graphs[case['name']]
            norm = (graph['nodes'] - graph['nodes'].mean(0)) / np.maximum(np.ptp(graph['nodes'], axis=0), 1e-6)
            _, td = compute_quad_descriptors(norm, [node_ids])
            if len(sd) and len(td): correct_quads.append(float(np.linalg.norm(sd[0] - td[0])))
    best = min(correct_quads) if correct_quads else None
    # Conservative rank proxy: number of generated scene descriptors no farther
    # from any true-class template descriptor than the best correct pair.
    rank = None
    if best is not None and len(sq.descriptors):
        td = pattern_index.template_quads[case['name']][1]
        nearest = np.min(np.linalg.norm(sq.descriptors[:, None, :] - td[None, :, :], axis=2), axis=1)
        rank = int(np.sum(nearest < best)) + 1
    return {'candidate_correspondences': len(available), 'correct_quad_can_form': bool(correct_quads),
            'best_correct_descriptor_distance': best, 'best_correct_descriptor_rank_proxy': rank,
            'enters_scene_quad_budget': bool(rank is not None and rank <= cfg['max_scene_quads']),
            'scene_quads': len(sq.ids)}


def _run_cases(seed: int, target: Path, model: dict | None, policy: dict,
               fit_mode: bool = False, say=print) -> dict:
    index = get_or_create_index()
    existing = json.loads(target.read_text()) if target.exists() else {
        'status': 'RUNNING', 'seed': int(seed), 'completed': {}, 'small_patterns': index.small_classes,
    }
    # Synthetic screens trade proposal depth for complete all-pattern coverage.
    # The full 90k configuration is retained for real development/validation;
    # this fixed screen evaluates 12k hypotheses per case and completed seeds
    # are never mixed with the preserved full-budget partial pilot.
    cfg = {**DEFAULT_CONFIG, 'max_scene_quads': 6000, 'proposal_budget': 4000,
           'progressive_budgets': [1500, 4000]}
    for pos, name in enumerate(index.usable_classes, 1):
        if name in existing['completed']:
            continue
        case_seed = int(np.random.SeedSequence([seed, pos]).generate_state(1)[0])
        case = _affine_case(name, index.graphs[name]['nodes'], case_seed)
        started = time.perf_counter()
        selected_arms = ('rank1', 'one_alt_top5', 'adaptive')
        result = solve(case['alternatives'], index, arms=selected_arms,
                       config=cfg, seed=case_seed)
        correct = _placement_correct(result, case)
        decisions = gate_decisions(result, model, policy)
        stream_winners = [v['winner'] for v in result['streams'].values() if v['winner']]
        decisions['subsample_stable'] = bool(stream_winners and
            max(stream_winners.count(x) for x in set(stream_winners)) >= 2)
        decisions['perturbation_stable'] = False
        if decisions['strict'] or decisions['primary']:
            prng=np.random.default_rng(case_seed+77001)
            perturbed=[]
            for row in case['alternatives']:
                perturbed.append([(c[0]+float(prng.normal(0,.5)),c[1]+float(prng.normal(0,.5)),*c[2:]) for c in row])
            small_cfg={**cfg,'max_scene_quads':3000,'proposal_budget':2000,
                       'progressive_budgets':[1000,2000]}
            repeat=solve(perturbed,index,arms=selected_arms,config=small_cfg,seed=case_seed)
            decisions['perturbation_stable']=repeat['winner']==result['winner']
        trace_arms = ARMS if fit_mode else selected_arms
        traces = {arm: proposal_trace(case, index, arm, cfg, case_seed + i)
                  for i, arm in enumerate(trace_arms)}
        existing['completed'][name] = {
            'pattern': name, 'winner': result['winner'], 'correct': correct,
            'features': feature_vector(result).tolist(), 'decisions': decisions,
            'margin': result['margin'], 'top': result['ranked'][0] if result['ranked'] else None,
            'proposal_trace': traces, 'proposals_evaluated': result['proposals_evaluated'],
            'runtime_seconds': time.perf_counter() - started,
            'peak_rss_platform_units': result['peak_rss_platform_units'],
            'generation_seed': case_seed, 'screen_config': cfg,
        }
        atomic_json(target, existing)
        say(f"{pos}/{len(index.usable_classes)} {name}: {result['winner']} correct={correct}")
    rows = list(existing['completed'].values())
    existing['status'] = 'COMPLETE'
    existing['n_patterns'] = len(rows)
    existing['per_pattern_complete'] = len(rows) == len(index.usable_classes)
    existing['raw_accuracy'] = sum(r['correct'] for r in rows) / max(len(rows), 1)
    gates=('strict','primary','exploratory','null_calibrated','logistic','isotonic',
           'agreement','subsample_stable','perturbation_stable')
    existing['selective'] = {g: selective_metrics(rows, g) for g in gates}
    existing['calibration_bins'] = calibration_bins(rows)
    atomic_json(target, existing)
    return existing


def run_fit(seed: int, say=print) -> dict:
    target = OUT / 'synthetic_fit.json'
    provisional_policy = {'primary_gate': DEFAULT_CONFIG['primary_gate'],
                          'exploratory_confidence_min': .75}
    result = _run_cases(seed, target, None, provisional_policy, fit_mode=True, say=say)
    rows = list(result['completed'].values())
    model = fit_logistic(rows)
    # Fit confidence first, then freeze an empirical threshold using this seed only.
    from experiments.exp5c_affine_recovery.confidence import (predict_logistic,
        empirical_threshold,fit_isotonic)
    scored = []
    for r in rows:
        fake = {'ranked': [r['top']] if r['top'] else [], 'margin': r['margin']}
        scored.append({**r, 'confidence': predict_logistic(model, fake)})
    empirical = empirical_threshold(scored, .05)
    iso_rows=[{'score':r['top']['score'] if r['top'] else -1e9,'correct':r['correct']} for r in rows]
    isotonic=fit_isotonic(iso_rows)
    policy = {
        'model': model,
        'primary_gate': {**DEFAULT_CONFIG['primary_gate'],
                         'confidence_min': max(.90, float(empirical['threshold']))},
        'exploratory_confidence_min': .75,
        'empirical_fit': empirical,
        'isotonic_model': isotonic,
        'null_significance_min': 2.0,
        'fit_seed': int(seed),
    }
    atomic_json(OUT / 'deployment_policy.json', policy)
    result['calibration_model'] = model; result['frozen_policy'] = policy
    atomic_json(target, result)
    return result


def run_final(seed: int, filename: str, say=print) -> dict:
    policy = json.loads((OUT / 'deployment_policy.json').read_text())
    return _run_cases(seed, OUT / filename, policy['model'], policy, say=say)

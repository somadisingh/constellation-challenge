"""Bounded affine-quad proposal retrieval and independent verification.

The proposal descriptor is used only to retrieve transform candidates.  Final
ranking deliberately excludes descriptor distance and uses independently
matched observations, real pattern edges, transform quality and a
hypothesis-count-adjusted spatial null.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import math
import time
import resource

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binom

from constellation.joint import fit_affine, valid
from constellation.slate import slate_from_candidates, QuerySlate
from experiments.exp5c_affine_recovery import DEFAULT_CONFIG, TOLERANCE
from experiments.exp5c_affine_recovery.hashing import compute_quad_descriptors
from experiments.exp5c_affine_recovery.pattern_index import PatternIndex


ARMS = (
    'rank1', 'one_alt_top3', 'one_alt_top5', 'two_alt_top3',
    'two_alt_top5', 'adaptive', 'prosac',
)


@dataclass
class SceneQuads:
    arm: str
    points: np.ndarray
    point_queries: np.ndarray
    point_ranks: np.ndarray
    point_scores: np.ndarray
    ids: np.ndarray
    descriptors: np.ndarray


def _stable_u64(*parts) -> int:
    raw = '|'.join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big')


def _query_quads(n_queries: int, limit: int, seed: int, confidences: np.ndarray) -> list[tuple[int, ...]]:
    """Bound query quadruples without constructing an unrestricted product."""
    if n_queries < 4:
        return []
    order = sorted(range(n_queries), key=lambda q: (-float(confidences[q]), q))
    head = order[:min(28, n_queries)]
    ranked = []
    for q in combinations(head, 4):
        span = max(q) - min(q)
        ranked.append((-sum(float(confidences[x]) for x in q), -span, q))
    ranked.sort()
    chosen = [x[2] for x in ranked[:limit]]
    seen = {tuple(sorted(x)) for x in chosen}
    rng = np.random.default_rng(seed)
    attempts = 0
    while len(chosen) < limit and attempts < limit * 30:
        q = tuple(sorted(map(int, rng.choice(n_queries, 4, replace=False))))
        attempts += 1
        if q not in seen:
            seen.add(q); chosen.append(q)
    return chosen


def build_scene_quads(
    alternatives: list,
    arm: str,
    max_quads: int = 30000,
    seed: int = 42,
    gap_threshold: float = 0.03,
) -> SceneQuads:
    """Create a bounded arm while retaining candidate rank and query identity."""
    if arm not in ARMS:
        raise ValueError(f'unknown arm {arm}')
    slates = [slate_from_candidates(x) for x in alternatives]
    slates = [s for s in slates if len(s)]
    if len(slates) < 4:
        empty = np.empty((0, 4), np.int32)
        return SceneQuads(arm, np.empty((0, 2)), np.empty(0, int),
                          np.empty(0, int), np.empty(0), empty, np.empty((0, 4)))

    depth = 1 if arm == 'rank1' else (3 if 'top3' in arm else 5)
    if arm in ('adaptive', 'prosac'):
        depth = 5
    points, point_queries, point_ranks, point_scores, by_query = [], [], [], [], []
    gaps = []
    for qi, s in enumerate(slates):
        ranked = list(map(int, s.order_by_rank()[:depth]))
        gap = s.ambiguity_gap
        gaps.append(gap)
        ids = []
        for rank, ci in enumerate(ranked):
            ids.append(len(points)); points.append(s.xy[ci]); point_queries.append(qi)
            point_ranks.append(rank); point_scores.append(float(s.rank[ci]))
        by_query.append(ids)
    points = np.asarray(points, float)
    point_queries = np.asarray(point_queries, int)
    point_ranks = np.asarray(point_ranks, int)
    point_scores = np.asarray(point_scores, float)
    confidence = np.asarray([float(s.rank[s.order_by_rank()[0]]) for s in slates])

    # Each query quadruple receives only a bounded set of rank patterns.  This
    # tests deeper ranks without ever constructing their Cartesian product.
    q_limit = max(max_quads * 2, 4000)
    qquads = _query_quads(len(slates), q_limit, seed, confidence)
    raw = []
    for serial, qs in enumerate(qquads):
        rank_patterns = []
        if arm == 'rank1':
            rank_patterns = [(0, 0, 0, 0)]
        elif arm.startswith('one_alt'):
            for pos in range(4):
                for r in range(1, depth):
                    z = [0, 0, 0, 0]; z[pos] = r; rank_patterns.append(tuple(z))
        elif arm.startswith('two_alt'):
            for a, b in combinations(range(4), 2):
                # Round-robin deeper pairs rather than an unrestricted product.
                for r in range(1, depth):
                    z = [0, 0, 0, 0]; z[a] = r; z[b] = 1 + ((r + serial) % (depth - 1))
                    rank_patterns.append(tuple(z))
        elif arm == 'adaptive':
            amb = [i for i, q in enumerate(qs) if gaps[q] < gap_threshold]
            rank_patterns = [(0, 0, 0, 0)]
            for pos in amb:
                for r in range(1, min(depth, len(by_query[qs[pos]]))):
                    z = [0, 0, 0, 0]; z[pos] = r; rank_patterns.append(tuple(z))
            for a, b in combinations(amb, 2):
                z = [0, 0, 0, 0]; z[a] = 1; z[b] = 1; rank_patterns.append(tuple(z))
        else:  # deterministic PROSAC-style progressive ranks
            stage = min(4, serial // max(1, len(qquads) // 5))
            rank_patterns = [(0, 0, 0, 0)]
            for pos in range(4):
                if stage and len(by_query[qs[pos]]) > 1:
                    z = [0, 0, 0, 0]; z[pos] = 1 + ((serial + pos) % min(stage, len(by_query[qs[pos]]) - 1))
                    rank_patterns.append(tuple(z))
        for ranks in rank_patterns:
            if any(r >= len(by_query[q]) for q, r in zip(qs, ranks)):
                continue
            ids = tuple(by_query[q][r] for q, r in zip(qs, ranks))
            # Query identity is asserted here and rechecked by tests.
            if len({int(point_queries[i]) for i in ids}) == 4:
                raw.append(ids)
        if len(raw) >= max_quads * 3:
            break

    # Deterministic priority: fewer alternates, stronger appearance, then a
    # stable hash.  Arm semantics still force the requested alternate count.
    unique = list(dict.fromkeys(raw))
    unique.sort(key=lambda ids: (
        sum(int(point_ranks[i] > 0) for i in ids),
        -sum(float(point_scores[i]) for i in ids),
        _stable_u64(seed, arm, *ids), ids))
    raw_ids = np.asarray(unique[:max_quads], np.int32).reshape(-1, 4)
    ids, desc = compute_quad_descriptors(points, raw_ids)
    return SceneQuads(arm, points, point_queries, point_ranks, point_scores, ids, desc)


def _template_catalog(index: PatternIndex):
    desc, names, ids, guided = [], [], [], []
    for name in sorted(index.usable_classes):
        ti, td = index.template_quads[name]
        edges = {tuple(sorted(e)) for e in index.graphs[name]['edges']}
        for q, d in zip(ti, td):
            edge_count = sum(tuple(sorted((int(a), int(b)))) in edges for a, b in combinations(q, 2))
            # At least two real edges makes this a graph-guided template quad.
            desc.append(d); names.append(name); ids.append(q); guided.append(edge_count >= 2)
    return np.asarray(desc, float), np.asarray(names, object), np.asarray(ids, object), np.asarray(guided, bool)


def _candidate_pool(alternatives: list, depth: int = 5):
    pts, query_ids, scores = [], [], []
    for qi, cands in enumerate(alternatives):
        for c in cands[:depth]:
            pts.append(c[:2]); query_ids.append(qi); scores.append(c[2])
    pts = np.asarray(pts, float).reshape(-1, 2)
    query_ids = np.asarray(query_ids, int)
    scores = np.asarray(scores, float)
    # Anchor grouping prevents repeated/nearly coincident candidates from
    # supplying multiple independent support observations.
    location_groups = []
    anchors = []
    for p in pts:
        d = np.linalg.norm(np.asarray(anchors) - p, axis=1) if anchors else np.empty(0)
        if len(d) and d.min() < 3.0:
            location_groups.append(int(d.argmin()))
        else:
            location_groups.append(len(anchors)); anchors.append(p)
    return pts, query_ids, scores, np.asarray(location_groups, int)


def _assign(mapped, pool, query_ids, location_groups, tolerance):
    if not len(pool):
        return [], []
    d = np.linalg.norm(mapped[:, None, :] - pool[None, :, :], axis=2)
    cand = np.argwhere(d < tolerance)
    order = np.argsort(d[cand[:, 0], cand[:, 1]], kind='stable')
    used_nodes, used_queries, used_locations = set(), set(), set()
    pairs, residuals = [], []
    for z in order:
        node, pi = map(int, cand[z]); q = int(query_ids[pi]); loc = int(location_groups[pi])
        if node in used_nodes or q in used_queries or loc in used_locations:
            continue
        used_nodes.add(node); used_queries.add(q); used_locations.add(loc)
        pairs.append((node, pi)); residuals.append(float(d[node, pi]))
    return pairs, residuals


def _graph_features(edges, matched_nodes):
    matched = set(map(int, matched_nodes))
    supported = [(int(a), int(b)) for a, b in edges if a in matched and b in matched]
    adjacency = {x: set() for x in matched}
    for a, b in supported:
        adjacency[a].add(b); adjacency[b].add(a)
    largest = 0; seen = set()
    for start in sorted(matched):
        if start in seen: continue
        stack = [start]; component = set()
        while stack:
            u = stack.pop()
            if u in component: continue
            component.add(u); seen.add(u); stack.extend(adjacency[u] - component)
        largest = max(largest, len(component))
    incident = sum(1 for a, b in edges if a in matched or b in matched)
    return {'supported_edges': len(supported),
            'supported_edge_fraction': len(supported) / max(incident, 1),
            'largest_component': largest}


def verify_transform(name, graph, template_ids, scene_ids, scene: SceneQuads,
                     pool, query_ids, pool_scores, location_groups, scene_shape,
                     tolerance=TOLERANCE):
    nodes = np.asarray(graph['nodes'], float)
    norm = (nodes - nodes.mean(0)) / np.maximum(np.ptp(nodes, axis=0), 1e-6)
    matrix = fit_affine(norm[np.asarray(template_ids, int)], scene.points[np.asarray(scene_ids, int)])
    if matrix is None or not valid(matrix):
        return None
    sv = np.linalg.svd(matrix[:2], compute_uv=False)
    condition = float(sv[0] / max(sv[-1], 1e-12))
    mapped = np.c_[norm, np.ones(len(norm))] @ matrix
    pairs, residuals = _assign(mapped, pool, query_ids, location_groups, tolerance)
    if len(pairs) < 4:
        return None
    seed_nodes = set(map(int, template_ids))
    held = [(a, b) for a, b in pairs if a not in seed_nodes]
    matched_nodes = [a for a, _ in pairs]
    graph_features = _graph_features(graph['edges'], matched_nodes)
    h, w = scene_shape
    inside = ((mapped[:, 0] >= 0) & (mapped[:, 0] < w) &
              (mapped[:, 1] >= 0) & (mapped[:, 1] < h))
    boundary_valid = float(np.mean(inside)) >= 0.5 and all(inside[a] for a, _ in pairs)
    if not boundary_valid:
        return None
    res = np.asarray(residuals, float)
    appearance = np.asarray([pool_scores[b] for _, b in pairs], float)
    unique_queries = len({int(query_ids[b]) for _, b in pairs})
    chance = min(0.8, len(np.unique(location_groups)) * math.pi * tolerance ** 2 / max(h * w, 1))
    raw_p = float(binom.sf(max(len(held) - 1, 0), max(len(nodes) - 4, 1), chance))
    raw_significance = -math.log10(max(raw_p, 1e-300))
    return {
        'name': name, 'matrix': matrix.tolist(), 'mapped_nodes': mapped.tolist(),
        'pairs': pairs, 'seed_nodes': sorted(seed_nodes),
        'seed_scene_queries': [int(scene.point_queries[i]) for i in scene_ids],
        'support': len(pairs), 'unique_matched_queries': unique_queries,
        'held_out_support': len(held), 'residual_mean': float(res.mean()),
        'residual_median': float(np.median(res)), 'residual_p90': float(np.quantile(res, .9)),
        'residual_max': float(res.max()), 'appearance_mean': float(appearance.mean()),
        'appearance_min': float(appearance.min()), 'condition_number': condition,
        'boundary_valid': boundary_valid, 'graph': graph_features,
        'null_probability_raw': raw_p, 'null_significance_raw': raw_significance,
        'proposal_stream': scene.arm,
    }


def _final_score(h, tested):
    adjusted_p = min(1.0, h['null_probability_raw'] * max(tested, 1))
    significance = -math.log10(max(adjusted_p, 1e-300))
    h['hypotheses_tested'] = int(tested)
    h['null_probability_adjusted'] = adjusted_p
    h['null_significance_adjusted'] = significance
    # No proposal descriptor term: it is retrieval evidence, not confidence.
    h['score'] = float(
        0.70 * h['held_out_support'] + 0.25 * h['support'] +
        0.35 * h['graph']['supported_edges'] +
        0.20 * h['graph']['largest_component'] +
        0.15 * h['appearance_mean'] + 0.30 * significance -
        0.55 * h['residual_p90'] / TOLERANCE -
        0.12 * math.log1p(h['condition_number']))
    return h


def solve(
    alternatives: list,
    pattern_index: PatternIndex,
    scene_shape=(3000, 3000),
    arms=('rank1', 'one_alt_top5', 'two_alt_top5', 'adaptive', 'prosac'),
    config: dict | None = None,
    seed: int = 42,
):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    started = time.perf_counter()
    template_desc, template_names, template_ids, graph_guided = _template_catalog(pattern_index)
    tree = cKDTree(template_desc) if len(template_desc) else None
    pool, query_ids, pool_scores, location_groups = _candidate_pool(alternatives, depth=5)
    all_best, stream_summaries = [], {}
    total_evaluated = 0
    for arm_pos, arm in enumerate(arms):
        sq = build_scene_quads(alternatives, arm, cfg['max_scene_quads'], seed + arm_pos)
        jobs = []
        if tree is not None and len(sq.descriptors):
            k = min(int(cfg['descriptor_neighbors']), len(template_desc))
            dist, ind = tree.query(sq.descriptors, k=k,
                                   distance_upper_bound=float(cfg['descriptor_max_distance']))
            dist, ind = np.atleast_2d(dist), np.atleast_2d(ind)
            if dist.shape[0] != len(sq.descriptors): dist, ind = dist.T, ind.T
            for si in range(len(sq.descriptors)):
                for d, ti in zip(dist[si], ind[si]):
                    if np.isfinite(d) and int(ti) < len(template_desc):
                        jobs.append((float(d), not bool(graph_guided[int(ti)]),
                                     str(template_names[int(ti)]), int(ti), si))
        jobs.sort(key=lambda x: (x[1], x[0], x[2], x[3], x[4]))
        jobs = jobs[:int(cfg['proposal_budget'])]
        best_by_class = {}
        valid_count = 0
        stop_budget = int(cfg['proposal_budget'])
        for pos, (_, _, name, ti, si) in enumerate(jobs):
            h = verify_transform(name, pattern_index.graphs[name], template_ids[ti],
                                 sq.ids[si], sq, pool, query_ids, pool_scores,
                                 location_groups, scene_shape, cfg['tolerance_px'])
            total_evaluated += 1
            if h is None: continue
            valid_count += 1
            old = best_by_class.get(name)
            # Temporary structural ordering; multiplicity correction is applied below.
            key = (h['held_out_support'], h['support'], h['graph']['supported_edges'],
                   h['graph']['largest_component'], -h['residual_p90'])
            old_key = None if old is None else (old['held_out_support'], old['support'],
                old['graph']['supported_edges'], old['graph']['largest_component'], -old['residual_p90'])
            if old is None or key > old_key:
                best_by_class[name] = h
            # Progressive widening / early termination uses only independent support.
            if pos + 1 in cfg['progressive_budgets']:
                leaders = sorted(best_by_class.values(), key=lambda x: (-x['held_out_support'], -x['support'], x['name']))
                if leaders and leaders[0]['held_out_support'] >= 8 and leaders[0]['support'] >= 11:
                    stop_budget = pos + 1; break
        ranked = [_final_score(h, max(total_evaluated, 1)) for h in best_by_class.values()]
        ranked.sort(key=lambda h: (-h['score'], h['name']))
        stream_summaries[arm] = {
            'scene_quads': len(sq.ids), 'retrieved_jobs': len(jobs),
            'evaluated': min(len(jobs), stop_budget), 'valid': valid_count,
            'winner': ranked[0]['name'] if ranked else None,
            'top': ranked[:3], 'early_terminated': stop_budget < len(jobs),
        }
        all_best.extend(ranked)

    # One best transform per class over independent proposal streams.
    by_class = {}
    for h in all_best:
        if h['name'] not in by_class or h['score'] > by_class[h['name']]['score']:
            by_class[h['name']] = h
    ranked = sorted(by_class.values(), key=lambda h: (-h['score'], h['name']))
    winners = [x['winner'] for x in stream_summaries.values() if x['winner']]
    for h in ranked:
        h['stream_agreement'] = int(sum(w == h['name'] for w in winners))
    margin = float(ranked[0]['score'] - ranked[1]['score']) if len(ranked) > 1 else 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        'winner': ranked[0]['name'] if ranked else 'unknown', 'margin': margin,
        'ranked': ranked, 'streams': stream_summaries,
        'proposals_evaluated': total_evaluated,
        'runtime_seconds': time.perf_counter() - started,
        'peak_rss_platform_units': int(peak),
    }


def alternatives_from_prediction_json(doc: dict) -> list:
    return [[tuple(map(float, c[:5])) for c in (q.get('candidates') or [])]
            for q in doc['diagnostics']['queries']]

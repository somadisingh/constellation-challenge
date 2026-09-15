"""Candidate-depth arms and scene quadruple generation."""
from __future__ import annotations

from itertools import combinations
import numpy as np

from constellation.slate import slate_from_candidates, QuerySlate
from constellation.geometry import consolidate
from constellation.joint import generation_points, build_pool
from experiments.exp5c_affine_recovery.hashing import compute_quad_descriptors


def scene_inputs_from_alternatives(alternatives: list):
    """Build standardized live slates, groups, pool and tags from candidate alternatives."""
    sl = [slate_from_candidates(x) for x in alternatives]
    live = [s for s in sl if len(s)]
    anchors = np.asarray([s.seed_xy() for s in live]).reshape(-1, 2)
    _, groups = consolidate(anchors) if len(anchors) else ([], [])
    rank1 = generation_points(live, groups)
    pool, tags, scores, ranks = build_pool(live, groups, top_k=8)
    return live, rank1, pool, tags, scores, groups


def get_query_candidates(slates: list[QuerySlate], max_depth: int = 5) -> list[list[dict]]:
    """Return candidates per query with metadata (query_idx, cand_idx, xy, score, gap)."""
    query_cands = []
    for q_idx, s in enumerate(slates):
        ranked_indices = s.order_by_rank()[:max_depth]
        scores = s.scores if s.scores is not None else [1.0] * len(s)
        gap = float(scores[ranked_indices[0]] - scores[ranked_indices[1]]) if len(ranked_indices) > 1 else 1.0
        cands = []
        for r, c_idx in enumerate(ranked_indices):
            cands.append({
                'query_idx': q_idx,
                'cand_idx': int(c_idx),
                'rank': r,
                'xy': np.array(s.xy[c_idx], dtype=float),
                'score': float(scores[c_idx]),
                'gap': gap,
            })
        query_cands.append(cands)
    return query_cands


def quads_rank1_only(slates: list[QuerySlate]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arm 1: Rank-one only."""
    points = np.array([s.xy[s.order_by_rank()[0]] for s in slates], dtype=float)
    n = len(points)
    if n < 4:
        return points, np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=float)
    raw_ids = np.array(list(combinations(range(n), 4)), dtype=np.int32)
    ids, desc = compute_quad_descriptors(points, raw_ids)
    return points, ids, desc


def quads_three_r1_one_alt(slates: list[QuerySlate], max_alt_rank: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arm 2 & 3: Three rank-1 plus one alternative up to rank (top-3 or top-5)."""
    points = []
    by_query = []
    for s in slates:
        q_ids = []
        for j in s.order_by_rank()[:max_alt_rank]:
            q_ids.append(len(points))
            points.append(s.xy[j])
        by_query.append(q_ids)
    points = np.asarray(points, dtype=float)
    n = len(by_query)
    if n < 4:
        return points, np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=float)

    primary = np.asarray([x[0] for x in by_query], dtype=np.int32)
    blocks = []
    for q, ids in enumerate(by_query):
        others = [i for i in range(n) if i != q]
        triples = np.asarray(list(combinations(others, 3)), dtype=np.int32)
        if not len(triples):
            continue
        for alt in ids[1:]:
            blocks.append(np.c_[np.full(len(triples), alt, dtype=np.int32), primary[triples]])
    raw_ids = np.vstack(blocks) if blocks else np.empty((0, 4), dtype=np.int32)
    ids, desc = compute_quad_descriptors(points, raw_ids)
    return points, ids, desc


def quads_two_r1_two_alt(
    slates: list[QuerySlate],
    max_alt_rank: int = 3,
    max_ambiguous_queries: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arm 4 & 5: Two rank-1 plus two alternatives (top-3 or top-5)."""
    points = []
    by_query = []
    gaps = []
    for q_idx, s in enumerate(slates):
        q_ids = []
        ranked = s.order_by_rank()[:max_alt_rank]
        scores = s.scores if s.scores is not None else [1.0] * len(s)
        gap = float(scores[ranked[0]] - scores[ranked[1]]) if len(ranked) > 1 else 1.0
        gaps.append((gap, q_idx))
        for j in ranked:
            q_ids.append(len(points))
            points.append(s.xy[j])
        by_query.append(q_ids)
    points = np.asarray(points, dtype=float)
    n = len(by_query)
    if n < 4:
        return points, np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=float)

    # Pick the most ambiguous queries to provide the alternate pairs
    gaps.sort(key=lambda x: x[0])
    ambiguous_q = [q for _, q in gaps[:max_ambiguous_queries]]

    primary = np.asarray([x[0] for x in by_query], dtype=np.int32)
    blocks = []
    for i_idx, q1 in enumerate(ambiguous_q):
        for q2 in ambiguous_q[i_idx + 1:]:
            alt_pairs = [(a1, a2) for a1 in by_query[q1][1:] for a2 in by_query[q2][1:]]
            if not alt_pairs:
                continue
            others = [i for i in range(n) if i != q1 and i != q2]
            r1_pairs = np.asarray(list(combinations(others, 2)), dtype=np.int32)
            if not len(r1_pairs):
                continue
            for a1, a2 in alt_pairs:
                blocks.append(np.c_[
                    np.full(len(r1_pairs), a1, dtype=np.int32),
                    np.full(len(r1_pairs), a2, dtype=np.int32),
                    primary[r1_pairs]
                ])

    raw_ids = np.vstack(blocks) if blocks else np.empty((0, 4), dtype=np.int32)
    ids, desc = compute_quad_descriptors(points, raw_ids)
    return points, ids, desc


def quads_adaptive_uncertainty(
    slates: list[QuerySlate],
    gap_threshold: float = 0.03,
    max_alt_rank: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arm 6: Adaptive candidate depth based only on candidate-score uncertainty."""
    points = []
    by_query = []
    ambiguous_queries = []
    for q_idx, s in enumerate(slates):
        ranked = s.order_by_rank()
        scores = s.scores if s.scores is not None else [1.0] * len(s)
        gap = float(scores[ranked[0]] - scores[ranked[1]]) if len(ranked) > 1 else 1.0
        q_ids = []
        # If ambiguous, include up to max_alt_rank; otherwise only rank 1
        limit = max_alt_rank if gap < gap_threshold else 1
        if gap < gap_threshold and len(ranked) > 1:
            ambiguous_queries.append(q_idx)
        for j in ranked[:limit]:
            q_ids.append(len(points))
            points.append(s.xy[j])
        by_query.append(q_ids)

    points = np.asarray(points, dtype=float)
    n = len(by_query)
    if n < 4:
        return points, np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=float)

    primary = np.asarray([x[0] for x in by_query], dtype=np.int32)
    blocks = []
    # 1. Base rank-1 triples with 1 alternate from ambiguous queries
    for q in ambiguous_queries:
        others = [i for i in range(n) if i != q]
        triples = np.asarray(list(combinations(others, 3)), dtype=np.int32)
        if not len(triples):
            continue
        for alt in by_query[q][1:]:
            blocks.append(np.c_[np.full(len(triples), alt, dtype=np.int32), primary[triples]])

    # 2. Pairs of alternates from ambiguous queries with 2 rank-1
    for i_idx, q1 in enumerate(ambiguous_queries):
        for q2 in ambiguous_queries[i_idx + 1:]:
            alt_pairs = [(a1, a2) for a1 in by_query[q1][1:] for a2 in by_query[q2][1:]]
            if not alt_pairs:
                continue
            others = [i for i in range(n) if i != q1 and i != q2]
            r1_pairs = np.asarray(list(combinations(others, 2)), dtype=np.int32)
            if not len(r1_pairs):
                continue
            for a1, a2 in alt_pairs:
                blocks.append(np.c_[
                    np.full(len(r1_pairs), a1, dtype=np.int32),
                    np.full(len(r1_pairs), a2, dtype=np.int32),
                    primary[r1_pairs]
                ])

    raw_ids = np.vstack(blocks) if blocks else np.empty((0, 4), dtype=np.int32)
    ids, desc = compute_quad_descriptors(points, raw_ids)
    return points, ids, desc


def quads_prosac_schedule(
    slates: list[QuerySlate],
    max_quads: int = 30000,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arm 7: Bounded PROSAC-style progressive schedule using query candidate confidence."""
    # Build candidate list sorted by confidence
    cands = []
    points = []
    query_map = []
    for q_idx, s in enumerate(slates):
        ranked = s.order_by_rank()[:5]
        scores = s.scores if s.scores is not None else [1.0] * len(s)
        for r, j in enumerate(ranked):
            p_idx = len(points)
            points.append(s.xy[j])
            query_map.append(q_idx)
            # Confidence metric: rank-0 receives high confidence; alternates discounted by rank & score
            conf = float(scores[j]) * (1.0 / (1.0 + r * 0.5))
            cands.append((conf, p_idx, q_idx))

    points = np.asarray(points, dtype=float)
    cands.sort(key=lambda x: -x[0])  # descending confidence

    # Progressive sampling
    sampled_ids = []
    seen = set()
    n_cands = len(cands)
    rng = np.random.default_rng(seed)

    # Initial deterministic top candidates
    for m in range(4, min(n_cands, 35)):
        # Quadruples containing the m-th candidate and 3 from the top m-1
        m_cand = cands[m]
        top_subset = cands[:m]
        # Group by query to enforce distinct queries
        queries_avail = {}
        for c in top_subset:
            q = c[2]
            if q != m_cand[2]:
                if q not in queries_avail:
                    queries_avail[q] = []
                queries_avail[q].append(c[1])

        if len(queries_avail) < 3:
            continue
        q_keys = list(queries_avail.keys())
        for q_triple in combinations(q_keys, 3):
            p1 = queries_avail[q_triple[0]][0]
            p2 = queries_avail[q_triple[1]][0]
            p3 = queries_avail[q_triple[2]][0]
            quad = tuple(sorted([m_cand[1], p1, p2, p3]))
            if quad not in seen:
                seen.add(quad)
                sampled_ids.append(quad)
                if len(sampled_ids) >= max_quads:
                    break
        if len(sampled_ids) >= max_quads:
            break

    raw_ids = np.asarray(sampled_ids, dtype=np.int32) if sampled_ids else np.empty((0, 4), dtype=np.int32)
    ids, desc = compute_quad_descriptors(points, raw_ids)
    return points, ids, desc

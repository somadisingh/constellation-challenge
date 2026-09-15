"""Stage 2 (Branch G): robust multi-candidate geometric hypothesis recovery.

Branch G was selected mechanically in `branch_decision.py` -- every real
labelled scene already has >=4 distinct correct figure-star candidates and
>=90% figure-query coverage in the frozen bank at 36px, so candidate
RETRIEVAL is not the bottleneck (see `oracle_audit.json`). This module fixes
the bottleneck the oracle audit and `taurus_failure_trace.json` actually
found: the EXISTING seed-triple search (`constellation.joint.SceneIndex` +
`independent_scorer.generate_and_score`) generates and keeps hypotheses that
tie or beat a hypothesis that recovers MORE correct physical placements,
because it only scores by held-out support count, with no independent
(fourth-point / graph) validation of WHICH held-out matches are geometrically
coherent versus coincidental.

Reused verbatim: `constellation.joint.fit_affine/valid/assignment/
build_pool/generation_points/SceneIndex`, `constellation.slate.
slate_from_candidates`, `constellation.geometry.consolidate`,
`experiments.exp4b_joint_solver.unqueried_star_evidence.score_unqueried_nodes`
(matched-null unqueried-star evidence), `experiments.exp1.env.derive_seed`
(deterministic seeding, never `hash()`).

New in this module: multi-candidate (K) representation, PROSAC-style
deterministic triple ordering with explicit degeneracy/condition-number/
duplicate rejection, barycentric fourth-point validation (`barycentric.py`),
partial graph-consistency scoring over the mapped template's own edges, and a
bounded deterministic beam search combining everything into one hypothesis
score with an explicit, auditable decomposition.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np

from scipy.stats import binom

from . import CORRECT_PX, K_GRID, ROOT
from .barycentric import barycentric_agreement, to_barycentric
from experiments.exp1.env import derive_seed

BEAM_WIDTH = 8
MAX_TRIPLES_PER_CLASS = 300      # bounded search: PROSAC ordering explores the
                                 # most-confident triples first and stops here
                                 # (kept small enough for full 48-pattern x
                                 # 3-scene x 2-seed x multi-ablation evaluation
                                 # to complete in feasible wall time on an M4
                                 # Pro, per the task's own runtime/memory gate)
NMS_TRANSFORM_RADIUS_PX = 10.0   # non-maximum suppression: two hypotheses of the
                                 # SAME class whose mapped centroids land within
                                 # this radius are treated as duplicates
DEFAULT_TOLERANCE = 18.0
MIN_SUPPORT = 4


# ─── MULTI-CANDIDATE REPRESENTATION (K candidates per query) ─────────────────

def build_candidate_records(scene: str, k: int, exp3_rows: list | None = None) -> list:
    """Up to `k` candidates per query (never selected on the held-out sky --
    `k` is a fixed, predeclared constant supplied by the caller from `K_GRID`).
    Each record carries every field the task requires where available; fields
    Exp3 does not supply (per-candidate presence probability at this
    granularity) are explicitly `None`, never fabricated as 0."""
    from experiments.exp1.data import query_records
    from experiments.exp1.evaluation import _aligned_set, load_real_aligned
    from experiments.exp1.stages import Paths

    node = load_real_aligned(Paths(ROOT / 'outputs' / 'exp1'), scene)
    records = query_records(scene)
    exp3_by_query = {r['query_id']: r for r in (exp3_rows or [])}

    out = []
    for r in records:
        aligned = _aligned_set(node, r['index'])
        valid = np.asarray(aligned.admissible) & ~np.asarray(aligned.low_info)
        if not valid.any():
            out.append({'query_id': r['query_id'], 'index': r['index'],
                       'present': r['present'], 'figure': r['figure'], 'candidates': []})
            continue
        ncc = np.where(valid, aligned.ncc, -np.inf)
        order = np.argsort(-ncc)[:k]
        exp3 = exp3_by_query.get(r['query_id'])
        cands = []
        for rank, i in enumerate(order):
            if not valid[i]:
                continue
            cands.append({
                'query_index': r['index'], 'xy': (float(aligned.xy[i, 0]), float(aligned.xy[i, 1])),
                'classical_score': float(aligned.ncc[i]), 'classical_rank': rank,
                'exp3_logit': (float(exp3.get('best_logit')) if exp3 and rank == 0
                              and exp3.get('best_logit') is not None else None),
                'exp3_disagreement': None,
                'presence_probability': None,
                'pose': (float(aligned.poses[i, 0]), float(aligned.poses[i, 1])),
            })
        out.append({'query_id': r['query_id'], 'index': r['index'],
                   'present': r['present'], 'figure': r['figure'], 'candidates': cands})
    return out


# ─── MULTI-CANDIDATE SEED POINTS (do not commit to rank-1 prematurely) ──────

def multi_candidate_generation_points(slates: list, groups: list, k: int,
                                      limit: int = 56) -> tuple:
    """`constellation.joint.generation_points` contributes exactly ONE point
    per query (`slate.seed_xy()`, always rank-0/index-0) -- confirmed by the
    oracle audit's `taurus_failure_trace.json` to be the actual mechanism
    that loses 2 of Taurus's 6 correct figure-star candidates entirely (their
    correct location exists in the bank at rank 3 and rank 7, never at rank
    0, so `generation_points` never offers it as a seed point at all,
    independent of any scoring downstream). This function instead emits up
    to `k` points PER QUERY (its own top-k by calibration/rank score),
    each tagged with that query's physical-source group id, so triangle
    seeding can draw from any of a query's top-k candidates rather than only
    its single most-confident one. The per-query cap (`limit`, matching
    `generation_points`'s own `GEN_LIMIT`) is applied to the NUMBER OF QUERIES
    contributing points (highest presence_score first), identical to the
    existing selection -- only the per-query POINT count changes.
    """
    # `groups[pos]` is indexed by POSITION within the live-slate list (the
    # same order `constellation.geometry.consolidate` was called with to
    # produce `groups`), not by the original scene-wide slate index -- match
    # that convention exactly here.
    live = [(pos, s) for pos, s in enumerate(slates) if len(s)]
    if len(live) > limit:
        live.sort(key=lambda item: -item[1].presence_score)
        live = live[:limit]
        live.sort(key=lambda item: item[0])
    pts, tags = [], []
    for pos, s in live:
        order = s.order_by_rank()[:k]
        for idx in order:
            pts.append(s.xy[idx])
            tags.append(groups[pos])
    return (np.asarray(pts, dtype=float).reshape(-1, 2), np.asarray(tags, dtype=int))


# ─── PROSAC-STYLE DETERMINISTIC TRIPLE ORDERING ──────────────────────────────

def _condition_number(matrix: np.ndarray) -> float:
    sv = np.linalg.svd(matrix[:2], compute_uv=False)
    return float(sv[0] / max(sv[-1], 1e-9))


def _is_collinear(pts: np.ndarray, min_area: float = 1e-6) -> bool:
    v = pts[1] - pts[0]
    w = pts[2] - pts[0]
    area = abs(v[0] * w[1] - v[1] * w[0])
    return area < min_area


def prosac_triples(template: np.ndarray, gen_pts: np.ndarray, groups_of_gen: np.ndarray,
                   index, per_class_budget: int, seed: int, class_name: str) -> list:
    """Deterministic, confidence-ordered candidate triples: reuses
    `SceneIndex.seeds()`'s own KD-tree nearest-invariant-first ordering
    (already deterministic and already confidence-weighted through the
    generation-point cap in `generation_points`, which retains the
    highest-`presence_score` queries first) as the PROSAC ranking, then
    rejects degenerate/duplicate/unstable triples up front rather than after
    a wasted fit.

    `groups_of_gen[i]` is the physical-source group id of `gen_pts[i]`, used
    to enforce "three candidates belonging to distinct queries / distinct
    physical-source groups" (a group already IS the physical-source
    consolidation Exp4B/Exp1 use, and every point in `gen_pts` is one query's
    own seed anchor, so distinct scene-index triples automatically come from
    distinct queries by construction of `generation_points`)."""
    p = (template - template.mean(axis=0)) / np.maximum(np.ptp(template, axis=0), 1e-6)
    raw_seeds = index.seeds(p, per_class_budget, quad_share=0.0)
    accepted = []
    seen_source_sets = set()
    for _, ti, si in raw_seeds:
        if len(ti) != 3 or len(si) != 3:
            continue
        src_groups = tuple(sorted(int(groups_of_gen[j]) for j in si))
        if len(set(src_groups)) != 3:
            continue   # duplicate physical-source/query assignment
        if _is_collinear(p[np.asarray(ti)]):
            continue   # nearly collinear template triple
        dst = gen_pts[np.asarray(si)]
        if _is_collinear(dst):
            continue   # nearly collinear scene triple
        key = (tuple(int(x) for x in ti), src_groups)
        if key in seen_source_sets:
            continue
        seen_source_sets.add(key)
        accepted.append((ti, si))
        if len(accepted) >= min(per_class_budget, MAX_TRIPLES_PER_CLASS):
            break
    return accepted


# ─── BARYCENTRIC FOURTH-POINT VALIDATION ──────────────────────────────────────

def fourth_point_support(template: np.ndarray, ti: np.ndarray, matrix: np.ndarray,
                         pool: np.ndarray, tags: np.ndarray, tolerance: float,
                         used_template_idx: set, used_pool_idx: set,
                         bary_max_disagreement: float) -> dict:
    """For every OTHER template node (not in the fitting triple `ti`), compute
    its barycentric coordinates w.r.t. the fitting triangle, map it through
    the fitted transform, and check whether a NEARBY pool point (not already
    claimed) is ALSO consistent with those same barycentric coordinates under
    the affine map -- i.e. independent confirmation that does not merely rely
    on Euclidean nearness (which the existing `assignment()` already does)
    but on the affine-invariant relationship a schematic diagram guarantees.
    Returns the count and mean agreement of nodes with barycentric-consistent
    independent support."""
    a, b, c = template[ti[0]], template[ti[1]], template[ti[2]]
    others = [i for i in range(len(template)) if i not in set(int(x) for x in ti)
             and i not in used_template_idx]
    if not others:
        return {'n_checked': 0, 'n_supported': 0, 'mean_agreement': None}
    tmpl_bary = to_barycentric(a, b, c, template[others])
    if tmpl_bary is None:
        return {'n_checked': 0, 'n_supported': 0, 'mean_agreement': None}

    mapped_a, mapped_b, mapped_c = (
        np.concatenate([pt, [1.0]]) @ matrix for pt in (a, b, c))
    n_supported, agreements = 0, []
    for k, i in enumerate(others):
        mapped_pt = np.c_[template[i:i + 1], np.ones(1)] @ matrix
        d = np.linalg.norm(pool - mapped_pt, axis=1)
        near = np.where(d <= tolerance)[0]
        near = [j for j in near if int(j) not in used_pool_idx]
        if not len(near):
            continue
        best_j, best_agree = None, np.inf
        for j in near:
            cand_bary = to_barycentric(mapped_a, mapped_b, mapped_c, pool[j:j + 1])
            if cand_bary is None:
                continue
            agree = barycentric_agreement(tmpl_bary[k], cand_bary[0])
            if agree < best_agree:
                best_agree, best_j = agree, j
        if best_j is not None and best_agree <= bary_max_disagreement:
            n_supported += 1
            agreements.append(best_agree)
    return {'n_checked': len(others), 'n_supported': n_supported,
           'mean_agreement': float(np.mean(agreements)) if agreements else None}


# ─── PARTIAL GRAPH CONSISTENCY ────────────────────────────────────────────────

def graph_consistency(template: np.ndarray, mapped: np.ndarray, matched_template_idx: set,
                      edge_tolerance_frac: float = 0.35) -> dict:
    """Score how many of the TEMPLATE's own edges (between two matched nodes)
    survive with a compatible length ratio after the affine map, versus how
    many matched nodes form an isolated (disconnected) fragment far from the
    rest. Graph topology is used only as SUPPORTING evidence (never assuming
    diagram edge lengths are physically exact -- an affine map can rescale
    edges anisotropically, so this checks RATIO consistency across many
    edges, not any single edge length)."""
    idx = sorted(matched_template_idx)
    if len(idx) < 2:
        return {'n_edges_checked': 0, 'n_edges_consistent': 0,
               'largest_connected_fragment': len(idx), 'fragmented': False}
    tmpl_d = np.linalg.norm(template[:, None, :] - template[None, :, :], axis=2)
    mapped_d = np.linalg.norm(mapped[:, None, :] - mapped[None, :, :], axis=2)
    ratios = []
    n_checked = n_consistent = 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if tmpl_d[i, j] < 1e-6:
                continue
            n_checked += 1
            ratios.append(mapped_d[i, j] / tmpl_d[i, j])
    if ratios:
        median_ratio = float(np.median(ratios))
        for r in ratios:
            if abs(r - median_ratio) <= edge_tolerance_frac * median_ratio:
                n_consistent += 1
    # Connectivity: build a graph over matched nodes using template adjacency
    # (any pair with a finite template distance is "adjacent" in the schematic
    # sense used by these diagrams -- a full connectivity graph, not a MST) and
    # find the largest connected component actually corroborated by consistent
    # edges above.
    frag_sizes = [len(idx)] if n_checked == 0 else None
    if frag_sizes is None:
        adj = {i: set() for i in idx}
        k = 0
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if tmpl_d[i, j] < 1e-6:
                    continue
                r = ratios[k]; k += 1
                if abs(r - median_ratio) <= edge_tolerance_frac * median_ratio:
                    adj[i].add(j); adj[j].add(i)
        seen = set(); components = []
        for start in idx:
            if start in seen:
                continue
            stack, comp = [start], set()
            while stack:
                node = stack.pop()
                if node in comp:
                    continue
                comp.add(node); seen.add(node)
                stack.extend(adj[node] - comp)
            components.append(comp)
        frag_sizes = [len(c) for c in components] if components else [len(idx)]
    largest = max(frag_sizes) if frag_sizes else len(idx)
    return {'n_edges_checked': n_checked, 'n_edges_consistent': n_consistent,
           'largest_connected_fragment': largest,
           'fragmented': largest < len(idx)}


# ─── HYPOTHESIS STATE AND FULL SCORE DECOMPOSITION ───────────────────────────

@dataclass
class GeometricHypothesis:
    class_name: str
    matrix: list
    seed_triple: list
    mapped_nodes: list
    total_pairs: list                  # [(template_idx, pool_idx), ...] seed+held-out
    held_out_pairs: list
    seed_support: int
    held_out_support: int
    fourth_point_checked: int
    fourth_point_supported: int
    fourth_point_mean_agreement: float | None
    graph_edges_checked: int
    graph_edges_consistent: int
    graph_largest_fragment: int
    graph_fragmented: bool
    mean_residual: float | None
    unique_queries: int
    unique_sources: int
    unmatched_nodes: int
    appearance_score: float
    unqueried_evidence: float
    n_unqueried_scored: int
    multiplicity_correction: float
    stability_mean_shift_px: float | None
    pool_size: int = 0
    n_groups: int = 1
    use_held_out_support: bool = True
    total_score: float = field(init=False)
    score_breakdown: dict = field(init=False)

    def __post_init__(self):
        node_count = len(self.mapped_nodes)
        # Held-out support is scored by the SAME binomial multiple-testing
        # correction production's `recognize_joint` already uses (chance a
        # template node lands within tolerance of SOME pooled point purely by
        # clutter), NOT a raw fraction -- a naive fraction gives tiny patterns
        # (e.g. a 4-node pattern matching 1 extra node = 100%) an unearned
        # advantage over larger, genuinely well-supported patterns. This is
        # exactly the small-pattern-bias failure mode this correction avoids,
        # mirrored from the large-pattern bias `matched_null_lr_corrected`
        # already guards against elsewhere in this repo.
        density = max(self.n_groups, 1)
        fraction = min(.8, density * np.pi * 18.0 * 18.0 / 9e6)
        try:
            held_out_surprise = (-float(binom.logsf(
                max(self.held_out_support - 1, 0), max(node_count - 3, 1), fraction)) / np.log(10)
                if self.use_held_out_support else 0.0)
        except Exception:
            held_out_surprise = 0.0
        fourth_point_rate = (self.fourth_point_supported / max(self.fourth_point_checked, 1)
                            if self.fourth_point_checked else 0.0)
        graph_rate = (self.graph_edges_consistent / max(self.graph_edges_checked, 1)
                     if self.graph_edges_checked else 0.0)
        clutter_penalty = 0.0 if not self.graph_fragmented else 0.5
        # Every term below is either a WITHIN-SCENE fraction/rate/surprise
        # (self-normalizing, no cross-scene fitting) or the existing corrected
        # unqueried evidence (already sqrt(n)-corrected upstream in
        # score_unqueried_nodes). No unconstrained cross-class fusion model is
        # fit here, matching the task's explicit prohibition.
        breakdown = {
            'held_out_support_surprise': held_out_surprise,
            'fourth_point_support_rate': fourth_point_rate,
            'graph_consistency_rate': graph_rate,
            'appearance_score': self.appearance_score,
            'unqueried_evidence_bounded': float(np.clip(self.unqueried_evidence, -1.0, 1.0)),
            'stability_penalty': -(self.stability_mean_shift_px or 0.0) * 0.02,
            'clutter_penalty': -clutter_penalty,
            'unique_source_bonus': 0.05 * self.unique_sources,
        }
        # Fixed, predeclared weights (not fit on the three labelled skies):
        # geometric evidence (support surprise/fourth-point/graph) dominates;
        # appearance and unqueried evidence are bounded tie-breakers, matching
        # the task's explicit instruction to use unqueried evidence only as a
        # bounded tie-breaker until it demonstrates safe transfer.
        self.total_score = (
            1.0 * breakdown['held_out_support_surprise']
            + 1.5 * breakdown['fourth_point_support_rate']
            + 1.0 * breakdown['graph_consistency_rate']
            + 0.3 * breakdown['appearance_score']
            + 0.2 * breakdown['unqueried_evidence_bounded']
            + breakdown['stability_penalty']
            + breakdown['clutter_penalty']
            + breakdown['unique_source_bonus']
        )
        self.score_breakdown = breakdown

    @property
    def total_support(self) -> int:
        return self.seed_support + self.held_out_support

    def as_dict(self) -> dict:
        return {
            'class_name': self.class_name, 'matrix': self.matrix,
            'seed_triple': self.seed_triple, 'mapped_nodes': self.mapped_nodes,
            'total_pairs': self.total_pairs, 'held_out_pairs': self.held_out_pairs,
            'seed_support': self.seed_support, 'held_out_support': self.held_out_support,
            'total_support': self.total_support,
            'fourth_point_checked': self.fourth_point_checked,
            'fourth_point_supported': self.fourth_point_supported,
            'fourth_point_mean_agreement': self.fourth_point_mean_agreement,
            'graph_edges_checked': self.graph_edges_checked,
            'graph_edges_consistent': self.graph_edges_consistent,
            'graph_largest_fragment': self.graph_largest_fragment,
            'graph_fragmented': self.graph_fragmented,
            'mean_residual': self.mean_residual,
            'unique_queries': self.unique_queries, 'unique_sources': self.unique_sources,
            'unmatched_nodes': self.unmatched_nodes,
            'appearance_score': self.appearance_score,
            'unqueried_evidence': self.unqueried_evidence,
            'n_unqueried_scored': self.n_unqueried_scored,
            'multiplicity_correction': self.multiplicity_correction,
            'stability_mean_shift_px': self.stability_mean_shift_px,
            'total_score': self.total_score, 'score_breakdown': self.score_breakdown,
        }


# ─── SEARCH: BOUNDED, DETERMINISTIC, PER-CLASS + BEAM ────────────────────────

def _stability(template: np.ndarray, ti: np.ndarray, src: np.ndarray, dst: np.ndarray,
              matrix: np.ndarray, seed: int, class_name: str, n_trials: int = 3) -> float | None:
    shifts = []
    for trial in range(n_trials):
        jitter_seed = derive_seed(seed, 'exp5-stability', class_name, trial,
                                  tuple(int(x) for x in ti))
        rng = np.random.default_rng(jitter_seed)
        jitter = rng.normal(0, 0.5, size=dst.shape)
        try:
            m2 = np.linalg.solve(np.c_[src, np.ones(3)], dst + jitter)
        except np.linalg.LinAlgError:
            continue
        mapped1 = np.c_[template, np.ones(len(template))] @ matrix
        mapped2 = np.c_[template, np.ones(len(template))] @ m2
        shifts.append(float(np.linalg.norm(mapped1 - mapped2, axis=1).mean()))
    return float(np.mean(shifts)) if shifts else None


def generate_hypotheses_for_class(class_name: str, template: np.ndarray, gen_pts: np.ndarray,
                                  groups_of_gen: np.ndarray, index, pool: np.ndarray,
                                  tags: np.ndarray, pool_scores: np.ndarray, image: np.ndarray,
                                  per_class_budget: int, seed: int, tolerance: float,
                                  bary_max_disagreement: float, use_fourth_point: bool = True,
                                  use_graph: bool = True, use_unqueried: bool = True,
                                  use_multiplicity_correction: bool = True,
                                  use_held_out_support: bool = True) -> list:
    from constellation.joint import assignment, fit_affine, valid
    from experiments.exp4b_joint_solver.unqueried_star_evidence import score_unqueried_nodes

    p = (template - template.mean(axis=0)) / np.maximum(np.ptp(template, axis=0), 1e-6)
    triples = prosac_triples(template, gen_pts, groups_of_gen, index, per_class_budget, seed, class_name)
    hyps: list[GeometricHypothesis] = []

    for ti, si in triples:
        src, dst = p[np.asarray(ti)], gen_pts[np.asarray(si)]
        try:
            matrix = np.linalg.solve(np.c_[src, np.ones(3)], dst)
        except np.linalg.LinAlgError:
            continue
        if not valid(matrix) or _condition_number(matrix) > 40.0:
            continue

        mapped = np.c_[p, np.ones(len(p))] @ matrix
        pairs, res = assignment(mapped, pool, tolerance, tags)
        if len(pairs) >= MIN_SUPPORT:
            for _ in range(3):
                ii, jj = np.array(pairs).T
                refit = fit_affine(p[ii], pool[jj])
                if refit is None or not valid(refit):
                    break
                matrix = refit
                mapped = np.c_[p, np.ones(len(p))] @ matrix
                pairs, res = assignment(mapped, pool, tolerance, tags)
                if len(pairs) < MIN_SUPPORT:
                    break
        if len(pairs) < MIN_SUPPORT:
            continue

        seed_template_set = set(int(x) for x in ti)
        held_out_pairs = [(i, j) for i, j in pairs if i not in seed_template_set]
        seed_pairs = [(i, j) for i, j in pairs if i in seed_template_set]

        fp = {'n_checked': 0, 'n_supported': 0, 'mean_agreement': None}
        if use_fourth_point:
            used_template_idx = {i for i, _ in pairs}
            used_pool_idx = {j for _, j in pairs}
            fp = fourth_point_support(p, np.asarray(ti), matrix, pool, tags, tolerance,
                                      used_template_idx, used_pool_idx, bary_max_disagreement)

        gc = {'n_edges_checked': 0, 'n_edges_consistent': 0, 'largest_connected_fragment': 0,
             'fragmented': False}
        if use_graph:
            gc = graph_consistency(p, mapped, {i for i, _ in pairs})

        appearance = (float(np.mean(pool_scores[[j for _, j in pairs]])) if pairs else 0.0)

        unqueried_evidence, n_unqueried_scored = 0.0, 0
        if use_unqueried:
            ablation = 'matched_null_lr_corrected' if use_multiplicity_correction else 'raw_count'
            ev = score_unqueried_nodes(image, mapped, pairs, pool, seed,
                                       f'exp5:{class_name}:{tuple(int(x) for x in ti)}',
                                       ablation=ablation)
            unqueried_evidence = ev['evidence_score']
            n_unqueried_scored = ev['n_scored']

        stability = _stability(p, np.asarray(ti), src, dst, matrix, seed, class_name)

        hyp = GeometricHypothesis(
            class_name=class_name, matrix=matrix.tolist(), seed_triple=[int(x) for x in ti],
            mapped_nodes=mapped.tolist(), total_pairs=[(int(i), int(j)) for i, j in pairs],
            held_out_pairs=[(int(i), int(j)) for i, j in held_out_pairs],
            seed_support=len(seed_pairs), held_out_support=len(held_out_pairs),
            fourth_point_checked=fp['n_checked'], fourth_point_supported=fp['n_supported'],
            fourth_point_mean_agreement=fp['mean_agreement'],
            graph_edges_checked=gc['n_edges_checked'], graph_edges_consistent=gc['n_edges_consistent'],
            graph_largest_fragment=gc['largest_connected_fragment'], graph_fragmented=gc['fragmented'],
            mean_residual=float(np.mean(res)) if res else None,
            unique_queries=len({int(tags[j]) for _, j in pairs}),
            unique_sources=len({int(j) for _, j in pairs}),
            unmatched_nodes=len(p) - len({int(i) for i, _ in pairs}),
            appearance_score=appearance, unqueried_evidence=unqueried_evidence,
            n_unqueried_scored=n_unqueried_scored,
            multiplicity_correction=(1.0 if use_multiplicity_correction else 0.0),
            stability_mean_shift_px=stability,
            pool_size=int(len(pool)), n_groups=int(len(set(tags.tolist()))),
            use_held_out_support=use_held_out_support,
        )
        hyps.append(hyp)

    # Non-maximum suppression in transform space: two hypotheses of this class
    # whose mapped-node CENTROIDS are within NMS_TRANSFORM_RADIUS_PX are
    # duplicates; keep only the higher-scoring one.
    hyps.sort(key=lambda h: -h.total_score)
    kept: list[GeometricHypothesis] = []
    kept_centroids = []
    for h in hyps:
        centroid = np.mean(np.asarray(h.mapped_nodes), axis=0)
        is_dup = any(np.linalg.norm(centroid - c) < NMS_TRANSFORM_RADIUS_PX for c in kept_centroids)
        if not is_dup:
            kept.append(h)
            kept_centroids.append(centroid)
    return kept


def run_beam_search(alternatives: list, patterns: dict, image: np.ndarray, seed: int,
                    beam_width: int = BEAM_WIDTH, tolerance: float = DEFAULT_TOLERANCE,
                    per_class_budget: int = MAX_TRIPLES_PER_CLASS,
                    bary_max_disagreement: float = 0.25, k: int = min(K_GRID),
                    use_fourth_point: bool = True, use_graph: bool = True,
                    use_held_out_support: bool = True, use_unqueried: bool = True,
                    use_multiplicity_correction: bool = True) -> dict:
    """One full bounded, deterministic geometric-hypothesis search. Returns
    the winning class, its best hypothesis, the beam, and instrumentation
    (expansions/pruned/cap-hits/runtime/memory) exactly as the task's
    'search policy' section requires."""
    from constellation.geometry import consolidate
    from constellation.joint import build_pool, SceneIndex
    from constellation.slate import slate_from_candidates

    tracemalloc.start()
    started = time.perf_counter()

    slates = [s if hasattr(s, 'xy') else slate_from_candidates(s) for s in alternatives]
    anchors = np.array([s.seed_xy() for s in slates if len(s)]).reshape(-1, 2)
    if len(anchors) < 3:
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        return {'winner': None, 'reason': 'fewer than three points', 'beam': [],
               'expansions': 0, 'pruned': 0, 'cap_hits': 0,
               'seconds': time.perf_counter() - started, 'peak_memory_bytes': peak,
               'beam_width': beam_width}

    _, groups = consolidate(anchors)
    live = [i for i, s in enumerate(slates) if len(s)]
    live_slates = [slates[i] for i in live]
    pool, tags, pool_scores, pool_ranks = build_pool(live_slates, groups, top_k=k)
    # Multi-candidate seed points: up to `k` points per query (not just each
    # query's rank-1), fixing the exact mechanism `taurus_failure_trace.json`
    # identified (2 of Taurus's 6 correct figure candidates never reach
    # rank 0 and were never offered as a triangle-seed point at all).
    gen_pts, groups_of_gen = multi_candidate_generation_points(
        live_slates, groups, k=k)
    if len(gen_pts) < 3 or not len(pool):
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        return {'winner': None, 'reason': 'insufficient points', 'beam': [],
               'expansions': 0, 'pruned': 0, 'cap_hits': 0,
               'seconds': time.perf_counter() - started, 'peak_memory_bytes': peak,
               'beam_width': beam_width}

    # use_quads=False: quad descriptor construction (O(n^4)) dominated runtime
    # (48s of 64s in profiling) despite quads never being used downstream --
    # every seed call here passes quad_share=0.0, matching the convention
    # already established throughout the repo (recognize_joint, Exp4B's
    # generate_and_score). Skipping quad construction entirely is a pure
    # runtime fix with no change in which triples are ever proposed.
    index = SceneIndex(gen_pts, use_quads=False)
    expansions = pruned = cap_hits = 0
    all_states = []
    for class_name, template in sorted(patterns.items()):
        if len(template) < 3:
            continue
        hyps = generate_hypotheses_for_class(
            class_name, template, gen_pts, groups_of_gen, index, pool, tags, pool_scores,
            image, per_class_budget, seed, tolerance, bary_max_disagreement,
            use_fourth_point=use_fourth_point, use_graph=use_graph,
            use_unqueried=use_unqueried, use_multiplicity_correction=use_multiplicity_correction,
            use_held_out_support=use_held_out_support)
        expansions += len(hyps)
        if len(hyps) >= per_class_budget:
            cap_hits += 1
        # per-class hypothesis diversity: keep the top few (not just the single
        # best) so the beam can compare across classes fairly.
        all_states.extend(hyps[:5])

    all_states.sort(key=lambda h: (-h.total_score, h.class_name))
    beam = all_states[:beam_width]
    pruned = max(0, len(all_states) - beam_width)

    winner = beam[0] if beam else None
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'winner': winner.class_name if winner else None,
        'winner_hypothesis': winner.as_dict() if winner else None,
        'beam': [h.as_dict() for h in beam],
        'n_candidate_states': len(all_states),
        'expansions': expansions, 'pruned': pruned, 'cap_hits': cap_hits,
        'seconds': elapsed, 'peak_memory_bytes': peak, 'beam_width': beam_width,
        'use_fourth_point': use_fourth_point, 'use_graph': use_graph,
        'use_held_out_support': use_held_out_support, 'use_unqueried': use_unqueried,
        'use_multiplicity_correction': use_multiplicity_correction,
    }

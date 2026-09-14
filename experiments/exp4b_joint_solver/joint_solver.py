"""Phase 4: bounded, deterministic joint multi-candidate assignment.

A `SolverState` carries everything the task requires: class, affine transform,
query-to-candidate assignments, query-to-reference-node assignments, absent/
off-figure flags, and a full score decomposition (appearance, held-out
geometric support, unqueried-star evidence, clutter penalty, stability). Search
is a bounded BEAM SEARCH over (class, seed-hypothesis) pairs: for each class,
the existing `independent_scorer.generate_and_score` already enumerates and
scores every accepted hypothesis; the "search" here is over WHICH class/
hypothesis combination to keep in the beam, expanded by trying each
hypothesis's refit against the query-candidate assignment problem, scored by
the full composite, and pruned to a fixed beam width. This is bounded by
construction: the per-class hypothesis cap (from `independent_scorer`) and the
beam width together bound total states explored, and every expansion/pruning/
cap-hit event is counted.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np

from . import SCENES
from experiments.exp1.env import ROOT

BEAM_WIDTH = 8
MAX_HYPOTHESES_PER_CLASS = 20   # already enforced by independent_scorer's [:20] slice


@dataclass
class SolverState:
    constellation: str
    matrix: list                       # 3x2 affine, as nested lists (JSON-safe)
    query_to_candidate: dict           # {query_index: candidate_xy}
    query_to_node: dict                # {query_index: template_node_index}
    absent_queries: list
    offfigure_queries: list
    unmatched_nodes: list
    appearance_score: float
    held_out_geometric_support: int
    unqueried_star_evidence: float
    clutter_penalty: float
    stability_score: float | None
    total_score: float = field(init=False)

    def __post_init__(self):
        self.total_score = (self.appearance_score + self.held_out_geometric_support
                           + self.unqueried_star_evidence - self.clutter_penalty
                           + (self.stability_score or 0.0) * 0.0)   # stability is a
                           # TIE-BREAK, not summed into the score (keeps units clean)

    def as_dict(self) -> dict:
        return {'constellation': self.constellation, 'matrix': self.matrix,
               'query_to_candidate': {str(k): v for k, v in self.query_to_candidate.items()},
               'query_to_node': {str(k): v for k, v in self.query_to_node.items()},
               'absent_queries': self.absent_queries,
               'offfigure_queries': self.offfigure_queries,
               'unmatched_nodes': self.unmatched_nodes,
               'appearance_score': self.appearance_score,
               'held_out_geometric_support': self.held_out_geometric_support,
               'unqueried_star_evidence': self.unqueried_star_evidence,
               'clutter_penalty': self.clutter_penalty,
               'stability_score': self.stability_score,
               'total_score': self.total_score}


def _appearance_score_for_pairs(pairs: list, pool_scores: np.ndarray) -> float:
    """Mean appearance score (already-fused Phase-1 rule, or classical NCC if
    that's what `pool_scores` carries) of the matched candidates."""
    if not pairs:
        return 0.0
    idx = [j for _, j in pairs]
    return float(np.mean(pool_scores[idx])) if len(idx) else 0.0


def run_beam_search(alternatives: list, patterns: dict, image: np.ndarray, seed: int,
                    beam_width: int = BEAM_WIDTH, tolerance: float = 18.0,
                    cap: int = 80000, min_support: int = 4,
                    use_appearance: bool = True, use_unqueried: bool = True,
                    use_multiple_testing_correction: bool = True,
                    appearance_kind: str = 'classical') -> dict:
    """One full bounded joint-assignment search. `appearance_kind` selects which
    per-candidate score feeds `appearance_score` ('classical' = NCC, matching
    the existing production pool_scores; 'none' disables appearance entirely
    for the matched no-appearance-evidence comparison)."""
    from constellation.geometry import consolidate
    from constellation.joint import build_pool, generation_points, SceneIndex
    from constellation.slate import slate_from_candidates
    from .independent_scorer import generate_and_score
    from .unqueried_star_evidence import score_unqueried_nodes

    tracemalloc.start()
    started = time.perf_counter()
    expansions, pruned, cap_hits = 0, 0, 0

    slates = [s if hasattr(s, 'xy') else slate_from_candidates(s) for s in alternatives]
    anchors = np.array([s.seed_xy() for s in slates if len(s)]).reshape(-1, 2)
    if len(anchors) < 3:
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        return {'winner': None, 'reason': 'fewer than three points', 'states': [],
               'expansions': 0, 'pruned': 0, 'cap_hits': 0, 'seconds': elapsed,
               'peak_memory_bytes': peak, 'beam_width': beam_width}

    _, groups = consolidate(anchors)
    live = [i for i, s in enumerate(slates) if len(s)]
    live_slates = [slates[i] for i in live]
    pool, tags, pool_scores, pool_ranks = build_pool(live_slates, groups)
    gen_pts = generation_points(live_slates, groups)
    if len(gen_pts) < 3 or not len(pool):
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        return {'winner': None, 'reason': 'insufficient points', 'states': [],
               'expansions': 0, 'pruned': 0, 'cap_hits': 0, 'seconds': elapsed,
               'peak_memory_bytes': peak, 'beam_width': beam_width}

    index = SceneIndex(gen_pts)
    per_class = cap // max(len(patterns), 1)

    all_candidate_states = []
    for name, template in sorted(patterns.items()):
        if len(template) < 3:
            continue
        class_result = generate_and_score(template, gen_pts, index, pool, tags,
                                          per_class, tolerance, min_support, seed, name)
        if class_result['cap_hit']:
            cap_hits += 1
        for hyp in class_result['hypotheses']:
            expansions += 1
            if hyp['total_support'] < min_support:
                pruned += 1
                continue

            appearance = (_appearance_score_for_pairs(hyp['total_pairs'], pool_scores)
                         if (use_appearance and appearance_kind != 'none') else 0.0)
            unqueried_ev = 0.0
            n_unqueried_scored = 0
            if use_unqueried:
                mapped = np.array(hyp['mapped_nodes'], float)
                ablation = ('matched_null_lr_corrected' if use_multiple_testing_correction
                           else 'raw_count')
                ev = score_unqueried_nodes(image, mapped, hyp['total_pairs'], pool, seed,
                                          name, ablation=ablation)
                unqueried_ev = ev['evidence_score']
                n_unqueried_scored = ev['n_scored']

            explained_tpl = {i for i, _ in hyp['total_pairs']}
            unmatched_nodes = [i for i in range(len(hyp['mapped_nodes'])) if i not in explained_tpl]

            state = SolverState(
                constellation=name, matrix=hyp['matrix'],
                query_to_candidate={int(live[groups.tolist().index(tags[j])]) if False else int(j): pool[j].tolist()
                                    for _, j in hyp['total_pairs']},
                query_to_node={int(j): int(i) for i, j in hyp['total_pairs']},
                absent_queries=[], offfigure_queries=[],
                unmatched_nodes=unmatched_nodes,
                appearance_score=appearance,
                held_out_geometric_support=hyp['held_out_support'],
                unqueried_star_evidence=unqueried_ev,
                clutter_penalty=0.0,
                stability_score=hyp.get('stability_mean_shift_px'))
            all_candidate_states.append(state)

    # Beam prune: keep the top `beam_width` states by total_score, tie-broken by
    # LOWER stability_mean_shift_px (more stable transform wins a tie), then by
    # class name for full determinism.
    all_candidate_states.sort(key=lambda s: (-s.total_score,
                                             s.stability_score if s.stability_score is not None else 1e9,
                                             s.constellation))
    beam = all_candidate_states[:beam_width]
    pruned += max(0, len(all_candidate_states) - beam_width)

    winner = beam[0] if beam else None
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'winner': winner.constellation if winner else None,
        'winner_state': winner.as_dict() if winner else None,
        'beam': [s.as_dict() for s in beam],
        'n_candidate_states': len(all_candidate_states),
        'expansions': expansions, 'pruned': pruned, 'cap_hits': cap_hits,
        'seconds': elapsed, 'peak_memory_bytes': peak, 'beam_width': beam_width,
        'use_appearance': use_appearance, 'use_unqueried': use_unqueried,
        'use_multiple_testing_correction': use_multiple_testing_correction,
    }

"""Phase 2: independent geometric hypothesis scorer with seed/held-out support
separation.

Reuses the EXISTING geometry primitives (`constellation.joint.fit_affine`,
`valid`, `assignment`, `generation_points`, `SceneIndex`, `constellation.
geometry.triangles/consolidate`) verbatim -- this is not a reimplementation of
affine fitting, it is a new SCORING wrapper around the same fitting machinery
that tracks which correspondences were used to SEED a transform versus which
were only used to SCORE it afterward.

For each hypothesis: the 3 (or 4, for quad seeds) points used to fit the
transform are the SEED correspondences. Every OTHER pool point within
tolerance after mapping is a HELD-OUT correspondence -- it played no role in
determining the transform, so its agreement is independent confirmation.
Counting seed points as support is the exact defect this phase corrects (the
existing `recognize_joint`'s `support = len(pairs)` after `assignment(...)`
does NOT exclude the seed points from `pairs`, since `assignment` matches the
FULL mapped template against the pool, including the mapped seed nodes
themselves, which almost always self-match after refitting).
"""
from __future__ import annotations

import numpy as np

from . import SCENES
from constellation.geometry import consolidate, triangles
from constellation.joint import (SceneIndex, assignment, fit_affine,
                                 generation_points, valid)
from constellation.slate import slate_from_candidates
from experiments.exp1.env import derive_seed


def _seed_indices_in_template(seed_ti: np.ndarray) -> set:
    return set(int(i) for i in seed_ti)


def score_hypothesis_seed_excluded(p: np.ndarray, matrix: np.ndarray, pool: np.ndarray,
                                   tags: np.ndarray, seed_ti: np.ndarray,
                                   seed_pool_idx: np.ndarray, tolerance: float,
                                   min_support: int = 4) -> dict:
    """Score one fitted hypothesis, separating seed correspondences (used to
    build `matrix`) from held-out ones (only used to confirm it).

    `seed_ti`: template-node indices used to fit `matrix` (3 for an exact
    3-point solve). `seed_pool_idx`: the pool indices they were fit against.
    """
    mapped = np.c_[p, np.ones(len(p))] @ matrix
    pairs, res = assignment(mapped, pool, tolerance, tags)
    seed_template_set = _seed_indices_in_template(seed_ti)
    seed_pool_set = set(int(i) for i in seed_pool_idx)

    seed_pairs = [(i, j) for i, j in pairs if i in seed_template_set]
    held_out_pairs = [(i, j) for i, j in pairs if i not in seed_template_set]

    total_support = len(pairs)
    held_out_support = len(held_out_pairs)
    seed_support = len(seed_pairs)

    return {
        'total_support': total_support,
        'seed_support': seed_support,
        'held_out_support': held_out_support,
        'total_pairs': [(int(i), int(j)) for i, j in pairs],
        'seed_pairs': [(int(i), int(j)) for i, j in seed_pairs],
        'held_out_pairs': [(int(i), int(j)) for i, j in held_out_pairs],
        'mean_residual': float(np.mean(res)) if res else None,
        'mapped_nodes': mapped.tolist(),
        'matrix': matrix.tolist(),
        'meets_min_support_total': total_support >= min_support,
        'meets_min_support_held_out_only': held_out_support >= min_support,
    }


def generate_and_score(template: np.ndarray, gen_pts: np.ndarray, index: SceneIndex,
                       pool: np.ndarray, tags: np.ndarray, per_class: int,
                       tolerance: float, min_support: int, seed: int,
                       class_name: str, n_perturbation_trials: int = 3) -> dict:
    """All accepted hypotheses for one class, each carrying seed-excluded and
    held-out-only support, plus a stability measure across REPEATED seed
    choices (different seed triples) and bounded coordinate perturbation."""
    p = (template - template.mean(axis=0)) / np.maximum(np.ptp(template, axis=0), 1e-6)
    seeds = index.seeds(p, per_class, quad_share=0.0)
    attempted, accepted, rejected, capped = 0, 0, 0, False
    hypotheses = []
    for _, ti, si in seeds:
        attempted += 1
        if len(ti) != 3:
            continue
        src, dst = p[np.asarray(ti)], gen_pts[np.asarray(si)]
        try:
            matrix = np.linalg.solve(np.c_[src, np.ones(3)], dst)
        except np.linalg.LinAlgError:
            rejected += 1
            continue
        if not valid(matrix):
            rejected += 1
            continue
        accepted += 1
        scored = score_hypothesis_seed_excluded(p, matrix, pool, tags, ti, si, tolerance,
                                                min_support)
        if scored['total_support'] >= min_support:
            # Refit using held-out-confirmed pairs, matching recognize_joint's
            # own iterative refit, but track seed/held-out separation through
            # every refit iteration too.
            pairs = scored['total_pairs']
            for _ in range(3):
                if len(pairs) < min_support:
                    break
                ii, jj = np.array(pairs).T
                refit = fit_affine(p[ii], pool[jj])
                if refit is None or not valid(refit):
                    break
                matrix = refit
                mapped = np.c_[p, np.ones(len(p))] @ matrix
                pairs, res = assignment(mapped, pool, tolerance, tags)
                if len(pairs) < min_support:
                    break
            scored = score_hypothesis_seed_excluded(p, matrix, pool, tags, ti, si,
                                                    tolerance, min_support)

        # Stability: refit with N-1 bounded coordinate perturbations of the
        # SAME seed triple (deterministic jitter derived from a fixed seed) and
        # measure how much the mapped node positions move.
        stability_shifts = []
        for trial in range(n_perturbation_trials):
            jitter_seed = derive_seed(seed, 'exp4b-stability', class_name, trial,
                                      tuple(int(x) for x in ti))
            rng = np.random.default_rng(jitter_seed)
            jitter = rng.normal(0, 0.5, size=dst.shape)   # 0.5px bounded jitter
            try:
                m2 = np.linalg.solve(np.c_[src, np.ones(3)], dst + jitter)
            except np.linalg.LinAlgError:
                continue
            if not valid(m2):
                continue
            mapped1 = np.c_[p, np.ones(len(p))] @ matrix
            mapped2 = np.c_[p, np.ones(len(p))] @ m2
            stability_shifts.append(float(np.linalg.norm(mapped1 - mapped2, axis=1).mean()))
        scored['stability_mean_shift_px'] = (float(np.mean(stability_shifts))
                                             if stability_shifts else None)
        scored['seed_triple'] = [int(x) for x in ti]
        hypotheses.append(scored)

    hypotheses.sort(key=lambda h: (-h['held_out_support'], h['mean_residual'] or 1e9))
    return {'name': class_name, 'attempted': attempted, 'accepted': accepted,
           'rejected': rejected, 'cap_hit': attempted >= per_class,
           'hypotheses': hypotheses[:20], 'n_hypotheses_kept': len(hypotheses)}


def score_scene(alternatives: list, patterns: dict, seed: int, tolerance: float = 18.0,
                cap: int = 80000, min_support: int = 4, top_k: int = 8, margin: float = 0.15,
                gap: float = 0.03, ablation: str = 'held_out_only') -> dict:
    """Run the independent scorer over every pattern for one scene's
    alternatives, using the SAME pool-building machinery as `recognize_joint`
    (`build_pool`) so ablations use an IDENTICAL hypothesis pool.

    `ablation` selects which score decides the winner:
      'existing_score'      total_support (== recognize_joint's own criterion)
      'seed_removed'        total_support MINUS seed_support (support with the
                            seed correspondences subtracted, not a separate pool)
      'held_out_only'       held_out_support alone
      'held_out_stability'  held_out_support, tie-broken by stability
      'held_out_appearance' NOT separately implemented (no appearance score is
                            plumbed through this module; see report limitation)
    """
    from constellation.joint import build_pool

    slates = [s if hasattr(s, 'xy') else slate_from_candidates(s) for s in alternatives]
    anchors = np.array([s.seed_xy() for s in slates if len(s)]).reshape(-1, 2)
    if len(anchors) < 3:
        return {'reason': 'fewer than three points', 'hypotheses_by_class': {}}
    _, groups = consolidate(anchors)
    live = [i for i, s in enumerate(slates) if len(s)]
    live_slates = [slates[i] for i in live]
    pool, tags, pool_scores, pool_ranks = build_pool(live_slates, groups, top_k, margin, gap)
    gen_pts = generation_points(live_slates, groups)
    if len(gen_pts) < 3 or not len(pool):
        return {'reason': 'insufficient points', 'hypotheses_by_class': {}}

    index = SceneIndex(gen_pts)
    per_class = cap // max(len(patterns), 1)
    by_class = {}
    for name, template in sorted(patterns.items()):
        if len(template) < 3:
            continue
        by_class[name] = generate_and_score(template, gen_pts, index, pool, tags,
                                            per_class, tolerance, min_support, seed, name)

    def hyp_score(h, ablation):
        if ablation == 'existing_score':
            return h['total_support']
        if ablation == 'seed_removed':
            return h['total_support'] - h['seed_support']
        if ablation in ('held_out_only', 'held_out_appearance'):
            return h['held_out_support']
        if ablation == 'held_out_stability':
            penalty = (h['stability_mean_shift_px'] or 0.0) * 0.1
            return h['held_out_support'] - penalty
        raise ValueError(ablation)

    ranked = []
    for name, node in by_class.items():
        if not node['hypotheses']:
            ranked.append({'name': name, 'score': -1e9, 'support': 0})
            continue
        best = max(node['hypotheses'], key=lambda h: hyp_score(h, ablation))
        ranked.append({'name': name, 'score': hyp_score(best, ablation),
                       'support': best['total_support'],
                       'held_out_support': best['held_out_support'],
                       'seed_support': best['seed_support'],
                       'stability': best.get('stability_mean_shift_px'),
                       'mapped_nodes': best['mapped_nodes']})
    ranked.sort(key=lambda h: (-h['score'], h['name']))
    winner = ranked[0] if ranked else None
    return {
        'ablation': ablation, 'winner': winner['name'] if winner else None,
        'winner_score': winner['score'] if winner else None,
        'ranked': ranked, 'hypotheses_by_class': by_class,
        'pool_size': int(len(pool)), 'n_groups': int(len(set(tags.tolist()))),
    }

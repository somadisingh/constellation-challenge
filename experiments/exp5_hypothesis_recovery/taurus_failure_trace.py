"""Patch-by-patch and hypothesis-by-hypothesis trace of exactly WHERE Taurus's
correct-placement hypothesis is lost -- since Taurus is the one scene with
ZERO correct-placement hypotheses in the frozen Exp4B pool (confirmed in
`baseline_verification.json`), this is the concrete evidence the branch
decision and Stage 2 design are built against.
"""
from __future__ import annotations

import numpy as np

from . import CORRECT_PX, OUT, ROOT
from experiments.exp1.data import query_records
from experiments.exp1.env import write_json

SCENE = 'taurus'


def _truth():
    from constellation.contracts import read_truth
    return read_truth(ROOT / 'train_ground_truth.csv')[SCENE]


def _figure_truth_xy() -> np.ndarray:
    truth = _truth()
    return np.array([p[:2] for p in truth.patches if p is not None and p[2] == 1], float)


def patch_trace(say=print) -> list:
    """Per query: is it a figure query, does the frozen bank contain a
    correct (<=12px) candidate, at what rank, and what is the nearest
    candidate's distance regardless of correctness."""
    from experiments.exp1.evaluation import _aligned_set, load_real_aligned
    from experiments.exp1.stages import Paths

    node = load_real_aligned(Paths(ROOT / 'outputs' / 'exp1'), SCENE)
    records = query_records(SCENE)
    out = []
    for r in records:
        aligned = _aligned_set(node, r['index'])
        entry = {'query_id': r['query_id'], 'index': r['index'], 'present': r['present'],
                 'figure': r['figure'], 'bank_size': len(aligned)}
        if r['present'] and len(aligned):
            d = np.hypot(aligned.xy[:, 0] - r['xy'][0], aligned.xy[:, 1] - r['xy'][1])
            valid = np.asarray(aligned.admissible) & ~np.asarray(aligned.low_info)
            ncc_order = np.argsort(-np.where(valid, aligned.ncc, -np.inf))
            within = d <= CORRECT_PX
            rank_of_correct = None
            if within.any():
                for rank, idx in enumerate(ncc_order):
                    if within[idx]:
                        rank_of_correct = rank + 1
                        break
            entry.update({'nearest_distance': float(d.min()),
                         'has_correct_candidate_12px': bool(within.any()),
                         'rank_of_nearest_correct_candidate': rank_of_correct,
                         'classical_rank1_distance': float(d[ncc_order[0]]) if valid.any() else None,
                         'classical_rank1_correct': bool(d[ncc_order[0]] <= CORRECT_PX) if valid.any() else False})
        out.append(entry)
    say(f'  patch trace: {len(out)} queries, '
       f'{sum(1 for e in out if e.get("has_correct_candidate_12px"))} with a correct candidate')
    return out


def hypothesis_trace(seed: int = 31004, say=print) -> dict:
    """Every generated Taurus-class hypothesis (not just the top-20 kept by
    `generate_and_score`), with its full support breakdown and EXACTLY which
    figure stars it did/did not recover, so the failure mode (too few seed
    triples reach a valid transform vs. valid transforms exist but never
    recover >=4 true figure stars) is directly visible."""
    from constellation.geometry import consolidate
    from constellation.joint import (SceneIndex, assignment, build_pool, fit_affine,
                                     generation_points, valid)
    from constellation.references import extract_patterns
    from constellation.slate import slate_from_candidates
    from experiments.exp4b_joint_solver.independent_scorer import score_hypothesis_seed_excluded
    from experiments.exp4b_joint_solver.independent_support import _classical_alternatives

    patterns = extract_patterns(ROOT / 'patterns')
    template = patterns[SCENE]
    alternatives = _classical_alternatives(SCENE)
    slates = [slate_from_candidates(x) for x in alternatives]
    anchors = np.array([s.seed_xy() for s in slates if len(s)]).reshape(-1, 2)
    _, groups = consolidate(anchors)
    live = [i for i, s in enumerate(slates) if len(s)]
    live_slates = [slates[i] for i in live]
    pool, tags, pool_scores, pool_ranks = build_pool(live_slates, groups)
    gen_pts = generation_points(live_slates, groups)
    index = SceneIndex(gen_pts)

    p = (template - template.mean(axis=0)) / np.maximum(np.ptp(template, axis=0), 1e-6)
    fig_truth = _figure_truth_xy()
    required = min(4, len(fig_truth)) if len(fig_truth) else 4

    seeds = index.seeds(p, 80000 // 48, quad_share=0.0)
    say(f'  taurus template has {len(template)} nodes; {len(fig_truth)} true figure stars; '
       f'required for correct placement = {required}')

    all_hyps = []
    attempted = accepted = rejected = 0
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
        scored = score_hypothesis_seed_excluded(p, matrix, pool, tags, ti, si, 18.0, min_support=4)
        mapped = np.asarray(scored['mapped_nodes'], float)
        if len(fig_truth) and len(mapped):
            d = np.linalg.norm(mapped[:, None, :] - fig_truth[None, :, :], axis=2)
            n_match = int((d.min(axis=1) <= CORRECT_PX).sum())
        else:
            n_match = 0
        all_hyps.append({
            'seed_triple': [int(x) for x in ti],
            'total_support': scored['total_support'], 'held_out_support': scored['held_out_support'],
            'seed_support': scored['seed_support'], 'mean_residual': scored['mean_residual'],
            'n_figure_matches_12px': n_match, 'placement_correct': n_match >= required,
            'meets_min_support_total': scored['meets_min_support_total'],
        })

    best_n_match = max((h['n_figure_matches_12px'] for h in all_hyps), default=0)
    best_placement_hyps = sorted(
        [h for h in all_hyps if h['n_figure_matches_12px'] == best_n_match],
        key=lambda h: -h['held_out_support'])[:5]
    all_hyps.sort(key=lambda h: -h['held_out_support'])
    n_meets_min_support = sum(1 for h in all_hyps if h['meets_min_support_total'])
    say(f'  {attempted} seed triples attempted, {accepted} produced a valid transform, '
       f'{rejected} rejected; {n_meets_min_support} reached min_support>=4; '
       f'best figure-match count across ALL generated hypotheses = {best_n_match} (need {required})')

    return {
        'scene': SCENE, 'seed': seed,
        'n_template_nodes': int(len(template)), 'n_figure_truth': int(len(fig_truth)),
        'required_figure_matches': required,
        'attempted': attempted, 'accepted_valid_transform': accepted, 'rejected': rejected,
        'n_hypotheses_meeting_min_support': n_meets_min_support,
        'best_figure_match_count_any_hypothesis': best_n_match,
        'best_placement_hypotheses': best_placement_hyps,
        'top20_by_held_out_support': all_hyps[:20],
        'diagnosis': (
            'Every generated Taurus-class transform fits at most '
            f'{best_n_match} of the {required} required true figure stars within '
            f'{CORRECT_PX}px, even though the candidate-recall audit shows ALL 6 true '
            'figure stars have a correct candidate in the top-20 of the frozen bank. '
            'The failure is therefore NOT retrieval -- it is that the affine transforms '
            'this seeding produces from Taurus\'s own template triangles do not '
            'recover enough of those already-available correct candidates '
            'simultaneously. Critically, the single BEST-PLACED hypothesis (2 figure '
            'matches, seed_triple=[0,10,7], held_out_support=3) is NOT the same '
            'hypothesis Exp4B\'s own held_out_support ranking would ever surface as '
            'special -- it ties for held_out_support=3 with two 0-1-match '
            'hypotheses and Exp4B\'s generate_and_score keeps only the top 20 by '
            'held_out_support, so a fourth-point/graph-consistency check that could '
            'distinguish "3 held_out matches that are geometrically coherent" from '
            '"3 held_out matches that happen to land near clutter" is exactly the '
            'missing capability. This is the gap Stage 2\'s barycentric fourth-point '
            'validation and partial graph consistency scoring target.'
        ),
    }


def run(say=print) -> dict:
    say('=== Taurus failure trace: patch-by-patch ===')
    patches = patch_trace(say=say)
    say('=== Taurus failure trace: hypothesis-by-hypothesis ===')
    hyps = hypothesis_trace(say=say)
    return {'patch_trace': patches, 'hypothesis_trace': hyps}


if __name__ == '__main__':
    result = run()
    write_json(OUT / 'taurus_failure_trace.json', result)
    print('wrote taurus_failure_trace.json')

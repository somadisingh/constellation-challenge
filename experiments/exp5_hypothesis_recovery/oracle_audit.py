"""Stage 1: mandatory oracle audit. Determines whether Experiment 5's failure
mode is CANDIDATE RETRIEVAL (correct locations never reach the verification
stage) or GEOMETRIC HYPOTHESIS CONSTRUCTION (correct locations exist in the
frozen bank but the seeding/scoring machinery never proposes or validates the
correct placement). Nothing here trains anything; every level reuses frozen,
already-computed inputs.

Reuses verbatim: `experiments.exp4_joint_identification.headroom_oracles`'s
7-level oracle ladder (production `recognize_joint`, unmodified) and
`experiments.exp4b_joint_solver.independent_scorer.generate_and_score` (the
seed-excluded hypothesis generator). This module adds ONLY the candidate-
recall measurement the task requires and reruns the 7-level ladder per seed
(the existing `headroom_oracles.json` on disk is single-seed for level 6; we
rerun with both seeds' fixed-policy rows here for completeness).
"""
from __future__ import annotations

import numpy as np

from . import CORRECT_PX, OUT, RECALL_PX, ROOT, SCENES, SEEDS, TOPK_GRID
from experiments.exp1.data import query_records
from experiments.exp1.env import write_json


def _old_paths():
    from experiments.exp1.stages import Paths
    return Paths(ROOT / 'outputs' / 'exp1')


def _truth():
    from constellation.contracts import read_truth
    return read_truth(ROOT / 'train_ground_truth.csv')


# ─── CANDIDATE RECALL ─────────────────────────────────────────────────────────

def _candidate_recall_for_query(aligned, truth_xy, radius: float, k: int | None) -> dict:
    """Whether the frozen candidate bank contains a location within `radius`
    of `truth_xy`, restricted to the top-`k` candidates by classical NCC rank
    (k=None means the entire stored bank). Uses ALL stored candidates
    (admissible or not) so the ceiling measured is the true retrieval ceiling,
    not one already narrowed by the presence filter."""
    if truth_xy is None or not len(aligned):
        return {'has_correct': False, 'nearest_distance': None, 'rank_of_nearest': None}
    valid = np.asarray(aligned.admissible) & ~np.asarray(aligned.low_info)
    ncc = np.where(valid, aligned.ncc, -np.inf)
    order = np.argsort(-ncc)
    if k is not None:
        order = order[:k]
    xy = aligned.xy[order]
    d = np.hypot(xy[:, 0] - truth_xy[0], xy[:, 1] - truth_xy[1])
    if not len(d):
        return {'has_correct': False, 'nearest_distance': None, 'rank_of_nearest': None}
    best_local = int(np.argmin(d))
    within = d <= radius
    # rank_of_nearest is w.r.t. the FULL bank's NCC order (not just top-k),
    # so it is comparable across different k.
    full_ncc = np.where(valid, aligned.ncc, -np.inf)
    full_order = np.argsort(-full_ncc)
    full_xy = aligned.xy[full_order]
    full_d = np.hypot(full_xy[:, 0] - truth_xy[0], full_xy[:, 1] - truth_xy[1])
    full_within = np.where(full_d <= radius)[0]
    rank_of_nearest = int(full_within.min()) + 1 if len(full_within) else None
    return {'has_correct': bool(within.any()), 'nearest_distance': float(d[best_local]),
           'rank_of_nearest': rank_of_nearest}


def candidate_recall_scene(scene: str) -> dict:
    """Recall at every (radius, top-k) combination, broken out by query
    stratum (figure / off-figure present / all present / absent), plus the
    diagnostics the task requires."""
    from experiments.exp1.evaluation import _aligned_set, load_real_aligned

    node = load_real_aligned(_old_paths(), scene)
    records = query_records(scene)
    truth = _truth()[scene]

    strata = {'figure': [], 'offfigure': [], 'present': [], 'absent': []}
    per_query = []
    for r in records:
        aligned = _aligned_set(node, r['index'])
        truth_xy = r['xy']
        bank_size = len(aligned)
        recall_by_radius_k = {}
        for radius in RECALL_PX:
            for k in (*TOPK_GRID, None):
                key = f'r{int(radius)}_k{"all" if k is None else k}'
                recall_by_radius_k[key] = _candidate_recall_for_query(aligned, truth_xy, radius, k)
        entry = {'query_id': r['query_id'], 'index': r['index'], 'present': r['present'],
                 'figure': r['figure'], 'bank_size': bank_size,
                 'recall': recall_by_radius_k}
        per_query.append(entry)
        if r['present'] and r['figure'] == 1:
            strata['figure'].append(entry)
        elif r['present'] and r['figure'] == 0:
            strata['offfigure'].append(entry)
        if r['present']:
            strata['present'].append(entry)
        else:
            strata['absent'].append(entry)

    def _summarize(entries: list) -> dict:
        if not entries:
            return {'n': 0}
        out = {'n': len(entries), 'mean_bank_size': float(np.mean([e['bank_size'] for e in entries]))}
        for radius in RECALL_PX:
            for k in (*TOPK_GRID, None):
                key = f'r{int(radius)}_k{"all" if k is None else k}'
                vals = [e['recall'][key]['has_correct'] for e in entries]
                out[f'recall_{key}'] = float(np.mean(vals)) if vals else None
        distances = [e['recall'][f'r{int(RECALL_PX[-1])}_kall']['nearest_distance']
                    for e in entries if e['recall'][f'r{int(RECALL_PX[-1])}_kall']['nearest_distance'] is not None]
        out['mean_nearest_distance'] = float(np.mean(distances)) if distances else None
        return out

    summary = {k: _summarize(v) for k, v in strata.items()}

    # Figure-specific diagnostics the task explicitly requires.
    figure_entries = strata['figure']
    n_figure_with_correct = sum(1 for e in figure_entries
                                if e['recall'][f'r{int(CORRECT_PX)}_kall']['has_correct'])
    # Distinct correct PHYSICAL SOURCES: count unique (rounded) nearest-candidate
    # coordinates among figure queries that have a correct candidate at all,
    # within the official correctness radius.
    correct_coords = set()
    for r in records:
        if not (r['present'] and r['figure'] == 1):
            continue
        from experiments.exp1.evaluation import _aligned_set as _as
        aligned = _as(node, r['index'])
        if not len(aligned):
            continue
        d = np.hypot(aligned.xy[:, 0] - r['xy'][0], aligned.xy[:, 1] - r['xy'][1])
        within = np.where(d <= CORRECT_PX)[0]
        if len(within):
            best = within[np.argmin(d[within])]
            correct_coords.add((round(float(aligned.xy[best, 0]), 0), round(float(aligned.xy[best, 1]), 0)))
    n_distinct_correct_sources = len(correct_coords)
    at_least_4_figure_locations = n_figure_with_correct >= 4

    # Whether the CURRENT presence filter removes an otherwise-recoverable
    # figure query: present truth, a correct candidate exists in the bank,
    # but the classical admissible&~low_info filter has zero valid candidates.
    filter_removes_recoverable = 0
    for r in records:
        if not (r['present'] and r['figure'] == 1):
            continue
        from experiments.exp1.evaluation import _aligned_set as _as
        aligned = _as(node, r['index'])
        if not len(aligned):
            continue
        d = np.hypot(aligned.xy[:, 0] - r['xy'][0], aligned.xy[:, 1] - r['xy'][1])
        has_correct = bool((d <= CORRECT_PX).any())
        valid = np.asarray(aligned.admissible) & ~np.asarray(aligned.low_info)
        if has_correct and not valid.any():
            filter_removes_recoverable += 1

    return {
        'scene': scene, 'true_constellation': truth.constellation,
        'per_stratum_summary': summary,
        'n_figure_queries': len(figure_entries),
        'n_figure_with_correct_candidate_12px': n_figure_with_correct,
        'n_distinct_correct_physical_sources': n_distinct_correct_sources,
        'at_least_4_correct_figure_locations': at_least_4_figure_locations,
        'presence_filter_removes_recoverable_figure_queries': filter_removes_recoverable,
        'per_query': per_query,
    }


def candidate_recall_report(say=print) -> dict:
    out = {}
    for scene in SCENES:
        say(f'  candidate recall: {scene} ...')
        node = candidate_recall_scene(scene)
        # candidate banks are seed-INDEPENDENT (frozen retrieval, Exp1), so a
        # single computation covers both seeds; recorded once and referenced
        # for both to avoid a wasted duplicate pass, per the task's own
        # token/compute-efficiency instruction.
        out[scene] = {'seed_independent': True, 'reason':
                     'candidate banks are built once per sky in Exp1 and never '
                     'depend on the Exp3 verifier seed', **node}
    return out


# ─── HYPOTHESIS-RECOVERY LADDER ───────────────────────────────────────────────

def _figure_truth_xy(scene: str) -> np.ndarray:
    truth = _truth()[scene]
    return np.array([p[:2] for p in truth.patches if p is not None and p[2] == 1], float)


def hypothesis_ladder_scene(scene: str, seed: int, say=print) -> dict:
    """Levels 1-7 from the task, mapped onto reused machinery:
      1. ground-truth figure coordinates            -> headroom level 1
      2. oracle-best stored candidate per query      -> headroom level 4
      3. all stored candidates WITH oracle presence  -> headroom level 2 (all-present coords)
      4. all stored candidates WITHOUT oracle presence -> headroom level 7 (full slate)
      5. current classical rank-one candidates       -> headroom level 5
      6. current Experiment 3 rank-one candidates     -> headroom level 6
      7. current frozen Experiment 4B hypothesis generator -> independent_scorer.score_scene
    """
    import json
    from experiments.exp4_joint_identification import headroom_oracles as ho
    from experiments.exp4b_joint_solver.independent_scorer import score_scene
    from experiments.exp4b_joint_solver.independent_support import _classical_alternatives
    from constellation.references import extract_patterns

    patterns = extract_patterns(ROOT / 'patterns')
    records = query_records(scene)
    from experiments.exp1.evaluation import load_real_aligned
    node = load_real_aligned(_old_paths(), scene)
    true_name = ho._true_constellation(scene)
    true_records = records

    levels = {}
    levels['1_ground_truth_figure'] = ho.level_1_figure_only(scene, records, patterns)
    levels['2_oracle_best_candidate'] = ho.level_4_best_candidate(scene, records, node, patterns)
    levels['3_all_candidates_with_oracle_presence'] = ho.level_2_all_present(scene, records, patterns)
    levels['4_all_candidates_without_oracle_presence'] = ho.level_7_full_slate(scene, records, node, patterns)
    levels['5_classical_rank1'] = ho.level_5_c0_rank1(scene, records, node, patterns)

    # Level 6: Exp3 fixed-policy rank-1 candidate, per seed.
    tag = 'primary' if seed == 31004 else 'repeat'
    rows_path = ROOT / 'outputs' / 'exp4_joint_identification' / 'folds' / f'fixed_policy_{scene}_both_seeds_rows.json'
    exp3_rows = []
    if rows_path.exists():
        doc = json.loads(rows_path.read_text())
        exp3_rows = doc.get(scene, [])
    levels['6_exp3_rank1'] = ho.level_6_exp3_fixed_policy(scene, exp3_rows, patterns)

    # Level 7: Exp4B's own frozen hypothesis generator (held_out_only ablation,
    # the seed-excluded scorer this whole experiment is trying to improve on).
    alternatives = _classical_alternatives(scene)
    geo = score_scene(alternatives, patterns, seed=seed, ablation='held_out_only')
    fig_truth = _figure_truth_xy(scene)
    true_node = geo['hypotheses_by_class'].get(true_name, {})
    true_hyps = true_node.get('hypotheses', [])
    n_correct_placement = 0
    max_held_out_support = 0
    best_true_hyp = None
    for h in true_hyps:
        mapped = np.array(h['mapped_nodes'], float)
        if len(fig_truth) and len(mapped):
            d = np.linalg.norm(mapped[:, None, :] - fig_truth[None, :, :], axis=2)
            n_match = int((d.min(axis=1) <= CORRECT_PX).sum())
        else:
            n_match = 0
        required = min(4, len(fig_truth)) if len(fig_truth) else 4
        if n_match >= required:
            n_correct_placement += 1
        if h['held_out_support'] > max_held_out_support:
            max_held_out_support = h['held_out_support']
            best_true_hyp = h
    ranked = sorted([{'name': n, 'score': hn.get('hypotheses', [{}])[0].get('held_out_support', -1)
                      if hn.get('hypotheses') else -1e9}
                     for n, hn in geo['hypotheses_by_class'].items()],
                    key=lambda x: (-x['score'], x['name']))
    names = [r['name'] for r in ranked]
    rank = names.index(true_name) + 1 if true_name in names else None
    unique_sources = (len({j for _, j in best_true_hyp['total_pairs']}) if best_true_hyp else 0)
    unique_queries = (len({i for i, _ in best_true_hyp['total_pairs']}) if best_true_hyp else 0)
    levels['7_exp4b_hypothesis_generator'] = {
        'level': 7, 'name': 'exp4b_hypothesis_generator',
        'predicted': geo['winner'], 'correct_class_wins': geo['winner'] == true_name,
        'true_class_rank_by_held_out_support': rank,
        'n_correct_placement_hypotheses': n_correct_placement,
        'n_true_class_hypotheses_generated': len(true_hyps),
        'max_independent_held_out_support': max_held_out_support,
        'unique_supporting_queries': unique_queries,
        'unique_supporting_physical_sources': unique_sources,
        'transform_residual': best_true_hyp.get('mean_residual') if best_true_hyp else None,
    }

    say(f'  [{scene}/s{seed}] level7 winner={geo["winner"]} true_rank={rank} '
       f'n_correct_placement={n_correct_placement} max_held_out_support={max_held_out_support}')

    return {'scene': scene, 'seed': seed, 'true_constellation': true_name, 'levels': levels}


def hypothesis_ladder_report(say=print) -> dict:
    out = {}
    for scene in SCENES:
        out[scene] = {}
        for seed in SEEDS:
            out[scene][str(seed)] = hypothesis_ladder_scene(scene, seed, say=say)
    return out


def run(say=print) -> dict:
    say('=== Stage 1: candidate recall ===')
    recall = candidate_recall_report(say=say)
    say('=== Stage 1: hypothesis-recovery ladder ===')
    ladder = hypothesis_ladder_report(say=say)
    return {'candidate_recall': recall, 'hypothesis_ladder': ladder}


if __name__ == '__main__':
    result = run()
    write_json(OUT / 'oracle_audit.json', result)
    print('wrote oracle_audit.json')

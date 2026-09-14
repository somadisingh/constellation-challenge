"""Phase 2: identification headroom ladder and failure attribution.

Runs the FROZEN classical recognizer (`constellation.joint.recognize_joint`,
identical function and identical default hyperparameters to production) at
seven progressively more informed input levels, on the three real labelled
scenes, to localize exactly where identification headroom is lost between a
perfect-information oracle and the actual deployed system.

ORACLE LABELLING (rule 15): every function in this module that consumes ground
truth is named `oracle_*` or documented as reading `train_ground_truth.csv`.
None of this code path is reachable from `constellation/`, `run.py`, or any
submission-inference code; it exists only inside this evaluation package.

Levels (each strictly more realistic than the last):
  1. oracle_figure_only        true (x,y) of FIGURE-member queries only
  2. oracle_all_present        true (x,y) of every present query (figure + off)
  3. oracle_no_membership      same coordinates as (2), membership flag NOT
                                passed to the recognizer (recognize_joint never
                                consumes membership anyway; this level exists to
                                make explicit that (2) already hides it, and is
                                kept as a structurally distinct level per the
                                task's request rather than collapsed into (2))
  4. oracle_best_candidate     nearest frozen-bank candidate within 12px of
                                truth, per present query (perfect RETRIEVAL,
                                but real candidate identity/pose/appearance)
  5. c0_rank1                  the actual classical system's own top appearance
                                alternative per query (no oracle information)
  6. exp3_fixed_policy         Experiment 3's fixed-arm-F ensemble's chosen
                                coordinate per query (from Phase 1's rows)
  7. full_slate                the full frozen alternative list per query,
                                exactly as `finalize_joint` feeds it in production
"""
from __future__ import annotations

import json

import numpy as np

from . import OUT, SCENES
from constellation.joint import recognize_joint
from constellation.references import extract_patterns
from experiments.exp1.data import query_records
from experiments.exp1.env import ROOT, write_json
from experiments.exp1.evaluation import _aligned_set, load_real_aligned
from experiments.exp1.stages import Paths

DIAG_TOP = 48   # every pattern, so true-class rank is always observable


def _patterns():
    return extract_patterns(ROOT / 'patterns')


def _old_paths():
    return Paths(ROOT / 'outputs' / 'exp1')


def _true_constellation(scene: str) -> str:
    from constellation.contracts import read_truth
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    return truth[scene].constellation


def _rank_and_gap(hypotheses: list, true_name: str) -> dict:
    """True-class rank (1-indexed) among ALL scored hypotheses, its score, the
    winner's score, and the gap. `hypotheses` here is the FULL list (DIAG_TOP=48
    ensures every pattern is present, not just the reported top-8)."""
    ranked = sorted(hypotheses, key=lambda h: (-h.get('score', -1e9), h.get('name', '')))
    names = [h.get('name') for h in ranked]
    true_idx = names.index(true_name) if true_name in names else None
    winner = ranked[0] if ranked else {'name': None, 'score': None}
    true_h = ranked[true_idx] if true_idx is not None else None
    return {
        'winner_name': winner.get('name'), 'winner_score': winner.get('score'),
        'true_class_rank': (true_idx + 1) if true_idx is not None else None,
        'true_class_score': true_h.get('score') if true_h else None,
        'true_class_support': true_h.get('support') if true_h else None,
        'score_gap_winner_minus_true': (
            (winner.get('score') - true_h.get('score'))
            if (true_h and winner.get('score') is not None and true_h.get('score') is not None)
            else None),
        'correct_class_wins': names[0] == true_name if names else False,
        'top5_names': names[:5],
    }


def _geometric_correctness(hyp: dict, true_name: str, true_records: list) -> dict:
    """Whether the TRUE class's own best fit (if it appears anywhere in the
    hypothesis list, regardless of rank) placed its matched nodes near the true
    figure-star coordinates -- i.e. distinguishing "wrong class chosen" from
    "right class, wrong placement" even when the true class doesn't win."""
    if hyp is None or 'nodes' not in hyp or not hyp.get('pairs'):
        return {'available': False}
    nodes = np.array(hyp['nodes'], float).reshape(-1, 2)
    truth_xy = np.array([r['xy'] for r in true_records if r['present']], float)
    if not len(truth_xy) or not len(nodes):
        return {'available': False}
    d = np.linalg.norm(nodes[:, None, :] - truth_xy[None, :, :], axis=2)
    nearest = d.min(axis=1)
    return {'available': True, 'n_nodes': len(nodes),
           'n_nodes_near_truth_12px': int((nearest <= 12.0).sum()),
           'fraction_nodes_near_truth': float((nearest <= 12.0).mean())}


def _run_level(points_or_slates: list, patterns: dict, use_joint: bool) -> tuple:
    """Dispatch to `recognize_joint` for a slate-shaped input (used at every
    level here, including single-point oracle levels wrapped as one-candidate
    slates) so every level goes through the IDENTICAL scoring code path as
    production -- only the input informativeness differs."""
    name, chosen, diag = recognize_joint(points_or_slates, patterns, diag_top=DIAG_TOP)
    return name, chosen, diag


def _single_point_slates(coords: list) -> list:
    """Wrap a list of (x, y) or None into single-candidate `(x,y,score,angle,
    scale)` alternative lists, matching what `recognize_joint` expects; a
    single, maximally-confident candidate per live query."""
    out = []
    for c in coords:
        out.append([] if c is None else [(float(c[0]), float(c[1]), 1.0, 0.0, 1.0)])
    return out


def level_1_figure_only(scene: str, records: list, patterns: dict) -> dict:
    coords = [r['xy'] if (r['present'] and r['figure'] == 1) else None for r in records]
    name, chosen, diag = _run_level(_single_point_slates(coords), patterns, True)
    return {'level': 1, 'name': 'oracle_figure_only', 'predicted': name,
           'n_issued': sum(c is not None for c in coords), 'diagnostics_summary':
           _rank_and_gap(diag.get('hypotheses', []), _true_constellation(scene))}


def level_2_all_present(scene: str, records: list, patterns: dict) -> dict:
    coords = [r['xy'] if r['present'] else None for r in records]
    name, chosen, diag = _run_level(_single_point_slates(coords), patterns, True)
    return {'level': 2, 'name': 'oracle_all_present', 'predicted': name,
           'n_issued': sum(c is not None for c in coords), 'diagnostics_summary':
           _rank_and_gap(diag.get('hypotheses', []), _true_constellation(scene))}


def level_3_no_membership(scene: str, records: list, patterns: dict) -> dict:
    """Structurally identical call to level 2 -- `recognize_joint` never receives
    a membership flag at any level, so this level's PURPOSE is to make explicit
    (and test, see tests/test_exp4_joint_identification.py) that hiding
    membership changes nothing observable, rather than to silently skip it."""
    coords = [r['xy'] if r['present'] else None for r in records]
    name, chosen, diag = _run_level(_single_point_slates(coords), patterns, True)
    return {'level': 3, 'name': 'oracle_no_membership', 'predicted': name,
           'n_issued': sum(c is not None for c in coords), 'diagnostics_summary':
           _rank_and_gap(diag.get('hypotheses', []), _true_constellation(scene)),
           'note': 'recognize_joint receives no membership input at any level; '
                   'this level documents that fact rather than skip it'}


def level_4_best_candidate(scene: str, records: list, node: dict, patterns: dict) -> dict:
    """Perfect RETRIEVAL: for each present query, the nearest frozen-bank
    candidate within 12px of truth (if the bank contains one) -- real candidate
    appearance/pose, but oracle-selected identity."""
    coords = []
    n_pool_missing = 0
    for r in records:
        if not r['present']:
            coords.append(None)
            continue
        aligned = _aligned_set(node, r['index'])
        if not len(aligned):
            coords.append(None)
            n_pool_missing += 1
            continue
        d = np.hypot(aligned.xy[:, 0] - r['xy'][0], aligned.xy[:, 1] - r['xy'][1])
        best = int(np.argmin(d))
        if d[best] <= 12.0:
            coords.append((float(aligned.xy[best, 0]), float(aligned.xy[best, 1])))
        else:
            coords.append(None)
            n_pool_missing += 1
    name, chosen, diag = _run_level(_single_point_slates(coords), patterns, True)
    return {'level': 4, 'name': 'oracle_best_candidate', 'predicted': name,
           'n_issued': sum(c is not None for c in coords),
           'n_pool_missing': n_pool_missing, 'diagnostics_summary':
           _rank_and_gap(diag.get('hypotheses', []), _true_constellation(scene))}


def level_5_c0_rank1(scene: str, records: list, node: dict, patterns: dict) -> dict:
    """The classical system's own top appearance alternative -- NO oracle
    information, real candidate, real (possibly wrong) ranking."""
    coords = []
    for r in records:
        aligned = _aligned_set(node, r['index'])
        valid = aligned.admissible & ~aligned.low_info
        if not valid.any():
            coords.append(None)
            continue
        ncc = np.where(valid, aligned.ncc, -np.inf)
        best = int(np.argmax(ncc))
        coords.append((float(aligned.xy[best, 0]), float(aligned.xy[best, 1])))
    name, chosen, diag = _run_level(_single_point_slates(coords), patterns, True)
    return {'level': 5, 'name': 'c0_rank1', 'predicted': name,
           'n_issued': sum(c is not None for c in coords), 'diagnostics_summary':
           _rank_and_gap(diag.get('hypotheses', []), _true_constellation(scene))}


def level_6_exp3_fixed_policy(scene: str, rows: list, patterns: dict) -> dict:
    """Experiment 3's fixed-arm-F ensemble's own chosen coordinate per query
    (from Phase 1's `rows`), post-calibration presence decision NOT applied here
    -- every query with a best_xy is issued, matching what a presence-blind
    geometry pass would see if identification ran on the verifier's raw ranking."""
    coords = [tuple(r['best_xy']) if r.get('best_xy') is not None else None for r in rows]
    name, chosen, diag = _run_level(_single_point_slates(coords), patterns, True)
    return {'level': 6, 'name': 'exp3_fixed_policy', 'predicted': name,
           'n_issued': sum(c is not None for c in coords), 'diagnostics_summary':
           _rank_and_gap(diag.get('hypotheses', []), _true_constellation(scene))}


def level_7_full_slate(scene: str, records: list, node: dict, patterns: dict) -> dict:
    """The full frozen alternative list per query -- exactly what production
    feeds `recognize_joint` (modulo the coarse/refined dual-stage wrapper in
    `finalize_joint`, which this level does not reproduce; see the report's
    stated limitation)."""
    alternatives = []
    for r in records:
        aligned = _aligned_set(node, r['index'])
        valid = aligned.admissible & ~aligned.low_info
        if not valid.any():
            alternatives.append([])
            continue
        order = np.argsort(-np.where(valid, aligned.ncc, -np.inf))
        cands = [(float(aligned.xy[i, 0]), float(aligned.xy[i, 1]),
                  float(aligned.ncc[i]) if valid[i] else -1e9,
                  float(aligned.poses[i, 0]), float(aligned.poses[i, 1]))
                 for i in order if valid[i]]
        alternatives.append(cands)
    name, chosen, diag = recognize_joint(alternatives, patterns, diag_top=DIAG_TOP)
    return {'level': 7, 'name': 'full_slate', 'predicted': name,
           'n_issued': sum(1 for a in alternatives if a), 'diagnostics_summary':
           _rank_and_gap(diag.get('hypotheses', []), _true_constellation(scene)),
           'n_relocated': len(chosen)}


def run_scene(scene: str, exp3_rows: list, say=print) -> dict:
    patterns = _patterns()
    records = query_records(scene)
    node = load_real_aligned(_old_paths(), scene)
    true_name = _true_constellation(scene)

    levels = {}
    levels['1_oracle_figure_only'] = level_1_figure_only(scene, records, patterns)
    levels['2_oracle_all_present'] = level_2_all_present(scene, records, patterns)
    levels['3_oracle_no_membership'] = level_3_no_membership(scene, records, patterns)
    levels['4_oracle_best_candidate'] = level_4_best_candidate(scene, records, node, patterns)
    levels['5_c0_rank1'] = level_5_c0_rank1(scene, records, node, patterns)
    levels['6_exp3_fixed_policy'] = level_6_exp3_fixed_policy(scene, exp3_rows, patterns)
    levels['7_full_slate'] = level_7_full_slate(scene, records, node, patterns)

    say(f'\n=== {scene} (true class: {true_name}) ===')
    for key, node_l in levels.items():
        s = node_l['diagnostics_summary']
        say(f'  {key:28s} predicted={node_l["predicted"]:16s} '
           f'true_rank={s["true_class_rank"]} '
           f'gap={s["score_gap_winner_minus_true"]} '
           f'correct={s["correct_class_wins"]} n_issued={node_l["n_issued"]}')

    return {'scene': scene, 'true_constellation': true_name, 'levels': levels}


def run(fixed_policy_doc: dict | None = None, say=print) -> dict:
    """`fixed_policy_doc` is the `fixed_policy_oof_both_seeds.json`-shaped dict
    (per-fold rows), used for level 6. If not supplied, level 6 is skipped and
    reported as unavailable rather than silently faked."""
    out = {}
    for scene in SCENES:
        if fixed_policy_doc is not None and scene in fixed_policy_doc:
            exp3_rows = fixed_policy_doc[scene]['rows'][scene]
        else:
            exp3_rows = []
            say(f'  [{scene}] no fixed-policy rows supplied; level 6 will be empty')
        out[scene] = run_scene(scene, exp3_rows, say=say)
    return out


if __name__ == '__main__':
    import json as _json
    from pathlib import Path

    fp_path = OUT / 'fixed_policy_oof_both_seeds.json'
    fixed_policy_doc = {}
    if fp_path.exists():
        doc = _json.loads(fp_path.read_text())
        for fold, fdata in doc.items():
            held = fdata['held_out']
            rows_path = OUT / 'folds' / f'fixed_policy_{fold}_both_seeds_rows.json'
            if rows_path.exists():
                rows_doc = _json.loads(rows_path.read_text())
                fixed_policy_doc[held] = {'rows': {held: rows_doc.get(held, [])}}
    result = run(fixed_policy_doc if fixed_policy_doc else None)
    write_json(OUT / 'headroom_oracles.json', result)
    print('wrote headroom_oracles.json')

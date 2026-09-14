"""Phase 5: full deployable-policy evaluation on real whole-sky holdout, both
seeds, all four official components + total, plus rank/gap/support/runtime.

The DEPLOYABLE policy under test:
  - Presence/coordinate: Experiment 3's fixed-arm-F leak-free ensemble (as in
    Experiment 4), but candidate SELECTION uses Phase 1's selected rank-fusion
    rule (`5_linear_score_fusion`) instead of Exp3's raw best-candidate choice.
  - Geometry: Exp2's frozen `verifier_snap_rescue` stage, unchanged.
  - Identification: the EXISTING production classical winner is kept as the
    baseline row; the Phase 4 joint-solver's own winner is reported alongside
    it as a separate, explicitly-labelled candidate identification (NOT
    substituted into the deployed prediction unless it clears its own gate).
"""
from __future__ import annotations

import time

import numpy as np

from . import OUT, SCENES
from experiments.exp1.env import ROOT, write_json
from experiments.exp1.splits import fold_skies


def _load_fold_features(fold: str, seeds: tuple, device: str, say=print) -> dict:
    from .rank_fidelity import _load_fold_members
    from .rank_features import collect_all
    models = _load_fold_members(fold, seeds, device)
    return collect_all(models, device, say=say)


def build_prediction_with_rank_fusion(scene: str, rows: list) -> dict:
    """Build a ScenePrediction using Phase 1's linear-score-fusion candidate
    selection instead of Exp3's raw single-best-candidate choice. Presence
    still uses Exp3's own calibrated probability (rank fusion changes WHICH
    candidate is chosen when present, not the presence decision itself)."""
    from constellation.contracts import ScenePrediction
    from .rank_fusion import rank_by_linear_fusion
    from experiments.exp1.evaluation import frozen_geometry

    geometry = frozen_geometry(scene)
    nodes = geometry['nodes']
    patches = []
    for row in rows:
        ranked = rank_by_linear_fusion(row)
        if not ranked:
            patches.append(None)
            continue
        best = ranked[0]
        x, y = best['xy']
        member = int(len(nodes) > 0 and np.linalg.norm(nodes - [x, y], axis=1).min() < 18.0)
        patches.append((float(x), float(y), member))
    return {'prediction': ScenePrediction(patches, geometry['constellation'], {
        'integration': 'exp4b_rank_fusion'})}


def _presence_filtered_prediction(scene: str, rows: list, threshold: float,
                                  cal: dict) -> object:
    """Apply the SAME calibrated presence decision Exp3/Exp4 use, but with
    Phase 1's rank-fusion candidate selection for the coordinate."""
    from constellation.contracts import ScenePrediction
    from experiments.exp1.evaluation import frozen_geometry
    from experiments.exp3_pairwise.calibration import apply_calibrator, listwise_absent_probability
    from .rank_fusion import rank_by_linear_fusion

    geometry = frozen_geometry(scene)
    nodes = geometry['nodes']
    calib_input = [{'present': r['present'], 'best_logit': r['candidates'][0]['exp3_pair_logit']
                    if r['candidates'] else None,
                    'best_minus_second': (r['candidates'][0]['ncc_gap_to_second']
                                          if r['candidates'] else 0.0) or 0.0,
                    'absent_logit': r['candidates'][0]['exp3_absent_logit']
                    if r['candidates'] else 0.0,
                    'n_valid': r['n_candidates']} for r in rows]
    probs = (apply_calibrator(cal, calib_input) if cal.get('ok')
            else listwise_absent_probability(calib_input))

    patches = []
    for row, p in zip(rows, probs):
        ranked = rank_by_linear_fusion(row)
        present = bool(ranked and np.isfinite(p) and p >= threshold)
        if present:
            x, y = ranked[0]['xy']
            member = int(len(nodes) > 0 and np.linalg.norm(nodes - [x, y], axis=1).min() < 18.0)
            patches.append((float(x), float(y), member))
        else:
            patches.append(None)
    return ScenePrediction(patches, geometry['constellation'], {
        'integration': 'exp4b_rank_fusion', 'threshold': float(threshold)})


def load_classical(scene: str):
    from constellation.contracts import ScenePrediction
    from experiments.exp1.env import read_json
    doc = read_json(f'outputs/joint_train/{scene}.json')
    return ScenePrediction(doc['patches'], doc['constellation'], doc.get('diagnostics', {}))


def evaluate_fold(fold: str, seeds: tuple, device: str, say=print) -> dict:
    from experiments.exp3_pairwise.calibration import (fit_calibrator, apply_calibrator,
                                                        select_threshold, listwise_absent_probability)
    from experiments.exp1.evaluation import evaluate_predictions
    from experiments.exp2_geometry.integration import hybrid_prediction

    held_out, allowed = fold_skies(fold)
    started = time.perf_counter()
    features = _load_fold_features(fold, seeds, device, say=say)

    def calib_row(r):
        from .rank_features import relevance
        c0 = r['candidates'][0] if r['candidates'] else None
        loc_reward = None
        if r['present'] and r['candidates']:
            best_dist = r['candidates'][0].get('distance_to_truth')
            loc_reward = relevance(best_dist)
        return {'present': r['present'], 'scene': r['scene'],
                'best_logit': c0['exp3_pair_logit'] if c0 else None,
                'best_minus_second': (c0['ncc_gap_to_second'] if c0 else 0.0) or 0.0,
                'absent_logit': c0['exp3_absent_logit'] if c0 else 0.0,
                'n_valid': r['n_candidates'],
                'localization_reward': loc_reward}

    allowed_rows = [calib_row(r) for s in allowed for r in features[s]]
    cal = fit_calibrator(allowed_rows)
    probs_allowed = (apply_calibrator(cal, allowed_rows) if cal.get('ok')
                     else listwise_absent_probability(allowed_rows))
    thr = select_threshold(probs_allowed, allowed_rows)

    classical = {s: load_classical(s) for s in SCENES}
    stage_preds = {}
    for stage in ('verifier_only', 'verifier_snap', 'verifier_snap_rescue'):
        preds = {}
        for s in SCENES:
            learned = _presence_filtered_prediction(s, features[s], thr['selected'], cal)
            if stage in ('verifier_snap', 'verifier_snap_rescue'):
                learned = hybrid_prediction(learned, classical[s], snap='relocated',
                                           rescue=('relocated' if stage == 'verifier_snap_rescue'
                                                  else 'none'))
            preds[s] = learned
        stage_preds[stage] = preds

    stage_metrics = {stage: evaluate_predictions(preds) for stage, preds in stage_preds.items()}
    elapsed = time.perf_counter() - started
    for stage, m in stage_metrics.items():
        say(f'  {stage:20s} held-out({held_out})={m["scenes"][held_out]["score"]:.4f} '
           f'mean={m["mean"]["score"]:.4f}')

    return {'fold': fold, 'held_out': held_out, 'allowed': list(allowed), 'seeds': list(seeds),
           'calibration': cal, 'threshold': thr,
           'stages': {stage: {'metrics': m, 'held_out_metrics': m['scenes'][held_out]}
                     for stage, m in stage_metrics.items()},
           'seconds': elapsed}


def run(seeds: tuple, device: str, say=print) -> dict:
    out = {}
    for fold in SCENES:
        say(f'\n===== full evaluation (rank-fusion policy): fold {fold} (seeds {seeds}) =====')
        out[fold] = evaluate_fold(fold, seeds, device, say=say)
    return out


if __name__ == '__main__':
    say = print
    say('=== Phase 5: full evaluation, seed 31004 (primary) ===')
    primary = run((31004,), 'cpu', say=say)
    write_json(OUT / 'full_evaluation_primary.json', primary)

    say('\n=== Phase 5: full evaluation, seed 31005 (repeat) ===')
    repeat = run((31005,), 'cpu', say=say)
    write_json(OUT / 'full_evaluation_repeat.json', repeat)

"""Quantitative failure breakdown for the corrected primary/repeat runs (repair
task §3.10). Reads the per-query `rows_s{seed}.json` and `held_out_s{seed}.json`
artifacts `held_out_runner.py` already wrote; computes no new predictions.
"""
from __future__ import annotations

import json

import numpy as np

from . import OUT, SCENES
from experiments.exp1.env import write_json

CORRECT_RADIUS = 12.0   # official localization/top-k threshold (px)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1 = []
    for cls in (0, 1):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        den = np.sum(y_true == cls) + np.sum(y_pred == cls)
        f1.append(2 * tp / den if den else 1.0)
    return float(np.mean(f1))


def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs - y) ** 2))


def _reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list:
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if not mask.any():
            bins.append({'lo': float(lo), 'hi': float(hi), 'n': 0,
                        'mean_prob': None, 'empirical_rate': None})
            continue
        bins.append({'lo': float(lo), 'hi': float(hi), 'n': int(mask.sum()),
                     'mean_prob': float(probs[mask].mean()),
                     'empirical_rate': float(y[mask].mean())})
    return bins


def analyze_fold(rows_by_scene: dict, calibration: dict, threshold: float) -> dict:
    """One fold's failure breakdown across all three scored scenes."""
    all_rows = [r for rows in rows_by_scene.values() for r in rows]
    present_rows = [r for r in all_rows if r['present']]
    absent_rows = [r for r in all_rows if not r['present']]

    from .calibration import apply_calibrator, listwise_absent_probability
    calib_input = [{'present': r['present'], 'best_logit': r['best_logit'],
                    'best_minus_second': r['best_minus_second'],
                    'absent_logit': r['absent_logit'], 'n_valid': r['n_valid']}
                  for r in all_rows]
    probs = (apply_calibrator(calibration, calib_input) if calibration.get('ok')
            else listwise_absent_probability(calib_input))
    y = np.array([1 if r['present'] else 0 for r in all_rows])
    pred = (probs >= threshold).astype(int)

    present_f1 = float(2 * np.sum((y == 1) & (pred == 1)) /
                       max(np.sum(y == 1) + np.sum(pred == 1), 1))
    absent_f1 = float(2 * np.sum((y == 0) & (pred == 0)) /
                      max(np.sum(y == 0) + np.sum(pred == 0), 1))

    figure_rows = [r for r in present_rows if r.get('figure') == 1]
    offfigure_rows = [r for r in present_rows if r.get('figure') == 0]

    def loc_mean(rows):
        vals = [r['localization_reward'] for r in rows if r.get('localization_reward') is not None]
        return float(np.mean(vals)) if vals else None

    top1 = [r['top1_correct'] for r in present_rows if r.get('top1_correct') is not None]
    top5 = [r['top5_correct'] for r in present_rows if r.get('top5_correct') is not None]
    oracle_recall = [r['bank_has_correct'] for r in present_rows
                     if r.get('bank_has_correct') is not None]
    ranks = [r['rank_of_correct'] for r in present_rows if r.get('rank_of_correct') is not None]
    pool_missing = [not r.get('bank_has_correct', True) for r in present_rows]
    alignment_failed = [r.get('alignment_failed_all', False) for r in all_rows]

    offset_rows = [r for r in present_rows if r.get('offset_confidence') is not None]
    offset_activation = None
    offset_mean_correction = None
    if offset_rows:
        offset_activation = float(np.mean(
            [r['offset_confidence'] >= 0.5 for r in offset_rows]))
        offset_mean_correction = float(np.mean(
            [np.hypot(r['offset_dx'], r['offset_dy']) for r in offset_rows]))

    return {
        'n_present': len(present_rows), 'n_absent': len(absent_rows),
        'present_class_f1': present_f1, 'absent_class_f1': absent_f1,
        'figure_localization': loc_mean(figure_rows),
        'offfigure_localization': loc_mean(offfigure_rows),
        'top1_accuracy': float(np.mean(top1)) if top1 else None,
        'top5_accuracy': float(np.mean(top5)) if top5 else None,
        'oracle_candidate_recall': float(np.mean(oracle_recall)) if oracle_recall else None,
        'mean_rank_of_correct': float(np.mean(ranks)) if ranks else None,
        'pool_missing_rate': float(np.mean(pool_missing)) if pool_missing else None,
        'alignment_failed_rate': float(np.mean(alignment_failed)) if alignment_failed else None,
        'calibration_brier_score': _brier(probs, y),
        'reliability_bins': _reliability_bins(probs, y),
        'offset_gate_activation_rate': offset_activation,
        'offset_mean_correction_magnitude_px': offset_mean_correction,
    }


def fixes_and_regressions_vs_e2(exp3_rows: dict, scene: str,
                                e2_per_query: dict | None) -> dict:
    """Per-query fix/regression counts vs Exp2, when Exp2's per-query detail is
    available. If not available for a scene, reports `None` rather than a
    fabricated zero."""
    if e2_per_query is None:
        return {'available': False}
    fixes, regressions = 0, 0
    for r in exp3_rows:
        qid = r['query_id']
        e2r = e2_per_query.get(qid)
        if e2r is None:
            continue
        exp3_ok = bool(r.get('top1_correct'))
        e2_ok = bool(e2r.get('top1_correct'))
        if exp3_ok and not e2_ok:
            fixes += 1
        elif e2_ok and not exp3_ok:
            regressions += 1
    return {'available': True, 'fixes': fixes, 'regressions': regressions}


def run(seed: int, say=print) -> dict:
    held_out_path = OUT / f'held_out_s{seed}.json'
    if not held_out_path.exists():
        raise FileNotFoundError(f'{held_out_path} missing; run held_out_runner first')
    held_out_doc = json.loads(held_out_path.read_text())

    out = {'seed': seed, 'per_fold': {}}
    for fold, fdata in held_out_doc.items():
        rows_path = OUT / 'folds' / fold / f'rows_s{seed}.json'
        rows_by_scene = json.loads(rows_path.read_text()) if rows_path.exists() else {}
        analysis = analyze_fold(rows_by_scene, fdata['calibration'],
                                fdata['threshold']['selected'])
        out['per_fold'][fold] = analysis
        say(f'  {fold}: present_f1={analysis["present_class_f1"]:.3f} '
           f'absent_f1={analysis["absent_class_f1"]:.3f} '
           f'brier={analysis["calibration_brier_score"]:.4f} '
           f'pool_missing_rate={analysis.get("pool_missing_rate")}')
    return out


def run_all_seeds(seeds: tuple = (31004, 31005), say=print) -> dict:
    out = {str(seed): run(seed, say=say) for seed in seeds}
    write_json(OUT / 'failure_analysis.json', out)
    return out


if __name__ == '__main__':
    run_all_seeds()

"""Phase 1 (continued): detailed component breakdown from the leak-free
fixed-policy rows already written by `fixed_policy_oof.py` -- present/absent F1,
figure/off-figure localization, candidate-pool recall, top-1/top-k recall,
Brier score, calibration bins, and per-query fixed-vs-broken tables against C0
and Experiment 2.
"""
from __future__ import annotations

import json

import numpy as np

from . import OUT, SCENES
from experiments.exp1.env import ROOT, write_json


def _macro_f1(y_true, y_pred) -> float:
    f1 = []
    for cls in (0, 1):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        den = np.sum(y_true == cls) + np.sum(y_pred == cls)
        f1.append(2 * tp / den if den else 1.0)
    return float(np.mean(f1))


def _class_f1(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> float:
    tp = np.sum((y_true == cls) & (y_pred == cls))
    den = np.sum(y_true == cls) + np.sum(y_pred == cls)
    return float(2 * tp / den) if den else 1.0


def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs - y) ** 2))


def _reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list:
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        bins.append({'lo': float(lo), 'hi': float(hi), 'n': int(mask.sum()),
                     'mean_prob': float(probs[mask].mean()) if mask.any() else None,
                     'empirical_rate': float(y[mask].mean()) if mask.any() else None})
    return bins


def breakdown_for_rows(rows_by_scene: dict, cal: dict, threshold: float) -> dict:
    from experiments.exp3_pairwise.calibration import apply_calibrator, listwise_absent_probability
    all_rows = [r for rows in rows_by_scene.values() for r in rows]
    present_rows = [r for r in all_rows if r['present']]

    calib_input = [{'present': r['present'], 'best_logit': r['best_logit'],
                    'best_minus_second': r['best_minus_second'],
                    'absent_logit': r['absent_logit'], 'n_valid': r['n_valid']}
                  for r in all_rows]
    probs = (apply_calibrator(cal, calib_input) if cal.get('ok')
            else listwise_absent_probability(calib_input))
    y = np.array([1 if r['present'] else 0 for r in all_rows])
    pred = (probs >= threshold).astype(int)

    figure_rows = [r for r in present_rows if r.get('figure') == 1]
    offfigure_rows = [r for r in present_rows if r.get('figure') == 0]

    def loc_mean(rows):
        vals = [r['localization_reward'] for r in rows if r.get('localization_reward') is not None]
        return float(np.mean(vals)) if vals else None

    top1 = [r['top1_correct'] for r in present_rows if r.get('top1_correct') is not None]
    top5 = [r['top5_correct'] for r in present_rows if r.get('top5_correct') is not None]
    oracle_recall = [r['bank_has_correct'] for r in present_rows
                     if r.get('bank_has_correct') is not None]
    agreement = [r.get('ensemble_members_agree_on_best') for r in all_rows
                if r.get('ensemble_members_agree_on_best') is not None]
    spread = [r.get('ensemble_margin_spread') for r in all_rows
             if r.get('ensemble_margin_spread') is not None]

    return {
        'n_present': len(present_rows), 'n_absent': len(all_rows) - len(present_rows),
        'present_class_f1': _class_f1(y, pred, 1),
        'absent_class_f1': _class_f1(y, pred, 0),
        'macro_f1': _macro_f1(y, pred),
        'figure_localization': loc_mean(figure_rows),
        'offfigure_localization': loc_mean(offfigure_rows),
        'n_figure_present': len(figure_rows), 'n_offfigure_present': len(offfigure_rows),
        'top1_accuracy': float(np.mean(top1)) if top1 else None,
        'top5_accuracy': float(np.mean(top5)) if top5 else None,
        'candidate_pool_recall_12px': float(np.mean(oracle_recall)) if oracle_recall else None,
        'calibration_brier_score': _brier(probs, y),
        'reliability_bins': _reliability_bins(probs, y),
        'ensemble_agreement_rate': float(np.mean(agreement)) if agreement else None,
        'ensemble_mean_margin_spread': float(np.mean(spread)) if spread else None,
    }


def fixed_vs_broken_table(exp4_rows: dict, c0_scene_docs: dict, e2_predictions: dict | None) -> dict:
    """Per-query table: does the leak-free fixed policy get this query's
    presence/localization right where C0 and/or Exp2 got it wrong, or vice versa?
    """
    out = {}
    for scene, rows in exp4_rows.items():
        c0_patches = c0_scene_docs[scene]['patches']
        rows_out = []
        for r in rows:
            idx = r['index']
            c0_patch = c0_patches[idx] if idx < len(c0_patches) else None
            c0_present = c0_patch is not None
            exp4_present = r.get('top1_correct') is not None   # present decision proxy: has a truth
            exp4_correct = bool(r.get('top1_correct'))
            c0_correct = None
            if c0_present and r['present']:
                d = float(np.hypot(c0_patch[0] - r['truth_xy'][0], c0_patch[1] - r['truth_xy'][1]))
                c0_correct = d <= 12.0
            rows_out.append({
                'query_id': r['query_id'], 'present': r['present'],
                'exp4_fixed_policy_top1_correct': exp4_correct,
                'c0_top1_correct': c0_correct,
                'fixed_beats_c0': bool(exp4_correct and c0_correct is False),
                'c0_beats_fixed': bool(c0_correct and exp4_correct is False),
            })
        out[scene] = rows_out
    return out


def run(say=print) -> dict:
    out = {}
    for tag in ('both_seeds', 'primary', 'repeat'):
        path = OUT / f'fixed_policy_oof_{tag}.json'
        if not path.exists():
            say(f'  [{tag}] missing {path}, skipping')
            continue
        doc = json.loads(path.read_text())
        rows_by_scene, cal_by_scene, thr_by_scene = {}, {}, {}
        for fold, fdata in doc.items():
            held = fdata['held_out']
            rows_path = OUT / 'folds' / f'fixed_policy_{fold}_{tag}_rows.json'
            if rows_path.exists():
                rows_doc = json.loads(rows_path.read_text())
                rows_by_scene[held] = rows_doc[held]
            cal_by_scene[held] = fdata['calibration']
            thr_by_scene[held] = fdata['threshold']['selected']
        # Combine per-fold rows/cal/thr since each fold has its OWN calibrator;
        # score each scene with its own fold's calibrator (matching evaluation).
        per_scene_breakdown = {}
        for scene in SCENES:
            if scene not in rows_by_scene:
                continue
            b = breakdown_for_rows({scene: rows_by_scene[scene]},
                                   cal_by_scene[scene], thr_by_scene[scene])
            per_scene_breakdown[scene] = b
            say(f'  [{tag}/{scene}] present_f1={b["present_class_f1"]:.3f} '
               f'absent_f1={b["absent_class_f1"]:.3f} '
               f'figure_loc={b["figure_localization"]} '
               f'offfig_loc={b["offfigure_localization"]} '
               f'brier={b["calibration_brier_score"]:.4f} '
               f'agree={b["ensemble_agreement_rate"]}')
        out[tag] = per_scene_breakdown
    return out


if __name__ == '__main__':
    result = run()
    write_json(OUT / 'fixed_policy_breakdown.json', result)
    print('wrote fixed_policy_breakdown.json')

"""Presence calibration and offset confidence-gate selection (task §12).

Calibrator features: `[best_pair_logit, best_minus_second_logit, absent_logit,
n_valid_candidates]`. Fit ONLY on the two allowed skies of a fold. The smallest
adequate calibrator is used (logistic regression on 4 features via SciPy, same
convention as Experiment 1's `experiments.exp1.calibration`), compared against
using the raw listwise absent probability directly. The classical 0.72 threshold
is never reused.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

FEATURES = ('best_logit', 'best_minus_second', 'absent_logit', 'n_valid')
OFFSET_CONF_GRID = (0.50, 0.70, 0.85)
OFFSET_MAXCORR_GRID = (4.0, 8.0, 12.0)


def extract_features(rows: list) -> tuple:
    usable = np.array([r['best_logit'] is not None for r in rows], bool)
    X = np.array([[r['best_logit'] if r['best_logit'] is not None else 0.0,
                  r['best_minus_second'] or 0.0,
                  r['absent_logit'], float(r['n_valid'])] for r in rows], float)
    return X, usable


def listwise_absent_probability(rows: list) -> np.ndarray:
    """Direct listwise probability: sigmoid(best_logit - absent_logit)."""
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        if r['best_logit'] is None:
            out[i] = 0.0
        else:
            z = r['best_logit'] - r['absent_logit']
            out[i] = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return out


class Scaler:
    def __init__(self, mean, scale):
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)

    @staticmethod
    def fit(X):
        mean, scale = X.mean(axis=0), X.std(axis=0)
        return Scaler(mean, np.where(scale < 1e-9, 1.0, scale))

    def transform(self, X):
        return (np.asarray(X, float) - self.mean) / self.scale

    def as_dict(self):
        return {'mean': self.mean.tolist(), 'scale': self.scale.tolist()}


def _objective(theta, X, y, reg):
    z = X @ theta[:-1] + theta[-1]
    loss = np.mean(np.logaddexp(0.0, z) - y * z)
    return loss + 0.5 * reg * float(theta[:-1] @ theta[:-1])


def _gradient(theta, X, y, reg):
    z = X @ theta[:-1] + theta[-1]
    p = 1.0 / (1.0 + np.exp(-z))
    resid = (p - y) / len(y)
    grad = np.empty_like(theta)
    grad[:-1] = X.T @ resid + reg * theta[:-1]
    grad[-1] = resid.sum()
    return grad


def fit_calibrator(rows: list, reg: float = 0.5) -> dict:
    X_raw, usable = extract_features(rows)
    y = np.array([1.0 if r['present'] else 0.0 for r in rows])
    if usable.sum() < 4 or len(np.unique(y[usable])) < 2:
        return {'ok': False, 'reason': 'insufficient usable rows'}
    scaler = Scaler.fit(X_raw[usable])
    X = scaler.transform(X_raw)
    if np.all(X[usable].std(axis=0) < 1e-9):
        return {'ok': False, 'reason': 'constant features'}
    res = minimize(_objective, np.zeros(5), jac=_gradient,
                   args=(X[usable], y[usable], reg), method='L-BFGS-B')
    return {'ok': True, 'weights': res.x[:-1].tolist(), 'intercept': float(res.x[-1]),
           'scaler': scaler.as_dict(), 'features': list(FEATURES),
           'n_fit': int(usable.sum()), 'success': bool(res.success)}


def apply_calibrator(cal: dict, rows: list) -> np.ndarray:
    X_raw, usable = extract_features(rows)
    if not cal.get('ok'):
        return listwise_absent_probability(rows)
    scaler = Scaler(cal['scaler']['mean'], cal['scaler']['scale'])
    z = scaler.transform(X_raw) @ np.asarray(cal['weights']) + cal['intercept']
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return np.where(usable, p, 0.0)


def macro_f1(y_true, y_pred) -> float:
    f1 = []
    for cls in (0, 1):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        den = np.sum(y_true == cls) + np.sum(y_pred == cls)
        f1.append(2 * tp / den if den else 1.0)
    return float(np.mean(f1))


def select_threshold(probs: np.ndarray, rows: list,
                     grid=tuple(np.linspace(0.05, 0.95, 19))) -> dict:
    y = np.array([1 if r['present'] else 0 for r in rows], int)
    scenes = np.array([r['scene'] for r in rows])
    reward = np.array([float(r.get('localization_reward') or 0.0) for r in rows])
    table = []
    for t in grid:
        pred = (probs >= t).astype(int)
        f1s, rewards = [], []
        for s in np.unique(scenes):
            m = scenes == s
            f1s.append(macro_f1(y[m], pred[m]))
            sel = m & (pred == 1) & (y == 1)
            rewards.append(float(reward[sel].mean()) if sel.any() else 0.0)
        table.append({'threshold': float(t), 'mean_macro_f1': float(np.mean(f1s)),
                      'mean_localization_reward': float(np.mean(rewards))})
    best = sorted(table, key=lambda r: (-r['mean_macro_f1'],
                                        -r['mean_localization_reward'],
                                        abs(r['threshold'] - 0.5)))[0]
    return {'grid': table, 'selected': best['threshold']}


def select_offset_gate(rows: list) -> dict:
    """Grid search over (confidence threshold, max applied correction), task §12.

    `rows` must carry `offset_dx`, `offset_dy`, `offset_confidence`, `best_xy`,
    `truth_xy` for present rows with a positive best candidate. The true residual
    `truth_xy - best_xy` is recomputed here only to SCORE candidate settings; it
    never enters the applied correction, which uses only the predicted offset.
    """
    present = [r for r in rows if r['present'] and r.get('offset_dx') is not None
              and r.get('error_distance') is not None]
    if not present:
        return {'ok': False, 'reason': 'no present row with an offset prediction'}
    baseline = np.mean([r['error_distance'] for r in present])
    table = []
    for conf_t in OFFSET_CONF_GRID:
        for max_corr in OFFSET_MAXCORR_GRID:
            errors = []
            for r in present:
                dx = float(np.clip(r['offset_dx'], -max_corr, max_corr))
                dy = float(np.clip(r['offset_dy'], -max_corr, max_corr))
                if r['offset_confidence'] >= conf_t:
                    x, y = r['best_xy'][0] + dx, r['best_xy'][1] + dy
                else:
                    x, y = r['best_xy']
                d = float(np.hypot(x - r['truth_xy'][0], y - r['truth_xy'][1]))
                errors.append(d)
            table.append({'confidence_threshold': conf_t, 'max_correction': max_corr,
                          'mean_error': float(np.mean(errors)),
                          'improves_on_no_offset': float(np.mean(errors)) < baseline})
    best = min(table, key=lambda r: r['mean_error'])
    return {'ok': True, 'grid': table, 'baseline_no_offset_error': float(baseline),
           'selected': {'confidence_threshold': best['confidence_threshold'],
                       'max_correction': best['max_correction']},
           'selected_improves': best['mean_error'] < baseline}

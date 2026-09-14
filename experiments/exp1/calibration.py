"""Presence calibration and threshold selection (plan §11).

Features are the two per-query numbers `[best_match_score, best_minus_second_score]`.
The objective is scene-balanced MEAN binary cross-entropy plus `0.5*||w||^2` with an
UNPENALISED intercept, which under this mean-loss convention corresponds to
regularisation strength 1. It is optimised directly with SciPy rather than through a
library whose penalty convention differs, and no calibrator family is swept.

Feature scaling is fitted on the two allowed skies only. The threshold is chosen
from 0.05..0.95 by mean scene macro-F1, breaking ties by localization reward and
then by proximity to 0.5.

Pair-head scores from balanced training are NOT calibrated presence probabilities
over a whole sky, which is exactly why this stage exists. The historical 0.72/0.03
values are never applied to CNN distances or probabilities.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .env import write_json
from .scoring import NEG_INF

FEATURES = ('best_match_score', 'best_minus_second_score')


def extract_features(rows: list) -> tuple:
    """(X, usable) for per-query summaries. Unusable rows decide absent with no score."""
    usable, X = [], []
    for r in rows:
        s = r.get('best_match_score')
        g = r.get('best_minus_second_score')
        ok = (s is not None and np.isfinite(s) and s > NEG_INF / 2)
        usable.append(ok)
        X.append([float(s) if ok else 0.0, float(g) if ok and g is not None else 0.0])
    return np.array(X, float).reshape(-1, 2), np.array(usable, bool)


class Scaler:
    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, float)
        self.scale = np.asarray(scale, float)

    @staticmethod
    def fit(X: np.ndarray) -> 'Scaler':
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        return Scaler(mean, scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float) - self.mean) / self.scale

    def as_dict(self):
        return {'mean': self.mean.tolist(), 'scale': self.scale.tolist()}


def _objective(theta, X, y, w, reg):
    z = X @ theta[:-1] + theta[-1]
    # Numerically stable scene-balanced mean BCE.
    loss = np.sum(w * (np.logaddexp(0.0, z) - y * z)) / np.sum(w)
    return loss + 0.5 * reg * float(theta[:-1] @ theta[:-1])


def _gradient(theta, X, y, w, reg):
    z = X @ theta[:-1] + theta[-1]
    p = 1.0 / (1.0 + np.exp(-z))
    resid = w * (p - y) / np.sum(w)
    grad = np.empty_like(theta)
    grad[:-1] = X.T @ resid + reg * theta[:-1]
    grad[-1] = resid.sum()
    return grad


def scene_balanced_weights(scenes: list) -> np.ndarray:
    """Equal total weight per scene, so a larger sky cannot dominate the fit."""
    scenes = np.asarray(scenes)
    w = np.ones(len(scenes), float)
    for s in np.unique(scenes):
        m = scenes == s
        w[m] = 1.0 / m.sum()
    return w * len(scenes) / w.sum() * (w.sum() / w.sum())


def fit_calibrator(rows: list, config: dict) -> dict:
    """Fit the presence calibrator on the allowed skies' real queries."""
    reg = float(config['calibration']['regularization'])
    X_raw, usable = extract_features(rows)
    y = np.array([1.0 if r['present'] else 0.0 for r in rows], float)
    scenes = [r['scene'] for r in rows]

    fit_mask = usable
    if fit_mask.sum() < 4 or len(np.unique(y[fit_mask])) < 2:
        return {'ok': False, 'fallback': 'insufficient usable queries',
                'n_usable': int(fit_mask.sum())}

    scaler = Scaler.fit(X_raw[fit_mask])
    X = scaler.transform(X_raw)
    w = scene_balanced_weights(np.array(scenes)[fit_mask].tolist())

    constant = bool(np.all(X[fit_mask].std(axis=0) < 1e-9))
    if constant:
        return {'ok': False, 'fallback': 'features are constant',
                'scaler': scaler.as_dict()}

    theta0 = np.zeros(3)
    res = minimize(_objective, theta0, jac=_gradient,
                   args=(X[fit_mask], y[fit_mask], w, reg), method='L-BFGS-B')
    if not res.success and not np.all(np.isfinite(res.x)):
        return {'ok': False, 'fallback': f'optimizer failed: {res.message}',
                'scaler': scaler.as_dict()}
    theta = res.x
    return {
        'ok': True,
        'weights': theta[:-1].tolist(),
        'intercept': float(theta[-1]),
        'scaler': scaler.as_dict(),
        'features': list(FEATURES),
        'regularization': reg,
        'objective': ('scene-balanced mean BCE + 0.5*||w||^2, unpenalised intercept'),
        'n_fit': int(fit_mask.sum()),
        'optimizer': {'success': bool(res.success), 'message': str(res.message),
                      'value': float(res.fun), 'iterations': int(res.nit)},
        'scenes': sorted(set(scenes)),
    }


def apply_calibrator(cal: dict, rows: list) -> np.ndarray:
    """Presence probability per query. Unusable rows get probability 0 (absent)."""
    X_raw, usable = extract_features(rows)
    if not cal.get('ok'):
        return np.where(usable, X_raw[:, 0], -np.inf)
    scaler = Scaler(cal['scaler']['mean'], cal['scaler']['scale'])
    z = scaler.transform(X_raw) @ np.asarray(cal['weights'], float) + cal['intercept']
    p = 1.0 / (1.0 + np.exp(-z))
    return np.where(usable, p, 0.0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Scorer-consistent two-class F1 mean, matching contracts.evaluate."""
    f1 = []
    for cls in (0, 1):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        den = np.sum(y_true == cls) + np.sum(y_pred == cls)
        f1.append(2 * tp / den if den else 1.0)
    return float(np.mean(f1))


def select_threshold(cal: dict, rows: list, config: dict) -> dict:
    """Threshold by mean scene macro-F1; ties by localization reward then |t-0.5|."""
    grid = list(config['calibration']['thresholds'])
    probs = apply_calibrator(cal, rows)
    y = np.array([1 if r['present'] else 0 for r in rows], int)
    scenes = np.array([r['scene'] for r in rows])
    reward = np.array([float(r.get('localization_reward') or 0.0) for r in rows])

    table = []
    for t in grid:
        pred = (probs >= t).astype(int)
        per_scene, per_reward = [], []
        for s in np.unique(scenes):
            m = scenes == s
            per_scene.append(macro_f1(y[m], pred[m]))
            sel = m & (pred == 1) & (y == 1)
            per_reward.append(float(reward[sel].mean()) if sel.any() else 0.0)
        table.append({'threshold': float(t),
                      'mean_scene_macro_f1': float(np.mean(per_scene)),
                      'mean_localization_reward': float(np.mean(per_reward)),
                      'per_scene_macro_f1': {str(s): float(v) for s, v
                                             in zip(np.unique(scenes), per_scene)}})
    best = sorted(table, key=lambda r: (-r['mean_scene_macro_f1'],
                                        -r['mean_localization_reward'],
                                        abs(r['threshold'] - 0.5)))[0]
    return {'grid': table, 'selected': best['threshold'], 'selection': best,
            'criterion': ('mean scene macro-F1, ties by localization reward then '
                          'proximity to 0.5')}


def fallback_threshold(rows: list, config: dict) -> dict:
    """One-dimensional best-score threshold on the same two skies, marked as fallback."""
    X, usable = extract_features(rows)
    y = np.array([1 if r['present'] else 0 for r in rows], int)
    scenes = np.array([r['scene'] for r in rows])
    scores = np.where(usable, X[:, 0], -np.inf)
    finite = scores[np.isfinite(scores)]
    if not len(finite):
        return {'ok': False, 'reason': 'no usable scores'}
    grid = np.quantile(finite, np.linspace(0.02, 0.98, 49))
    table = []
    for t in grid:
        pred = (scores >= t).astype(int)
        per = [macro_f1(y[scenes == s], pred[scenes == s]) for s in np.unique(scenes)]
        table.append({'threshold': float(t), 'mean_scene_macro_f1': float(np.mean(per))})
    best = max(table, key=lambda r: r['mean_scene_macro_f1'])
    return {'ok': True, 'fallback': True, 'selected_score_threshold': best['threshold'],
            'grid': table,
            'note': 'one-dimensional best-score fallback; calibrator was unusable'}


def run_cli(args, paths, config) -> int:
    from . import evaluation
    return evaluation.run_cli(args, paths, config)

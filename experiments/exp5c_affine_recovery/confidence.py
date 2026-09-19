"""Synthetic-only confidence fitting and frozen selective-prediction gates."""
from __future__ import annotations

import math
import numpy as np


FEATURES = (
    'margin', 'support', 'held_out_support', 'supported_edges',
    'largest_component', 'appearance_mean', 'residual_p90',
    'condition_log', 'null_significance', 'stream_agreement',
)


def feature_vector(result: dict) -> np.ndarray:
    if not result.get('ranked'):
        return np.zeros(len(FEATURES), float)
    h = result['ranked'][0]
    return np.asarray([
        result.get('margin', 0.0), h.get('support', 0), h.get('held_out_support', 0),
        h.get('graph', {}).get('supported_edges', 0),
        h.get('graph', {}).get('largest_component', 0), h.get('appearance_mean', 0),
        h.get('residual_p90', 99), math.log1p(h.get('condition_number', 99)),
        h.get('null_significance_adjusted', 0), h.get('stream_agreement', 0),
    ], float)


def fit_logistic(rows: list[dict], l2: float = 1.0, iterations: int = 80) -> dict:
    """Small deterministic iteratively-reweighted least-squares calibrator."""
    x = np.asarray([r['features'] for r in rows], float)
    y = np.asarray([bool(r['correct']) for r in rows], float)
    if not len(x):
        return {'status': 'unavailable', 'reason': 'no fitting rows'}
    med = np.median(x, axis=0)
    scale = np.median(np.abs(x - med), axis=0) * 1.4826
    scale[scale < 1e-6] = 1.0
    z = (x - med) / scale
    z = np.c_[np.ones(len(z)), z]
    beta = np.zeros(z.shape[1])
    penalty = np.eye(z.shape[1]) * l2; penalty[0, 0] = 0
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(z @ beta, -30, 30)))
        w = np.maximum(p * (1 - p), 1e-5)
        h = z.T @ (w[:, None] * z) + penalty
        g = z.T @ (y - p) - penalty @ beta
        try: step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError: step = np.linalg.lstsq(h, g, rcond=None)[0]
        beta += step
        if np.linalg.norm(step) < 1e-7: break
    return {'status': 'fit', 'features': list(FEATURES), 'median': med.tolist(),
            'scale': scale.tolist(), 'coefficients': beta.tolist(), 'n': len(rows),
            'positives': int(y.sum())}


def predict_logistic(model: dict, result: dict) -> float:
    if model.get('status') != 'fit':
        return 0.0
    x = feature_vector(result)
    z = (x - np.asarray(model['median'])) / np.asarray(model['scale'])
    score = float(np.r_[1.0, z] @ np.asarray(model['coefficients']))
    return float(1 / (1 + math.exp(-max(-30, min(30, score)))))


def empirical_threshold(rows: list[dict], max_false_accept_rate: float = 0.05) -> dict:
    """Choose the broadest score threshold meeting a fitting-set false-accept cap."""
    candidates = sorted({float(r['confidence']) for r in rows}, reverse=True)
    best = None
    for threshold in candidates:
        accepted = [r for r in rows if r['confidence'] >= threshold]
        if not accepted: continue
        false_rate = sum(not r['correct'] for r in accepted) / len(accepted)
        if false_rate <= max_false_accept_rate:
            best = {'threshold': threshold, 'accepted': len(accepted),
                    'precision': 1 - false_rate, 'false_accept_rate': false_rate}
    return best or {'threshold': 1.000001, 'accepted': 0, 'precision': None,
                    'false_accept_rate': 0.0}


def fit_isotonic(rows: list[dict]) -> dict:
    """Fit a deterministic pool-adjacent-violators calibrator when n>=30."""
    if len(rows) < 30:
        return {'status': 'unavailable', 'reason': 'fewer than 30 independent fitting scenes'}
    ordered = sorted((float(r['score']), float(bool(r['correct']))) for r in rows)
    blocks = [[x, x, y, 1] for x, y in ordered]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] / blocks[i][3] <= blocks[i+1][2] / blocks[i+1][3]:
            i += 1; continue
        a, b = blocks[i], blocks[i+1]
        blocks[i:i+2] = [[a[0], b[1], a[2]+b[2], a[3]+b[3]]]
        i = max(0, i-1)
    return {'status':'fit','blocks':[{'lo':a,'hi':b,'p':s/n,'n':n} for a,b,s,n in blocks]}


def predict_isotonic(model: dict, score: float) -> float:
    if model.get('status') != 'fit': return 0.0
    for b in model['blocks']:
        if score <= b['hi']: return float(b['p'])
    return float(model['blocks'][-1]['p'])


def gate_decisions(result: dict, model: dict | None, policy: dict) -> dict:
    top = result['ranked'][0] if result.get('ranked') else {}
    confidence = predict_logistic(model or {}, result)
    strict = (result.get('margin', 0) >= 2 and top.get('support', 0) >= 9 and
              top.get('held_out_support', 0) >= 5)
    primary_cfg = policy.get('primary_gate', policy)
    primary = (
        confidence >= float(primary_cfg.get('confidence_min', 1.000001)) and
        top.get('support', 0) >= int(primary_cfg.get('support_min', 7)) and
        top.get('held_out_support', 0) >= int(primary_cfg.get('held_out_min', 3)) and
        top.get('graph', {}).get('largest_component', 0) >= int(primary_cfg.get('largest_component_min', 3)) and
        top.get('stream_agreement', 0) >= int(primary_cfg.get('stream_agreement_min', 2)))
    exploratory = (confidence >= float(policy.get('exploratory_confidence_min', 0.75)) and
                   top.get('held_out_support', 0) >= 3)
    iso = predict_isotonic(policy.get('isotonic_model', {}), top.get('score', -1e9))
    null_calibrated = top.get('null_significance_adjusted', 0) >= float(policy.get('null_significance_min', 2.0))
    agreement = top.get('stream_agreement', 0) >= 2
    return {'confidence': confidence, 'isotonic_confidence': iso,
            'strict': bool(strict), 'primary': bool(primary),
            'exploratory': bool(exploratory), 'null_calibrated': bool(null_calibrated),
            'logistic': bool(confidence >= .90), 'isotonic': bool(iso >= .90),
            'agreement': bool(agreement)}


def selective_metrics(rows: list[dict], gate: str) -> dict:
    accepted = [r for r in rows if r['decisions'][gate]]
    correct = sum(bool(r['correct']) for r in accepted)
    wrong = len(accepted) - correct
    return {
        'n': len(rows), 'accepted': len(accepted),
        'coverage': len(accepted) / max(len(rows), 1),
        'precision': correct / len(accepted) if accepted else None,
        'false_accept_rate': wrong / len(accepted) if accepted else 0.0,
        'correct_rescue_rate': correct / max(len(rows), 1),
        'wrong_overwrite_rate': wrong / max(len(rows), 1),
    }


def calibration_bins(rows: list[dict], width: float = .2) -> list[dict]:
    out=[]
    for lo in np.arange(0,1,width):
        group=[r for r in rows if lo <= r['decisions']['confidence'] < lo+width or
               (lo+width>=1 and r['decisions']['confidence']==1)]
        if group: out.append({'lo':float(lo),'hi':float(min(1,lo+width)),'n':len(group),
                              'mean_confidence':float(np.mean([r['decisions']['confidence'] for r in group])),
                              'accuracy':float(np.mean([r['correct'] for r in group]))})
    return out

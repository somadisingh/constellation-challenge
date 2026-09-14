"""Inner-validation scoring and the selection statistic (task §12).

Selection metric: `0.25 * mean_scene_presence + 0.20 * mean_scene_localization`,
computed on ALLOWED-SKY groups only. Ties break by (1) higher localization,
(2) higher candidate top-1 reward, (3) earlier checkpoint, (4) alphabetical arm.
"""
from __future__ import annotations

import numpy as np
import torch

from .batch import collate
from .forward import score_batch

CORRECT_RADIUS = 12.0


@torch.no_grad()
def score_groups(model, groups: list, device: str) -> list:
    """Per-group summary: best candidate, presence decision inputs, localization."""
    if not groups:
        return []
    batch = collate(groups)
    out = score_batch(model, batch, device)
    logits = out['pair_logits'].cpu().numpy()
    absent_logit = out['absent_logit'].cpu().numpy()
    valid = batch['valid'].numpy()
    rows = []
    for i, g in enumerate(groups):
        row_logits = np.where(valid[i], logits[i], -np.inf)
        has_valid = bool(valid[i].any())
        best_idx = int(np.argmax(row_logits)) if has_valid else None
        best_logit = float(row_logits[best_idx]) if has_valid else None
        second = None
        if has_valid and int(valid[i].sum()) >= 2:
            order = np.argsort(-row_logits)
            second = float(row_logits[order[1]])
        entry = {
            'group_id': g.group_id, 'kind': g.kind, 'fold': g.fold,
            'target_scene': g.target_scene, 'source_scene': g.source_scene,
            'pool_missing': g.pool_missing,
            'best_index': best_idx, 'best_logit': best_logit,
            'second_logit': second,
            'absent_logit': float(absent_logit[i]),
            'best_xy': (g.xy[best_idx].tolist() if best_idx is not None else None),
            'n_candidates': len(g), 'n_valid': int(valid[i].sum()),
        }
        if g.kind == 'present' and g.centre is not None and best_idx is not None:
            d = float(np.hypot(g.xy[best_idx, 0] - g.centre[0],
                               g.xy[best_idx, 1] - g.centre[1]))
            entry['error_distance'] = d
            entry['localization_reward'] = float(np.clip((36.0 - d) / 24.0, 0, 1))
            entry['top1_correct'] = bool(d <= CORRECT_RADIUS)
            has_correct = bool((np.hypot(g.xy[:, 0] - g.centre[0],
                                         g.xy[:, 1] - g.centre[1]) <= CORRECT_RADIUS).any())
            entry['bank_has_correct'] = has_correct
        elif g.kind == 'present':
            entry.update({'error_distance': None, 'localization_reward': 0.0,
                         'top1_correct': False, 'bank_has_correct': not g.pool_missing})
        rows.append(entry)
    return rows


def presence_decision(rows: list, threshold: float) -> np.ndarray:
    """1 if predicted present (best pair logit beats absent logit AND >= threshold
    after calibration is applied elsewhere); here: raw score margin > threshold."""
    margin = np.array([(r['best_logit'] - r['absent_logit'])
                       if r['best_logit'] is not None else -np.inf for r in rows])
    return (margin >= threshold).astype(int)


def inner_selection_metrics(rows_by_scene: dict, threshold: float = 0.0) -> dict:
    """0.25*presence + 0.20*localization, equal-scene mean (task §12)."""
    per_scene = {}
    for scene, rows in rows_by_scene.items():
        y = np.array([1 if r['kind'] == 'present' else 0 for r in rows])
        pred = presence_decision(rows, threshold)
        presence = _macro_f1(y, pred)
        present_rows = [r for r in rows if r['kind'] == 'present']
        loc = float(np.mean([r['localization_reward'] for r in present_rows])) \
            if present_rows else 1.0
        top1 = float(np.mean([r['top1_correct'] for r in present_rows])) \
            if present_rows else 1.0
        has = [r.get('bank_has_correct', False) for r in present_rows]
        per_scene[scene] = {'presence': presence, 'localization': loc,
                            'top1_correct': top1,
                            'candidate_recall@12': float(np.mean(has)) if has else 1.0,
                            'n_present': len(present_rows),
                            'n_absent': len(rows) - len(present_rows)}
    scenes = sorted(per_scene)
    mean_presence = float(np.mean([per_scene[s]['presence'] for s in scenes]))
    mean_localization = float(np.mean([per_scene[s]['localization'] for s in scenes]))
    mean_top1 = float(np.mean([per_scene[s]['top1_correct'] for s in scenes]))
    return {
        'per_scene': per_scene,
        'presence': mean_presence, 'localization': mean_localization,
        'top1_correct': mean_top1,
        'selection_metric': 0.25 * mean_presence + 0.20 * mean_localization,
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1 = []
    for cls in (0, 1):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        den = np.sum(y_true == cls) + np.sum(y_pred == cls)
        f1.append(2 * tp / den if den else 1.0)
    return float(np.mean(f1))


def select_arm(results: dict) -> dict:
    """Break ties per task §12: localization, then top1, then step, then name."""
    ranked = sorted(
        results.items(),
        key=lambda kv: (-kv[1]['metric'], -kv[1].get('localization', 0.0),
                        -kv[1].get('top1_correct', 0.0), kv[1].get('step', 1 << 30),
                        kv[0]))
    winner = ranked[0][0]
    return {'selected': winner, 'ranking': [k for k, _ in ranked],
           'metrics': {k: v for k, v in results.items()}}

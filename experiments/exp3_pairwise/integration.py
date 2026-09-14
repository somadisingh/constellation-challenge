"""Build ScenePredictions and integrate with Experiment 2's frozen hybrid rule
(task §13).

Stages, evaluated SEPARATELY so any gain is attributable:

  1. verifier_only         calibrated presence + best candidate, no offset
  2. verifier_offset       + bounded residual correction where the gate passes
  3. verifier_snap         + Exp2 geometry snap only (relocated support)
  4. verifier_snap_rescue  + Exp2 geometry snap AND rescue (Exp2's selected rule)

The neural residual offset is NEVER applied after a geometry snap or rescue
(task §13): those coordinates come from C0's own verified geometry, and the
learned correction was never trained or gated against that source.
"""
from __future__ import annotations

import numpy as np

from constellation.contracts import ScenePrediction
from experiments.exp1.evaluation import frozen_geometry
from experiments.exp2_geometry.integration import hybrid_prediction

MEMBER_RADIUS = 18.0


def build_prediction(scene: str, rows: list, probs: np.ndarray, threshold: float,
                     offset_gate: dict | None = None) -> dict:
    """One ScenePrediction from calibrated presence + (optional) offset correction."""
    geometry = frozen_geometry(scene)
    nodes = geometry['nodes']
    patches, detail = [], []
    for r, p in zip(rows, probs):
        chosen = r.get('best_xy')
        usable = chosen is not None and np.isfinite(p)
        present = bool(usable and p >= threshold)
        offset_applied = False
        if present:
            x, y = float(chosen[0]), float(chosen[1])
            if offset_gate is not None and r.get('offset_confidence') is not None:
                conf_t = offset_gate['confidence_threshold']
                max_corr = offset_gate['max_correction']
                if r['offset_confidence'] >= conf_t:
                    dx = float(np.clip(r['offset_dx'], -max_corr, max_corr))
                    dy = float(np.clip(r['offset_dy'], -max_corr, max_corr))
                    x, y = x + dx, y + dy
                    offset_applied = True
            member = int(len(nodes) > 0 and
                         np.linalg.norm(nodes - [x, y], axis=1).min() < MEMBER_RADIUS)
            patches.append((x, y, member))
        else:
            patches.append(None)
        detail.append({'query_id': r['query_id'], 'present_pred': present,
                       'probability': float(p) if np.isfinite(p) else None,
                       'selected_xy': chosen, 'offset_applied': offset_applied,
                       'empty_bank': r['empty_bank'],
                       'alignment_failed_all': r['alignment_failed_all']})
    prediction = ScenePrediction(patches, geometry['constellation'], {
        'integration': 'exp3_pairwise_verifier',
        'threshold': float(threshold),
        'offset_enabled': offset_gate is not None,
        'frozen_nodes': int(len(nodes)),
        'frozen_class_source': geometry['source']})
    return {'prediction': prediction, 'detail': detail}


def apply_geometry_stage(learned: ScenePrediction, classical: ScenePrediction,
                         stage: str) -> ScenePrediction:
    """`stage` in {'verifier_only', 'verifier_snap', 'verifier_snap_rescue'}.

    Reuses Experiment 2's `hybrid_prediction` verbatim: `snap`/`rescue` support
    modes are keyed on C0's own `relocated_queries`, exactly as Experiment 2 uses
    them. `verifier_offset` is not a geometry stage; it is applied upstream in
    `build_prediction` before this function ever sees the coordinate, and the
    offset is never re-applied to a geometry-snapped or rescued coordinate because
    `hybrid_prediction` REPLACES the learned coordinate wholesale with C0's when it
    snaps or rescues.
    """
    if stage == 'verifier_only':
        return learned
    if stage == 'verifier_snap':
        return hybrid_prediction(learned, classical, snap='relocated', rescue='none')
    if stage == 'verifier_snap_rescue':
        return hybrid_prediction(learned, classical, snap='relocated', rescue='relocated')
    raise ValueError(f'unknown stage {stage!r}')

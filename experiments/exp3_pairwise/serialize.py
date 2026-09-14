"""JSON-safe conversion for experiment output structures."""
from __future__ import annotations

import numpy as np


def to_json_safe(obj):
    """Recursively convert obj to JSON-serialisable Python types."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    # ScenePrediction — convert patches only
    if hasattr(obj, 'patches') and hasattr(obj, 'constellation'):
        return {'patches': to_json_safe(obj.patches),
                'constellation': str(obj.constellation)}
    return str(obj)   # last-resort fallback; never silently drops data


def slim_eval_result(result: dict) -> dict:
    """Trim a held_out.evaluate_fold result to a JSON-safe summary."""
    out = {}
    for key, val in result.items():
        if key == 'rows':
            continue   # per-query rows go to a separate per-fold JSON
        if key == 'predictions':
            # Convert ScenePrediction objects
            out[key] = {
                stage: {s: to_json_safe(p) for s, p in stage_preds.items()}
                for stage, stage_preds in val.items()
            }
        else:
            out[key] = to_json_safe(val)
    return out


def rows_to_json(rows_by_scene: dict) -> dict:
    return {s: to_json_safe(rows) for s, rows in rows_by_scene.items()}

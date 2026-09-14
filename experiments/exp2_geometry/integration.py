"""Label-free hybrid rules and candidate-level learned geometry utilities.

The public functions in this module never inspect ground truth.  Evaluation and
leave-one-scene-out selection live in ``run.py`` so leakage is easy to audit.
"""
from __future__ import annotations

import numpy as np

from constellation.contracts import ScenePrediction
from constellation.slate import QuerySlate


def supported_queries(prediction: ScenePrediction, mode: str) -> set[int]:
    """Return query indices independently supported by C0 geometry."""
    relocated = set(prediction.diagnostics.get("relocated_queries", []))
    members = {i for i, p in enumerate(prediction.patches)
               if p is not None and len(p) >= 3 and int(p[2]) == 1}
    if mode == "none":
        return set()
    if mode == "relocated":
        return relocated
    if mode == "member":
        return members
    if mode == "either":
        return relocated | members
    raise ValueError(f"unknown support mode {mode!r}")


def hybrid_prediction(learned: ScenePrediction, classical: ScenePrediction,
                      *, snap: str = "relocated", rescue: str = "relocated") -> ScenePrediction:
    """Combine learned presence with independently verified C0 correspondences.

    ``snap`` controls when a learned-present location is replaced by C0's
    geometry-supported coordinate. ``rescue`` controls when a learned-absent
    query is restored. Both decisions use only inference-time C0 diagnostics.
    """
    if len(learned.patches) != len(classical.patches):
        raise ValueError("learned/classical query counts differ")
    snap_ids = supported_queries(classical, snap)
    rescue_ids = supported_queries(classical, rescue)
    patches, actions = [], []
    for i, (lp, cp) in enumerate(zip(learned.patches, classical.patches)):
        action = "learned"
        if lp is None and cp is not None and i in rescue_ids:
            out, action = cp, "geometry_rescue"
        elif lp is not None and cp is not None and i in snap_ids:
            out, action = cp, "geometry_snap"
        else:
            out = lp
        patches.append(out)
        actions.append(action)
    return ScenePrediction(patches, classical.constellation, {
        "integration": "learned_presence_with_verified_geometry",
        "snap": snap, "rescue": rescue, "actions": actions,
        "n_snap": actions.count("geometry_snap"),
        "n_rescue": actions.count("geometry_rescue"),
        "class_source": "C0 frozen geometric winner",
    })


def rank_utility(scores: np.ndarray) -> np.ndarray:
    """Convert descriptor scores to a per-query [0,1] rank utility."""
    scores = np.asarray(scores, float)
    valid = np.isfinite(scores) & (scores > -1e8)
    out = np.zeros(len(scores), float)
    order = np.argsort(-np.where(valid, scores, -np.inf), kind="stable")
    n = int(valid.sum())
    if n == 1:
        out[order[0]] = 1.
    elif n > 1:
        out[order[:n]] = 1. - np.arange(n) / (n - 1)
    return out


def map_scores(source_xy: np.ndarray, source_scores: np.ndarray,
               target_xy: np.ndarray, tolerance: float = 2.) -> tuple[np.ndarray, dict]:
    """Map union-bank learned scores to branch candidates by coordinates."""
    source_xy = np.asarray(source_xy, float).reshape(-1, 2)
    target_xy = np.asarray(target_xy, float).reshape(-1, 2)
    if not len(target_xy):
        return np.zeros(0), {"mapped": 0, "max_distance": 0.}
    d = np.linalg.norm(target_xy[:, None, :] - source_xy[None, :, :], axis=2)
    nearest = d.argmin(1)
    dist = d[np.arange(len(target_xy)), nearest]
    if np.any(dist > tolerance):
        raise ValueError(f"candidate provenance mismatch: max distance {dist.max():.3f}px")
    return np.asarray(source_scores, float)[nearest], {
        "mapped": int(len(target_xy)), "max_distance": float(dist.max()),
        "mean_distance": float(dist.mean()),
    }


def learned_slate(candidates: list, source_xy: np.ndarray,
                  source_scores: np.ndarray) -> tuple[QuerySlate, dict]:
    """Keep classical calibration/seed while assigning learned ranking."""
    arr = np.asarray(candidates, float).reshape(-1, 5)
    learned, audit = map_scores(source_xy, source_scores, arr[:, :2])
    return QuerySlate(xy=arr[:, :2], pose=arr[:, 3:5], calib=arr[:, 2],
                      rank=rank_utility(learned), seed=0), audit


"""One scoring path shared by every arm (plan §6, §8, §11).

Attribution depends on the arms differing in exactly one place. So:

* the candidate locations come from the frozen bank,
* the pose is chosen once by the masked classical criterion and CACHED, so every
  model sees identical crops and identical pose trial counts,
* the aligned crops are built once,
* only the scoring function differs.

Score direction is always higher-is-better:
    C1 / ncc     masked NCC, already higher-is-better
    descriptor   NEGATIVE Euclidean distance
    pair head    logit

An empty candidate bank returns absent with no fabricated score. A single-candidate
bank uses gap 0 and is flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import PATCH
from .pose import (AA_FACTOR, SceneReps, aligned_candidate, masked_ncc,
                   prepare_query, select_pose, N_POSE_TRIALS)

# Torch is imported lazily: alignment, the classical control and the per-query
# summary are pure NumPy, so they must work in an environment without torch.
# Kept numerically identical to models.LOW_INFO_STD, which is asserted in tests.
LOW_INFO_STD = 1e-3

NEG_INF = -1e9


@dataclass
class AlignedSet:
    """Aligned crops and cached poses for one query against its candidate bank."""
    query_id: str
    xy: np.ndarray                  # (k, 2) candidate coordinates, unchanged
    crops: np.ndarray               # (k, 32, 32) float32 in [0,1], query frame
    ncc: np.ndarray                 # (k,) masked classical NCC, the C1 score
    poses: np.ndarray               # (k, 2) selected angle, scale
    admissible: np.ndarray          # (k,) bool: pose read entirely inside the image
    low_info: np.ndarray            # (k,) bool: crop carries no usable structure
    pose_trials: int
    rejected_poses: int
    query_raw: np.ndarray = field(repr=False, default=None)

    def __len__(self):
        return len(self.xy)

    @property
    def n_admissible(self):
        return int(self.admissible.sum())


def align_query(reps: SceneReps, patch: np.ndarray, candidate_xy,
                base_poses, query_id: str = '', aa_factor: float = AA_FACTOR,
                oracle_poses=None) -> AlignedSet:
    """Align every candidate of one query. Model independent, so cache the result.

    `oracle_poses`, when given, replaces the image-estimated pose with a known
    synthetic pose. That is a DIAGNOSTIC path only (plan §6); the reported
    comparison always uses the estimated pose.
    """
    qraw, qblur = prepare_query(patch)
    xy = np.asarray(candidate_xy, float).reshape(-1, 2)
    base = np.asarray(base_poses, float).reshape(-1, 2)
    n = len(xy)
    crops = np.zeros((n, PATCH, PATCH), np.float32)
    ncc = np.full(n, np.nan, np.float64)
    poses = np.zeros((n, 2), np.float64)
    admissible = np.zeros(n, bool)
    low_info = np.zeros(n, bool)
    rejected = 0

    for i in range(n):
        if oracle_poses is not None:
            chosen = (float(oracle_poses[i][0]), float(oracle_poses[i][1]))
            crop, mask = aligned_candidate(reps.blur, xy[i], chosen, aa_factor)
            score = masked_ncc(qblur, crop, mask) if mask.all() else np.nan
            selection = {'pose': chosen if mask.all() else None, 'ncc': score,
                         'rejected': 0 if mask.all() else 1}
        else:
            selection = select_pose(reps, qblur, xy[i], base[i], aa_factor)
        rejected += selection['rejected']
        if selection['pose'] is None:
            continue
        admissible[i] = True
        poses[i] = selection['pose']
        ncc[i] = selection['ncc']
        crop, _ = aligned_candidate(reps.raw, xy[i], selection['pose'], aa_factor)
        crops[i] = crop / 255.0 if crop.max() > 1.5 else crop
        low_info[i] = float(crops[i].std()) < LOW_INFO_STD

    return AlignedSet(query_id=query_id, xy=xy, crops=crops, ncc=ncc, poses=poses,
                      admissible=admissible, low_info=low_info,
                      pose_trials=N_POSE_TRIALS if oracle_poses is None else 1,
                      rejected_poses=rejected, query_raw=qraw)


# --- arms -------------------------------------------------------------------------
def score_classical(aligned: AlignedSet) -> np.ndarray:
    """C1: the documented common masked NCC on the aligned crops."""
    out = np.where(aligned.admissible, aligned.ncc, np.nan)
    return np.where(np.isfinite(out), out, NEG_INF)


def encode_batch(model, crops: np.ndarray, device: str, microbatch: int = 512):
    """Descriptors for a stack of 32x32 float crops."""
    import torch
    if not len(crops):
        return torch.zeros(0, 128)
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(crops), microbatch):
            chunk = crops[start:start + microbatch]
            x = torch.from_numpy(np.ascontiguousarray(chunk)).float()[:, None, :, :]
            out.append(model(x.to(device)).cpu())
    return torch.cat(out, dim=0)


def score_descriptor(model, aligned: AlignedSet, device: str) -> np.ndarray:
    """Negative descriptor Euclidean distance, so higher is better."""
    import torch
    if not len(aligned):
        return np.zeros(0)
    with torch.no_grad():
        zq = encode_batch(model, aligned.query_raw[None, ...], device)
        zc = encode_batch(model, aligned.crops, device)
        dist = torch.sqrt(torch.clamp(((zc - zq) ** 2).sum(1), min=1e-12)).numpy()
    scores = -dist.astype(np.float64)
    scores[~aligned.admissible] = NEG_INF
    scores[aligned.low_info] = NEG_INF
    return scores


def score_pair_head(model, head, aligned: AlignedSet, device: str) -> np.ndarray:
    """Pair-head logit, higher is better."""
    import torch
    if not len(aligned):
        return np.zeros(0)
    head.eval()
    with torch.no_grad():
        zq = encode_batch(model, aligned.query_raw[None, ...], device).to(device)
        zc = encode_batch(model, aligned.crops, device).to(device)
        logits = head(zq.expand(len(zc), -1), zc).cpu().numpy().astype(np.float64)
    logits[~aligned.admissible] = NEG_INF
    logits[aligned.low_info] = NEG_INF
    return logits


# --- per-query summary ------------------------------------------------------------
def summarise_query(scores: np.ndarray, aligned: AlignedSet,
                    truth_xy=None) -> dict:
    """Per-query features and localization outcome for one arm.

    `best_match_score` and `best_minus_second_score` are the two calibration
    features (plan §11).
    """
    n = len(scores)
    valid = np.isfinite(scores) & (scores > NEG_INF / 2)
    out = {
        'n_candidates': int(n),
        'n_valid': int(valid.sum()),
        'empty_bank': n == 0,
        'single_candidate': int(valid.sum()) == 1,
        'alignment_failed_all': n > 0 and int(valid.sum()) == 0,
    }
    if not valid.any():
        out.update({'best_match_score': None, 'best_minus_second_score': None,
                    'best_index': None, 'best_xy': None, 'flagged': True})
        if truth_xy is not None:
            out.update({'error_distance': None, 'localization_reward': 0.0,
                        'top1_correct': False, 'top5_correct': False,
                        'rank_of_correct': None})
        return out

    order = np.argsort(-np.where(valid, scores, -np.inf), kind='stable')
    best = int(order[0])
    best_score = float(scores[best])
    second = float(scores[order[1]]) if valid.sum() >= 2 else None
    out.update({
        'best_match_score': best_score,
        'best_minus_second_score': (best_score - second) if second is not None else 0.0,
        'best_index': best,
        'best_xy': [float(aligned.xy[best, 0]), float(aligned.xy[best, 1])],
        'best_pose': [float(aligned.poses[best, 0]), float(aligned.poses[best, 1])],
        'flagged': second is None,
    })
    if truth_xy is not None:
        truth = np.asarray(truth_xy, float)
        d = np.linalg.norm(aligned.xy - truth, axis=1)
        err = float(d[best])
        ranked = [int(i) for i in order]
        correct = [i for i in ranked if d[i] <= 12.0]
        out.update({
            'error_distance': err,
            'localization_reward': float(np.clip((36.0 - err) / 24.0, 0, 1)),
            'top1_correct': bool(d[best] <= 12.0),
            'top5_correct': bool(any(d[i] <= 12.0 for i in ranked[:5])),
            'rank_of_correct': (ranked.index(correct[0]) if correct else None),
            'bank_has_correct': bool((d <= 12.0).any()),
            'nearest_bank_distance': float(d.min()),
        })
    return out

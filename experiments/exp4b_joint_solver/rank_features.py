"""Phase 1: per-query-candidate feature collection from frozen banks.

Every feature listed in the task is collected where the existing pipeline
actually computes it; features the pipeline does not compute (ECC convergence
diagnostics, HardNet distance per candidate, centre/annulus score, local
source response, saturation/background statistics, explicit duplicate-source
grouping beyond `consolidate`) are recorded as `None`/absent rather than
fabricated, per rule 14 ("every claimed feature or scoring term must have an
executed ablation proving it was active" -- an absent feature cannot be
"active" and must say so).
"""
from __future__ import annotations

import numpy as np

from . import SCENES
from experiments.exp1.data import query_records
from experiments.exp1.evaluation import _aligned_set, load_real_aligned
from experiments.exp1.stages import Paths
from experiments.exp1.env import ROOT as EXP1_ROOT

AVAILABLE_FEATURES = (
    'classical_ncc',        # masked NCC on the classical blur representation
    'pose_angle', 'pose_scale',
    'candidate_rank_by_ncc',
    'ncc_gap_to_second',
    'exp3_pair_logit',      # Exp3's per-candidate pair-logit (this fold/seed member)
    'exp3_absent_logit',    # Exp3's per-query absent logit (broadcast to every row)
    'exp3_ensemble_mean_logit',
    'exp3_ensemble_disagreement',  # std across ensemble members' logit for this row
)
UNAVAILABLE_FEATURES = (
    'ecc_score', 'ecc_convergence_diagnostics',   # ECC refinement is not cached
                                                   # per-candidate in this pipeline
    'hardnet_distance_per_candidate',             # only fused inside Exp3's model,
                                                   # never exposed as a standalone
                                                   # per-candidate scalar
    'centre_annulus_score',                       # not computed by this pipeline
    'local_source_response',                      # computed separately in Phase 3
                                                   # (unqueried-star search), not
                                                   # cached per RETRIEVED candidate
    'saturation_background_stats',                # not computed per candidate
    'explicit_duplicate_source_id',                # `consolidate()` groups QUERIES
                                                   # by proximity, not CANDIDATES;
                                                   # no per-candidate duplicate id
                                                   # is computed by the existing code
)


def _old_paths():
    return Paths(EXP1_ROOT / 'outputs' / 'exp1')


def collect_query_features(scene: str, models: list, device: str) -> list:
    """One dict per real query: candidate-level feature arrays + truth.

    `models` is a list of loaded Exp3 arm-F models (one or more seeds); if more
    than one, ensemble mean/disagreement are also recorded per candidate.
    """
    import torch
    from experiments.exp3_pairwise.real_scoring import score_aligned_set

    node = load_real_aligned(_old_paths(), scene)
    records = query_records(scene)
    out = []
    for r in records:
        aligned = _aligned_set(node, r['index'])
        k = len(aligned)
        if k == 0:
            out.append({'scene': scene, 'index': r['index'], 'query_id': r['query_id'],
                       'present': r['present'], 'figure': r['figure'],
                       'truth_xy': r['xy'], 'n_candidates': 0, 'candidates': []})
            continue

        valid = aligned.admissible & ~aligned.low_info
        ncc = np.where(valid, aligned.ncc, -np.inf)
        rank_by_ncc = np.full(k, -1, int)
        order = np.argsort(-ncc)
        rank_by_ncc[order] = np.arange(k)
        second = np.sort(ncc)[::-1][1] if valid.sum() >= 2 else -np.inf
        ncc_gap = ncc - second

        member_logits, member_absent = [], []
        for model in models:
            scored = score_aligned_set(model, aligned, device)
            member_logits.append(np.where(scored['valid'], scored['pair_logits'], np.nan))
            member_absent.append(scored['absent_logit'])
        stacked = np.stack(member_logits) if member_logits else np.zeros((0, k))
        if stacked.size and not np.all(np.isnan(stacked)):
            mean_logit = np.nanmean(stacked, axis=0)
        else:
            mean_logit = np.full(k, np.nan)
        disagreement = (np.nanstd(stacked, axis=0) if stacked.shape[0] >= 2
                       else np.zeros(k))
        mean_absent = float(np.mean(member_absent)) if member_absent else None

        candidates = []
        for i in range(k):
            d = None
            if r['present']:
                d = float(np.hypot(aligned.xy[i, 0] - r['xy'][0],
                                   aligned.xy[i, 1] - r['xy'][1]))
            candidates.append({
                'index': i, 'xy': aligned.xy[i].tolist(), 'valid': bool(valid[i]),
                'distance_to_truth': d,
                'classical_ncc': float(aligned.ncc[i]) if valid[i] else None,
                'pose_angle': float(aligned.poses[i, 0]),
                'pose_scale': float(aligned.poses[i, 1]),
                'candidate_rank_by_ncc': int(rank_by_ncc[i]) if valid[i] else None,
                'ncc_gap_to_second': float(ncc_gap[i]) if valid[i] else None,
                'exp3_pair_logit': (float(mean_logit[i]) if valid[i]
                                    and not np.isnan(mean_logit[i]) else None),
                'exp3_absent_logit': mean_absent,
                'exp3_ensemble_disagreement': (float(disagreement[i]) if valid[i]
                                              and not np.isnan(disagreement[i]) else None),
            })
        out.append({'scene': scene, 'index': r['index'], 'query_id': r['query_id'],
                   'present': r['present'], 'figure': r['figure'],
                   'truth_xy': r['xy'], 'n_candidates': k, 'candidates': candidates})
    return out


def relevance(distance_to_truth: float | None) -> float:
    """Official localization-reward relevance target: full credit at <=12px,
    zero at >=36px, linear between -- identical formula to constellation.contracts.reward."""
    if distance_to_truth is None:
        return 0.0
    return float(np.clip((36.0 - distance_to_truth) / 24.0, 0.0, 1.0))


def collect_all(models_by_seed: dict, device: str, say=print) -> dict:
    """`models_by_seed`: {seed: model} for a SINGLE fold's own-fold checkpoints
    (leak-free membership -- caller must ensure this)."""
    out = {}
    for scene in SCENES:
        say(f'  collecting features for {scene} ...')
        out[scene] = collect_query_features(scene, list(models_by_seed.values()), device)
    return out

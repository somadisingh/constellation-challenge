"""Pad a list of QueryGroups into fixed-K tensors for one training/eval step."""
from __future__ import annotations

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from experiments.exp1.mining import NEGATIVE, POSITIVE as POS_LABEL


def collate(groups: list, kmax: int | None = None) -> dict:
    """Stack groups into (B, K) padded tensors. K is the max candidate count.

    `true_dx`/`true_dy` are the residual `truth - candidate_xy` for every
    candidate of a present group (NaN where truth is unknown, i.e. absent groups
    or a missing centre), so the offset loss can index them directly without
    reconstructing truth from distance.
    """
    import torch
    kmax = kmax or max((len(g) for g in groups), default=1) or 1
    b = len(groups)
    query = np.zeros((b, 32, 32), np.float32)
    crops = np.zeros((b, kmax, 32, 32), np.float32)
    valid = np.zeros((b, kmax), bool)
    positive = np.zeros((b, kmax), bool)
    xy = np.zeros((b, kmax, 2), np.float32)
    distance = np.full((b, kmax), np.nan, np.float32)
    true_dx = np.full((b, kmax), np.nan, np.float32)
    true_dy = np.full((b, kmax), np.nan, np.float32)
    is_present = np.zeros(b, bool)
    pool_missing = np.zeros(b, bool)
    for i, g in enumerate(groups):
        k = len(g)
        query[i] = g.query
        if k:
            crops[i, :k] = g.crops
            valid[i, :k] = g.admissible
            positive[i, :k] = np.array([l == POS_LABEL for l in g.labels], bool)
            xy[i, :k] = g.xy
            distance[i, :k] = g.distance
            if g.centre is not None:
                true_dx[i, :k] = g.centre[0] - g.xy[:, 0]
                true_dy[i, :k] = g.centre[1] - g.xy[:, 1]
        is_present[i] = g.kind == 'present'
        pool_missing[i] = g.pool_missing
    return {
        'query': torch.from_numpy(query),
        'crops': torch.from_numpy(crops),
        'valid': torch.from_numpy(valid),
        'positive': torch.from_numpy(positive),
        'xy': xy, 'distance': distance,
        'true_dx': torch.from_numpy(true_dx), 'true_dy': torch.from_numpy(true_dy),
        'is_present': torch.from_numpy(is_present),
        'pool_missing': torch.from_numpy(pool_missing),
        'kmax': kmax,
        'group_ids': [g.group_id for g in groups],
    }


def usable_present_mask(batch: dict) -> torch.Tensor:
    """Present rows with at least one valid positive under the mask, or absent rows.

    Excludes pool-missing present rows from the classification loss mean: the true
    location was never retrieved, so no target logit exists to push up, and the
    listwise loss for such a row is a numerical artefact (huge, not informative).
    """
    has_positive = (batch['positive'] & batch['valid']).any(dim=1)
    return (~batch['is_present']) | has_positive

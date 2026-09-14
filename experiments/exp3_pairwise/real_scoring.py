"""Score REAL queries against the frozen Experiment 1 union bank (task §13).

Training/inner-validation use synthetic `QueryGroup`s (groups.py). Held-out
evaluation must score the actual 116 real queries against the actual frozen
candidate banks, via `experiments.exp1.evaluation.load_real_aligned` and
`_aligned_set` -- exactly the artifact Experiment 1B and Experiment 2 both reuse
verbatim. No candidate coordinate here is ever recomputed or injected.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.exp1.data import query_records
from experiments.exp1.evaluation import _aligned_set, load_real_aligned
from experiments.exp1.stages import Paths
from experiments.exp1 import SCENES as EXP1_SCENES

OLD = None  # set lazily to avoid import cycles; see `_old_paths`


def _old_paths():
    from experiments.exp1.env import ROOT as EXP1_ROOT
    return Paths(EXP1_ROOT / 'outputs' / 'exp1')


@torch.no_grad()
def score_aligned_set(model, aligned, device: str) -> dict:
    """Pair logits for every candidate + one absent logit, for one real query."""
    k = len(aligned)
    if k == 0:
        query = torch.from_numpy(aligned.query_raw)[None].to(device)
        embed = model.query_self_embed(query)
        empty_logits = torch.zeros((1, 0), device=device)
        empty_valid = torch.zeros((1, 0), dtype=torch.bool, device=device)
        absent_logit = model.absent_head(embed, empty_logits, empty_valid)
        return {'pair_logits': np.zeros(0), 'absent_logit': float(absent_logit.item()),
               'valid': np.zeros(0, bool)}
    query = torch.from_numpy(aligned.query_raw)[None].expand(k, -1, -1).to(device)
    crops = torch.from_numpy(aligned.crops).to(device)
    zq = zc = None
    if model.hardnet_fusion:
        zq = model.hardnet_descriptors(query)
        zc = model.hardnet_descriptors(crops)
    logits = model.pair_logit(query, crops, zq, zc)
    query1 = torch.from_numpy(aligned.query_raw)[None].to(device)
    embed = model.query_self_embed(query1)
    valid_t = torch.from_numpy(aligned.admissible & ~aligned.low_info)[None].to(device)
    absent_logit = model.absent_head(embed, logits[None], valid_t)
    return {'pair_logits': logits.cpu().numpy(),
           'absent_logit': float(absent_logit.item()),
           'valid': (aligned.admissible & ~aligned.low_info)}


def score_real_scene(model, scene: str, device: str, data=None,
                     offset_model=None) -> list:
    """Per-query rows for one scene: presence-decision inputs + localization truth."""
    node = load_real_aligned(_old_paths(), scene)
    records = query_records(scene, data)
    rows = []
    for r in records:
        aligned = _aligned_set(node, r['index'])
        scored = score_aligned_set(model, aligned, device)
        logits = np.where(scored['valid'], scored['pair_logits'], -np.inf)
        has_valid = bool(scored['valid'].any())
        best_idx = int(np.argmax(logits)) if has_valid else None
        best_logit = float(logits[best_idx]) if has_valid else None
        second_logit = None
        if has_valid and int(scored['valid'].sum()) >= 2:
            order = np.argsort(-logits)
            second_logit = float(logits[order[1]])
        row = {
            'scene': scene, 'index': r['index'], 'query_id': r['query_id'],
            'present': r['present'], 'stratum': r['stratum'], 'truth_xy': r['xy'],
            'figure': r['figure'],
            'best_index': best_idx, 'best_logit': best_logit,
            'second_logit': second_logit,
            'best_minus_second': (best_logit - second_logit)
            if second_logit is not None else 0.0,
            'absent_logit': scored['absent_logit'],
            'margin': (best_logit - scored['absent_logit'])
            if best_logit is not None else -np.inf,
            'n_valid': int(scored['valid'].sum()),
            'best_xy': (aligned.xy[best_idx].tolist() if best_idx is not None else None),
            'best_pose': (aligned.poses[best_idx].tolist() if best_idx is not None else None),
            'empty_bank': len(aligned) == 0,
            'alignment_failed_all': len(aligned) > 0 and not has_valid,
        }
        if r['present'] and best_idx is not None:
            d = float(np.hypot(aligned.xy[best_idx, 0] - r['xy'][0],
                               aligned.xy[best_idx, 1] - r['xy'][1]))
            row['error_distance'] = d
            row['localization_reward'] = float(np.clip((36.0 - d) / 24.0, 0, 1))
            row['top1_correct'] = bool(d <= 12.0)
            dist_all = np.hypot(aligned.xy[:, 0] - r['xy'][0], aligned.xy[:, 1] - r['xy'][1])
            row['bank_has_correct'] = bool((dist_all <= 12.0).any())
            top5_idx = np.argsort(logits)[::-1][:5]
            row['top5_correct'] = bool((dist_all[top5_idx] <= 12.0).any())
            ranked = np.argsort(-logits)
            correct = [i for i in ranked if dist_all[i] <= 12.0]
            row['rank_of_correct'] = int(np.where(ranked == correct[0])[0][0]) \
                if correct else None
        elif r['present']:
            row.update({'error_distance': None, 'localization_reward': 0.0,
                       'top1_correct': False, 'bank_has_correct': False,
                       'top5_correct': False, 'rank_of_correct': None})
        if offset_model is not None and best_idx is not None and has_valid:
            row.update(_predict_offset(offset_model, aligned, best_idx, device))
        rows.append(row)
    return rows


@torch.no_grad()
def _predict_offset(model, aligned, best_idx: int, device: str) -> dict:
    query = torch.from_numpy(aligned.query_raw)[None].to(device)
    candidate = torch.from_numpy(aligned.crops[best_idx])[None].to(device)
    zq = zc = None
    if model.hardnet_fusion:
        zq = model.hardnet_descriptors(query)
        zc = model.hardnet_descriptors(candidate)
    dx, dy, conf = model.offset(query, candidate, zq, zc)
    return {'offset_dx': float(dx.item()), 'offset_dy': float(dy.item()),
           'offset_confidence': float(conf.item())}

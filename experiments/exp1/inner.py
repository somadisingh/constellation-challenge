"""Inner-validation alignment cache and checkpoint-selection metrics (plan §10, §11).

Alignment does not depend on the model, so the inner bank is aligned once per fold
and cached. Every checkpoint of every arm is then scored against byte-identical
crops with identical pose trial counts.

The selection statistic is plan §10 step 3: equal-source-sky mean top1 localization
reward over TRULY PRESENT inner-validation queries, with no learned presence
threshold. Absent queries are scored too, but only for the false-accept curve.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import load_scene
from .env import read_json, write_json
from .pose import SceneReps
from .scoring import (NEG_INF, AlignedSet, align_query, score_classical,
                      score_descriptor, score_pair_head, summarise_query)

CORRECT_RADIUS = 12.0


def build_inner_aligned(fold: str, inner_doc: dict, config: dict,
                        data: str | None = None, progress=None) -> dict:
    """Aligned crops for every inner-validation query of one fold."""
    aa = float(config['pose']['aa_factor'])
    out = {'fold': fold, 'skies': {}}
    for scene, node in inner_doc['skies'].items():
        reps = SceneReps(load_scene(scene, data).image)
        queries = node['queries']
        images = node['images']
        kmax = max((len(q['candidates']) for q in queries), default=0)
        n = len(queries)
        crops = np.zeros((n, kmax, 32, 32), np.float32)
        ncc = np.full((n, kmax), np.nan, np.float32)
        poses = np.zeros((n, kmax, 2), np.float32)
        adm = np.zeros((n, kmax), bool)
        low = np.zeros((n, kmax), bool)
        xy = np.zeros((n, kmax, 2), np.float32)
        counts = np.zeros(n, int)
        qimg = np.zeros((n, 32, 32), np.float32)
        rejected = np.zeros(n, int)
        for i, q in enumerate(queries):
            cands = q['candidates']
            aligned = align_query(
                reps, images[q['image_slot']],
                [[c['x'], c['y']] for c in cands],
                [[c['angle'], c['scale']] for c in cands],
                query_id=f'{scene}:inner:{i}', aa_factor=aa)
            k = len(cands)
            counts[i] = k
            qimg[i] = aligned.query_raw
            rejected[i] = aligned.rejected_poses
            if k:
                crops[i, :k] = aligned.crops
                ncc[i, :k] = aligned.ncc
                poses[i, :k] = aligned.poses
                adm[i, :k] = aligned.admissible
                low[i, :k] = aligned.low_info
                xy[i, :k] = aligned.xy
            if progress and (i + 1) % 64 == 0:
                progress(f'    {fold}/{scene} align {i + 1}/{n}')
        out['skies'][scene] = {
            'crops': crops, 'ncc': ncc, 'poses': poses, 'admissible': adm,
            'low_info': low, 'xy': xy, 'counts': counts, 'query': qimg,
            'rejected': rejected,
            'kind': [q['kind'] for q in queries],
            'centre': [q['centre'] for q in queries],
            'stratum': [q['stratum'] for q in queries],
            'source_id': [q['source_id'] for q in queries],
            'pose_trials': int(aligned.pose_trials) if n else 0,
        }
    return out


def save_inner_aligned(paths, fold: str, doc: dict) -> Path:
    target = Path(paths.root) / 'inner' / f'{fold}_aligned.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = {}
    meta = {'fold': fold, 'skies': {}}
    for scene, node in doc['skies'].items():
        for key in ('crops', 'ncc', 'poses', 'admissible', 'low_info', 'xy',
                    'counts', 'query', 'rejected'):
            blob[f'{scene}/{key}'] = node[key]
        meta['skies'][scene] = {k: node[k] for k in
                                ('kind', 'centre', 'stratum', 'source_id',
                                 'pose_trials')}
    np.savez_compressed(target, **blob)
    write_json(target.with_suffix('.meta.json'), meta)
    return target


def load_inner_aligned(paths, fold: str) -> dict:
    target = Path(paths.root) / 'inner' / f'{fold}_aligned.npz'
    meta = read_json(target.with_suffix('.meta.json'))
    out = {'fold': fold, 'skies': {}}
    with np.load(target) as blob:
        for scene, node in meta['skies'].items():
            entry = dict(node)
            for key in ('crops', 'ncc', 'poses', 'admissible', 'low_info', 'xy',
                        'counts', 'query', 'rejected'):
                entry[key] = blob[f'{scene}/{key}']
            out['skies'][scene] = entry
    return out


def _aligned_set(node: dict, i: int) -> AlignedSet:
    k = int(node['counts'][i])
    return AlignedSet(
        query_id=f'inner:{i}', xy=node['xy'][i, :k].astype(float),
        crops=node['crops'][i, :k], ncc=node['ncc'][i, :k].astype(float),
        poses=node['poses'][i, :k].astype(float),
        admissible=node['admissible'][i, :k], low_info=node['low_info'][i, :k],
        pose_trials=int(node.get('pose_trials', 9)),
        rejected_poses=int(node['rejected'][i]), query_raw=node['query'][i])


def score_inner(aligned_doc: dict, arm: str, model=None, head=None,
                device: str = 'cpu') -> dict:
    """Score every inner query with one arm. Returns per-sky per-query summaries."""
    out = {}
    for scene, node in aligned_doc['skies'].items():
        rows = []
        n = len(node['kind'])
        for i in range(n):
            aligned = _aligned_set(node, i)
            if arm == 'classical':
                scores = score_classical(aligned)
            elif arm == 'descriptor':
                scores = score_descriptor(model, aligned, device)
            elif arm == 'pair_head':
                scores = score_pair_head(model, head, aligned, device)
            else:
                raise ValueError(f'unknown arm {arm!r}')
            truth = node['centre'][i]
            summary = summarise_query(scores, aligned,
                                      truth_xy=truth if truth is not None else None)
            summary.update({'kind': node['kind'][i], 'stratum': node['stratum'][i],
                            'source_id': node['source_id'][i], 'index': i})
            rows.append(summary)
        out[scene] = rows
    return out


def inner_metrics(scored: dict) -> dict:
    """Selection statistic plus the diagnostics plan §11 asks for."""
    per_sky = {}
    for scene, rows in scored.items():
        present = [r for r in rows if r['kind'] == 'present']
        absent = [r for r in rows if r['kind'] == 'absent']
        reward = [r['localization_reward'] for r in present]
        top1 = [r['top1_correct'] for r in present]
        top5 = [r['top5_correct'] for r in present]
        has = [r.get('bank_has_correct', False) for r in present]
        cond = [r['top1_correct'] for r in present if r.get('bank_has_correct')]
        per_sky[scene] = {
            'n_present': len(present), 'n_absent': len(absent),
            'top1_localization_reward': float(np.mean(reward)) if reward else 0.0,
            'top1_correct': float(np.mean(top1)) if top1 else 0.0,
            'top5_correct': float(np.mean(top5)) if top5 else 0.0,
            'candidate_recall@12': float(np.mean(has)) if has else 0.0,
            'top1_correct_given_bank_has_it': float(np.mean(cond)) if cond else 0.0,
            'mean_best_score_present': _safe_mean([r['best_match_score'] for r in present]),
            'mean_best_score_absent': _safe_mean([r['best_match_score'] for r in absent]),
            'alignment_failed': int(sum(r['alignment_failed_all'] for r in rows)),
        }
    skies = sorted(per_sky)
    metric = float(np.mean([per_sky[s]['top1_localization_reward'] for s in skies]))
    out = {
        'per_sky': per_sky,
        'equal_sky_top1_localization_reward': metric,
        'equal_sky_top1_correct': float(np.mean(
            [per_sky[s]['top1_correct'] for s in skies])),
        'separation': float(np.mean([
            (per_sky[s]['mean_best_score_present'] or 0.0)
            - (per_sky[s]['mean_best_score_absent'] or 0.0) for s in skies])),
    }
    out['false_accept'] = false_accept_curve(scored)
    return out


def _safe_mean(values):
    v = [x for x in values if x is not None and np.isfinite(x) and x > NEG_INF / 2]
    return float(np.mean(v)) if v else None


def false_accept_curve(scored: dict, points: int = 21) -> dict:
    """Present-recall against absent false-accept over the pooled best score.

    Small regional searches underestimate full-sky false-match opportunity, so this
    is a checkpoint-ranking aid, not a presence-quality claim.
    """
    present, absent = [], []
    for rows in scored.values():
        for r in rows:
            s = r['best_match_score']
            if s is None or not np.isfinite(s) or s <= NEG_INF / 2:
                continue
            (present if r['kind'] == 'present' else absent).append(float(s))
    if not present or not absent:
        return {'ok': False}
    lo = min(min(present), min(absent))
    hi = max(max(present), max(absent))
    curve = []
    for t in np.linspace(lo, hi, points):
        curve.append({
            'threshold': float(t),
            'recall': float(np.mean(np.array(present) >= t)),
            'false_accept': float(np.mean(np.array(absent) >= t)),
        })
    fa95 = [c['false_accept'] for c in curve if c['recall'] >= 0.95]
    return {'ok': True, 'curve': curve,
            'n_present': len(present), 'n_absent': len(absent),
            'false_accept_at_95_recall': (min(fa95) if fa95 else 1.0),
            'auc': _auc(present, absent)}


def _auc(present, absent) -> float:
    """Probability a present query outranks an absent one (rank based)."""
    a = np.array(present)
    b = np.array(absent)
    order = np.argsort(np.concatenate([a, b]), kind='stable')
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    ra = ranks[:len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))

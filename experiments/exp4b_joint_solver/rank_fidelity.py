"""Phase 1: leave-one-sky-out evaluation of the 7 rank-fusion rules, both seeds.

For each fold (held-out sky S), features are collected using ONLY the arm-F
checkpoint(s) whose own training fold equals S (the same leak-free membership
rule Experiment 4 established) -- i.e. features for scene S in fold S's run are
never produced by a model that trained on S.
"""
from __future__ import annotations

import time

import numpy as np

from . import OUT, SCENES
from .rank_features import collect_all, relevance
from .rank_fusion import (RULES, TrainedRankingHead, rank_by_absent_separated)
from experiments.exp1.env import write_json
from experiments.exp1.splits import fold_skies


def _load_fold_members(fold: str, seeds: tuple, device: str) -> dict:
    import torch
    from experiments.exp3_pairwise.hardnet_source import frozen_hardnet
    from experiments.exp3_pairwise.training import _load_trainable_state_dict, build_model
    from experiments.exp3_pairwise.paths import checkpoints_dir
    from experiments.exp3_pairwise import ARM_SPEC

    spec = ARM_SPEC['F']
    models = {}
    for seed in seeds:
        hardnet_model, _ = frozen_hardnet(device, fold, seed)
        model = build_model('F', hardnet_backbone=hardnet_model).to(device)
        ckpt = checkpoints_dir(fold, 'F', seed) / 'best.pt'
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        _load_trainable_state_dict(model, blob['model'])
        model.eval()
        models[seed] = model
    return models


def _metrics_for_ranking(rows: list, ranking_fn, is_rule7: bool = False) -> dict:
    top1_reward, top1_correct, top5_correct, top20_correct, mrr = [], [], [], [], []
    figure_reward, offfigure_reward = [], []
    pool_missing = 0
    per_query = []
    for row in rows:
        if not row['present']:
            continue
        if is_rule7:
            ranked, _ = ranking_fn(row)
        else:
            ranked = ranking_fn(row)
        if not ranked:
            pool_missing += 1
            top1_reward.append(0.0); top1_correct.append(False)
            top5_correct.append(False); top20_correct.append(False)
            mrr.append(0.0)
            per_query.append({'query_id': row['query_id'], 'top1_correct': False,
                             'reward': 0.0})
            if row.get('figure') == 1:
                figure_reward.append(0.0)
            elif row.get('figure') == 0:
                offfigure_reward.append(0.0)
            continue
        best = ranked[0]
        r0 = relevance(best['distance_to_truth'])
        top1_reward.append(r0)
        top1_correct.append(bool(best['distance_to_truth'] is not None
                                 and best['distance_to_truth'] <= 12.0))
        top5 = ranked[:5]
        top20 = ranked[:20]
        top5_correct.append(any(c['distance_to_truth'] is not None
                               and c['distance_to_truth'] <= 12.0 for c in top5))
        top20_correct.append(any(c['distance_to_truth'] is not None
                                and c['distance_to_truth'] <= 12.0 for c in top20))
        rank_of_correct = next((i for i, c in enumerate(ranked)
                               if c['distance_to_truth'] is not None
                               and c['distance_to_truth'] <= 12.0), None)
        mrr.append(1.0 / (rank_of_correct + 1) if rank_of_correct is not None else 0.0)
        per_query.append({'query_id': row['query_id'], 'top1_correct': top1_correct[-1],
                         'reward': r0})
        if row.get('figure') == 1:
            figure_reward.append(r0)
        elif row.get('figure') == 0:
            offfigure_reward.append(r0)

    return {
        'n_present': len(top1_reward),
        'top1_localization_reward': float(np.mean(top1_reward)) if top1_reward else None,
        'top1_recall': float(np.mean(top1_correct)) if top1_correct else None,
        'top5_recall': float(np.mean(top5_correct)) if top5_correct else None,
        'top20_recall': float(np.mean(top20_correct)) if top20_correct else None,
        'mean_reciprocal_rank': float(np.mean(mrr)) if mrr else None,
        'figure_reward': float(np.mean(figure_reward)) if figure_reward else None,
        'offfigure_reward': float(np.mean(offfigure_reward)) if offfigure_reward else None,
        'pool_missing_rate': float(pool_missing / max(len(top1_reward), 1)),
        'per_query': per_query,
    }


def evaluate_fold(fold: str, seeds: tuple, device: str, say=print) -> dict:
    held_out, allowed = fold_skies(fold)
    started = time.perf_counter()
    models = _load_fold_members(fold, seeds, device)
    say(f'  loaded {len(models)} leak-free member(s) for fold={fold}')

    features = collect_all(models, device, say=say)
    allowed_rows = [r for s in allowed for r in features[s]]

    head = TrainedRankingHead()
    fit_result = head.fit(allowed_rows, seed=seeds[0])
    say(f'  trained ranking head: {fit_result.get("ok")} n_pairs={fit_result.get("n_pairs")}')

    results = {}
    for name, fn in RULES.items():
        m = _metrics_for_ranking(features[held_out], fn)
        results[name] = m
        say(f'    {name:28s} top1_reward={m["top1_localization_reward"]} '
           f'top1_recall={m["top1_recall"]} mrr={m["mean_reciprocal_rank"]}')

    if fit_result.get('ok'):
        m6 = _metrics_for_ranking(features[held_out], head.rank)
        results['6_trained_ranking_head'] = m6
        say(f'    6_trained_ranking_head      top1_reward={m6["top1_localization_reward"]} '
           f'top1_recall={m6["top1_recall"]} mrr={m6["mean_reciprocal_rank"]}')
    else:
        results['6_trained_ranking_head'] = {'ok': False, 'reason': fit_result.get('reason')}

    m7 = _metrics_for_ranking(features[held_out], rank_by_absent_separated, is_rule7=True)
    results['7_absent_separated'] = m7
    say(f'    7_absent_separated           top1_reward={m7["top1_localization_reward"]} '
       f'top1_recall={m7["top1_recall"]} mrr={m7["mean_reciprocal_rank"]}')

    elapsed = time.perf_counter() - started
    return {
        'fold': fold, 'held_out': held_out, 'allowed': list(allowed), 'seeds': list(seeds),
        'ranking_head_fit': fit_result, 'results': results,
        'seconds': elapsed,
    }


def run(seeds: tuple, device: str, say=print) -> dict:
    out = {}
    for fold in SCENES:
        say(f'\n===== rank fidelity: fold {fold} (seeds {seeds}) =====')
        out[fold] = evaluate_fold(fold, seeds, device, say=say)
    return out


def select_deployable_rule(primary_doc: dict, repeat_doc: dict, say=print) -> dict:
    """Select ONE fixed rule using ONLY the mean top1_localization_reward across
    all folds and BOTH seeds (allowed-sky-equivalent aggregate evidence -- this
    is the held-out metric itself here because rank fidelity IS the held-out
    task, but the SELECTION uses the aggregate across all folds/seeds, never a
    single scene's own result, matching the "no per-scene routing" rule)."""
    rule_names = list(RULES) + ['6_trained_ranking_head', '7_absent_separated']
    scores = {name: [] for name in rule_names}
    for doc in (primary_doc, repeat_doc):
        for fold, fdata in doc.items():
            for name in rule_names:
                r = fdata['results'].get(name, {})
                v = r.get('top1_localization_reward')
                if v is not None:
                    scores[name].append(v)
    means = {name: float(np.mean(vs)) if vs else None for name, vs in scores.items()}
    ranked = sorted([n for n in means if means[n] is not None], key=lambda n: -means[n])
    selected = ranked[0] if ranked else None
    say(f'Rule selection (mean top1_localization_reward across all folds/seeds): {means}')
    say(f'SELECTED: {selected}')
    return {'means': means, 'ranking': ranked, 'selected': selected}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    say = print
    say('=== Phase 1: rank fidelity, seed 31004 (primary) ===')
    primary = run((31004,), args.device, say=say)
    write_json(OUT / 'rank_fidelity_primary.json', primary)

    say('\n=== Phase 1: rank fidelity, seed 31005 (repeat) ===')
    repeat = run((31005,), args.device, say=say)
    write_json(OUT / 'rank_fidelity_repeat.json', repeat)

    say('\n=== Phase 1: selecting one fixed deployable rule ===')
    selection = select_deployable_rule(primary, repeat, say=say)
    write_json(OUT / 'rank_fidelity_ablation.json', {
        'primary': primary, 'repeat': repeat, 'selection': selection})

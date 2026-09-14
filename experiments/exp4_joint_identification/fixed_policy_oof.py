"""Phase 1: honest, leak-free evaluation of the ACTUAL deployable arm-F policy.

The corrected Experiment 3 headline (0.8170) evaluates each fold's ORACLE-SELECTED
arm (D/E/F, chosen per fold from allowed-sky evidence) -- not the single fixed
architecture (arm F, six-checkpoint ensemble) that `deployment_policy.json`
actually deploys on an unseen scene. Those are different systems. This module
evaluates the fixed-arm-F system, honestly, and with a leak-free membership rule
that the deployment ensemble itself does not satisfy for real-scene evaluation.

LEAK-FREE MEMBERSHIP RULE (rule 14 of the task):
`deployment_policy.json`'s ensemble has 6 members: one arm-F checkpoint per
(fold, seed). Fold X's checkpoint is trained on the two ALLOWED skies of fold X
(`experiments.exp1.splits.fold_skies`), i.e. every sky except X. So if we scored
the FULL 6-member ensemble on sky S, four of the six members (the two OTHER
folds' checkpoints, both seeds) would have trained on S -- a direct label leak.
Only the two members whose OWN fold equals S never saw S during training or
calibration. This module therefore evaluates, for each held-out sky S, an
ensemble of EXACTLY those two eligible members (fold=S, seed in {31004,31005}),
combined by the same calibrated-log-odds-averaging rule `deployment.py` intends
for the real (six-member) system.

This is NOT a perfect estimate of the six-member ensemble's unseen-scene
accuracy: the six-member ensemble also benefits from the OTHER four members'
different training data (more diverse features), which a 2-member evaluation
cannot capture. It IS a leak-free lower/comparable bound on what the SAME
architecture and aggregation rule achieve on a genuinely unseen sky, which the
6-member evaluation on a labelled sky cannot honestly claim to be.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import DEPLOYED_ARM, OUT, SCENES
from experiments.exp1.env import read_json, sha256_file, write_json
from experiments.exp1.evaluation import evaluate_predictions
from experiments.exp1.splits import fold_skies
from experiments.exp3_pairwise import ARM_SPEC
from experiments.exp3_pairwise.calibration import (apply_calibrator, fit_calibrator,
                                                    listwise_absent_probability,
                                                    select_threshold)
from experiments.exp3_pairwise.hardnet_source import frozen_hardnet
from experiments.exp3_pairwise.integration import apply_geometry_stage, build_prediction
from experiments.exp3_pairwise.paths import checkpoints_dir
from experiments.exp3_pairwise.real_scoring import _aligned_set as _exp3_aligned_set
from experiments.exp3_pairwise.training import _load_trainable_state_dict, build_model

# Deployment drops the offset head (zero measured held-out gain, per Exp3's own
# repair); the honest fixed-policy evaluation follows the ACTUAL deployed
# behaviour, so no `verifier_offset` stage is scored here.
STAGES = ('verifier_only', 'verifier_snap', 'verifier_snap_rescue')


def _load_member(fold: str, seed: int, device: str) -> tuple:
    spec = ARM_SPEC[DEPLOYED_ARM]
    hardnet_model, hardnet_prov = frozen_hardnet(device, fold, seed)
    model = build_model(DEPLOYED_ARM, hardnet_backbone=hardnet_model).to(device)
    ckpt = checkpoints_dir(fold, DEPLOYED_ARM, seed) / 'best.pt'
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    _load_trainable_state_dict(model, blob['model'])
    model.eval()
    return model, {'fold': fold, 'seed': seed, 'checkpoint': str(ckpt),
                  'checkpoint_sha256': sha256_file(ckpt),
                  'hardnet_source': hardnet_prov}


@torch.no_grad()
def _score_member(model, aligned, device: str) -> dict:
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


def score_scene_ensemble(members: list, scene: str, device: str, data=None) -> list:
    """Score every real query of `scene` with each of `members`, average in
    logit space, and additionally record per-member disagreement diagnostics."""
    from experiments.exp1.data import query_records
    from experiments.exp1.evaluation import load_real_aligned
    from experiments.exp1.stages import Paths
    from experiments.exp1.env import ROOT as EXP1_ROOT

    node = load_real_aligned(Paths(EXP1_ROOT / 'outputs' / 'exp1'), scene)
    records = query_records(scene, data)
    rows = []
    for r in records:
        aligned = _exp3_aligned_set(node, r['index'])
        per_member = [_score_member(m, aligned, device) for m in members]
        valid = per_member[0]['valid']   # identical candidate bank for every member
        stacked = np.stack([np.where(pm['valid'], pm['pair_logits'], np.nan)
                            for pm in per_member])
        if stacked.size and not np.all(np.isnan(stacked)):
            mean_logits = np.nanmean(stacked, axis=0)
        else:
            mean_logits = np.zeros(stacked.shape[-1] if stacked.ndim > 1 else 0)
        mean_absent = float(np.mean([pm['absent_logit'] for pm in per_member]))
        masked = np.where(valid, mean_logits, -np.inf) if len(mean_logits) else mean_logits
        has_valid = bool(valid.any())
        best_idx = int(np.argmax(masked)) if has_valid else None
        best_logit = float(masked[best_idx]) if has_valid else None
        second_logit = None
        if has_valid and int(valid.sum()) >= 2:
            order = np.argsort(-masked)
            second_logit = float(masked[order[1]])

        # Ensemble disagreement: per-member best-candidate agreement, and spread
        # of each member's own best-logit-minus-absent-logit margin.
        member_best_idx = []
        member_margins = []
        for pm in per_member:
            pm_masked = np.where(pm['valid'], pm['pair_logits'], -np.inf) \
                if len(pm['pair_logits']) else pm['pair_logits']
            pm_best = int(np.argmax(pm_masked)) if pm['valid'].any() else None
            member_best_idx.append(pm_best)
            pm_best_logit = float(pm_masked[pm_best]) if pm_best is not None else None
            member_margins.append((pm_best_logit - pm['absent_logit'])
                                  if pm_best_logit is not None else None)
        agree = (len(set(x for x in member_best_idx if x is not None)) <= 1
                and all(x is not None for x in member_best_idx)) if member_best_idx else True
        margin_values = [m for m in member_margins if m is not None]
        margin_spread = float(np.std(margin_values)) if len(margin_values) >= 2 else 0.0

        row = {
            'scene': scene, 'index': r['index'], 'query_id': r['query_id'],
            'present': r['present'], 'stratum': r['stratum'], 'truth_xy': r['xy'],
            'figure': r['figure'],
            'best_index': best_idx, 'best_logit': best_logit,
            'second_logit': second_logit,
            'best_minus_second': (best_logit - second_logit)
            if second_logit is not None else 0.0,
            'absent_logit': mean_absent,
            'n_valid': int(valid.sum()),
            'best_xy': (aligned.xy[best_idx].tolist() if best_idx is not None else None),
            'empty_bank': len(aligned) == 0,
            'alignment_failed_all': len(aligned) > 0 and not has_valid,
            'ensemble_members_agree_on_best': agree,
            'ensemble_margin_spread': margin_spread,
            'ensemble_member_best_index': member_best_idx,
        }
        if r['present'] and best_idx is not None:
            d = float(np.hypot(aligned.xy[best_idx, 0] - r['xy'][0],
                               aligned.xy[best_idx, 1] - r['xy'][1]))
            row['error_distance'] = d
            row['localization_reward'] = float(np.clip((36.0 - d) / 24.0, 0, 1))
            row['top1_correct'] = bool(d <= 12.0)
            dist_all = np.hypot(aligned.xy[:, 0] - r['xy'][0], aligned.xy[:, 1] - r['xy'][1])
            row['bank_has_correct'] = bool((dist_all <= 12.0).any())
            top5_idx = np.argsort(masked)[::-1][:5] if has_valid else np.array([], int)
            row['top5_correct'] = bool((dist_all[top5_idx] <= 12.0).any()) if len(top5_idx) else False
        elif r['present']:
            row.update({'error_distance': None, 'localization_reward': 0.0,
                       'top1_correct': False, 'bank_has_correct': False,
                       'top5_correct': False})
        rows.append(row)
    return rows


def load_classical(scene: str):
    from constellation.contracts import ScenePrediction
    doc = read_json(f'outputs/joint_train/{scene}.json')
    return ScenePrediction(doc['patches'], doc['constellation'], doc.get('diagnostics', {}))


def evaluate_leak_free_fold(fold: str, seed_pair: tuple, device: str,
                            data: str | None = None, say=print) -> dict:
    """Evaluate the fixed arm-F, 2-member (leak-free) ensemble on fold `fold`'s
    held-out sky. `seed_pair` is normally (31004, 31005) -- both seeds' own
    fold-`fold` checkpoints, never a checkpoint trained on the held-out sky."""
    held_out, allowed = fold_skies(fold)
    members, member_provenance = [], []
    for seed in seed_pair:
        model, prov = _load_member(fold, seed, device)
        members.append(model)
        member_provenance.append(prov)
        say(f'  loaded leak-free member fold={fold} seed={seed} '
           f'checkpoint={prov["checkpoint"]}')

    rows_by_scene = {s: score_scene_ensemble(members, s, device, data) for s in SCENES}
    allowed_rows = [r for s in allowed for r in rows_by_scene[s]]

    cal = fit_calibrator(allowed_rows)
    probs_allowed = (apply_calibrator(cal, allowed_rows) if cal.get('ok')
                     else listwise_absent_probability(allowed_rows))
    thr = select_threshold(probs_allowed, allowed_rows)
    say(f'  calibrator ok={cal.get("ok")} threshold={thr["selected"]:.3f}')

    classical = {s: load_classical(s) for s in SCENES}

    stage_predictions = {}
    for stage in STAGES:
        preds = {}
        for s in SCENES:
            probs = (apply_calibrator(cal, rows_by_scene[s]) if cal.get('ok')
                    else listwise_absent_probability(rows_by_scene[s]))
            built = build_prediction(s, rows_by_scene[s], probs, thr['selected'],
                                     offset_gate=None)
            learned = built['prediction']
            if stage in ('verifier_snap', 'verifier_snap_rescue'):
                learned = apply_geometry_stage(learned, classical[s], stage)
            preds[s] = learned
        stage_predictions[stage] = preds

    stage_metrics = {stage: evaluate_predictions(preds)
                     for stage, preds in stage_predictions.items()}
    for stage, m in stage_metrics.items():
        say(f'  {stage:20s} held-out({held_out})={m["scenes"][held_out]["score"]:.4f} '
           f'mean={m["mean"]["score"]:.4f}')

    disagreement = {
        s: {
            'mean_margin_spread': float(np.mean([r['ensemble_margin_spread']
                                                 for r in rows_by_scene[s]])),
            'agreement_rate': float(np.mean([r['ensemble_members_agree_on_best']
                                             for r in rows_by_scene[s]])),
        } for s in SCENES
    }

    return {
        'fold': fold, 'held_out': held_out, 'allowed': list(allowed),
        'arm': DEPLOYED_ARM, 'seeds': list(seed_pair),
        'members': member_provenance,
        'leak_free_membership_rule': (
            f'only checkpoints whose OWN training fold == held-out sky {held_out!r} '
            f'are used (both members trained on {list(allowed)}, held out {held_out!r}); '
            f'the full 6-member deployment ensemble is NOT used here because 4 of its '
            f'6 members were trained on {held_out!r} and would leak'),
        'calibration': cal, 'threshold': thr,
        'ensemble_disagreement': disagreement,
        'stages': {stage: {'metrics': m, 'held_out_metrics': m['scenes'][held_out]}
                  for stage, m in stage_metrics.items()},
        'predictions': {stage: {s: {'patches': p.patches,
                                    'constellation': p.constellation,
                                    'diagnostics': p.diagnostics}
                               for s, p in preds.items()}
                       for stage, preds in stage_predictions.items()},
        'rows': {s: rows_by_scene[s] for s in SCENES},
    }


def run_all_folds(device: str, data: str | None = None, say=print) -> dict:
    """Both seeds (31004, 31005) are BOTH used as ensemble members within one
    leak-free evaluation (there is no separate 'primary'/'repeat' seed split
    for the fixed policy, since deployment always uses both seeds together as
    2 ensemble members per fold). We report this as a single result, plus a
    single-seed ablation for comparison against Exp3's own primary/repeat
    convention."""
    out = {}
    for fold in SCENES:
        say(f'\n===== leak-free fixed-arm-F eval: fold {fold} =====')
        out[fold] = evaluate_leak_free_fold(fold, (31004, 31005), device, data, say)
    return out


def single_seed_ablation(device: str, seed: int, data: str | None = None,
                         say=print) -> dict:
    """Single-seed variant (1 member per fold instead of 2) -- lets us report a
    primary/repeat-style pair using the SAME leak-free rule, matching Exp3's own
    seed convention for a fair side-by-side comparison."""
    out = {}
    for fold in SCENES:
        say(f'\n===== leak-free fixed-arm-F eval (seed {seed} only): fold {fold} =====')
        out[fold] = evaluate_leak_free_fold(fold, (seed,), device, data, say)
    return out


def _mean_across_folds(doc: dict, stage: str) -> dict:
    per_scene = {}
    for fold, fdata in doc.items():
        held = fdata['held_out']
        per_scene[held] = fdata['stages'][stage]['held_out_metrics']
    keys = ('presence', 'localization', 'recovery', 'identification', 'score')
    mean = {k: float(np.mean([per_scene[s][k] for s in per_scene])) for k in keys}
    mean['per_scene'] = per_scene
    return mean


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    from experiments.exp3_pairwise.serialize import rows_to_json, slim_eval_result

    say = print
    say('=== Phase 1: leak-free fixed-arm-F OOF (both seeds as ensemble) ===')
    both_seeds = run_all_folds(args.device, say=say)
    (OUT / 'folds').mkdir(parents=True, exist_ok=True)
    for fold, fdata in both_seeds.items():
        write_json(OUT / 'folds' / f'fixed_policy_{fold}_both_seeds.json',
                  slim_eval_result(fdata))
        write_json(OUT / 'folds' / f'fixed_policy_{fold}_both_seeds_rows.json',
                  rows_to_json(fdata['rows']))
    write_json(OUT / 'fixed_policy_oof_both_seeds.json',
              {f: slim_eval_result(d) for f, d in both_seeds.items()})

    say('\n=== Phase 1: single-seed ablations (primary=31004, repeat=31005) ===')
    primary = single_seed_ablation(args.device, 31004, say=say)
    write_json(OUT / 'fixed_policy_oof_primary.json',
              {f: slim_eval_result(d) for f, d in primary.items()})
    repeat = single_seed_ablation(args.device, 31005, say=say)
    write_json(OUT / 'fixed_policy_oof_repeat.json',
              {f: slim_eval_result(d) for f, d in repeat.items()})

    say('\n=== Summary ===')
    for tag, doc in (('both-seed ensemble', both_seeds), ('primary (31004)', primary),
                     ('repeat (31005)', repeat)):
        for stage in STAGES:
            m = _mean_across_folds(doc, stage)
            say(f'{tag:20s} {stage:20s} score={m["score"]:.4f} pres={m["presence"]:.3f} '
               f'loc={m["localization"]:.3f} rec={m["recovery"]:.3f}')

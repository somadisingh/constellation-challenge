"""Fixed A-F matrix on allowed-sky inner validation (task §10, §9).

For each fold, independently (never consulting the held-out sky):
  1. build/cache the fit and val group streams (shared across all six arms);
  2. train each arm for the common step budget, evaluating on val every
     `eval_every` steps with `metrics.inner_selection_metrics`;
  3. select one arm by the frozen tie-break rule in `metrics.select_arm`.

Arms are never added after this step, and the selection uses only the metrics
computed here -- never the held-out sky.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from . import ARM_SPEC, ARMS, OUT
from .hardnet_source import frozen_hardnet
from .metrics import (calibrated_selection_metrics, inner_selection_metrics,
                      score_groups, select_arm, to_calibration_rows)
from .paths import checkpoints_dir
from .streams import StepSampler, load_or_build_stream
from .training import train_arm
from experiments.exp1.env import derive_seed, seed_torch, write_json
from experiments.exp1.splits import fold_skies

COMMON_STEPS = 2000
EVAL_EVERY = 500
BATCH_SIZE = 16

FIT_PRESENT_PER_SKY = 60
FIT_ABSENT_PER_SKY = 60
SYNTHCAL_PRESENT_PER_SKY = 32
SYNTHCAL_ABSENT_PER_SKY = 32
VAL_PRESENT_PER_SKY = 32
VAL_ABSENT_PER_SKY = 32


def build_eval_fn(synthcal_groups: list, val_groups: list, allowed: tuple, device: str):
    """Calibrate on `synthcal`, then score `val` with that FROZEN calibrator.

    Repair task §3.3: prior code compared raw `best_logit - absent_logit` margins
    at threshold 0 across BOTH the binary-BCE arms (A, B) and the listwise arms
    (C-F), whose logit scales are not directly comparable, and it fit calibration
    and select checkpoints on the SAME `val` pool. This function separates the
    two: `synthcal_groups` (the previously-unused third partition,
    `experiments.exp1.splits.SYNTH_CELLS`) is scored, a presence calibrator and
    threshold are fit on it via `calibration.fit_calibrator`/`select_threshold`
    (the SAME mechanism used for the real held-out calibrator, so BCE and
    listwise arms produce comparable probabilities), and ONLY THEN is that frozen
    calibrator/threshold applied to `val` to compute the equal-sky presence and
    localization checkpoint-selection metric.
    """
    from .calibration import fit_calibrator, select_threshold
    synthcal_by_scene = {s: [g for g in synthcal_groups if g.target_scene == s]
                         for s in allowed}
    val_by_scene = {s: [g for g in val_groups if g.target_scene == s] for s in allowed}

    def eval_fn(model, hardnet_model, dev):
        model.eval()
        synthcal_rows_by_scene = {s: score_groups(model, gs, dev)
                                  for s, gs in synthcal_by_scene.items()}
        synthcal_rows = [r for rows in synthcal_rows_by_scene.values() for r in rows]
        calib_rows = to_calibration_rows(synthcal_rows)
        cal = fit_calibrator(calib_rows)
        if cal.get('ok'):
            from .calibration import apply_calibrator
            probs = apply_calibrator(cal, calib_rows)
        else:
            from .calibration import listwise_absent_probability
            probs = listwise_absent_probability(calib_rows)
        thr = select_threshold(probs, calib_rows)

        val_rows_by_scene = {s: score_groups(model, gs, dev)
                             for s, gs in val_by_scene.items()}
        metric_doc = calibrated_selection_metrics(val_rows_by_scene, cal, thr['selected'])
        metric_doc['calibrator'] = cal
        metric_doc['threshold'] = thr
        return metric_doc
    return eval_fn


def run_fold(fold: str, device: str, seed: int, data: str | None = None,
            steps: int = COMMON_STEPS, say=print) -> dict:
    held_out, allowed = fold_skies(fold)
    say(f'\n===== fold {fold} (held out {held_out}, allowed {allowed}) =====')

    fit_groups = load_or_build_stream(fold, 'fit', f's{seed}', FIT_PRESENT_PER_SKY,
                                      FIT_ABSENT_PER_SKY, seed, data, progress=say)
    synthcal_groups = load_or_build_stream(fold, 'synthcal', f's{seed}',
                                           SYNTHCAL_PRESENT_PER_SKY,
                                           SYNTHCAL_ABSENT_PER_SKY, seed, data,
                                           progress=say)
    val_groups = load_or_build_stream(fold, 'val', f's{seed}', VAL_PRESENT_PER_SKY,
                                      VAL_ABSENT_PER_SKY, seed, data, progress=say)
    say(f'  fit groups: {len(fit_groups)}, synthcal groups: {len(synthcal_groups)}, '
       f'val groups: {len(val_groups)}')

    sampler = StepSampler(fit_groups, seed=derive_seed(seed, fold, 'sampler'),
                          batch_size=BATCH_SIZE)
    eval_fn = build_eval_fn(synthcal_groups, val_groups, allowed, device)
    hardnet_model, hardnet_provenance = frozen_hardnet(device, fold, seed)
    say(f'  frozen HardNet source: {hardnet_provenance["source"]} '
       f'checkpoint={hardnet_provenance["checkpoint_path"]} '
       f'sha256={hardnet_provenance["checkpoint_sha256"][:12]}...')

    results = {}
    for arm in ARMS:
        say(f'  training arm {arm} ...')
        ckpt_dir = checkpoints_dir(fold, arm, seed)
        result = train_arm(fold, arm, sampler, hardnet_model, device, seed, steps,
                           EVAL_EVERY, eval_fn, say=say, checkpoint_dir=ckpt_dir)
        results[arm] = result
        write_json(ckpt_dir / 'result.json', {
            k: v for k, v in result.items() if k not in ('history',)})

    # Repair task §3.4: `best` is now ALREADY an atomic, single-step record
    # (step, selection_metric, presence, localization, top1_correct, calibrator,
    # threshold) written by train_arm at the exact step it was saved -- no
    # separate lookup into `evaluations[-1]` (which could be a LATER, unrelated
    # step) is needed or performed here.
    selection_inputs = {arm: r['best'] for arm, r in results.items()}
    selection = select_arm(selection_inputs)
    say(f'  SELECTED arm for {fold}: {selection["selected"]} '
       f'(ranking {selection["ranking"]})')

    return {'fold': fold, 'held_out': held_out, 'allowed': list(allowed),
           'seed': seed, 'steps': steps,
           'hardnet_source': hardnet_provenance,
           'results': {arm: {'best': r['best'], 'nan_steps': r['nan_steps'],
                             'seconds': r['seconds'],
                             'steps_per_second': r['steps_per_second'],
                             'negative_selection_digest': r.get('negative_selection_digest')}
                       for arm, r in results.items()},
           'selection': selection, 'n_fit_groups': len(fit_groups),
           'n_synthcal_groups': len(synthcal_groups), 'n_val_groups': len(val_groups)}


def run_all(device: str, seed: int, data: str | None = None,
           steps: int = COMMON_STEPS, say=print) -> dict:
    from . import SCENES
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for fold in SCENES:
        out[fold] = run_fold(fold, device, seed, data, steps, say)
        write_json(OUT / 'folds' / fold / f'matrix_s{seed}.json', out[fold])
    write_json(OUT / f'matrix_s{seed}.json', out)
    return out

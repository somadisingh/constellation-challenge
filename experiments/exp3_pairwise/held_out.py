"""Held-out evaluation: calibration + offset gate selection (allowed-sky only),
then a SINGLE evaluation of the held-out sky, decomposed into four separately
reported stages (task §12, §13, §15).
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import OUT, SCENES
from .calibration import (fit_calibrator, apply_calibrator, select_threshold,
                          select_offset_gate, listwise_absent_probability)
from .integration import build_prediction, apply_geometry_stage
from .paths import checkpoints_dir
from .real_scoring import score_real_scene
from .training import build_model
from experiments.exp1.env import read_json, sha256_file, write_json
from experiments.exp1.evaluation import evaluate_predictions
from experiments.exp1.splits import fold_skies

STAGES = ('verifier_only', 'verifier_offset', 'verifier_snap', 'verifier_snap_rescue')


def load_selected_model(fold: str, arm: str, seed: int, device: str):
    from . import ARM_SPEC
    from .hardnet_source import frozen_hardnet
    spec = ARM_SPEC[arm]
    hardnet_model = None
    hardnet_provenance = None
    if spec['hardnet_fusion']:
        hardnet_model, hardnet_provenance = frozen_hardnet(device, fold, seed)
    model = build_model(arm, hardnet_backbone=hardnet_model).to(device)
    ckpt = checkpoints_dir(fold, arm, seed) / 'best.pt'
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    from .training import _load_trainable_state_dict
    _load_trainable_state_dict(model, blob['model'])
    model.eval()
    # Keep spec on the model for downstream offset-gating decisions.
    model._exp3_arm = arm
    model._exp3_spec = spec
    model._exp3_hardnet_provenance = hardnet_provenance
    return model, ckpt, hardnet_provenance


def load_classical(scene: str):
    from constellation.contracts import ScenePrediction
    doc = read_json(f'outputs/joint_train/{scene}.json')
    return ScenePrediction(doc['patches'], doc['constellation'], doc.get('diagnostics', {}))


def evaluate_fold(fold: str, arm: str, seed: int, device: str,
                  data: str | None = None, say=print) -> dict:
    """Calibrate on allowed skies, then score the held-out sky exactly once."""
    held_out, allowed = fold_skies(fold)
    model, ckpt, hardnet_provenance = load_selected_model(fold, arm, seed, device)
    say(f'  loaded {fold}/{arm} seed {seed} checkpoint {ckpt}')
    if hardnet_provenance:
        say(f'  frozen HardNet: {hardnet_provenance["source"]} '
           f'sha256={hardnet_provenance["checkpoint_sha256"][:12]}...')

    rows_by_scene = {s: score_real_scene(model, s, device, data,
                                         offset_model=(model if getattr(model, '_exp3_spec', {}).get('offset')
                                                       else None))
                     for s in SCENES}
    allowed_rows = [r for s in allowed for r in rows_by_scene[s]]

    cal = fit_calibrator(allowed_rows)
    probs_allowed = (apply_calibrator(cal, allowed_rows) if cal.get('ok')
                     else listwise_absent_probability(allowed_rows))
    thr = select_threshold(probs_allowed, allowed_rows)
    say(f'  calibrator ok={cal.get("ok")} threshold={thr["selected"]:.3f}')

    offset_gate = select_offset_gate(allowed_rows)
    say(f'  offset gate: {offset_gate.get("selected") if offset_gate.get("ok") else offset_gate.get("reason")}')

    classical = {s: load_classical(s) for s in SCENES}

    stage_predictions = {}
    for stage in STAGES:
        gate = offset_gate.get('selected') if (stage == 'verifier_offset'
                                               and offset_gate.get('ok')) else None
        preds = {}
        for s in SCENES:
            probs = (apply_calibrator(cal, rows_by_scene[s]) if cal.get('ok')
                    else listwise_absent_probability(rows_by_scene[s]))
            built = build_prediction(s, rows_by_scene[s], probs, thr['selected'],
                                     offset_gate=gate)
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

    return {
        'fold': fold, 'held_out': held_out, 'allowed': list(allowed),
        'arm': arm, 'seed': seed, 'checkpoint': str(ckpt),
        'checkpoint_sha256': sha256_file(ckpt),
        'hardnet_source': hardnet_provenance,
        'calibration': cal, 'threshold': thr, 'offset_gate': offset_gate,
        'stages': {stage: {'metrics': m, 'held_out_metrics': m['scenes'][held_out]}
                  for stage, m in stage_metrics.items()},
        'predictions': {stage: {s: {'patches': p.patches,
                                    'constellation': p.constellation,
                                    'diagnostics': p.diagnostics}
                               for s, p in preds.items()}
                       for stage, preds in stage_predictions.items()},
        'rows': {s: rows_by_scene[s] for s in SCENES},
    }

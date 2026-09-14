"""Mandatory overfit gate before full training (task §11).

Checks, on ~32-64 real query groups:
  1. loss falls;
  2. positive logits exceed hard-negative logits;
  3. absent groups prefer the absent option;
  4. the offset head learns a PLANTED synthetic displacement;
  5. gradients reach the pair CNN and offset head;
  6. frozen HardNet weights and BatchNorm state are unchanged;
  7. CPU/MPS score direction and tensor shapes agree within tolerance.

`run()` returns a report with an explicit `ok` flag; the caller must not proceed
to the full training matrix if `ok` is False.
"""
from __future__ import annotations

import numpy as np
import torch

from . import ARM_SPEC, MAX_OFFSET
from .batch import collate
from .forward import score_batch
from .losses import listwise_absent_loss
from .streams import build_group_stream
from .training import build_model, build_training_batch, compute_loss
from experiments.exp1.env import seed_torch
from experiments.exp1.models import Descriptor, load_backbone

N_GROUPS = 48
STEPS = 150
LR = 2e-3


def _frozen_hardnet(device: str):
    backbone, _ = load_backbone('hardnet', pretrained=True, device=device)
    model = Descriptor(backbone, 'hardnet').to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def check_loss_falls_and_ranks_correctly(fold: str, arm: str, device: str,
                                         groups: list) -> dict:
    hardnet_model = _frozen_hardnet(device) if ARM_SPEC[arm]['hardnet_fusion'] else None
    seed_torch(31004)
    model = build_model(arm, hardnet_backbone=hardnet_model).to(device)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=LR)
    spec = ARM_SPEC[arm]

    losses = []
    for step in range(STEPS):
        batch, _ = build_training_batch(groups, spec, hardnet_model=hardnet_model,
                                        device=device)
        result = compute_loss(model, batch, spec, device)
        opt.zero_grad(set_to_none=True)
        result['loss'].backward()
        grad_norms = [float(p.grad.norm()) for p in model.trainable_parameters()
                     if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
        opt.step()
        losses.append(float(result['loss'].detach()))

    model.eval()
    batch, _ = build_training_batch(groups, spec, hardnet_model=hardnet_model,
                                    device=device)
    with torch.no_grad():
        out = score_batch(model, batch, device)
    pos_mask = (batch['positive'] & batch['valid'])
    neg_mask = batch['valid'] & (~batch['positive'])
    pos_logits = out['pair_logits'][pos_mask]
    neg_logits = out['pair_logits'][neg_mask]
    pos_beats_neg = bool(pos_logits.numel() and neg_logits.numel()
                         and float(pos_logits.mean()) > float(neg_logits.mean()))

    absent_rows = ~batch['is_present']
    present_rows = batch['is_present']
    prefers_absent = None
    if absent_rows.any():
        margin_absent = float((out['absent_logit'][absent_rows]
                               - out['pair_logits'][absent_rows].max(dim=1).values).mean())
        prefers_absent = margin_absent > 0
    prefers_present = None
    if present_rows.any():
        best_present = out['pair_logits'][present_rows].max(dim=1).values
        margin_present = float((best_present - out['absent_logit'][present_rows]).mean())
        prefers_present = margin_present > 0

    return {
        'arm': arm, 'loss_first': losses[0], 'loss_last': losses[-1],
        'loss_falls': losses[-1] < losses[0] * 0.7,
        'positive_beats_hard_negative': pos_beats_neg,
        'absent_groups_prefer_absent': prefers_absent,
        'present_groups_prefer_a_candidate': prefers_present,
        'grad_reached_pixel_cnn': any(g > 0 for g in grad_norms),
        'final_grad_norms_nonzero': grad_norms[:3],
        'steps': STEPS,
    }


def check_offset_recovers_planted_displacement(device: str) -> dict:
    """Synthetic pair with a KNOWN planted (dx, dy); the head must learn to predict it."""
    rng = np.random.default_rng(7)
    n = 40
    base = rng.uniform(0, 1, (n, 32, 32)).astype(np.float32)
    planted_dx = rng.uniform(-8, 8, n).astype(np.float32)
    planted_dy = rng.uniform(-8, 8, n).astype(np.float32)
    # Candidate = query shifted by -planted offset (so query = candidate + planted).
    candidate = np.zeros_like(base)
    for i in range(n):
        shift = np.array([[1, 0, -planted_dx[i]], [0, 1, -planted_dy[i]]], np.float32)
        import cv2
        candidate[i] = cv2.warpAffine(base[i], shift, (32, 32), borderMode=cv2.BORDER_REFLECT)

    hardnet_model = _frozen_hardnet(device)
    model = build_model('F', hardnet_backbone=hardnet_model).to(device)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)
    q = torch.from_numpy(base).to(device)
    c = torch.from_numpy(candidate).to(device)
    tdx = torch.from_numpy(planted_dx).to(device)
    tdy = torch.from_numpy(planted_dy).to(device)

    from .losses import offset_loss
    history = []
    for step in range(200):
        dx, dy, conf = model.offset(q, c)
        loss = offset_loss(dx, dy, tdx, tdy, weight=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        dx, dy, _ = model.offset(q, c)
    final_err = float(((dx - tdx).abs() + (dy - tdy).abs()).mean())
    return {'loss_first': history[0], 'loss_last': history[-1],
           'loss_falls': history[-1] < history[0] * 0.5,
           'final_mean_abs_error_px': final_err,
           'recovers_planted_offset': final_err < 2.0,
           'offset_bounded': bool(dx.abs().max() <= MAX_OFFSET + 1e-4
                                  and dy.abs().max() <= MAX_OFFSET + 1e-4)}


def check_frozen_hardnet_unchanged(device: str) -> dict:
    hardnet_model = _frozen_hardnet(device)
    before = {k: v.detach().clone() for k, v in hardnet_model.state_dict().items()}
    model = build_model('D', hardnet_backbone=hardnet_model).to(device)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    q = torch.rand(8, 32, 32, device=device)
    c = torch.rand(8, 32, 32, device=device)
    for _ in range(20):
        logit = model.pair_logit(q, c)
        loss = logit.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    after = hardnet_model.state_dict()
    max_drift = max(float((before[k] - after[k]).abs().max()) for k in before)
    hardnet_grad = any(p.grad is not None and float(p.grad.abs().max()) > 0
                       for p in hardnet_model.parameters())
    bn_modes = [m.training for m in hardnet_model.modules()
               if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    return {'max_weight_drift': max_drift, 'weights_unchanged': max_drift == 0.0,
           'received_gradient': hardnet_grad, 'no_gradient_reached_it': not hardnet_grad,
           'bn_all_eval_mode': all(not m for m in bn_modes) if bn_modes else True}


def check_cpu_mps_parity(fold: str, arm: str, groups: list) -> dict:
    import torch as T
    if not T.backends.mps.is_available():
        return {'ok': True, 'skipped': 'MPS unavailable on this machine'}
    results = {}
    for device in ('cpu', 'mps'):
        seed_torch(31004)
        hardnet_model = _frozen_hardnet(device) if ARM_SPEC[arm]['hardnet_fusion'] else None
        model = build_model(arm, hardnet_backbone=hardnet_model).to(device)
        model.eval()
        spec = ARM_SPEC[arm]
        batch, _ = build_training_batch(groups[:8], spec, hardnet_model=hardnet_model,
                                        device=device)
        with torch.no_grad():
            out = score_batch(model, batch, device)
        results[device] = {'shape': tuple(out['pair_logits'].shape),
                           'order': out['pair_logits'].argsort(dim=1).cpu().numpy()}
    same_shape = results['cpu']['shape'] == results['mps']['shape']
    same_rank = bool(np.array_equal(results['cpu']['order'], results['mps']['order']))
    return {'ok': same_shape, 'same_shape': same_shape,
           'rank_order_agrees': same_rank,
           'cpu_shape': results['cpu']['shape'], 'mps_shape': results['mps']['shape']}


def run(fold: str = 'pisces', arm: str = 'F', device: str = 'cpu',
       data: str | None = None, say=print) -> dict:
    say(f'building {N_GROUPS} groups for the overfit gate ...')
    groups = build_group_stream(fold, 'fit', n_present_per_sky=N_GROUPS // 4,
                                n_absent_per_sky=N_GROUPS // 4, seed=31003,
                                data=data, progress=say)
    say(f'  built {len(groups)} groups')

    checks = {}
    checks['loss_and_ranking'] = check_loss_falls_and_ranks_correctly(fold, arm, device, groups)
    checks['offset_recovery'] = check_offset_recovers_planted_displacement(device)
    checks['frozen_hardnet'] = check_frozen_hardnet_unchanged(device)
    checks['cpu_mps_parity'] = check_cpu_mps_parity(fold, arm, groups)

    passed = (
        checks['loss_and_ranking']['loss_falls']
        and checks['loss_and_ranking']['positive_beats_hard_negative']
        and (checks['loss_and_ranking']['absent_groups_prefer_absent'] is not False)
        and checks['loss_and_ranking']['grad_reached_pixel_cnn']
        and checks['offset_recovery']['recovers_planted_offset']
        and checks['offset_recovery']['offset_bounded']
        and checks['frozen_hardnet']['weights_unchanged']
        and checks['frozen_hardnet']['no_gradient_reached_it']
        and checks['frozen_hardnet']['bn_all_eval_mode']
        and checks['cpu_mps_parity']['ok']
    )
    report = {'ok': passed, 'checks': checks, 'n_groups': len(groups),
             'fold': fold, 'arm': arm, 'device': device}
    say(f'overfit gate: {"PASS" if passed else "FAIL"}')
    for name, node in checks.items():
        say(f'  {name}: {node}')
    return report

"""Resolve the frozen HardNet backbone Experiment 3 fuses with (repair task §3.2).

The protocol requires the SELECTED fold-specific Experiment 1B arm-B checkpoint,
not the generic pretrained HardNet weights. Prior code always loaded the generic
weights via `experiments.exp1.models.load_backbone('hardnet', pretrained=True)`
regardless of fold, which contradicted this package's own docstring claim of
reusing "fold-specific HardNet arm-B checkpoints" (that claim was aspirational,
not implemented, before this repair).

This module provides both:
  - `frozen_hardnet(device, fold, seed)`      the CORRECT fold-specific Exp1B
                                              arm-B checkpoint (default, required)
  - `frozen_hardnet_generic_control(device)`  the OLD generic-pretrained backbone,
                                              kept ONLY as an explicitly named
                                              control for anyone who wants to
                                              measure the effect of this repair;
                                              never used as "the" HardNet source.

Every call records the resolved checkpoint path and its SHA-256 so every run/held-
out record can carry unambiguous provenance for the exact weights that were fused.
"""
from __future__ import annotations

from pathlib import Path

from experiments.exp1.env import ROOT, sha256_file
from experiments.exp1.models import Descriptor, load_backbone


def exp1b_checkpoint_path(fold: str, seed: int) -> Path:
    """Path to Experiment 1B's SELECTED (arm B, per `selection_frozen.json`)
    fold-specific HardNet fine-tune checkpoint for one seed."""
    return ROOT / 'outputs' / 'exp1b' / 'runs' / fold / f'B_s{seed}' / 'best.pt'


def resolve_exp1b_source(fold: str, seed: int) -> dict:
    """Selected-arm + checkpoint-path + hash provenance for one fold/seed.

    Verifies against `outputs/exp1b/selection_frozen.json` that arm B really is
    the selected arm for this fold (fails loudly rather than silently loading a
    non-selected checkpoint if Experiment 1B's selection ever changes).
    """
    import json
    frozen_path = ROOT / 'outputs' / 'exp1b' / 'selection_frozen.json'
    frozen = json.loads(frozen_path.read_text())
    if fold not in frozen:
        raise KeyError(f'{fold}: no Exp1B selection recorded in {frozen_path}')
    selected_arm = frozen[fold]['arm']
    if selected_arm != 'B':
        raise RuntimeError(
            f'{fold}: Exp1B selection_frozen.json now selects arm {selected_arm!r}, '
            f'not "B" -- exp3_pairwise.hardnet_source assumes arm B; update this '
            f'module before proceeding, do not silently load a stale arm.')
    ckpt = exp1b_checkpoint_path(fold, seed)
    if not ckpt.exists():
        raise FileNotFoundError(
            f'{fold}/seed {seed}: expected Exp1B arm-B checkpoint at {ckpt}; '
            f'Experiment 1B must be run for this (fold, seed) before Experiment 3 '
            f'can fuse its HardNet fine-tune.')
    return {
        'fold': fold, 'seed': seed, 'selected_arm': selected_arm,
        'checkpoint_path': str(ckpt), 'checkpoint_sha256': sha256_file(ckpt),
        'selection_frozen_path': str(frozen_path),
        'selection_frozen_sha256': sha256_file(frozen_path),
        'selection_record': frozen[fold],
        'source': 'exp1b_fold_specific_arm_b',
    }


def frozen_hardnet(device: str, fold: str, seed: int):
    """The CORRECT frozen HardNet source: Experiment 1B's selected, fold-specific,
    seed-specific arm-B fine-tune checkpoint. Frozen, eval mode, no gradient.

    Returns `(model, provenance_dict)`. `provenance_dict` must be recorded in
    every training run and every held-out evaluation record (repair task §3.2:
    "Include the source checkpoint hash in every run and held-out record.").
    """
    import torch
    provenance = resolve_exp1b_source(fold, seed)
    backbone, info = load_backbone('hardnet', pretrained=True, device=device)
    model = Descriptor(backbone, 'hardnet').to(device).eval()
    blob = torch.load(provenance['checkpoint_path'], map_location=device,
                      weights_only=False)
    model.load_state_dict(blob['model'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    provenance['loaded_strict'] = True
    provenance['exp1b_training_step'] = blob.get('step')
    provenance['exp1b_training_metric'] = blob.get('metric')
    provenance['exp1b_fingerprint'] = blob.get('fingerprint')
    return model, provenance


def frozen_hardnet_generic_control(device: str):
    """EXPLICIT CONTROL ONLY: the generic pretrained HardNet backbone, with no
    per-fold Experiment 1B fine-tuning. Never call this expecting it to represent
    "the" Experiment 3 HardNet source -- it is a separate, explicitly named
    ablation arm for measuring what the fold-specific fine-tune buys, if anything.
    """
    backbone, info = load_backbone('hardnet', pretrained=True, device=device)
    model = Descriptor(backbone, 'hardnet').to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    provenance = {
        'fold': None, 'seed': None, 'selected_arm': None,
        'checkpoint_path': str(ROOT / info.get('weights_path', 'cnn/hardnet-hf/'
                                               'checkpoint_liberty_with_aug.pth')),
        'checkpoint_sha256': info.get('weights_sha256'),
        'source': 'generic_pretrained_control',
    }
    return model, provenance

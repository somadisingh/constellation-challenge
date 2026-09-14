"""Training loop for one arm of the fixed matrix (task §10, §11).

Every arm shares: the SAME stream of query groups (retrieval, alignment, labels),
the SAME optimizer step budget, the SAME evaluation checkpoints, and the SAME
seeds. Only the pieces named in `ARM_SPEC` differ (task §10): whether the frozen
HardNet fusion stream is present, the classification objective, which negative
policy feeds the classification loss, and whether the offset head is trained.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import ARM_SPEC, MAX_OFFSET, OUT
from .batch import collate, usable_present_mask
from .forward import score_batch
from .losses import binary_bce_loss, listwise_absent_loss, offset_loss
from .model import PairwiseVerifier
from .negatives import select_hard, select_random, hardnet_similarity
from experiments.exp1.env import (code_hash, derive_seed, peak_memory, mps_sync,
                                  seed_torch, sha256_json, write_json)

GRAD_CLIP = 5.0
LR = 1e-3
WEIGHT_DECAY = 1e-4
OFFSET_WEIGHT = 0.1


def _trainable_state_dict(model) -> dict:
    """State dict EXCLUDING the frozen HardNet submodule.

    The frozen backbone must never be part of an Exp3 checkpoint: saving it would
    let a later `load_state_dict` on a freshly loaded (and therefore correctly
    un-drifted) `hardnet_model` instance silently overwrite it with whatever that
    checkpoint captured, defeating the frozen-backbone guarantee on resume.
    """
    return {k: v for k, v in model.state_dict().items() if not k.startswith('hardnet.')}


def _load_trainable_state_dict(model, state: dict) -> None:
    """Load a trainable-only checkpoint; the only expected missing keys are HardNet's."""
    result = model.load_state_dict(state, strict=False)
    unexpected_missing = [k for k in result.missing_keys if not k.startswith('hardnet.')]
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(f'checkpoint mismatch: missing={unexpected_missing} '
                           f'unexpected={result.unexpected_keys}')


def build_model(arm: str, hardnet_backbone=None) -> PairwiseVerifier:
    spec = ARM_SPEC[arm]
    return PairwiseVerifier(hardnet_fusion=spec['hardnet_fusion'],
                            offset=spec['offset'], hardnet_backbone=hardnet_backbone)


def _resample_group_negatives(group, spec: dict, seed: int, refresh_generation: int,
                              network_scores=None, hardnet_sim=None):
    """Restrict `group`'s stored candidates to positives + selected negatives.

    Every arm's classification loss should see the same SIZE of candidate set
    regardless of policy, so hard and random controls are directly comparable: the
    ignored-band and unselected-negative candidates are masked out of `valid`,
    while every positive candidate stays.

    `seed`/`refresh_generation` make the negative choice deterministic and
    reproducible across processes and resumed runs (task §5.1 repair): they are
    threaded straight through to `negatives.select_hard`/`select_random`, which
    derive an RNG via `derive_seed` rather than Python's process-salted `hash()`.
    """
    if spec['negatives'] == 'hard':
        sel = select_hard(group, seed=seed, refresh_generation=refresh_generation,
                          network_scores=network_scores, hardnet_sim=hardnet_sim)
    else:
        sel = select_random(group, seed=seed, refresh_generation=refresh_generation)
    keep = set(sel['indices'].tolist()) | set(group.positive_indices().tolist())
    mask = np.zeros(len(group), bool)
    mask[list(keep)] = True
    return mask, sel


def build_training_batch(groups: list, spec: dict, seed: int = 0,
                         refresh_generation: int = 0, hardnet_model=None,
                         device: str = 'cpu', network_model=None) -> tuple:
    """Collate `groups`, restricting each group's valid candidates per `spec`.

    `seed`/`refresh_generation`: see `_resample_group_negatives`. A caller that
    omits `seed` gets `seed=0`, which is still fully deterministic (just not tied
    to a particular experiment run) -- every production call site in this package
    passes the real run seed explicitly.
    """
    kmax = max((len(g) for g in groups), default=1) or 1
    restricted_valid = []
    diagnostics = []
    for g in groups:
        hardnet_sim = None
        if spec['negatives'] == 'hard' and hardnet_model is not None and network_model is None:
            hardnet_sim = hardnet_similarity(hardnet_model, device, g.query, g.crops)
        network_scores = None
        if spec['negatives'] == 'hard' and network_model is not None:
            network_scores = network_model(g)
        mask, sel = _resample_group_negatives(g, spec, seed, refresh_generation,
                                              network_scores, hardnet_sim)
        restricted_valid.append(mask & g.admissible)
        diagnostics.append(sel)
    batch = collate(groups, kmax=kmax)
    restricted = np.zeros((len(groups), kmax), bool)
    for i, m in enumerate(restricted_valid):
        restricted[i, :len(m)] = m
    batch['valid'] = torch.from_numpy(restricted)
    batch['negative_selection'] = diagnostics
    return batch, diagnostics


def compute_loss(model: PairwiseVerifier, batch: dict, spec: dict, device: str) -> dict:
    out = score_batch(model, batch, device)
    valid = batch['valid'].to(device)
    positive = (batch['positive'] & batch['valid']).to(device)
    is_present = batch['is_present'].to(device)
    usable = usable_present_mask(batch).to(device)

    if spec['objective'] == 'binary_bce':
        pos_logits = out['pair_logits'][positive]
        neg_mask = valid & (~batch['positive'].to(device))
        neg_logits = out['pair_logits'][neg_mask]
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            cls_loss = torch.zeros((), device=device)
        else:
            cls_loss = binary_bce_loss(pos_logits, neg_logits)
        per_row = None
    else:
        per_row = listwise_absent_loss(out['pair_logits'], valid, positive,
                                       out['absent_logit'], is_present)
        cls_loss = per_row[usable].mean() if usable.any() else per_row.mean() * 0.0

    total = cls_loss
    off_loss = torch.zeros((), device=device)
    if spec['offset'] and 'dx' in out:
        # Offset loss applies only to known positive candidates (task §9).
        true_dx = batch['true_dx'].to(device)
        true_dy = batch['true_dy'].to(device)
        target_valid = positive & torch.isfinite(true_dx) & torch.isfinite(true_dy)
        if target_valid.any():
            off_loss = offset_loss(out['dx'][target_valid], out['dy'][target_valid],
                                   true_dx[target_valid], true_dy[target_valid],
                                   weight=OFFSET_WEIGHT)
        total = total + off_loss

    finite = bool(torch.isfinite(total))
    return {'loss': total, 'classification_loss': float(cls_loss.detach()),
           'offset_loss': float(off_loss.detach()), 'finite': finite,
           'n_usable_present': int(usable.sum()), 'n_rows': int(usable.numel()),
           'pair_logits': out['pair_logits'].detach(),
           'absent_logit': out['absent_logit'].detach()}


def train_arm(fold: str, arm: str, groups_by_step, hardnet_model, device: str,
             seed: int, steps: int, eval_every: int, eval_fn, say=print,
             checkpoint_dir: Path | None = None, resume: bool = True) -> dict:
    """Train one arm for `steps` optimizer steps, evaluating every `eval_every`."""
    spec = ARM_SPEC[arm]
    seed_torch(derive_seed(seed, fold, arm, 'exp3'))
    model = build_model(arm, hardnet_backbone=hardnet_model if spec['hardnet_fusion']
                        else None).to(device)
    model.train()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)

    frozen_state = None
    if hardnet_model is not None:
        frozen_state = {k: v.detach().clone() for k, v in hardnet_model.state_dict().items()}

    history, evaluations = [], []
    best = {'metric': -np.inf, 'selection_metric': -np.inf, 'step': None,
           'presence': None, 'localization': None, 'top1_correct': None,
           'calibrator': None, 'threshold': None, 'per_scene': None}
    start_step = 0
    ckpt_path = (checkpoint_dir / 'last.pt') if checkpoint_dir else None
    if resume and ckpt_path and ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        _load_trainable_state_dict(model, blob['model'])
        opt.load_state_dict(blob['optimizer'])
        start_step = blob['step']
        history, evaluations, best = blob['history'], blob['evaluations'], blob['best']
        say(f'  resumed {fold}/{arm} at step {start_step}')

    started = time.perf_counter()
    nan_steps = 0
    negative_digests = []
    for step in range(start_step, steps):
        groups = groups_by_step(step)
        batch, diagnostics = build_training_batch(groups, spec, seed=seed,
                                                   refresh_generation=0,
                                                   hardnet_model=hardnet_model,
                                                   device=device)
        if step % 20 == 0 or step + 1 == steps:
            negative_digests.append({'step': step + 1,
                                     'digests': [d.get('digest') for d in diagnostics]})
        result = compute_loss(model, batch, spec, device)
        if not result['finite']:
            nan_steps += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        result['loss'].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),
                                                          GRAD_CLIP))
        opt.step()
        if step % 20 == 0 or step + 1 == steps:
            history.append({'step': step + 1, 'loss': float(result['loss'].detach()),
                            'classification_loss': result['classification_loss'],
                            'offset_loss': result['offset_loss'],
                            'grad_norm': grad_norm,
                            'n_usable_present': result['n_usable_present']})
        due = (step + 1) % eval_every == 0 or (step + 1) == steps
        if due:
            metric_doc = eval_fn(model, hardnet_model, device)
            model.train()
            metric = metric_doc['selection_metric']
            evaluations.append({'step': step + 1, 'metric': metric, **metric_doc})
            improved = metric > best['metric'] + 1e-9
            if improved:
                # Repair task §3.4: `best` is now ONE atomic record built entirely
                # from THIS SAME `metric_doc` -- step, selection_metric, presence,
                # localization, top1_correct, calibrator and threshold can never
                # be pulled from a later, unrelated evaluation, because they are
                # all read out of the identical dict that triggered `improved`.
                best = {'step': step + 1, 'selection_metric': metric,
                       'metric': metric,   # back-compat alias, same value
                       'presence': metric_doc.get('presence'),
                       'localization': metric_doc.get('localization'),
                       'top1_correct': metric_doc.get('top1_correct'),
                       'calibrator': metric_doc.get('calibrator'),
                       'threshold': metric_doc.get('threshold'),
                       'per_scene': metric_doc.get('per_scene')}
                if checkpoint_dir:
                    torch.save({'model': _trainable_state_dict(model), 'step': step + 1,
                               'metric': metric, 'best': best, 'arm': arm, 'fold': fold,
                               'seed': seed, 'code_hash': code_hash()},
                              checkpoint_dir / 'best.pt')
            say(f'    {fold}/{arm} step {step + 1:4d} loss={float(result["loss"].detach()):.4f} '
               f'presence={metric_doc.get("presence", float("nan")):.4f} '
               f'localization={metric_doc.get("localization", float("nan")):.4f}'
               f'{" *" if improved else ""}')
            if checkpoint_dir:
                torch.save({'model': _trainable_state_dict(model),
                           'optimizer': opt.state_dict(),
                           'step': step + 1, 'history': history,
                           'evaluations': evaluations, 'best': best},
                          checkpoint_dir / 'last.pt')

    if frozen_state is not None:
        drift = max(float((frozen_state[k] - v).abs().max())
                   for k, v in hardnet_model.state_dict().items())
        assert drift == 0.0, f'frozen HardNet drifted by {drift}'

    elapsed = time.perf_counter() - started
    mps_sync(device)
    negative_digest_summary = sha256_json(negative_digests) if negative_digests else None
    return {
        'fold': fold, 'arm': arm, 'seed': seed, 'steps': steps,
        'best': best, 'history': history, 'evaluations': evaluations,
        'nan_steps': nan_steps, 'seconds': elapsed,
        'steps_per_second': steps / max(elapsed, 1e-9),
        'memory': peak_memory(),
        'hardnet_unchanged': frozen_state is not None,
        'negative_selection_digest': negative_digest_summary,
        'negative_digests_sample': negative_digests[:3] + negative_digests[-3:]
        if len(negative_digests) > 6 else negative_digests,
    }

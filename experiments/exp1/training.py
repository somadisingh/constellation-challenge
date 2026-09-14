"""Descriptor training, screening and continuation (plan §9, §10).

Loss: symmetric hard-negative triplet `relu(margin + d_pos - d_neg)` on
L2-normalised descriptors, with an explicit distance epsilon. Negatives are mined
in BOTH row and column directions of the in-batch distance matrix, plus the cached
hard negatives. The diagonal and every same-or-ambiguous physical-source pair are
excluded. An anchor with no valid negative contributes no loss and is counted.

Both streams go through the shared encoder in ONE concatenated forward pass, so
batch-dependent layers see the pair jointly rather than updating their statistics
separately.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from . import SCENES
from .env import (SEED_MODEL, append_jsonl, code_hash, derive_seed, peak_memory,
                  read_json, resolve_device, seed_torch, sha256_json, write_json,
                  mps_sync)
from .inner import (build_inner_aligned, inner_metrics, load_inner_aligned,
                    save_inner_aligned, score_inner)
from .mining import load_mined
from .models import (build_for_training, descriptor_distance, pairwise_distances,
                     restore_train_mode)
from .pairs import build_training_tensors, identity_mask
from .splits import fold_skies

ARMS = {'hardnet': 'T-H', 'hynet': 'T-Y', 'sosnet': 'T-S'}
PRETRAINED_LABEL = {'hardnet': 'P-H', 'hynet': 'P-Y', 'sosnet': 'P-S'}


# --- batch assembly ---------------------------------------------------------------
class TripletSampler:
    """Draws anchors across the two allowed skies and assembles one step's batch."""

    def __init__(self, tensors: dict, config: dict, seed: int,
                 hard: bool = True):
        self.tensors = tensors                      # {scene: tensor dict}
        self.scenes = sorted(tensors)
        self.config = config
        self.hard = hard
        self.rng = np.random.default_rng(seed)
        self.ignore_radius = float(config['mining']['ignore_radius'])
        mix = config['mining']['mix']
        self.mix = (float(mix['classical_hard']), float(mix['network_hard']),
                    float(mix['random']))
        self.network_rank: dict = {}                # scene -> (n, k) network scores
        self.state = {'draws': 0}
        self._pools = {s: np.where(self.tensors[s]['positive_ok'])[0]
                       for s in self.scenes}
        for s, pool in self._pools.items():
            if not len(pool):
                raise RuntimeError(f'{s}: no usable positive after alignment')

    def set_network_scores(self, scene: str, scores: np.ndarray) -> None:
        self.network_rank[scene] = scores

    def sample(self, anchors: int, warmup: bool):
        """Balanced anchors across skies, with their positives and cached negatives."""
        per = [anchors // len(self.scenes)] * len(self.scenes)
        for i in range(anchors - sum(per)):
            per[i] += 1
        idx = {}
        for scene, count in zip(self.scenes, per):
            pool = self._pools[scene]
            idx[scene] = pool[self.rng.integers(0, len(pool), size=count)]
        self.state['draws'] += anchors

        q, p, centres, scene_tags, negs = [], [], [], [], []
        for scene in self.scenes:
            t = self.tensors[scene]
            for i in idx[scene]:
                q.append(t['queries'][i])
                p.append(t['positive_warmup'][i] if warmup else t['positive'][i])
                centres.append(t['centres'][i])
                scene_tags.append(scene)
                negs.append(self._pick_negatives(scene, int(i), warmup))
        return {
            'query': np.stack(q),
            'positive': np.stack(p),
            'centres': np.stack(centres),
            'scenes': scene_tags,
            'cached_negatives': negs,
            'indices': {s: idx[s].tolist() for s in self.scenes},
        }

    def _pick_negatives(self, scene: str, i: int, warmup: bool):
        t = self.tensors[scene]
        ok = np.where(t['negative_ok'][i])[0]
        if not len(ok):
            return np.zeros((0, 32, 32), np.float32)
        if warmup or not self.hard:
            # Warmup and the random-negative control both draw uniformly.
            take = min(2, len(ok))
            pick = self.rng.choice(ok, size=take, replace=False)
            return t['negatives'][i][pick]

        classical = t['negative_classical'][i]
        kinds = t['negative_kind'][i]
        cls_pool = [j for j in ok if kinds[j] == 0 and np.isfinite(classical[j])]
        rnd_pool = [j for j in ok if kinds[j] == 1] or list(ok)
        cls_pool.sort(key=lambda j: -classical[j])

        want_total = 4
        n_cls = max(1, int(round(self.mix[0] * want_total)))
        n_net = int(round(self.mix[1] * want_total))
        n_rnd = max(0, want_total - n_cls - n_net)

        chosen = list(cls_pool[:n_cls])
        net = self.network_rank.get(scene)
        if n_net and net is not None:
            remaining = [j for j in ok if j not in chosen]
            remaining.sort(key=lambda j: -net[i, j])
            chosen += remaining[:n_net]
        elif n_net:
            # Before a network exists, its quarter is replaced by classical hardness.
            chosen += [j for j in cls_pool[n_cls:n_cls + n_net]]
        if n_rnd:
            pool = [j for j in rnd_pool if j not in chosen]
            if pool:
                chosen += list(self.rng.choice(
                    pool, size=min(n_rnd, len(pool)), replace=False))
        chosen = chosen[:want_total] or [int(ok[0])]
        return t['negatives'][i][np.array(chosen, int)]


# --- loss -------------------------------------------------------------------------
def triplet_loss(za: torch.Tensor, zp: torch.Tensor, invalid: torch.Tensor,
                 cached: list, margin: float, eps: float):
    """Symmetric hard-negative triplet with identity masking.

    `invalid[i, j]` is True where `j` may not serve as a negative for `i`.
    `cached[i]` holds already-encoded cached hard negatives for anchor `i`.
    """
    n = za.shape[0]
    d = pairwise_distances(za, zp, eps)
    d_pos = torch.diagonal(d)

    big = torch.full_like(d, 1e4)
    blocked = invalid | torch.eye(n, dtype=torch.bool, device=d.device)
    row = torch.where(blocked, big, d)                      # anchor i vs positive j
    col = torch.where(blocked, big, d.t())                  # anchor j vs positive i
    row_min, _ = row.min(dim=1)
    col_min, _ = col.min(dim=1)
    in_batch = torch.minimum(row_min, col_min)
    has_in_batch = (~blocked).any(dim=1)

    cached_min = torch.full_like(d_pos, 1e4)
    has_cached = torch.zeros(n, dtype=torch.bool, device=d.device)
    for i, zc in enumerate(cached):
        if zc is None or zc.shape[0] == 0:
            continue
        dist = torch.sqrt(torch.clamp(((zc - za[i:i + 1]) ** 2).sum(1), min=eps))
        cached_min[i] = dist.min()
        has_cached[i] = True

    d_neg = torch.minimum(in_batch, cached_min)
    usable = has_in_batch | has_cached
    raw = torch.relu(margin + d_pos - d_neg)
    loss = raw[usable].mean() if bool(usable.any()) else d_pos.sum() * 0.0
    with torch.no_grad():
        stats = {
            'd_pos': float(d_pos.mean()),
            'd_neg': float(d_neg[usable].mean()) if bool(usable.any()) else float('nan'),
            'active_fraction': float((raw[usable] > 0).float().mean())
            if bool(usable.any()) else 0.0,
            'anchors_without_negative': int((~usable).sum()),
            'n_anchors': n,
            'descriptor_variance': float(za.var(dim=0).mean()),
            'descriptor_norm_min': float(za.norm(dim=1).min()),
            'cached_used': int(has_cached.sum()),
        }
    return loss, stats


def lr_at(step: int, total: int, base: float, warmup: int, floor: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * progress))


# --- run --------------------------------------------------------------------------
def _load_tensors(paths, scenes, config, data, say) -> dict:
    out = {}
    for scene in scenes:
        cache = Path(paths.root) / 'tensors' / f'{scene}.npz'
        if cache.exists():
            with np.load(cache, allow_pickle=True) as blob:
                out[scene] = {k: blob[k] for k in blob.files}
            out[scene]['source_ids'] = list(out[scene]['source_ids'])
            out[scene]['stats'] = read_json(cache.with_suffix('.stats.json'))
            say(f'  tensors {scene:9s} cached')
        else:
            say(f'  building tensors for {scene} ...')
            mined = load_mined(paths, scene)
            t = build_training_tensors(scene, mined, config, data, progress=say)
            cache.parent.mkdir(parents=True, exist_ok=True)
            stats = t.pop('stats')
            scene_name = t.pop('scene')
            savable = {k: v for k, v in t.items() if k != 'source_ids'}
            savable['source_ids'] = np.array(t['source_ids'], dtype=object)
            np.savez_compressed(cache, **savable)
            write_json(cache.with_suffix('.stats.json'), stats)
            t['stats'] = stats
            t['scene'] = scene_name
            out[scene] = t
            say(f'  tensors {scene:9s} built: {stats}')
    return out


def _ensure_inner_aligned(paths, fold, config, data, say):
    target = Path(paths.root) / 'inner' / f'{fold}_aligned.npz'
    if target.exists():
        return load_inner_aligned(paths, fold)
    from .mining import load_inner
    say(f'  aligning inner bank for fold {fold} ...')
    doc = build_inner_aligned(fold, load_inner(paths, fold), config, data,
                              progress=say)
    save_inner_aligned(paths, fold, doc)
    return doc


def train_arm(paths, config, fold: str, model_name: str, *, device: str,
              seed: int, total_steps: int, pretrained: bool = True,
              hard: bool = True, tag: str, data=None, say=print,
              tensors=None, inner_aligned=None, resume_from=None) -> dict:
    """Train one arm and evaluate it on the fold's inner bank at intervals."""
    tcfg = config['train']
    held_out, allowed = fold_skies(fold)
    tensors = tensors or _load_tensors(paths, allowed, config, data, say)
    inner_aligned = inner_aligned or _ensure_inner_aligned(paths, fold, config,
                                                           data, say)
    freeze_bn = bool(tcfg['freeze_bn_pretrained']) and pretrained
    seed_torch(derive_seed(seed, fold, model_name, tag))
    model, info = build_for_training(model_name, pretrained=pretrained,
                                     device=device, freeze_bn=freeze_bn)
    base_lr = float(tcfg['lr_pretrained'] if pretrained else tcfg['lr_scratch'])
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                            weight_decay=float(tcfg['weight_decay']))
    sampler = TripletSampler(tensors, config,
                             derive_seed(seed, fold, model_name, tag, 'sampler'),
                             hard=hard)

    anchors = int(tcfg['anchors_per_step'])
    warmup_steps = int(tcfg['warmup_random_steps']) if hard else 0
    margin = float(tcfg['margin'])
    eps = float(tcfg['distance_eps'])
    ckpt_dir = Path(paths.checkpoints(fold)) / f'{model_name}_{tag}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    best = {'metric': -np.inf, 'step': None}
    history, evaluations = [], []
    if resume_from and Path(resume_from).exists():
        blob = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(blob['model'])
        opt.load_state_dict(blob['optimizer'])
        start_step = int(blob['step'])
        best = blob.get('best', best)
        history = blob.get('history', [])
        evaluations = blob.get('evaluations', [])
        say(f'  resumed {model_name}/{tag} from step {start_step}')

    nan_steps = 0
    stalled = 0
    stopped_early = None
    patience = int(tcfg['early_stop_patience'])
    min_steps = int(tcfg['min_steps'])
    started = time.perf_counter()
    for step in range(start_step, total_steps):
        warm = step < warmup_steps
        batch = sampler.sample(anchors, warm)
        lr = lr_at(step, total_steps, base_lr, int(tcfg['lr_warmup_steps']),
                   float(tcfg['lr_min']))
        for group in opt.param_groups:
            group['lr'] = lr

        q = torch.from_numpy(batch['query'])[:, None].to(device)
        p = torch.from_numpy(batch['positive'])[:, None].to(device)
        # ONE concatenated forward pass for the anchor stream, the positive stream
        # and every cached negative, so batch-dependent layers see the whole step
        # jointly rather than updating their statistics separately (plan §9). This
        # is also what makes the step fast: encoding cached negatives per anchor
        # meant 64 tiny forward passes per step.
        neg_arrays = [a for a in batch['cached_negatives'] if a.shape[0]]
        neg_counts = [a.shape[0] for a in batch['cached_negatives']]
        if neg_arrays:
            neg_stack = torch.from_numpy(np.concatenate(neg_arrays))[:, None].to(device)
            z = model(torch.cat([q, p, neg_stack], dim=0))
        else:
            z = model(torch.cat([q, p], dim=0))
        za, zp = z[:anchors], z[anchors:2 * anchors]
        zneg_flat = z[2 * anchors:]

        cached, cursor = [], 0
        for count in neg_counts:
            if count == 0:
                cached.append(None)
            else:
                cached.append(zneg_flat[cursor:cursor + count])
                cursor += count

        invalid_np = identity_mask(batch['centres'], batch['scenes'],
                                  batch['centres'], batch['scenes'],
                                  sampler.ignore_radius)
        invalid = torch.from_numpy(invalid_np).to(device)
        loss, stats = triplet_loss(za, zp, invalid, cached, margin, eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(tcfg['grad_clip'])))
        opt.step()

        value = float(loss.detach())
        if not np.isfinite(value):
            nan_steps += 1
        if step % 50 == 0 or step == total_steps - 1:
            history.append({'step': step, 'loss': value, 'lr': lr,
                            'grad_norm': grad_norm, 'warmup': warm, **stats})

        due = ((step + 1) % int(tcfg['eval_every']) == 0) or (step + 1 == total_steps)
        if due:
            metrics = evaluate_checkpoint(model, inner_aligned, device)
            restore_train_mode(model, freeze_bn)
            metric = metrics['equal_sky_top1_localization_reward']
            evaluations.append({'step': step + 1, 'metric': metric,
                                'metrics': metrics})
            improved = metric > best['metric'] + 1e-9
            if improved:
                best = {'metric': metric, 'step': step + 1}
                torch.save({'model': model.state_dict(), 'step': step + 1,
                            'metric': metric, 'model_name': model_name,
                            'tag': tag, 'fold': fold, 'seed': seed,
                            'pretrained': pretrained, 'hard': hard,
                            'code_hash': code_hash(),
                            'config_sha256': sha256_json(config)},
                           ckpt_dir / 'best.pt')
            say(f'    {model_name}/{tag} step {step + 1:5d} loss={value:.4f} '
                f'active={stats["active_fraction"]:.3f} '
                f'inner_reward={metric:.4f}{" *" if improved else ""}')
            # Plan §10.4: after the screening budget, stop when three consecutive
            # inner checks fail to improve, with a minimum total of `min_steps`.
            if total_steps > int(tcfg['screen_steps']) and \
                    (step + 1) > int(tcfg['screen_steps']):
                stalled = 0 if improved else stalled + 1
                if stalled >= patience and (step + 1) >= min_steps:
                    stopped_early = {'step': step + 1, 'checks_without_improvement':
                                     stalled, 'reason':
                                     'three inner checks without improvement'}
                    say(f'    early stop at step {step + 1} '
                        f'({stalled} checks without improvement)')
                    break
        if (step + 1) % int(tcfg['checkpoint_every']) == 0:
            _save_resume(ckpt_dir / 'last.pt', model, opt, step + 1, best,
                         history, evaluations, sampler)
        if (step + 1) == warmup_steps or (step + 1) == int(config['mining']['refresh_step']):
            _refresh_network_negatives(sampler, model, device, say)

    mps_sync(device)
    completed = (stopped_early['step'] if stopped_early else total_steps)
    _save_resume(ckpt_dir / 'last.pt', model, opt, completed, best, history,
                 evaluations, sampler)
    elapsed = time.perf_counter() - started
    active_tail = [h['active_fraction'] for h in history[-10:]]
    result = {
        'arm': ARMS.get(model_name, model_name), 'model': model_name, 'tag': tag,
        'fold': fold, 'held_out': held_out, 'allowed': list(allowed),
        'seed': seed, 'pretrained': pretrained, 'hard_negatives': hard,
        'total_steps': completed, 'requested_steps': total_steps,
        'stopped_early': stopped_early,
        'saturated': bool(active_tail) and float(np.mean(active_tail)) < 0.01,
        'mean_active_fraction_tail': float(np.mean(active_tail)) if active_tail else None,
        'anchors_per_step': anchors,
        'unique_sources': {s: int(len(tensors[s]['positive_ok'])) for s in allowed},
        'presentations': completed * anchors,
        'best': best, 'evaluations': evaluations, 'history': history,
        'nan_steps': nan_steps, 'seconds': elapsed,
        'steps_per_second': completed / max(elapsed, 1e-9),
        'checkpoint_dir': str(ckpt_dir.relative_to(paths.root)),
        'model_info': {k: v for k, v in info.items() if k != 'bn_frozen'},
        'n_bn_frozen': len(info.get('bn_frozen', [])),
        'memory': peak_memory(),
    }
    append_jsonl(paths.runs, {'stage': 'train', **{k: result[k] for k in
                 ('arm', 'model', 'tag', 'fold', 'seed', 'total_steps',
                  'nan_steps', 'seconds')},
                 'best_metric': best['metric'], 'best_step': best['step']})
    write_json(Path(paths.checkpoints(fold)) / f'{model_name}_{tag}.json', result)
    return result


def _save_resume(path, model, opt, step, best, history, evaluations, sampler):
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'step': step, 'best': best, 'history': history,
                'evaluations': evaluations,
                'sampler_state': sampler.state,
                'numpy_rng': sampler.rng.bit_generator.state,
                'torch_rng': torch.get_rng_state(),
                'code_hash': code_hash()}, path)


def _refresh_network_negatives(sampler, model, device, say):
    """Score the whole negative pool with the current network (plan §7 refresh)."""
    model.eval()
    with torch.no_grad():
        for scene, t in sampler.tensors.items():
            negs = t['negatives']
            n, k = negs.shape[0], negs.shape[1]
            scores = np.full((n, k), -np.inf, np.float32)
            q = torch.from_numpy(t['queries'])[:, None].to(device)
            zq = model(q)
            flat = negs.reshape(-1, 32, 32)
            zc = []
            for start in range(0, len(flat), 1024):
                chunk = torch.from_numpy(flat[start:start + 1024])[:, None].to(device)
                zc.append(model(chunk))
            zc = torch.cat(zc).reshape(n, k, -1)
            d = torch.sqrt(torch.clamp(
                ((zc - zq[:, None, :]) ** 2).sum(-1), min=1e-12)).cpu().numpy()
            scores = np.where(t['negative_ok'], -d, -np.inf)
            sampler.set_network_scores(scene, scores)
    model.train()
    say('    refreshed network hard negatives')


@torch.no_grad()
def evaluate_checkpoint(model, inner_aligned, device) -> dict:
    scored = score_inner(inner_aligned, 'descriptor', model=model, device=device)
    return inner_metrics(scored)


# --- CLI entry points -------------------------------------------------------------
def run_screen(args, paths, config) -> int:
    from .stages import folds_for, _device, _seed
    device = _device(args, config)
    seed = _seed(args, config)
    data = args.data
    say = (lambda m: None) if args.quiet else (lambda m: print(m, flush=True))
    models = [args.model] if getattr(args, 'model', None) else ['hardnet', 'hynet']
    steps = int(config['train']['screen_steps'])

    out = {}
    for fold in folds_for(args):
        _, allowed = fold_skies(fold)
        tensors = _load_tensors(paths, allowed, config, data, say)
        inner_aligned = _ensure_inner_aligned(paths, fold, config, data, say)
        say(f'fold {fold}: screening {models} for {steps} steps (seed {seed})')
        results = {}
        for name in models:
            results[name] = train_arm(
                paths, config, fold, name, device=device, seed=seed,
                total_steps=steps, pretrained=True, hard=True, tag=f'screen_s{seed}',
                data=data, say=say, tensors=tensors, inner_aligned=inner_aligned)
        out[fold] = {n: {'best': r['best'], 'seconds': r['seconds'],
                         'steps_per_second': r['steps_per_second']}
                     for n, r in results.items()}
        write_json(Path(paths.fold(fold)) / f'screen_s{seed}.json', out[fold])
    write_json(Path(paths.root) / f'screen_s{seed}.json', out)
    return 0


def run_train(args, paths, config) -> int:
    from .stages import folds_for, _device, _seed
    device = _device(args, config)
    seed = _seed(args, config)
    data = args.data
    say = (lambda m: None) if args.quiet else (lambda m: print(m, flush=True))
    steps = int(getattr(args, 'steps', None) or config['train']['max_steps'])
    scratch = bool(getattr(args, 'scratch', False))
    random_neg = bool(getattr(args, 'random_negatives', False))
    tag = ('scratch' if scratch else 'randomneg' if random_neg else 'continue')
    models = [args.model] if getattr(args, 'model', None) else ['hardnet']

    out = {}
    for fold in folds_for(args):
        _, allowed = fold_skies(fold)
        tensors = _load_tensors(paths, allowed, config, data, say)
        inner_aligned = _ensure_inner_aligned(paths, fold, config, data, say)
        results = {}
        for name in models:
            results[name] = train_arm(
                paths, config, fold, name, device=device, seed=seed,
                total_steps=steps, pretrained=not scratch,
                hard=not random_neg, tag=f'{tag}_s{seed}', data=data, say=say,
                tensors=tensors, inner_aligned=inner_aligned)
        out[fold] = {n: r['best'] for n, r in results.items()}
    write_json(Path(paths.root) / f'train_{tag}_s{seed}.json', out)
    return 0


# --- pair head (plan §9) ----------------------------------------------------------
def _encode_frozen(model, arrays, device, microbatch=1024):
    """Descriptors from a FROZEN backbone. No gradient reaches the encoder."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(arrays), microbatch):
            chunk = arrays[start:start + microbatch]
            x = torch.from_numpy(np.ascontiguousarray(chunk)).float()[:, None]
            out.append(model(x.to(device)).cpu())
    return torch.cat(out) if out else torch.zeros(0, 128)


def train_pair_head(paths, config, fold: str, model_name: str, checkpoint: Path, *,
                    device: str, seed: int, data=None, say=print, tensors=None,
                    inner_aligned=None) -> dict:
    """Train the pair head on top of a FROZEN selected backbone (plan §9)."""
    from .models import PairHead
    pcfg = config['pair_head']
    _, allowed = fold_skies(fold)
    tensors = tensors or _load_tensors(paths, allowed, config, data, say)
    inner_aligned = inner_aligned or _ensure_inner_aligned(paths, fold, config,
                                                           data, say)
    model, _ = build_for_training(model_name, pretrained=True, device=device)
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(blob['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    seed_torch(derive_seed(seed, fold, model_name, 'pairhead'))
    head = PairHead(dim=int(pcfg['dim']), dropout=float(pcfg['dropout'])).to(device)
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=float(pcfg['lr']),
                            weight_decay=float(pcfg['weight_decay']))
    lossfn = torch.nn.BCEWithLogitsLoss()

    say(f'  building pair dataset for {model_name} (frozen backbone) ...')
    data_pairs = _pair_arrays(tensors, model, device, config,
                              derive_seed(seed, fold, 'pairdata'))
    zq = data_pairs['zq'].to(device)
    zp = data_pairs['zpos'].to(device)
    zn_hard = data_pairs['zn_hard'].to(device)
    zn_easy = data_pairs['zn_easy'].to(device)
    n = zq.shape[0]
    rng = np.random.default_rng(derive_seed(seed, fold, 'pairbatch'))

    batch = 128
    history, evaluations = [], []
    best = {'metric': -np.inf, 'step': None}
    started = time.perf_counter()
    for step in range(int(pcfg['max_steps'])):
        sel = rng.integers(0, n, size=batch // 2)
        anchor = zq[sel]
        pos = zp[sel]
        use_hard = rng.random(batch // 2) < float(pcfg['hard_fraction'])
        neg = torch.where(torch.from_numpy(use_hard).to(device)[:, None],
                          zn_hard[sel], zn_easy[sel])
        logits = torch.cat([head(anchor, pos), head(anchor, neg)])
        target = torch.cat([torch.ones(batch // 2, device=device),
                            torch.zeros(batch // 2, device=device)])
        loss = lossfn(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            history.append({'step': step, 'loss': float(loss.detach())})
        if (step + 1) % int(pcfg['eval_every']) == 0 or step + 1 == int(pcfg['max_steps']):
            scored = score_inner(inner_aligned, 'pair_head', model=model, head=head,
                                 device=device)
            metrics = inner_metrics(scored)
            head.train()
            metric = metrics['equal_sky_top1_localization_reward']
            evaluations.append({'step': step + 1, 'metric': metric})
            if metric > best['metric'] + 1e-9:
                best = {'metric': metric, 'step': step + 1}
                torch.save({'head': head.state_dict(), 'step': step + 1,
                            'metric': metric, 'backbone': model_name,
                            'backbone_checkpoint': str(checkpoint),
                            'fold': fold, 'seed': seed, 'code_hash': code_hash()},
                           Path(checkpoint).parent / 'pair_head.pt')
            say(f'    pairhead step {step + 1:5d} loss={float(loss.detach()):.4f} '
                f'inner_reward={metric:.4f}')

    # Compare against descriptor distance on the same inner bank.
    desc = inner_metrics(score_inner(inner_aligned, 'descriptor', model=model,
                                     device=device))
    result = {
        'fold': fold, 'backbone': model_name, 'seed': seed,
        'checkpoint': str(checkpoint),
        'best': best, 'evaluations': evaluations, 'history': history,
        'descriptor_baseline': desc['equal_sky_top1_localization_reward'],
        'pair_head_best': best['metric'],
        'pair_head_beats_distance': bool(
            best['metric'] > desc['equal_sky_top1_localization_reward'] + 1e-9),
        'decision': ('adopt pair head' if best['metric'] >
                     desc['equal_sky_top1_localization_reward'] + 1e-9
                     else 'keep descriptor distance'),
        'seconds': time.perf_counter() - started,
        'note': ('pair scores from balanced training are not calibrated presence '
                 'probabilities; calibration is a separate stage'),
    }
    write_json(Path(paths.checkpoints(fold)) / f'{model_name}_pairhead_s{seed}.json',
               result)
    append_jsonl(paths.runs, {'stage': 'pair_head', **{k: result[k] for k in
                 ('fold', 'backbone', 'seed', 'pair_head_best',
                  'descriptor_baseline', 'decision')}})
    return result


def _pair_arrays(tensors, model, device, config, seed):
    """One hard and one easy negative descriptor per usable anchor."""
    rng = np.random.default_rng(seed)
    hard_fraction = float(config['pair_head']['hard_fraction'])
    zq, zp, zh, ze = [], [], [], []
    for scene in sorted(tensors):
        t = tensors[scene]
        idx = np.where(t['positive_ok'])[0]
        keep, hard_crops, easy_crops = [], [], []
        for i in idx:
            valid = np.where(t['negative_ok'][i])[0]
            if not len(valid):
                continue
            cls = t['negative_classical'][i]
            order = sorted(valid, key=lambda j: -(cls[j] if np.isfinite(cls[j])
                                                  else -np.inf))
            n_hard = max(1, int(round(hard_fraction * len(order))))
            hard_crops.append(t['negatives'][i][order[0]])
            rest = order[n_hard:] or order
            easy_crops.append(t['negatives'][i][int(rng.choice(rest))])
            keep.append(i)
        keep = np.array(keep, int)
        zq.append(_encode_frozen(model, t['queries'][keep], device))
        zp.append(_encode_frozen(model, t['positive'][keep], device))
        zh.append(_encode_frozen(model, np.stack(hard_crops), device))
        ze.append(_encode_frozen(model, np.stack(easy_crops), device))
    return {'zq': torch.cat(zq), 'zpos': torch.cat(zp),
            'zn_hard': torch.cat(zh), 'zn_easy': torch.cat(ze)}


# --- oracle-pose diagnostic (plan §6, §8; diagnostic only) -------------------------
def oracle_pose_diagnostic(paths, config, fold: str, model, device: str,
                           data=None) -> dict:
    """Would perfect pose estimation change the ranking?

    For each present inner query whose bank holds a correct candidate, the correct
    candidate is re-aligned with the KNOWN synthetic pose and rescored. This bounds
    what pose estimation costs. It is never used for selection or reporting of the
    headline result.
    """
    from .inner import _aligned_set
    from .mining import load_inner
    from .pose import SceneReps, aligned_candidate, masked_ncc, prepare_query
    from .scoring import score_descriptor
    from .data import load_scene

    inner_doc = load_inner(paths, fold)
    aligned_doc = _ensure_inner_aligned(paths, fold, config, data, lambda m: None)
    aa = float(config['pose']['aa_factor'])
    rows = []
    for scene, node in aligned_doc['skies'].items():
        reps = SceneReps(load_scene(scene, data).image)
        queries = inner_doc['skies'][scene]['queries']
        images = inner_doc['skies'][scene]['images']
        for i, q in enumerate(queries):
            if q['kind'] != 'present' or q['centre'] is None:
                continue
            aligned = _aligned_set(node, i)
            if not len(aligned):
                continue
            truth = np.array(q['centre'], float)
            d = np.linalg.norm(aligned.xy - truth, axis=1)
            j = int(d.argmin())
            if d[j] > 12.0:
                continue
            est = score_descriptor(model, aligned, device)
            qraw, qblur = prepare_query(images[q['image_slot']])
            crop, mask = aligned_candidate(reps.raw, aligned.xy[j],
                                           (q['pose']['angle'], q['pose']['scale']), aa)
            if not mask.all():
                continue
            swapped = aligned.crops.copy()
            swapped[j] = crop / 255.0 if crop.max() > 1.5 else crop
            oracle_set = type(aligned)(
                query_id=aligned.query_id, xy=aligned.xy, crops=swapped,
                ncc=aligned.ncc, poses=aligned.poses,
                admissible=aligned.admissible, low_info=aligned.low_info,
                pose_trials=1, rejected_poses=0, query_raw=aligned.query_raw)
            orc = score_descriptor(model, oracle_set, device)
            rows.append({
                'scene': scene, 'index': i,
                'estimated_top1_correct': bool(int(np.argmax(est)) == j),
                'oracle_top1_correct': bool(int(np.argmax(orc)) == j),
                'estimated_score': float(est[j]), 'oracle_score': float(orc[j]),
                'estimated_pose': aligned.poses[j].tolist(),
                'oracle_pose': [q['pose']['angle'], q['pose']['scale']],
            })
    if not rows:
        return {'ok': False, 'reason': 'no present query with a correct candidate'}
    return {
        'ok': True, 'n': len(rows),
        'estimated_top1': float(np.mean([r['estimated_top1_correct'] for r in rows])),
        'oracle_top1': float(np.mean([r['oracle_top1_correct'] for r in rows])),
        'mean_score_gain': float(np.mean([r['oracle_score'] - r['estimated_score']
                                          for r in rows])),
        'fixed_by_oracle': int(sum(1 for r in rows if r['oracle_top1_correct']
                                   and not r['estimated_top1_correct'])),
        'broken_by_oracle': int(sum(1 for r in rows if r['estimated_top1_correct']
                                    and not r['oracle_top1_correct'])),
        'note': ('diagnostic only: the reported comparison always uses the '
                 'image-estimated pose'),
        'rows': rows[:200],
    }


# --- full per-fold schedule (plan §10) --------------------------------------------
def run_schedule(args, paths, config) -> int:
    """Steps 1-7 of plan §10 for each fold, without consulting held-out labels.

    Selection uses only the fold's inner validation. If neither screened backbone
    beats C1 on the inner bank, the bounded SOSNet rescue is allowed once; if that
    also fails, the 8,000-step continuation is SKIPPED and the failed gate is
    reported rather than launching an open-ended search.
    """
    from .inner import inner_metrics, load_inner_aligned, score_inner
    from .stages import folds_for, _device, _seed
    device = _device(args, config)
    seed = _seed(args, config)
    data = args.data
    say = (lambda m: None) if getattr(args, 'quiet', False) else \
        (lambda m: print(m, flush=True))
    tcfg = config['train']
    summary = {}

    for fold in folds_for(args):
        held_out, allowed = fold_skies(fold)
        say(f'\n===== fold {fold} (held out {held_out}; allowed '
            f'{", ".join(allowed)}) =====')
        tensors = _load_tensors(paths, allowed, config, data, say)
        inner_aligned = _ensure_inner_aligned(paths, fold, config, data, say)

        # 1. controls on the inner bank, before any training result is consulted.
        c1 = inner_metrics(score_inner(inner_aligned, 'classical'))
        c1_metric = c1['equal_sky_top1_localization_reward']
        pretrained = {}
        for name in ('hardnet', 'hynet', 'sosnet'):
            from .models import Descriptor, load_backbone
            backbone, _ = load_backbone(name, pretrained=True, device=device)
            enc = Descriptor(backbone, name).to(device).eval()
            pretrained[name] = inner_metrics(
                score_inner(inner_aligned, 'descriptor', model=enc, device=device)
            )['equal_sky_top1_localization_reward']
        say(f'  inner controls  C1={c1_metric:.4f}  ' + '  '.join(
            f'{PRETRAINED_LABEL[n]}={v:.4f}' for n, v in pretrained.items()))

        # 2. screen T-H and T-Y for the screening budget.
        screened = {}
        for name in ('hardnet', 'hynet'):
            screened[name] = train_arm(
                paths, config, fold, name, device=device, seed=seed,
                total_steps=int(tcfg['screen_steps']), pretrained=True, hard=True,
                tag=f'screen_s{seed}', data=data, say=say, tensors=tensors,
                inner_aligned=inner_aligned)

        # 3. pick the backbone/checkpoint by inner validation only.
        ranked = sorted(screened.items(),
                        key=lambda kv: (-kv[1]['best']['metric'],
                                        kv[1]['seconds'],
                                        0 if kv[0] == 'hardnet' else 1))
        chosen, chosen_res = ranked[0]
        improves = chosen_res['best']['metric'] > c1_metric + 1e-9
        rescue = None
        if not improves:
            say(f'  neither screened backbone beats C1 ({c1_metric:.4f}); '
                f'best={chosen_res["best"]["metric"]:.4f}')
            if pretrained['sosnet'] > max(pretrained['hardnet'], pretrained['hynet']):
                say('  bounded rescue: SOSNet had the strongest pretrained control')
                rescue = train_arm(
                    paths, config, fold, 'sosnet', device=device, seed=seed,
                    total_steps=int(tcfg['screen_steps']), pretrained=True,
                    hard=True, tag=f'rescue_s{seed}', data=data, say=say,
                    tensors=tensors, inner_aligned=inner_aligned)
                if rescue['best']['metric'] > c1_metric + 1e-9:
                    chosen, chosen_res, improves = 'sosnet', rescue, True

        # 4. continuation, only for an improving backbone.
        continued = None
        if improves:
            continued = train_arm(
                paths, config, fold, chosen, device=device, seed=seed,
                total_steps=int(tcfg['max_steps']), pretrained=True, hard=True,
                tag=f'continue_s{seed}', data=data, say=say, tensors=tensors,
                inner_aligned=inner_aligned)
        else:
            say('  gate failed: skipping the 8,000-step continuation (plan §10.3)')

        # 5. equal-budget controls on the selected backbone.
        controls = {}
        for tag, kwargs in (('randomneg', {'pretrained': True, 'hard': False}),
                            ('scratch', {'pretrained': False, 'hard': True})):
            controls[tag] = train_arm(
                paths, config, fold, chosen, device=device, seed=seed,
                total_steps=int(tcfg['screen_steps']), tag=f'{tag}_s{seed}',
                data=data, say=say, tensors=tensors,
                inner_aligned=inner_aligned, **kwargs)

        # 6. pair head on the frozen selected checkpoint.
        best_dir = Path(paths.checkpoints(fold)) / (
            f'{chosen}_continue_s{seed}' if continued else f'{chosen}_screen_s{seed}')
        head_result = None
        if (best_dir / 'best.pt').exists():
            head_result = train_pair_head(
                paths, config, fold, chosen, best_dir / 'best.pt', device=device,
                seed=seed, data=data, say=say, tensors=tensors,
                inner_aligned=inner_aligned)
            say(f'  pair head: {head_result["decision"]} '
                f'({head_result["pair_head_best"]:.4f} vs distance '
                f'{head_result["descriptor_baseline"]:.4f})')

        # oracle-pose diagnostic on the selected checkpoint.
        oracle = None
        if (best_dir / 'best.pt').exists():
            model, _ = build_for_training(chosen, pretrained=True, device=device)
            blob = torch.load(best_dir / 'best.pt', map_location=device,
                              weights_only=False)
            model.load_state_dict(blob['model'])
            model.eval()
            oracle = oracle_pose_diagnostic(paths, config, fold, model, device, data)
            if oracle.get('ok'):
                say(f'  oracle pose: estimated top1={oracle["estimated_top1"]:.4f} '
                    f'vs oracle={oracle["oracle_top1"]:.4f} '
                    f'(+{oracle["fixed_by_oracle"]}/-{oracle["broken_by_oracle"]})')

        summary[fold] = {
            'held_out': held_out, 'allowed': list(allowed), 'seed': seed,
            'inner_controls': {'C1': c1_metric,
                               **{f'P-{n}': v for n, v in pretrained.items()}},
            'screened': {n: r['best'] for n, r in screened.items()},
            'selected_backbone': chosen,
            'selection_basis': 'fold inner validation only',
            'beats_c1_on_inner': improves,
            'rescue': (rescue['best'] if rescue else None),
            'continued': (continued['best'] if continued else None),
            'continuation_skipped': not improves,
            'controls': {t: r['best'] for t, r in controls.items()},
            'pair_head': head_result and {
                k: head_result[k] for k in
                ('pair_head_best', 'descriptor_baseline',
                 'pair_head_beats_distance', 'decision')},
            'oracle_pose': oracle and {k: oracle[k] for k in
                                       ('ok', 'n', 'estimated_top1', 'oracle_top1',
                                        'fixed_by_oracle', 'broken_by_oracle')
                                       if k in oracle},
        }
        write_json(Path(paths.fold(fold)) / f'schedule_s{seed}.json', summary[fold])
    write_json(Path(paths.root) / f'schedule_s{seed}.json', summary)
    return 0

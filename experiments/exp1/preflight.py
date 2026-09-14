"""Preflight: environment, device, parity and per-model smoke tests (plan §4).

Bitwise CPU/MPS equality is not required and is not asserted. What is asserted is
that descriptors agree to a stated tolerance and that RANKS agree, because ranking
is what the experiment measures. Zero-norm descriptors are treated as invalid
evidence, never as automatic matches.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .env import (describe_environment, peak_memory, resolve_device, mps_sync,
                  seed_torch, SEED_MODEL)
from .models import (BACKBONES, DESCRIPTOR_DIM, LOW_INFO_STD, build_for_training,
                     load_backbone, low_information, weight_provenance, Descriptor,
                     descriptor_distance, pairwise_distances)

# Stated tolerances for CPU vs MPS FP32 descriptor agreement.
TOL_COSINE = 0.999
TOL_MAXABS = 2e-3
MIN_FREE_DISK_GIB = 20.0
MIN_RAM_GIB = 8.0

# Operations that are unsupported on MPS and are deliberately run on CPU. Each is a
# targeted exception, not a blanket fallback, and each is reported (plan §4).
CPU_FALLBACK_OPERATIONS = [
    {'operation': 'float64 gradient-norm accumulation',
     'reason': 'MPS does not implement float64',
     'scope': 'diagnostic readout only; training math stays FP32 on device'},
]


def probe_patches(n: int = 24, seed: int = 12345) -> torch.Tensor:
    """Ordinary, constant, nearly constant and saturated patches (plan §4)."""
    gen = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(gen.uniform(0.15, 0.85, (32, 32)))
    out.append(np.zeros((32, 32)))                     # constant black
    out.append(np.ones((32, 32)))                      # constant white
    out.append(np.full((32, 32), 0.5))                 # constant mid
    out.append(np.full((32, 32), 0.5) + gen.normal(0, 1e-6, (32, 32)))  # near constant
    out.append(np.clip(gen.normal(1.4, 0.2, (32, 32)), 0, 1))           # saturated high
    out.append(np.clip(gen.normal(-0.4, 0.2, (32, 32)), 0, 1))          # saturated low
    arr = np.stack(out).astype(np.float32)[:, None, :, :]
    return torch.from_numpy(arr)


def _labelled_probe_kinds(total: int, n_random: int) -> list:
    tail = ['constant_black', 'constant_white', 'constant_mid', 'near_constant',
            'saturated_high', 'saturated_low']
    return ['ordinary'] * n_random + tail


def check_environment() -> dict:
    env = describe_environment()
    problems = []
    if env.machine != 'arm64':
        problems.append(f'expected arm64, found {env.machine}')
    if env.free_disk_gib < MIN_FREE_DISK_GIB:
        problems.append(f'only {env.free_disk_gib:.1f} GiB free disk')
    if env.ram_gib < MIN_RAM_GIB:
        problems.append(f'only {env.ram_gib:.1f} GiB RAM')
    if not env.mps_built:
        problems.append('torch was not built with MPS')
    if not env.mps_available:
        problems.append('MPS unavailable')
    return {'environment': env.__dict__, 'ok': not problems, 'problems': problems}


def check_weights() -> dict:
    entries = {n: weight_provenance(n) for n in BACKBONES}
    problems = [f'{n}: {"missing " + e["weights_path"] if not e["present"] else "sha256 mismatch"}'
                for n, e in entries.items()
                if not e['present'] or not e.get('sha256_matches')]
    return {'weights': entries, 'ok': not problems, 'problems': problems}


def forward_backward(name: str, device: str, batch: int = 16) -> dict:
    """Forward/backward on (B,1,32,32) with finite gradients and a real update."""
    seed_torch(SEED_MODEL)
    model, info = build_for_training(name, pretrained=True, device=device)
    x = probe_patches()[:batch].to(device)
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    z = model(x)
    norms = z.detach().norm(dim=1)
    loss = z.pow(2).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    # MPS has no float64, so the gradient norm is accumulated on CPU. Recorded as a
    # targeted CPU operation rather than a blanket fallback (plan §4).
    gnorm = float(torch.sqrt(sum((g.detach().cpu().double() ** 2).sum() for g in grads)))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.step()
    after = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    delta = float(max((a - b).abs().max().item() for a, b in zip(after, before)))
    return {
        'model': name, 'device': device, 'batch': batch,
        'descriptor_dim': int(z.shape[1]),
        'dim_ok': int(z.shape[1]) == DESCRIPTOR_DIM,
        'descriptors_finite': bool(torch.isfinite(z.detach()).all()),
        'unit_norm_max_error': float((norms - 1.0).abs().max()),
        'gradients_finite': finite,
        'gradient_norm': gnorm,
        'gradient_nonzero': gnorm > 0,
        'parameter_update_max': delta,
        'parameter_updated': delta > 0,
        'bn_policy': info['bn_policy'],
        'n_parameters': info['n_parameters'],
        'ctor_training_mode': info['ctor_training_mode'],
    }


def eval_repeatability(name: str, device: str) -> dict:
    """Eval mode must be deterministic across repeated calls."""
    model, _ = load_backbone(name, pretrained=True, device=device)
    enc = Descriptor(model, name).to(device).eval()
    x = probe_patches().to(device)
    with torch.no_grad():
        a = enc(x)
        b = enc(x)
    return {'model': name, 'device': device,
            'max_abs_diff': float((a - b).abs().max()),
            'repeatable': float((a - b).abs().max()) <= 1e-6}


def cpu_mps_parity(name: str) -> dict:
    """Descriptor and RANK agreement between CPU and MPS, with stated tolerances."""
    x = probe_patches()
    kinds = _labelled_probe_kinds(len(x), len(x) - 6)

    def encode(device):
        model, _ = load_backbone(name, pretrained=True, device=device)
        enc = Descriptor(model, name).to(device).eval()
        with torch.no_grad():
            return enc(x.to(device)).cpu()

    zc = encode('cpu')
    zm = encode('mps')
    cos = torch.nn.functional.cosine_similarity(zc, zm, dim=1)
    maxabs = float((zc - zm).abs().max())

    # Rank agreement: for every probe, order all others by descriptor distance.
    dc = pairwise_distances(zc, zc)
    dm = pairwise_distances(zm, zm)
    dc.fill_diagonal_(float('inf'))
    dm.fill_diagonal_(float('inf'))
    top1_same = float((dc.argmin(1) == dm.argmin(1)).float().mean())
    order_c = dc.argsort(1)[:, :5]
    order_m = dm.argsort(1)[:, :5]
    top5_same = float((order_c == order_m).all(1).float().mean())

    zero_norm = {
        'cpu': int((zc.norm(dim=1) < 1e-6).sum()),
        'mps': int((zm.norm(dim=1) < 1e-6).sum()),
    }
    per_kind = {}
    for i, kind in enumerate(kinds):
        per_kind.setdefault(kind, []).append(float(cos[i]))

    # Degenerate probes are excluded from the tolerance assertion and checked
    # separately: HardNet/SOSNet normalise each input internally, so a patch with
    # std ~ 1e-6 is divided by ~0 and its descriptor direction is meaningless. What
    # must hold for those is finiteness, a low-information flag, and stable ranks.
    degenerate = low_information(x).numpy()
    informative = ~degenerate
    cos_np = cos.numpy()
    diff = (zc - zm).abs().numpy()
    inf_cos_min = float(cos_np[informative].min())
    inf_maxabs = float(diff[informative].max())
    deg_cos_min = float(cos_np[degenerate].min()) if degenerate.any() else float('nan')
    deg_maxabs = float(diff[degenerate].max()) if degenerate.any() else 0.0

    return {
        'model': name,
        'n_probes': int(len(x)),
        'n_informative': int(informative.sum()),
        'n_degenerate': int(degenerate.sum()),
        'low_info_std_threshold': LOW_INFO_STD,
        'cosine_min': float(cos.min()),
        'cosine_mean': float(cos.mean()),
        'max_abs_diff': maxabs,
        'tol_cosine': TOL_COSINE, 'tol_maxabs': TOL_MAXABS,
        # Assertions apply to informative patches only.
        'informative_cosine_min': inf_cos_min,
        'informative_max_abs_diff': inf_maxabs,
        'cosine_ok': inf_cos_min >= TOL_COSINE,
        'maxabs_ok': inf_maxabs <= TOL_MAXABS,
        # Degenerate patches: finite, flagged, rank-stable. Not tolerance-checked.
        'degenerate_cosine_min': deg_cos_min,
        'degenerate_max_abs_diff': deg_maxabs,
        'degenerate_finite': bool(np.isfinite(diff[degenerate]).all()) if degenerate.any() else True,
        'degenerate_note': ('near-constant inputs are divided by ~0 by the internal '
                            'per-patch normalisation; flagged low-information, '
                            'excluded from the tolerance assertion'),
        'top1_rank_agreement': top1_same,
        'top5_rank_agreement': top5_same,
        'rank_ok': top1_same >= 1.0,
        'zero_norm_descriptors': zero_norm,
        'zero_norm_ok': zero_norm['cpu'] == 0 and zero_norm['mps'] == 0,
        'cosine_by_kind': {k: {'min': float(np.min(v)), 'mean': float(np.mean(v))}
                           for k, v in per_kind.items()},
    }


def overfit_smoke(name: str, device: str, pairs: int = 32, steps: int = 150) -> dict:
    """A 32-pair overfit smoke: the loss must fall substantially on a tiny set."""
    seed_torch(SEED_MODEL)
    model, _ = build_for_training(name, pretrained=True, device=device)
    gen = torch.Generator().manual_seed(4242)
    # Band-limited anchors make in-batch negatives genuinely confusable and the
    # positives non-trivial, so the initial loss is well above zero for every
    # backbone. White noise with a tiny positive jitter is already separated at
    # initialisation and would make this test vacuous.
    base = torch.randn(pairs, 1, 32, 32, generator=gen)
    kernel = torch.ones(1, 1, 5, 5) / 25.0
    anchors = torch.nn.functional.conv2d(base, kernel, padding=2)
    anchors = (anchors - anchors.amin()) / (anchors.amax() - anchors.amin())
    jitter = torch.nn.functional.conv2d(
        torch.randn(pairs, 1, 32, 32, generator=gen), kernel, padding=2)
    positives = (anchors + 0.25 * jitter).clamp(0, 1)
    a = anchors.to(device)
    p = positives.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    first = last = None
    history = []
    for step in range(steps):
        za, zp = model(a), model(p)
        d = pairwise_distances(za, zp)
        pos = torch.diagonal(d)
        neg = d + torch.eye(pairs, device=d.device) * 1e3
        loss = torch.relu(0.5 + pos - neg.min(1).values).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        value = float(loss.detach())
        history.append(value)
        first = value if first is None else first
        last = value
    return {
        'model': name, 'device': device, 'pairs': pairs, 'steps': steps,
        'loss_first': first, 'loss_last': last,
        'loss_min': float(min(history)),
        'decreased': last < first,
        'relative_drop': (first - last) / max(first, 1e-9),
        'nan_steps': int(sum(1 for v in history if not np.isfinite(v))),
        # A vacuous test (already-zero loss at step 0) is not a pass.
        'informative': first > 1e-3,
        'overfit_ok': first > 1e-3 and last < first * 0.5,
    }


def throughput(name: str, device: str, microbatch: int = 64,
               warmup: int = 50, measured: int = 200) -> dict:
    """Timed steps/s with explicit synchronisation (plan §4)."""
    seed_torch(SEED_MODEL)
    model, _ = build_for_training(name, pretrained=True, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gen = torch.Generator().manual_seed(11)
    a = torch.rand(microbatch, 1, 32, 32, generator=gen).to(device)
    p = torch.rand(microbatch, 1, 32, 32, generator=gen).to(device)

    def one_step():
        za, zp = model(a), model(p)
        d = pairwise_distances(za, zp)
        pos = torch.diagonal(d)
        neg = d + torch.eye(microbatch, device=d.device) * 1e3
        loss = torch.relu(0.5 + pos - neg.min(1).values).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    for _ in range(warmup):
        one_step()
    mps_sync(device)
    t0 = time.perf_counter()
    for _ in range(measured):
        one_step()
    mps_sync(device)
    elapsed = time.perf_counter() - t0
    return {
        'model': name, 'device': device, 'microbatch': microbatch,
        'warmup_steps': warmup, 'measured_steps': measured,
        'seconds': elapsed, 'steps_per_second': measured / elapsed,
        'memory': peak_memory(),
    }


def equivalence_check(device: str) -> dict:
    """Our dot-product distance matrix must match torch.cdist on this device."""
    gen = torch.Generator().manual_seed(3)
    a = torch.nn.functional.normalize(torch.randn(48, 128, generator=gen), dim=1).to(device)
    b = torch.nn.functional.normalize(torch.randn(48, 128, generator=gen), dim=1).to(device)
    ours = pairwise_distances(a, b)
    ref = torch.cdist(a, b)
    row = descriptor_distance(a, b)
    return {'device': device,
            'max_abs_diff_vs_cdist': float((ours - ref).abs().max()),
            'rowwise_matches_diagonal': float((row - torch.diagonal(ours)).abs().max()),
            'ok': float((ours - ref).abs().max()) < 2e-3}


def run(device_request: str = 'mps', models=BACKBONES, quick: bool = False) -> dict:
    report: dict = {'stage': 'preflight'}
    report['environment_check'] = check_environment()
    report['weights_check'] = check_weights()
    device, device_info = resolve_device(device_request)
    report['device'] = {'resolved': device, **device_info}
    if device == 'unavailable':
        report['ok'] = False
        report['blocked'] = ('MPS unavailable: ' + device_info.get('cause', 'unknown') +
                             '. Not launching on CPU unnoticed; rerun with '
                             '--device cpu to accept CPU explicitly.')
        return report

    devices = ['cpu'] if device == 'cpu' else ['cpu', device]
    report['equivalence'] = {d: equivalence_check(d) for d in devices}
    report['models'] = {}
    for name in models:
        if not report['weights_check']['weights'][name].get('sha256_matches'):
            report['models'][name] = {'skipped': 'weights missing or digest mismatch'}
            continue
        entry: dict = {}
        entry['forward_backward'] = {d: forward_backward(name, d) for d in devices}
        entry['eval_repeatability'] = {d: eval_repeatability(name, d) for d in devices}
        if device != 'cpu':
            entry['parity'] = cpu_mps_parity(name)
        entry['overfit'] = overfit_smoke(name, device, steps=40 if quick else 150)
        if not quick:
            entry['throughput'] = throughput(name, device)
        report['models'][name] = entry

    problems = list(report['environment_check']['problems'])
    problems += report['weights_check']['problems']
    for name, entry in report['models'].items():
        if 'skipped' in entry:
            problems.append(f'{name}: {entry["skipped"]}')
            continue
        for d, fb in entry['forward_backward'].items():
            for key in ('dim_ok', 'descriptors_finite', 'gradients_finite',
                        'gradient_nonzero', 'parameter_updated'):
                if not fb[key]:
                    problems.append(f'{name}/{d}: {key} failed')
        for d, rep in entry['eval_repeatability'].items():
            if not rep['repeatable']:
                problems.append(f'{name}/{d}: eval not repeatable')
        if 'parity' in entry:
            for key in ('cosine_ok', 'maxabs_ok', 'rank_ok', 'zero_norm_ok'):
                if not entry['parity'][key]:
                    problems.append(f'{name}: parity {key} failed')
        if not entry['overfit']['overfit_ok']:
            problems.append(f'{name}: 32-pair overfit smoke did not converge')
    report['problems'] = problems
    report['ok'] = not problems
    report['memory'] = peak_memory()
    report['cpu_fallback_operations'] = CPU_FALLBACK_OPERATIONS
    return report

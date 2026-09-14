"""Fixed group streams shared across paired ablations (task §11: "fixed training-
example streams shared across paired ablations").

Every arm for a given fold sees the SAME sequence of query groups at the SAME
step. Only the negative-selection policy inside `training.build_training_batch`
differs per arm. This makes the A-vs-B (hard vs random), C-vs-D (fusion),
D-vs-E (hard vs random with fusion) comparisons attributable to that one change.

Groups are expensive to build (real blind retrieval per group), so a fold's group
set is built once and cached to disk, then resampled by index for each step.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from . import OUT, SCENES
from .groups import assert_fold_isolated, build_absent_group, build_present_group
from experiments.exp1.env import derive_seed, rng, write_json
from experiments.exp1.splits import build_source_bank, fold_skies, partition_records


def stratified_sources(records: list, count: int, generator) -> list:
    """Round-robin over source strata, matching experiments.exp1.mining convention."""
    buckets: dict = {}
    for r in records:
        buckets.setdefault((r['source_class'], r['stratum']), []).append(r)
    keys = sorted(buckets)
    for k in keys:
        arr = buckets[k]
        order = generator.permutation(len(arr))
        buckets[k] = [arr[i] for i in order]
    out, cursor = [], {k: 0 for k in keys}
    while len(out) < count:
        progressed = False
        for k in keys:
            if len(out) >= count:
                break
            if cursor[k] < len(buckets[k]):
                out.append(buckets[k][cursor[k]])
                cursor[k] += 1
                progressed = True
        if not progressed:
            break
    return out


def build_group_stream(fold: str, partition: str, n_present_per_sky: int,
                       n_absent_per_sky: int, seed: int, data: str | None = None,
                       progress=None) -> list:
    """Present + absent groups for one fold's allowed skies, one partition."""
    held_out, allowed = fold_skies(fold)
    groups = []
    for scene in allowed:
        bank = build_source_bank(scene, data)
        records = partition_records(bank, partition)
        gen = rng(seed, 'exp3-present', fold, scene, partition)
        chosen = stratified_sources(records, n_present_per_sky, gen)
        for i, r in enumerate(chosen):
            groups.append(build_present_group(scene, r, gen, None, fold, partition, data))
            if progress and (i + 1) % 50 == 0:
                progress(f'    {fold}/{scene}/{partition} present {i + 1}/{len(chosen)}')

        donor = [s for s in allowed if s != scene][0]
        donor_bank = build_source_bank(donor, data)
        donor_records = partition_records(donor_bank, partition)
        gen_a = rng(seed, 'exp3-absent', fold, scene, partition)
        chosen_a = stratified_sources(donor_records, n_absent_per_sky, gen_a)
        for i, r in enumerate(chosen_a):
            groups.append(build_absent_group(scene, r, gen_a, None, fold, partition, data))
            if progress and (i + 1) % 50 == 0:
                progress(f'    {fold}/{scene}/{partition} absent {i + 1}/{len(chosen_a)}')

    check = assert_fold_isolated(fold, groups)
    if not check['ok']:
        raise RuntimeError(f'{fold}: isolation violated: {check["problems"]}')
    return groups


def cache_path(fold: str, partition: str, tag: str) -> Path:
    return OUT / 'groups' / f'{fold}_{partition}_{tag}.pkl'


def load_or_build_stream(fold: str, partition: str, tag: str, n_present_per_sky: int,
                         n_absent_per_sky: int, seed: int, data: str | None = None,
                         progress=None) -> list:
    path = cache_path(fold, partition, tag)
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    started = time.perf_counter()
    groups = build_group_stream(fold, partition, n_present_per_sky, n_absent_per_sky,
                                seed, data, progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(groups, f, protocol=5)
    if progress:
        progress(f'  {fold}/{partition}/{tag}: {len(groups)} groups in '
                 f'{time.perf_counter() - started:.1f}s')
    return groups


class StepSampler:
    """Deterministic per-step minibatch draw from a fixed, cached group stream."""

    def __init__(self, groups: list, seed: int, batch_size: int = 16):
        self.groups = groups
        self.seed = seed
        self.batch_size = batch_size
        # Index by position, never by dataclass equality (QueryGroup holds numpy
        # arrays, so `list.index()`/`==` would raise an ambiguous-truth-value error).
        self.present_idx = [i for i, g in enumerate(groups) if g.kind == 'present']
        self.absent_idx = [i for i, g in enumerate(groups) if g.kind == 'absent']

    def __call__(self, step: int) -> list:
        gen = np.random.default_rng(derive_seed(self.seed, 'exp3-step', step))
        half = self.batch_size // 2
        p = gen.choice(self.present_idx, size=min(half, len(self.present_idx)),
                      replace=len(self.present_idx) < half)
        a = gen.choice(self.absent_idx, size=min(self.batch_size - half, len(self.absent_idx)),
                      replace=len(self.absent_idx) < (self.batch_size - half))
        return [self.groups[i] for i in np.concatenate([p, a])]

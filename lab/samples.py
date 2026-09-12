"""Resample the scene into each cached candidate's patch frame, once, and cache it.

Every appearance-ranking variant then becomes pure arithmetic on a cached tensor,
so the expensive global search is never repeated.

Cached candidates carry (x, y, score, angle, scale). The verification stage built
its templates by sampling the *patch* at scene-aligned offsets,

    patch_coord = 15.5 + R(angle) . offset / scale

which restricts support to the largest disc that stays inside the 32x32 patch. This
module inverts that map instead, sampling the *scene* onto the patch grid,

    offset = scale . R(-angle) . (patch_coord - 15.5)

so the full 32x32 patch is usable and no patch pixel is discarded. Caches are keyed
by scene, candidate-source digest and sampler version.
"""
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.cache import TRAIN, ROOT

VERSION = 'sampler-v1-full32'
CACHE = ROOT / 'outputs/lab/samples'
SIZE = 32
CENTRE = (SIZE - 1) / 2.0


def sample_candidates(image, candidates):
    """(n_candidates, 32, 32) float32 scene content in the patch frame."""
    if not len(candidates):
        return np.zeros((0, SIZE, SIZE), np.float32)
    v, u = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    du, dv = (u - CENTRE), (v - CENTRE)
    out = np.empty((len(candidates), SIZE, SIZE), np.float32)
    for k, c in enumerate(candidates):
        x, y = float(c[0]), float(c[1])
        angle = float(c[3]) if len(c) > 3 else 0.
        scale = float(c[4]) if len(c) > 4 else 1.
        a = np.deg2rad(angle)
        ca, sa = np.cos(a), np.sin(a)
        mx = (x + scale * (ca * du + sa * dv)).astype(np.float32)
        my = (y + scale * (-sa * du + ca * dv)).astype(np.float32)
        out[k] = cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    return out


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     default=float).encode()).hexdigest()[:16]


def build(scene, split, candidate_lists, patches, scene_dir=None):
    """Cache aligned samples for one scene. Returns (samples, patches, mask)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = digest([VERSION, scene, [[list(map(float, c)) for c in q]
                                   for q in candidate_lists]])
    path = CACHE / f'{scene}.{key}.npz'
    if path.exists():
        z = np.load(path)
        return z['samples'], z['patches'], z['counts']
    d = scene_dir or scene
    image = cv2.imread(str(ROOT / split / d / f'{d}_image.png'), 0)
    if image is None:
        raise FileNotFoundError(f'missing sky for {scene}')
    n_alt = max((len(q) for q in candidate_lists), default=0)
    samples = np.zeros((len(candidate_lists), n_alt, SIZE, SIZE), np.float32)
    counts = np.zeros(len(candidate_lists), np.int32)
    for i, q in enumerate(candidate_lists):
        s = sample_candidates(image.astype(np.float32), q)
        samples[i, :len(s)] = s
        counts[i] = len(s)
    patch_stack = np.stack([p.astype(np.float32) for p in patches])
    np.savez_compressed(path, samples=samples, patches=patch_stack, counts=counts)
    return samples, patch_stack, counts


def load_scene_patches(scene, split, n):
    out = []
    for i in range(1, n + 1):
        p = cv2.imread(str(ROOT / split / scene / 'patches' / f'patch_{i:02}.png'), 0)
        if p is None:
            raise FileNotFoundError(f'missing patch {i} for {scene}')
        out.append(p)
    return out


def build_train(cache, branch='refined'):
    """Aligned samples for the three labelled scenes.

    The coarse branch is also needed: when it wins the geometry competition its pool
    holds coarse coordinates, so any guard scoring a proposed location must have
    samples for those candidates too.
    """
    out = {}
    for n in TRAIN:
        lists = cache[n][branch]
        patches = load_scene_patches(n, 'train', len(lists))
        out[n] = build(f'{n}.{branch}' if branch != 'refined' else n,
                       'train', lists, patches, scene_dir=n)
    return out


if __name__ == '__main__':
    from lab.cache import load_train
    data = build_train(load_train())
    for n, (s, p, c) in data.items():
        print(f'{n:9s} samples={s.shape} patches={p.shape} '
              f'alternatives per query: min={c.min()} max={c.max()}')

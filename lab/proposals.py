"""Cache the proposal set entering `verify`, per labelled query.

Attribution showed all seven uncovered present queries reach the proposals and are
lost inside `verify`. Proposal generation (harmonic retrieval plus the exhaustive
half-resolution search) is the expensive part, so it is cached once and every
verification variant is then measured against an identical input.

Keyed by scene and a digest of the generating configuration.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.retrieval import build_harmonic_index, retrieve_harmonic
from constellation.dense import dense_candidates
from lab.cache import TRAIN, ROOT
from lab.samples import load_scene_patches

CONFIG = {'index': 'harmonic stride4 + dog peaks', 'budget': 2000,
          'dense': 'half-res, 5 scales, 24 angles, 3 per pose', 'version': 'v1'}
CACHE = ROOT / 'outputs/lab/proposals'


def key():
    return hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]


def build(scene, split='train'):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'{scene}.{key()}.npz'
    if path.exists():
        z = np.load(path)
        return [z[f'q{i}'] for i in range(int(z['n']))]
    image = cv2.imread(str(ROOT / split / scene / f'{scene}_image.png'), 0)
    n = len(list((ROOT / split / scene / 'patches').glob('patch_*.png')))
    patches = load_scene_patches(scene, split, n)
    points, desc, _ = build_harmonic_index(image)
    out, t0 = [], time.perf_counter()
    for i, patch in enumerate(patches):
        shortlist = retrieve_harmonic(patch, points, desc)
        dc = dense_candidates(image, patch)
        out.append(np.unique(np.vstack([shortlist, dc]), axis=0).astype(np.float32))
    np.savez_compressed(path, n=len(out), **{f'q{i}': v for i, v in enumerate(out)})
    print(f'  {scene}: {len(out)} queries, mean {np.mean([len(v) for v in out]):.0f} '
          f'proposals, {time.perf_counter()-t0:.0f}s', flush=True)
    return out


def build_train():
    return {n: build(n) for n in TRAIN}


if __name__ == '__main__':
    data = build_train()
    for n, v in data.items():
        print(f'{n:9s} queries={len(v)} proposals min={min(len(q) for q in v)} '
              f'max={max(len(q) for q in v)}')

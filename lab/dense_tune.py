"""Chance fits dominate in dense query fields. Sweep the levers that suppress them.

Note this benchmark supplies no auxiliary star map, so it is a lower bound on the
production configuration, which does use one.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.joint import recognize_joint
from constellation.references import extract_patterns
from lab.cache import ROOT
from lab.bigscene import build

PATTERNS = None
BASE = dict(models=('affine',), gap=.03, score_mode='binom', cap=80000,
            top_k=8, quad_share=0., tolerance=18.)
N = 48


def _init():
    global PATTERNS
    PATTERNS = extract_patterns(ROOT / 'patterns')


def _one(args):
    scene, cfg = args
    name, _, _ = recognize_joint(scene['alternatives'], PATTERNS, **cfg)
    return name == scene['name']


def score(scenes, cfg, label):
    with ProcessPoolExecutor(max_workers=8, initializer=_init) as pool:
        res = list(pool.map(_one, [(s, cfg) for s in scenes], chunksize=1))
    acc = float(np.mean(res))
    print(f'  {label:34s} id={acc:.3f}+-{np.sqrt(acc*(1-acc)/len(res)):.3f}', flush=True)
    return acc


def main():
    _init()
    scenes = build(N, 45, patterns=PATTERNS)
    print(f'dense scenes (~{int(np.mean([len(s["alternatives"]) for s in scenes]))} queries)',
          flush=True)
    for ms in (4, 5, 6, 7, 8):
        score(scenes, {**BASE, 'min_support': ms}, f'min_support={ms}')


if __name__ == '__main__':
    main()

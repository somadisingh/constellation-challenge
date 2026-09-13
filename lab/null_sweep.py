"""Identification under corrected chance-fit nulls.

Measured on the synthetic geometry benchmark, which is the only place identification has
real sample size, at a seed not used for earlier tuning. Real-scene corroboration is run
separately in lab/null_real.py; a synthetic win alone does not settle anything.

  groups                 shipped null: chance density from query-group count
  pool                   chance density from the pooled point count the assignment
                         actually draws from
  decorrelate            shipped null, then the per-scene score-versus-log-size trend
                         subtracted
  pool+decorrelate       both
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.references import extract_patterns
from constellation.joint import recognize_joint
from lab.cache import ROOT
from lab.synth import dataset

PATTERNS = None
BASE = dict(models=('affine',), gap=.03, score_mode='binom', tolerance=18.,
            cap=80000, top_k=8, quad_share=0.)
MODES = ('groups', 'pool', 'decorrelate', 'pool+decorrelate')


def _init():
    global PATTERNS
    PATTERNS = extract_patterns(ROOT / 'patterns')


def _one(args):
    scene, mode = args
    name, _, _ = recognize_joint(scene['alternatives'], PATTERNS,
                                 null_mode=mode, **BASE)
    return (name == scene['name'], scene['template_nodes'], scene['n_figure'])


def run(mode, scenes, workers=8):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        return list(pool.map(_one, [(s, mode) for s in scenes], chunksize=1))


def report(label, res):
    acc = float(np.mean([r[0] for r in res]))
    se = np.sqrt(acc * (1 - acc) / len(res))
    small = [r for r in res if r[1] <= 9]
    large = [r for r in res if r[1] >= 14]
    few = [r for r in res if r[2] <= 6]
    many = [r for r in res if r[2] >= 10]
    print(f'{label:22s} id={acc:.3f}+-{se:.3f}  '
          f'small(<=9 nodes)={np.mean([r[0] for r in small]):.3f}(n={len(small)})  '
          f'large(>=14)={np.mean([r[0] for r in large]):.3f}(n={len(large)})  '
          f'few issued(<=6)={np.mean([r[0] for r in few]):.3f}  '
          f'many(>=10)={np.mean([r[0] for r in many]):.3f}', flush=True)
    return acc, se


def main():
    _init()
    out = {}
    for seed, tag in ((2027, 'tuning seed 2027'), (4099, 'fresh seed 4099')):
        for min_nodes, regime in ((7, 'templates >=7 nodes'), (4, 'all >=4 nodes')):
            scenes = dataset(PATTERNS, 192, seed, min_nodes)
            print(f'\n-- {tag}, {regime}, n={len(scenes)} --')
            for mode in MODES:
                acc, se = report(mode, run(mode, scenes))
                out[f'{seed}|{min_nodes}|{mode}'] = {'accuracy': acc, 'se': se}
    (ROOT / 'outputs/lab/null_sweep.json').write_text(
        json.dumps(out, indent=2, default=float))


if __name__ == '__main__':
    main()

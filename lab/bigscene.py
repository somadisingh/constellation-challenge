"""Validation scenes carry 21-87 queries but no labelled scene exceeds 41, and the
tuning ran at ~35-50. Check the configuration still holds when clutter grows, and
whether the 56-point cap on hypothesis seeding is now a limitation (it was set to
bound O(n^4) quad enumeration, which triangles-only seeding no longer needs).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import constellation.joint as J
from constellation.references import extract_patterns
from lab.cache import ROOT
from lab.synth import make_scene

PATTERNS = None
CFG = dict(models=('affine',), gap=.03, score_mode='binom', tolerance=18.,
           cap=80000, top_k=8, quad_share=0.)


def _init():
    global PATTERNS
    PATTERNS = extract_patterns(ROOT / 'patterns')


def _one(args):
    scene, gen_limit = args
    J.GEN_LIMIT = gen_limit
    name, _, _ = J.recognize_joint(scene['alternatives'], PATTERNS, **CFG)
    return name == scene['name']


def build(n_scenes, extra_off, seed=31, patterns=None):
    """Scenes with additional off-figure clutter queries."""
    rng = np.random.default_rng(seed)
    patterns = patterns if patterns is not None else PATTERNS
    usable = [n for n, v in sorted(patterns.items()) if len(v) >= 7]
    out = []
    for i in range(n_scenes):
        s = make_scene(rng, patterns, usable[i % len(usable)])
        if not s:
            continue
        for _ in range(extra_off):
            xy = rng.uniform(60., 2940., 2)
            pts = rng.uniform(0., 3000., size=(20, 2))
            pts[0] = xy
            sc = np.linspace(.93, .70, 20)
            s['alternatives'].append([(float(x), float(y), float(v), 0., 1.)
                                      for (x, y), v in zip(pts, sc)])
            s['truth'].append(tuple(xy))
        out.append(s)
    return out


def main():
    _init()
    for extra, label in [(0, 'as tuned (~35-50 queries)'),
                         (20, '+20 clutter queries'),
                         (45, '+45 clutter queries (~87 total)')]:
        scenes = build(96, extra)
        n = int(np.mean([len(s['alternatives']) for s in scenes]))
        for gen_limit in (56, 120):
            with ProcessPoolExecutor(max_workers=8, initializer=_init) as pool:
                res = list(pool.map(_one, [(s, gen_limit) for s in scenes], chunksize=1))
            acc = float(np.mean(res))
            print(f'{label:34s} queries~{n:3d} GEN_LIMIT={gen_limit:3d} '
                  f'id={acc:.3f}+-{np.sqrt(acc*(1-acc)/len(res)):.3f}')


if __name__ == '__main__':
    main()

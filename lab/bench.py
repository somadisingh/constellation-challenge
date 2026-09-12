"""Measure recognizer identification accuracy on the synthetic geometry benchmark."""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.references import extract_patterns
from constellation.joint import recognize_joint
from constellation.geometry import recognize as recognize_frozen
from lab.cache import ROOT
from lab.synth import dataset

PATTERNS = None


def _init():
    global PATTERNS
    PATTERNS = extract_patterns(ROOT / 'patterns')


def _one(args):
    scene, kind, cfg = args
    alts = scene['alternatives']
    if kind == 'frozen':
        name, _, _ = recognize_frozen([a[0][:2] for a in alts], PATTERNS,
                                      use_quads=True, shear_penalty=2., tolerance=18.)
        chosen = {}
    else:
        name, chosen, _ = recognize_joint(alts, PATTERNS, **cfg)
    # Snapping only matters where it changes correctness: count fixes (base wrong,
    # relocated right) against regressions (base right, relocated wrong).
    fixes = regressions = 0
    for i, xy in chosen.items():
        t = scene['truth'][i]
        base = np.array(alts[i][0][:2])
        if np.linalg.norm(np.array(xy) - base) < .5 or t is None:
            continue
        was = np.linalg.norm(base - np.array(t)) <= 12
        now = np.linalg.norm(np.array(xy) - np.array(t)) <= 12
        if now and not was:
            fixes += 1
        elif was and not now:
            regressions += 1
    return name == scene['name'], fixes, regressions


def run(label, kind='joint', cfg=None, n_scenes=96, seed=17, workers=8, min_nodes=7):
    cfg = cfg or {}
    _init()
    scenes = dataset(PATTERNS, n_scenes, seed, min_nodes)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        res = list(pool.map(_one, [(s, kind, cfg) for s in scenes], chunksize=1))
    acc = np.mean([r[0] for r in res])
    fixes = sum(r[1] for r in res); regr = sum(r[2] for r in res)
    se = np.sqrt(acc * (1 - acc) / len(res))
    print(f'{label:38s} id={acc:.3f}+-{se:.3f} (n={len(res)}) '
          f'snap fix={fixes} regress={regr} [{time.perf_counter()-t0:.0f}s]')
    return acc, se


if __name__ == '__main__':
    run('frozen recognizer', 'frozen')
    run('joint (defaults)', 'joint')

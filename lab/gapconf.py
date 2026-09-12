"""Does the winner-to-runner-up score gap predict whether identification is right?

If it does, coordinate relocation can be withheld when the class is uncertain,
which is where snapping does damage (a wrong class maps nodes to wrong places).
"""
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
CFG = dict(models=('affine',), gap=.03, score_mode='binom', tolerance=18.,
           cap=80000, top_k=8, quad_share=0.)


def _init():
    global PATTERNS
    PATTERNS = extract_patterns(ROOT / 'patterns')


def _one(scene):
    name, chosen, diag = recognize_joint(scene['alternatives'], PATTERNS, **CFG)
    fixes = regressions = 0
    for i, xy in chosen.items():
        t = scene['truth'][i]
        base = np.array(scene['alternatives'][i][0][:2])
        if t is None or np.linalg.norm(np.array(xy) - base) < .5:
            continue
        was = np.linalg.norm(base - np.array(t)) <= 12
        now = np.linalg.norm(np.array(xy) - np.array(t)) <= 12
        fixes += now and not was
        regressions += was and not now
    return (name == scene['name'], diag.get('score_gap'), fixes, regressions)


def main():
    _init()
    scenes = dataset(PATTERNS, 192, 99, 7)
    with ProcessPoolExecutor(max_workers=8, initializer=_init) as pool:
        res = list(pool.map(_one, scenes, chunksize=1))
    ok = [r[1] for r in res if r[0] and r[1] is not None]
    no = [r[1] for r in res if not r[0] and r[1] is not None]
    print(f'correct n={len(ok)} gap median={np.median(ok):.2f} p10={np.percentile(ok,10):.2f}')
    print(f'wrong   n={len(no)} gap median={np.median(no):.2f} p90={np.percentile(no,90):.2f}')
    print('\nwithholding relocation below a gap G:')
    print(f"{'G':>6} {'precision above G':>18} {'scenes above G':>16} "
          f"{'fixes kept':>11} {'regr kept':>10}")
    tot_f = sum(r[2] for r in res); tot_r = sum(r[3] for r in res)
    for G in (0., .5, 1., 2., 3., 5.):
        sel = [r for r in res if r[1] is not None and r[1] >= G]
        if not sel:
            continue
        prec = np.mean([r[0] for r in sel])
        f = sum(r[2] for r in sel); rr = sum(r[3] for r in sel)
        print(f'{G:6.1f} {prec:18.3f} {len(sel):16d} {f:6d}/{tot_f:<4d} {rr:5d}/{tot_r:<4d}')


if __name__ == '__main__':
    main()

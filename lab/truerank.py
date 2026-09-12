"""Where does the true class rank among all 48 hypotheses?

rank 0            correct
rank 1-5          scoring problem: the fit exists and is competitive
rank 6+ / no fit  generation problem: no verified transform was ever found
"""
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.references import extract_patterns
from constellation.joint import recognize_joint
from lab.cache import ROOT
from lab.synth import dataset

PATTERNS = None
CFG = dict(models=('affine',), tolerance=12., cap=30000, gap=.03, diag_top=48)


def _init():
    global PATTERNS
    PATTERNS = extract_patterns(ROOT / 'patterns')


def _one(scene):
    _, _, diag = recognize_joint(scene['alternatives'], PATTERNS, **CFG)
    hyps = diag.get('hypotheses', [])
    verified = [h for h in hyps if h.get('support', 0) >= 4]
    names = [h['name'] for h in verified]
    true = scene['name']
    rank = names.index(true) if true in names else None
    return rank, len(verified), scene['template_nodes'], scene['n_figure']


def main():
    _init()
    scenes = dataset(PATTERNS, 96, 17, 7)
    with ProcessPoolExecutor(max_workers=8, initializer=_init) as pool:
        res = list(pool.map(_one, scenes, chunksize=1))
    buckets = Counter()
    for rank, nver, nodes, nfig in res:
        if rank is None:
            buckets['no verified fit for true class'] += 1
        elif rank == 0:
            buckets['rank 0 (correct)'] += 1
        elif rank <= 5:
            buckets['rank 1-5 (scoring)'] += 1
        else:
            buckets['rank 6+ (scoring, weak)'] += 1
    for k, v in buckets.most_common():
        print(f'  {k:34s} {v:3d}  ({v/len(res):.2f})')
    nver = [r[1] for r in res]
    print(f'\ncompeting verified classes per scene: median={np.median(nver):.0f} '
          f'max={max(nver)}')
    lost = [(r[2], r[3]) for r in res if r[0] is None]
    if lost:
        print(f'no-fit cases: template nodes median={np.median([a for a,_ in lost]):.0f}, '
              f'issued figure queries median={np.median([b for _,b in lost]):.0f}')


if __name__ == '__main__':
    main()

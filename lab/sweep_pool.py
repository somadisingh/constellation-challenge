"""Pool width is the dominant identification lever; sweep it in the realistic regime."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.bench import run

BASE = dict(models=('affine',), tolerance=12., cap=30000)
N = 96


def main():
    print('-- ambiguity gate (pool widened only for near-tied queries) --')
    for gap in (.01, .02, .03, .045, .06):
        run(f'gap={gap:.3f}', 'joint', {**BASE, 'gap': gap}, n_scenes=N)

    print('\n-- alternatives per ambiguous query --')
    for k in (4, 8, 12, 20):
        run(f'top_k={k}', 'joint', {**BASE, 'top_k': k}, n_scenes=N)

    print('\n-- appearance margin bounding the pool --')
    for m in (.05, .10, .15, .30):
        run(f'margin={m:.2f}', 'joint', {**BASE, 'margin': m}, n_scenes=N)

    print('\n-- scoring weights on pooled evidence --')
    for aw in (0., .5, 1.5):
        run(f'appearance_weight={aw}', 'joint', {**BASE, 'appearance_weight': aw}, n_scenes=N)
    for rw in (0., .04, .15):
        run(f'rank_weight={rw}', 'joint', {**BASE, 'rank_weight': rw}, n_scenes=N)

    print('\n-- shear penalty --')
    for sp in (0., 2., 6.):
        run(f'shear_penalty={sp}', 'joint', {**BASE, 'shear_penalty': sp}, n_scenes=N)


if __name__ == '__main__':
    main()

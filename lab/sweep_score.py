"""Ranking, not generation, limits identification. Sweep the scoring function."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.bench import run

BASE = dict(models=('affine',), tolerance=12., cap=30000, gap=.03)
N = 96


def main():
    run('binomial surprise (previous)', 'joint', {**BASE, 'score_mode': 'binom'}, n_scenes=N)
    print('\n-- likelihood ratio: inlier sigma --')
    for s in (1.0, 1.5, 1.8, 2.5, 4.0):
        run(f'sigma={s}', 'joint', {**BASE, 'sigma': s}, n_scenes=N)
    print('\n-- likelihood ratio: cost per offered template node --')
    for sp in (0., .5, .9, 1.5, 2.5):
        run(f'size_penalty={sp}', 'joint', {**BASE, 'size_penalty': sp}, n_scenes=N)


if __name__ == '__main__':
    main()

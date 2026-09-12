"""Tolerance sets both the assignment radius and the chance-density normaliser,
so it interacts with the inlier sigma. Sweep them jointly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.bench import run

BASE = dict(models=('affine',), cap=30000, gap=.03, size_penalty=0.)
N = 96


def main():
    for tol in (8., 12., 18., 24.):
        for sigma in (1.0, 1.5, 2.2):
            run(f'tol={tol:.0f} sigma={sigma}', 'joint',
                {**BASE, 'tolerance': tol, 'sigma': sigma}, n_scenes=N)
        print()


if __name__ == '__main__':
    main()

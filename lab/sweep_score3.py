"""Re-tune with realistic ~5px template-to-sky model error in the benchmark."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.bench import run

BASE = dict(models=('affine',), cap=30000, gap=.03, size_penalty=0.)
N = 96


def main():
    run('frozen recognizer', 'frozen', n_scenes=N)
    run('joint + binomial', 'joint', {**BASE, 'score_mode': 'binom', 'tolerance': 18.},
        n_scenes=N)
    print()
    for tol in (12., 18., 24., 32.):
        for sigma in (3., 5., 7., 10.):
            run(f'tol={tol:.0f} sigma={sigma:.0f}', 'joint',
                {**BASE, 'tolerance': tol, 'sigma': sigma}, n_scenes=N)
        print()


if __name__ == '__main__':
    main()

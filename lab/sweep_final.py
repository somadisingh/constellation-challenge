"""Tune the binomial-score joint recognizer under realistic model error."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.bench import run

BASE = dict(models=('affine',), cap=30000, gap=.03, score_mode='binom', tolerance=18.)
N = 96


def main():
    run('frozen', 'frozen', n_scenes=N)
    run('joint base', 'joint', BASE, n_scenes=N)

    print('\n-- tolerance --')
    for tol in (14., 18., 22., 26.):
        run(f'tolerance={tol:.0f}', 'joint', {**BASE, 'tolerance': tol}, n_scenes=N)

    print('\n-- ambiguity gate --')
    for gap in (0., .02, .03, .05):
        run(f'gap={gap:.2f}', 'joint', {**BASE, 'gap': gap}, n_scenes=N)

    print('\n-- minimum support (chance matches are easier at 5px model error) --')
    for ms in (4, 5, 6, 7):
        run(f'min_support={ms}', 'joint', {**BASE, 'min_support': ms}, n_scenes=N)

    print('\n-- alternatives per ambiguous query --')
    for k in (8, 14, 20):
        run(f'top_k={k}', 'joint', {**BASE, 'top_k': k}, n_scenes=N)

    print('\n-- hypothesis budget --')
    for cap in (30000, 80000):
        run(f'cap={cap}', 'joint', {**BASE, 'cap': cap}, n_scenes=N)


if __name__ == '__main__':
    main()

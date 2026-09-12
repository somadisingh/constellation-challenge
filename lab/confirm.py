"""Confirm the tuned recognizer on fresh synthetic scenes (unseen seed, larger n)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.bench import run

TUNED = dict(models=('affine',), tolerance=12., cap=30000, gap=.03,
             score_mode='likelihood', sigma=2.2, size_penalty=0.)
N, SEED = 192, 99


def main():
    print(f'-- fresh data: n={N}, seed={SEED} (tuning used seed 17) --')
    run('frozen recognizer', 'frozen', n_scenes=N, seed=SEED)
    run('joint + binomial score', 'joint',
        {**TUNED, 'score_mode': 'binom'}, n_scenes=N, seed=SEED)
    run('joint + likelihood score (tuned)', 'joint', TUNED, n_scenes=N, seed=SEED)
    print('\n-- sensitivity of the tuned config on fresh data --')
    for tol in (8., 12., 18.):
        run(f'  tolerance={tol:.0f}', 'joint', {**TUNED, 'tolerance': tol},
            n_scenes=N, seed=SEED)


if __name__ == '__main__':
    main()

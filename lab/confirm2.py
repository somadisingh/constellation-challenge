"""Confirm the retuned recognizer on fresh synthetic scenes and check sensitivity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.bench import run

TUNED = dict(models=('affine',), gap=.03, score_mode='binom', tolerance=18.,
             cap=80000, top_k=8, quad_share=0.)
N, SEED = 192, 99


def main():
    print(f'-- fresh data: n={N}, seed={SEED} (tuning used seed 17) --')
    run('frozen recognizer', 'frozen', n_scenes=N, seed=SEED)
    run('joint, tuned', 'joint', TUNED, n_scenes=N, seed=SEED)
    run('joint, no pool widening', 'joint', {**TUNED, 'gap': 0.}, n_scenes=N, seed=SEED)
    run('joint, with quads (as frozen)', 'joint', {**TUNED, 'quad_share': .6},
        n_scenes=N, seed=SEED)
    run('joint, small budget', 'joint', {**TUNED, 'cap': 30000}, n_scenes=N, seed=SEED)
    print('\n-- also check the harder all-templates regime --')
    run('tuned, min_nodes=4', 'joint', TUNED, n_scenes=N, seed=SEED, min_nodes=4)
    run('frozen, min_nodes=4', 'frozen', n_scenes=N, seed=SEED, min_nodes=4)


if __name__ == '__main__':
    main()

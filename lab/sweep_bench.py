"""Tune the recognizer where identification can actually be measured."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.bench import run

N = 96


def main():
    run('frozen recognizer', 'frozen', n_scenes=N)
    run('joint baseline', 'joint', {}, n_scenes=N)

    print('\n-- verification tolerance --')
    for tol in (12., 18., 24., 30.):
        run(f'tolerance={tol:.0f}', 'joint', {'tolerance': tol}, n_scenes=N)

    print('\n-- minimum independent support --')
    for ms in (4, 5, 6):
        run(f'min_support={ms}', 'joint', {'min_support': ms}, n_scenes=N)

    print('\n-- model set --')
    run('affine only', 'joint', {'models': ('affine',)}, n_scenes=N)
    run('similarity only', 'joint', {'models': ('similarity',)}, n_scenes=N)

    print('\n-- hypothesis budget --')
    for cap in (30000, 120000):
        run(f'cap={cap}', 'joint', {'cap': cap}, n_scenes=N)

    print('\n-- ambiguity gate --')
    for gap in (0., .02, .06):
        run(f'gap={gap:.2f}', 'joint', {'gap': gap}, n_scenes=N)


if __name__ == '__main__':
    main()

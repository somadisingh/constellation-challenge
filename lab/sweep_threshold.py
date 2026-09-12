"""Presence threshold trade-off.

Presence F1 wants precision, but localization scores 0 for a truth-present query
predicted absent, and recovery is a greedy one-to-one match over predicted-present
points, so extra points can only help it. The optimum is therefore lower than a
pure-presence fit would suggest.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.harness import experiment

def main():
    print('-- presence threshold (refined-stage cutoff) --')
    for t in (.55, .60, .65, .68, .72, .76, .80):
        experiment(f'threshold={t:.2f}', {'threshold': t})

    print('\n-- alternatives per ambiguous query (drives snap coverage) --')
    for k in (8, 14, 20):
        experiment(f'top_k={k}', {'top_k': k})

    print('\n-- membership radius --')
    for r in (12., 18., 26., 36.):
        experiment(f'member_radius={r:.0f}', {'member_radius': r})


if __name__ == '__main__':
    main()

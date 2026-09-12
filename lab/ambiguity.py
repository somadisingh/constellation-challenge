"""Does an appearance-confidence signal separate queries whose rank-0 location is
already correct from those needing a geometric override?

If it does, the pool should only be widened for ambiguous queries, which both
shrinks the pool and stops geometry from moving confidently located queries.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from lab.cache import load_train, TRAIN, ROOT


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    cache = load_train()
    ok, bad = [], []
    for n in TRAIN:
        t = truth[n].patches
        for i, tp in enumerate(t):
            q = cache[n]['refined'][i]
            if tp is None or not len(q):
                continue
            s = np.array([c[2] for c in q])
            d0 = np.linalg.norm(np.array(q[0][:2]) - np.array(tp[:2]))
            rec = dict(score=float(s[0]),
                       gap=float(s[0] - s[1]) if len(s) > 1 else 1.,
                       figure=tp[2])
            (ok if d0 <= 12 else bad).append(rec)
    print(f'rank-0 correct: {len(ok)}   rank-0 wrong: {len(bad)}')
    for key in ('score', 'gap'):
        a = np.array([r[key] for r in ok]); b = np.array([r[key] for r in bad])
        print(f'  {key:6s} correct: median={np.median(a):.3f} p10={np.percentile(a,10):.3f}'
              f'   wrong: median={np.median(b):.3f} p90={np.percentile(b,90):.3f}')

    print('\ngate: treat a query as ambiguous when score < S (widen its pool)')
    print(f"{'S':>6} {'amb&wrong':>10} {'amb&correct':>12} {'missed wrong':>13}")
    for S in (.80, .85, .90, .93, .95, .97, 1.01):
        aw = sum(1 for r in bad if r['score'] < S)
        ac = sum(1 for r in ok if r['score'] < S)
        print(f'{S:6.2f} {aw:10d} {ac:12d} {len(bad)-aw:13d}')

    print('\ngate: ambiguous when gap < G')
    print(f"{'G':>6} {'amb&wrong':>10} {'amb&correct':>12} {'missed wrong':>13}")
    for G in (.01, .02, .05, .10, .20, 1.01):
        aw = sum(1 for r in bad if r['gap'] < G)
        ac = sum(1 for r in ok if r['gap'] < G)
        print(f'{G:6.2f} {aw:10d} {ac:12d} {len(bad)-aw:13d}')


if __name__ == '__main__':
    main()

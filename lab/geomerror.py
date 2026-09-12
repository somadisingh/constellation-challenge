"""How well does an affine map actually take reference nodes onto true sky stars?

The synthetic benchmark assumed ~1px residuals, but the reference diagrams are
schematics whose extracted node centroids need not be an exact affine image of
the sky. This measures the real model error, using ground-truth figure points, by
searching correspondences for the true class only.
"""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from constellation.references import extract_patterns
from constellation.joint import fit_affine, fit_similarity, assignment, valid
from lab.cache import TRAIN, ROOT


def best_fit(template, points, model='affine', tolerance=60., iters=4):
    """Exhaustive triangle seeding against known truth points for one class."""
    p = (template - template.mean(axis=0)) / np.maximum(np.ptp(template, axis=0), 1e-6)
    tags = np.arange(len(points))
    best = None
    for ti in combinations(range(len(p)), 3):
        for si in combinations(range(len(points)), 3):
            for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                src = p[list(ti)]
                dst = points[[si[k] for k in perm]]
                m = (fit_affine(src, dst) if model == 'affine'
                     else fit_similarity(src, dst))
                if m is None or not valid(m):
                    continue
                pairs, res = assignment(np.c_[p, np.ones(len(p))] @ m, points, tolerance, tags)
                if len(pairs) < 4:
                    continue
                for _ in range(iters):
                    ii, jj = np.array(pairs).T
                    m2 = (fit_affine(p[ii], points[jj]) if model == 'affine'
                          else fit_similarity(p[ii], points[jj]))
                    if m2 is None:
                        break
                    m = m2
                    pairs, res = assignment(np.c_[p, np.ones(len(p))] @ m, points,
                                            tolerance, tags)
                    if len(pairs) < 4:
                        break
                if len(pairs) < 4:
                    continue
                key = (len(pairs), -float(np.mean(res)))
                if best is None or key > best[0]:
                    best = (key, len(pairs), np.array(res), m)
    return best


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    pat = extract_patterns(ROOT / 'patterns')
    for n in TRAIN:
        t = truth[n].patches
        fig = np.array([p[:2] for p in t if p is not None and p[2] == 1]).reshape(-1, 2)
        tmpl = pat[truth[n].constellation]
        print(f'\n{n} ({truth[n].constellation}): {len(fig)} true figure points, '
              f'{len(tmpl)} template nodes')
        for model in ('affine', 'similarity'):
            b = best_fit(tmpl, fig, model)
            if not b:
                print(f'  {model:10s} no fit'); continue
            _, sup, res, _ = b
            print(f'  {model:10s} support={sup}/{len(fig)} residual '
                  f'median={np.median(res):.1f} mean={res.mean():.1f} '
                  f'p90={np.percentile(res,90):.1f} max={res.max():.1f}')


if __name__ == '__main__':
    main()

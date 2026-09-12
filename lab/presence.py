"""Presence is 25% of the score and is currently a single threshold on the
refined appearance correlation. Evaluate richer rules over all 116 labelled
queries, scored with the competition's two-class mean F1.

Leave-one-scene-out is used for anything with a fitted constant, so the reported
number is not the value the constant was chosen on.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from lab.cache import load_train, TRAIN, ROOT


def presence_f1(y, z):
    """Competition convention: mean F1 over the absent and present classes."""
    y, z = np.asarray(y, bool), np.asarray(z, bool)
    out = []
    for cls in (False, True):
        tp = np.sum((y == cls) & (z == cls))
        den = np.sum(y == cls) + np.sum(z == cls)
        out.append(2 * tp / den if den else 1.)
    return float(np.mean(out))


def features(cache, truth):
    """Per-query signals plus the present/absent label."""
    rows = []
    for n in TRAIN:
        for i, tp in enumerate(truth[n].patches):
            r = cache[n]['refined'][i]
            c = cache[n]['coarse'][i]
            if not len(r):
                continue
            s = np.array([q[2] for q in r])
            # Coarse agreement: best coarse score near the chosen refined point.
            rp = np.array(r[0][:2])
            cp = np.array([q[:2] for q in c]); cs = np.array([q[2] for q in c])
            near = np.linalg.norm(cp - rp, axis=1) < 8 if len(cp) else np.array([False])
            rows.append(dict(scene=n, present=tp is not None, score=float(s[0]),
                             gap=float(s[0] - s[1]) if len(s) > 1 else 1.,
                             coarse=float(cs[near].max()) if near.any() else 0.,
                             spread=float(s[:5].std())))
    return rows


def evaluate_rule(rows, rule, folds=True):
    """Score a rule; `rule(train_rows, row) -> bool`."""
    if not folds:
        z = [rule(rows, r) for r in rows]
        return presence_f1([r['present'] for r in rows], z)
    per = []
    for held in TRAIN:
        tr = [r for r in rows if r['scene'] != held]
        te = [r for r in rows if r['scene'] == held]
        z = [rule(tr, r) for r in te]
        per.append(presence_f1([r['present'] for r in te], z))
    return float(np.mean(per))


def fit_threshold(train, key, extra=None):
    best, bt = -1, .72
    for t in np.arange(.40, .99, .005):
        z = [(r[key] >= t) if extra is None else (r[key] >= t or extra(r)) for r in train]
        f = presence_f1([r['present'] for r in train], z)
        if f > best:
            best, bt = f, t
    return bt


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    rows = features(load_train(), truth)
    print(f'{len(rows)} queries, {sum(r["present"] for r in rows)} present\n')

    print('separation of each signal (present vs absent medians):')
    for k in ('score', 'gap', 'coarse', 'spread'):
        a = np.median([r[k] for r in rows if r['present']])
        b = np.median([r[k] for r in rows if not r['present']])
        print(f'  {k:8s} present={a:.3f} absent={b:.3f}')

    print('\nrules (leave-one-scene-out where a constant is fitted):')
    print(f"  {'fixed 0.72 on refined score':44s} "
          f"{evaluate_rule(rows, lambda tr, r: r['score'] >= .72, folds=False):.4f}")
    print(f"  {'fitted threshold on refined score':44s} "
          f"{evaluate_rule(rows, lambda tr, r: r['score'] >= fit_threshold(tr,'score')):.4f}")
    print(f"  {'fitted threshold on coarse score':44s} "
          f"{evaluate_rule(rows, lambda tr, r: r['coarse'] >= fit_threshold(tr,'coarse')):.4f}")

    # Sum of the two independently computed appearance scores.
    for r in rows:
        r['combo'] = r['score'] + .5 * r['coarse']
    print(f"  {'fitted threshold on refined+0.5*coarse':44s} "
          f"{evaluate_rule(rows, lambda tr, r: r['combo'] >= fit_threshold(tr,'combo')):.4f}")

    # A large rank-0/rank-1 gap means one location clearly won, which is itself
    # evidence the patch exists in the scene.
    def gap_rule(tr, r):
        t = fit_threshold(tr, 'score')
        return r['score'] >= t or (r['gap'] >= .25 and r['score'] >= t - .10)
    print(f"  {'refined threshold OR decisive gap':44s} {evaluate_rule(rows, gap_rule):.4f}")

    print('\nbest achievable with one fitted threshold per signal (in-sample ceiling):')
    for k in ('score', 'coarse', 'combo'):
        t = fit_threshold(rows, k)
        f = presence_f1([r['present'] for r in rows], [r[k] >= t for r in rows])
        print(f'  {k:8s} t={t:.3f} f1={f:.4f}')


if __name__ == '__main__':
    main()

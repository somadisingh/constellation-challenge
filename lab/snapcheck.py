"""Is snapping to the geometrically supported pool point better or worse than
keeping the top appearance alternative? Broken out by figure vs off-figure."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from lab.harness import experiment, ROOT


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    from lab.cache import load_train, TRAIN
    cache = load_train()
    m, preds, detail = experiment('snap-diagnostic', verbose=False)
    buckets = {}
    for n in TRAIN:
        t = truth[n].patches
        for i, tp in enumerate(t):
            if tp is None:
                continue
            base = np.array(cache[n]['refined'][i][0][:2])
            got = preds[n].patches[i]
            if got is None:
                continue
            new = np.array(got[:2])
            moved = np.linalg.norm(new - base) > .5
            if not moved:
                continue
            d_old = np.linalg.norm(base - np.array(tp[:2]))
            d_new = np.linalg.norm(new - np.array(tp[:2]))
            key = 'figure' if tp[2] == 1 else 'off-figure'
            b = buckets.setdefault(key, {'better': 0, 'worse': 0, 'neutral': 0,
                                         'gain': [], 'loss': []})
            if d_new < d_old - 1:
                b['better'] += 1; b['gain'].append((d_old, d_new))
            elif d_new > d_old + 1:
                b['worse'] += 1; b['loss'].append((d_old, d_new))
            else:
                b['neutral'] += 1
    for k, b in buckets.items():
        print(f"{k:11s} snapped better={b['better']:3d} worse={b['worse']:3d} "
              f"neutral={b['neutral']:3d}")
        for tag in ('gain', 'loss'):
            v = b[tag]
            if v:
                inside = sum(1 for o, nn in v if nn <= 12)
                print(f'    {tag:5s}: n={len(v):3d} -> within12 after={inside} '
                      f'median {np.median([o for o,_ in v]):.0f}px to '
                      f'{np.median([nn for _,nn in v]):.0f}px')


if __name__ == '__main__':
    main()

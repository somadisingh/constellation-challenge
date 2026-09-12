"""Is the auxiliary star-evidence map informative?

It scores template nodes that no query supports. That only discriminates between
competing classes if real stars are sparse enough for a wrong prediction to miss.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from lab.harness import aux_for
from lab.cache import TRAIN, ROOT


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    rng = np.random.default_rng(0)
    for n in TRAIN:
        m = aux_for(n)
        fig = np.array([p[:2] for p in truth[n].patches
                        if p is not None and p[2] == 1]).reshape(-1, 2).astype(int)
        pres = np.array([p[:2] for p in truth[n].patches
                         if p is not None]).reshape(-1, 2).astype(int)
        rand = rng.integers(0, 3000, size=(20000, 2))
        vf = m[fig[:, 1], fig[:, 0]]
        vp = m[pres[:, 1], pres[:, 0]]
        vr = m[rand[:, 1], rand[:, 0]]
        print(f'{n:9s} figure median={np.median(vf):.3f}  present median={np.median(vp):.3f}  '
              f'random median={np.median(vr):.3f}')
        print(f'{"":9s} fraction of random points above 0.9: {np.mean(vr > .9):.3f}  '
              f'above 0.99: {np.mean(vr > .99):.3f}')
        print(f'{"":9s} fraction of figure points above 0.9: {np.mean(vf > .9):.3f}')


if __name__ == '__main__':
    main()

"""Real figure geometry and candidate-noise statistics, used to parameterise the
synthetic geometry benchmark so tuning transfers."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from constellation.references import extract_patterns
from lab.cache import load_train, TRAIN, ROOT


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    pat = extract_patterns(ROOT / 'patterns')
    cache = load_train()
    print('--- figure extent and node usage ---')
    for n in TRAIN:
        t = truth[n].patches
        fig = np.array([p[:2] for p in t if p is not None and p[2] == 1])
        off = sum(1 for p in t if p is not None and p[2] == 0)
        absent = sum(1 for p in t if p is None)
        tmpl = pat[truth[n].constellation]
        print(f'  {n:9s} figure_q={len(fig):2d} off={off:2d} absent={absent:2d} '
              f'template_nodes={len(tmpl):2d} '
              f'span={np.ptp(fig,axis=0).astype(int) if len(fig) else None} '
              f'centroid={fig.mean(0).astype(int) if len(fig) else None}')

    print('\n--- accuracy of the best alternative when it is correct ---')
    errs, wrong_d = [], []
    for n in TRAIN:
        for q, tp in zip(cache[n]['refined'], truth[n].patches):
            if tp is None or not len(q):
                continue
            xy = np.array([c[:2] for c in q])
            d = np.linalg.norm(xy - np.array(tp[:2]), axis=1)
            if d.min() <= 12:
                errs.append(float(d.min()))
            wrong_d += [float(v) for v in d if v > 12]
    print(f'  correct-alternative error: n={len(errs)} median={np.median(errs):.2f} '
          f'p90={np.percentile(errs,90):.2f}')
    print(f'  wrong-alternative distance: median={np.median(wrong_d):.0f} '
          f'p10={np.percentile(wrong_d,10):.0f}')

    print('\n--- how many alternatives per query survive the ambiguity gate ---')
    for g in (.02, .03):
        sizes = []
        for n in TRAIN:
            for q in cache[n]['refined']:
                if len(q) < 2:
                    continue
                sizes.append(8 if (q[0][2] - q[1][2]) < g else 1)
        print(f'  gap={g}: mean pool per query={np.mean(sizes):.2f} '
              f'ambiguous={np.mean(np.array(sizes)>1):.2f}')


if __name__ == '__main__':
    main()

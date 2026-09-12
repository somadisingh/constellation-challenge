"""How much score is recoverable from the existing cached candidates?

Separates three loss sources per the priority list:
  1. proposal/verification failure  - no alternative within tolerance at all
  2. ranking failure               - correct alternative exists but is not rank 0
  3. presence-threshold failure    - correct rank-0 alternative dropped by threshold
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate, ScenePrediction
from lab.cache import load_train, TRAIN, ROOT


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    cache = load_train()
    print(f"{'scene':10s} {'stage':8s} {'present':>8s} {'hit@1':>7s} {'hit@20':>7s} {'medrank':>8s}")
    for stage in ('refined', 'coarse'):
        for n in TRAIN:
            t = truth[n].patches
            cands = cache[n][stage]
            hit1 = hit20 = npres = 0
            ranks = []
            for q, tp in zip(cands, t):
                if tp is None:
                    continue
                npres += 1
                d = np.linalg.norm(np.array([c[:2] for c in q]) - np.array(tp[:2]), axis=1)
                good = np.where(d <= 12)[0]
                if len(good):
                    hit20 += 1
                    ranks.append(int(good[0]))
                    if good[0] == 0:
                        hit1 += 1
            print(f'{n:10s} {stage:8s} {npres:8d} {hit1/npres:7.3f} {hit20/npres:7.3f} '
                  f'{np.median(ranks) if ranks else -1:8.1f}')

    # Oracle ceiling: pick the best of 20 refined alternatives per query, keep
    # existing presence decisions. Isolates pure ranking headroom.
    for label, chooser in [('as-is rank0', 'rank0'), ('oracle-rerank', 'oracle')]:
        preds = {}
        for n in TRAIN:
            t = truth[n].patches
            patches = []
            for q, tp in zip(cache[n]['refined'], t):
                pool = np.array([c[:2] for c in q])
                if chooser == 'oracle' and tp is not None:
                    j = int(np.argmin(np.linalg.norm(pool - np.array(tp[:2]), axis=1)))
                else:
                    j = 0
                x, y = pool[j]
                keep = q[j][2] if chooser == 'oracle' else q[0][2]
                patches.append((float(x), float(y), 1 if (tp is not None and tp[2] == 1) else 0)
                               if (q[0][2] >= .72) else None)
            preds[n] = ScenePrediction(patches, truth[n].constellation)
        m = evaluate(preds, truth)['mean']
        print(f"{label:16s} loc={m['localization']:.3f} rec={m['recovery']:.3f} "
              f"pres={m['presence']:.3f} score={m['score']:.3f}")


if __name__ == '__main__':
    main()

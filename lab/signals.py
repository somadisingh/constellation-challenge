"""Which blind signals separate the correct alternative from the other 19?

Candidate tuple layout: (x, y, score, angle, scale)
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from lab.cache import load_train, TRAIN, ROOT


def pair(refined, coarse, radius=6.):
    """For each refined alternative, the best coarse score within `radius`."""
    r = np.array([c[:2] for c in refined])
    c = np.array([x[:2] for x in coarse])
    cs = np.array([x[2] for x in coarse])
    if not len(c):
        return np.zeros(len(r))
    d = np.linalg.norm(r[:, None, :] - c[None, :, :], axis=2)
    out = []
    for i in range(len(r)):
        j = np.where(d[i] < radius)[0]
        out.append(float(cs[j].max()) if len(j) else 0.)
    return np.array(out)


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    cache = load_train()
    rows = []
    for n in TRAIN:
        t = truth[n].patches
        for qi, (rq, cq) in enumerate(zip(cache[n]['refined'], cache[n]['coarse'])):
            tp = t[qi]
            if tp is None:
                continue
            xy = np.array([c[:2] for c in rq])
            d = np.linalg.norm(xy - np.array(tp[:2]), axis=1)
            if d.min() > 12:
                continue
            sc = np.array([c[2] for c in rq])
            cons = pair(rq, cq)
            correct = int(np.argmin(d))
            rows.append(dict(scene=n, q=qi, figure=tp[2], correct=correct,
                             refined=sc, consensus=cons,
                             combo=sc + .35 * cons,
                             margin=sc - np.sort(sc)[-2] if len(sc) > 1 else 0.))
    print(f'{len(rows)} present queries with a correct alternative in the pool')
    fig = [r for r in rows if r['figure'] == 1]
    print(f'  of which {len(fig)} are figure stars')
    for key in ('refined', 'consensus', 'combo'):
        top1 = np.mean([r['correct'] == int(np.argmax(r[key])) for r in rows])
        top1f = np.mean([r['correct'] == int(np.argmax(r[key])) for r in fig])
        print(f'  rank-by-{key:10s} top1={top1:.3f}  top1(figure)={top1f:.3f}')

    # How often does the true figure-star answer sit in the pool but lose?
    lost = [r for r in rows if r['correct'] != int(np.argmax(r['refined']))]
    print(f'\n{len(lost)} ranking losses ({sum(r["figure"]==1 for r in lost)} on figure stars)')
    gaps = [float(np.max(r['refined']) - r['refined'][r['correct']]) for r in lost]
    print(f'  score gap to correct: median={np.median(gaps):.3f} max={np.max(gaps):.3f}')
    ranks = [int(np.argsort(-r['refined']).tolist().index(r['correct'])) for r in lost]
    print(f'  rank of correct alternative: {sorted(ranks)}')


if __name__ == '__main__':
    main()

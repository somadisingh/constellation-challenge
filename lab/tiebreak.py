"""Scale-preserving tie-break using a second appearance representation.

Replacing the ranking score outright was measured worse end-to-end (leave-one-scene-out
0.589 and 0.577 against 0.686), because the presence threshold, the ambiguity gate and
the pool margin are all calibrated to the incumbent score distribution.

This changes only *which location* each score value is attached to. For a query the
incumbent already calls ambiguous, the alternatives are reordered by a second
representation, then the incumbent's own descending score values are reassigned in
that new order. The score multiset per query is therefore unchanged, so presence
decisions, the rank0-rank1 gap and the pool margin are all bit-identical, and any
measured difference comes purely from choosing a better location.

Non-ambiguous queries are untouched.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.cache import TRAIN
from lab.recache import COMPONENTS
from lab.rescore import score_scene


def tiebreak_cache(cache, parts, key, gap=.03, restrict_to_ambiguous=True):
    """New cache with reordered locations for ambiguous queries only."""
    out = {}
    for n in TRAIN:
        refined = []
        for i, q in enumerate(cache[n]['refined']):
            inc = np.array([c[2] for c in q], float)
            ambiguous = len(q) > 1 and (inc[0] - inc[1]) < gap
            if not (ambiguous or not restrict_to_ambiguous) or len(q) < 2:
                refined.append([tuple(map(float, c)) for c in q]); continue
            alt = np.asarray(parts[n][key][i], float)
            order = np.argsort(-alt, kind='stable')
            values = np.sort(inc)[::-1]           # unchanged score multiset
            refined.append([(float(q[j][0]), float(q[j][1]), float(values[r]),
                             float(q[j][3]), float(q[j][4]))
                            for r, j in enumerate(order)])
        out[n] = {'refined': refined, 'coarse': cache[n]['coarse']}
    return out


def ambiguous_mask(cache, gap=.03):
    m = {}
    for n in TRAIN:
        m[n] = [len(q) > 1 and (q[0][2] - q[1][2]) < gap for q in cache[n]['refined']]
    return m


if __name__ == '__main__':
    from constellation.contracts import read_truth
    from lab.cache import load_train, ROOT
    from lab.samples import build_train
    from lab.recache import component_cache
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    cache = load_train()
    data = build_train(cache)
    parts = component_cache(cache, data)
    mask = ambiguous_mask(cache)
    n_amb = sum(sum(v) for v in mask.values())
    print(f'{n_amb} ambiguous queries of '
          f'{sum(len(v) for v in mask.values())} total\n')
    print(f"{'tie-break key':14s} {'amb figure top1':>16s} {'amb off top1':>13s} "
          f"{'amb absent unchanged':>21s}")
    for key in ['incumbent'] + list(COMPONENTS):
        rc = (cache if key == 'incumbent'
              else tiebreak_cache(cache, parts, key))
        fig, off = [], []
        for n in TRAIN:
            for i, q in enumerate(rc[n]['refined']):
                t = truth[n].patches[i]
                if t is None or not mask[n][i]:
                    continue
                d = np.linalg.norm(np.array(q[0][:2]) - np.array(t[:2]))
                (fig if t[2] == 1 else off).append(d <= 12)
        print(f'{key:14s} {np.mean(fig) if fig else float("nan"):16.3f} '
              f'{np.mean(off) if off else float("nan"):13.3f} '
              f'{"yes (multiset preserved)":>21s}')

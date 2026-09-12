"""Rank quality of each appearance-ranking variant over the fixed candidate sets.

Reported per variant:
  top1 / top3      fraction of present queries whose correct alternative ranks first
                   (or in the first three), among queries that have one at all
  figure / off     the same, split by category, because relocation and recovery
                   depend on figure queries while localization is dominated by
                   off-figure ones
  margin           median best-minus-runner-up, a calibration proxy
  absent sep       median top score on truth-absent queries; a good ranker should
                   not raise these
Queries whose correct location is absent from the pool are excluded from the rank
statistics and counted separately, so the denominator is explicit.
"""
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from lab.cache import load_train, TRAIN, ROOT
from lab.samples import build_train
from lab.rescore import VARIANTS, score_scene, category

RADIUS = 12.


def collect(cache, data, truth):
    """Per-query truth distances to every candidate, plus category."""
    rows = {}
    for n in TRAIN:
        samples, patches, counts = data[n]
        info = []
        for i, q in enumerate(cache[n]['refined']):
            t = truth[n].patches[i]
            xy = np.array([c[:2] for c in q], float)
            d = (np.linalg.norm(xy - np.array(t[:2]), axis=1) if t is not None
                 else np.full(len(q), np.inf))
            info.append(dict(dist=d, cat=category(t), n_alt=int(counts[i])))
        rows[n] = info
    return rows


def evaluate_variant(name, fn, cache, data, truth, rows):
    t0 = time.perf_counter()
    per_cat = {'figure': [], 'off-figure': []}
    margins, absent_top, missing = [], [], 0
    top3 = []
    for n in TRAIN:
        samples, patches, counts = data[n]
        scores = score_scene(fn, patches, samples, counts, cache[n]['refined'])
        for i, meta in enumerate(rows[n]):
            s = scores[i]
            if meta['cat'] == 'absent':
                absent_top.append(float(s.max()))
                continue
            good = np.where(meta['dist'] <= RADIUS)[0]
            if not len(good):
                missing += 1
                continue
            order = np.argsort(-s, kind='stable')
            rank = int(np.where(np.isin(order, good))[0][0])
            per_cat[meta['cat']].append(rank == 0)
            top3.append(rank < 3)
            srt = np.sort(s)[::-1]
            margins.append(float(srt[0] - srt[1]) if len(srt) > 1 else 0.)
    fig = np.mean(per_cat['figure']); off = np.mean(per_cat['off-figure'])
    both = np.mean(per_cat['figure'] + per_cat['off-figure'])
    return dict(name=name, top1=both, top3=np.mean(top3), figure=fig, off=off,
                margin=float(np.median(margins)),
                absent=float(np.median(absent_top)), missing=missing,
                seconds=time.perf_counter() - t0,
                n_fig=len(per_cat['figure']), n_off=len(per_cat['off-figure']))


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    cache = load_train()
    data = build_train(cache)
    rows = collect(cache, data, truth)
    n_missing = sum(1 for n in TRAIN for m in rows[n]
                    if m['cat'] != 'absent' and not (m['dist'] <= RADIUS).any())
    n_present = sum(1 for n in TRAIN for m in rows[n] if m['cat'] != 'absent')
    print(f'{n_present} present queries; {n_missing} have no candidate within '
          f'{RADIUS:.0f}px and are excluded from rank statistics '
          f'(denominator {n_present - n_missing})\n')
    print(f"{'variant':26s} {'top1':>6s} {'top3':>6s} {'figure':>7s} {'off':>6s} "
          f"{'margin':>7s} {'absent':>7s} {'sec':>5s}")
    results = []
    for name, fn in VARIANTS.items():
        r = evaluate_variant(name, fn, cache, data, truth, rows)
        results.append(r)
        print(f"{r['name']:26s} {r['top1']:6.3f} {r['top3']:6.3f} {r['figure']:7.3f} "
              f"{r['off']:6.3f} {r['margin']:7.3f} {r['absent']:7.3f} {r['seconds']:5.1f}")
    base = next(r for r in results if r['name'].startswith('incumbent'))
    print(f"\nfigure-query top1 is the lever for relocation and recovery; "
          f"incumbent {base['figure']:.3f} on n={base['n_fig']}, "
          f"off-figure {base['off']:.3f} on n={base['n_off']}")
    best = max(results, key=lambda r: r['top1'])
    print(f"best overall top1: {best['name']} {best['top1']:.3f} "
          f"vs incumbent {base['top1']:.3f}")


if __name__ == '__main__':
    main()

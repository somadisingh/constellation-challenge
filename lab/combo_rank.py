"""Combine the incumbent score with the representations that help figure queries.

The incumbent leads on off-figure queries (top1 0.902) and trails on figure queries
(0.609). Figure queries drive relocation, recovery and identification; off-figure
queries dominate the localization denominator. A combination is therefore the
natural test, not a replacement.

Weights are swept here at rank level only. Anything carried forward is refitted
under leave-one-scene-out in the end-to-end script.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from lab.cache import load_train, TRAIN, ROOT
from lab.samples import build_train
from lab.rescore import (ncc, rep_bgsub, rep_lcn, rep_dog, centre_annulus,
                         multiblur_ncc, score_scene, category)
from lab.rank_report import collect, RADIUS

PARTS = {
    'annulus': lambda p, s: centre_annulus(p, s, rep_bgsub, 'annulus'),
    'ca_min': lambda p, s: centre_annulus(p, s, rep_bgsub, 'min'),
    'lcn': lambda p, s: ncc(p, s, rep_lcn, 'disc14'),
    'dog32': lambda p, s: ncc(p, s, rep_dog, 'full'),
    'full32': lambda p, s: ncc(p, s, rep_bgsub, 'full'),
    'multiblur': lambda p, s: multiblur_ncc(p, s, rep_bgsub, 'disc14'),
}


def part_scores(cache, data):
    """Cache every component score once; combinations are then free."""
    out = {}
    for n in TRAIN:
        samples, patches, counts = data[n]
        d = {'incumbent': score_scene(None, patches, samples, counts,
                                      cache[n]['refined'])}
        for key, fn in PARTS.items():
            d[key] = score_scene(fn, patches, samples, counts, cache[n]['refined'])
        out[n] = d
    return out


def rank_stats(parts, rows, combine):
    per_cat = {'figure': [], 'off-figure': []}
    margins, absent = [], []
    for n in TRAIN:
        for i, meta in enumerate(rows[n]):
            s = combine({k: v[i] for k, v in parts[n].items()})
            if meta['cat'] == 'absent':
                absent.append(float(s.max())); continue
            good = np.where(meta['dist'] <= RADIUS)[0]
            if not len(good):
                continue
            order = np.argsort(-s, kind='stable')
            per_cat[meta['cat']].append(int(np.where(np.isin(order, good))[0][0]) == 0)
            srt = np.sort(s)[::-1]
            margins.append(float(srt[0] - srt[1]))
    fig, off = np.mean(per_cat['figure']), np.mean(per_cat['off-figure'])
    both = np.mean(per_cat['figure'] + per_cat['off-figure'])
    return both, fig, off, float(np.median(margins)), float(np.median(absent))


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    cache = load_train()
    data = build_train(cache)
    rows = collect(cache, data, truth)
    parts = part_scores(cache, data)

    print(f"{'combination':34s} {'top1':>6s} {'figure':>7s} {'off':>6s} "
          f"{'margin':>7s} {'absent':>7s}")

    def show(label, fn):
        b, f, o, m, a = rank_stats(parts, rows, fn)
        print(f'{label:34s} {b:6.3f} {f:7.3f} {o:6.3f} {m:7.3f} {a:7.3f}')
        return b, f, o

    show('incumbent alone', lambda d: d['incumbent'])
    for key in PARTS:
        show(f'{key} alone', lambda d, k=key: d[k])
    print()
    for key in ('annulus', 'ca_min', 'lcn', 'dog32', 'full32', 'multiblur'):
        for w in (.25, .5, 1., 2.):
            show(f'incumbent + {w:g}*{key}',
                 lambda d, k=key, w=w: d['incumbent'] + w * d[k])
    print()
    show('mean(incumbent, full32, lcn)',
         lambda d: (d['incumbent'] + d['full32'] + d['lcn']) / 3)
    show('incumbent + 0.5*annulus + 0.5*lcn',
         lambda d: d['incumbent'] + .5 * d['annulus'] + .5 * d['lcn'])
    show('min(incumbent, annulus)',
         lambda d: np.minimum(d['incumbent'], d['annulus']))
    show('incumbent * relu(annulus)',
         lambda d: d['incumbent'] * np.maximum(d['annulus'], 0.))


if __name__ == '__main__':
    main()

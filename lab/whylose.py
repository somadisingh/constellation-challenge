"""When identification is wrong, is the true class unfound or merely outscored?

Distinguishes:
  no-fit    true class never reached min_support -> hypothesis generation problem
  outscored true class verified but ranked below another -> scoring problem
"""
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.references import extract_patterns
from constellation.joint import recognize_joint
from lab.cache import ROOT
from lab.synth import dataset

PATTERNS = None


def _init():
    global PATTERNS
    PATTERNS = extract_patterns(ROOT / 'patterns')


def _one(args):
    scene, cfg = args
    name, _, diag = recognize_joint(scene['alternatives'], PATTERNS, **cfg)
    hyps = {h['name']: h for h in diag.get('hypotheses', [])}
    true = scene['name']
    win = hyps.get(name, {})
    t = hyps.get(true)
    if name == true:
        return ('correct', None, scene['n_figure'], scene['template_nodes'])
    if t is None:
        return ('true-class-not-in-top8', win.get('support'), scene['n_figure'],
                scene['template_nodes'])
    if t.get('support', 0) < 4:
        return ('no-fit', win.get('support'), scene['n_figure'], scene['template_nodes'])
    return ('outscored', (round(t['score'], 2), round(win.get('score', 0), 2),
                          t['support'], win.get('support')),
            scene['n_figure'], scene['template_nodes'])


def main():
    _init()
    cfg = dict(models=('affine',), tolerance=12., cap=30000)
    scenes = dataset(PATTERNS, 96, 17)
    with ProcessPoolExecutor(max_workers=8, initializer=_init) as pool:
        res = list(pool.map(_one, [(s, cfg) for s in scenes], chunksize=1))
    print(Counter(r[0] for r in res).most_common())
    print('\naccuracy by issued figure-query count:')
    by = {}
    for kind, _, nfig, nodes in res:
        by.setdefault(nfig, [0, 0])
        by[nfig][0] += kind == 'correct'; by[nfig][1] += 1
    for k in sorted(by):
        c, n = by[k]
        print(f'  {k:2d} issued queries: {c}/{n} = {c/n:.2f}')
    print('\naccuracy by template node count:')
    by = {}
    for kind, _, nfig, nodes in res:
        b = 'small 4-6' if nodes <= 6 else ('mid 7-12' if nodes <= 12 else 'large 13+')
        by.setdefault(b, [0, 0])
        by[b][0] += kind == 'correct'; by[b][1] += 1
    for k, (c, n) in sorted(by.items()):
        print(f'  {k:10s}: {c}/{n} = {c/n:.2f}')
    print('\nsample outscored cases (true_score, win_score, true_sup, win_sup):')
    for kind, info, _, _ in res:
        if kind == 'outscored':
            print('  ', info)


if __name__ == '__main__':
    main()

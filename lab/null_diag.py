"""Is the chance-fit null calibrated? Diagnose before changing it.

The verification score is

    surprise = -log10 P[Binomial(len(p)-3, fraction) >= support - 4]
    fraction = min(.8, n_groups * pi * tolerance^2 / 9e6)

Two suspected faults:

1. `fraction` counts query *groups*, but with widened pools a template node can match
   any point in the pool, and the pool holds several points per ambiguous query. If the
   chance density is understated, every class looks more surprising than it is, and the
   understatement is not uniform across scenes.
2. The binomial treats node matches as independent draws with replacement, while the
   assignment is one-to-one. Sampling without replacement is hypergeometric, and the
   discrepancy grows with support.

If the null is right, the score of a *wrong* class should have no systematic relation to
that class's node count. This measures exactly that, over the synthetic benchmark, using
a seed not used for any earlier tuning.
"""
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.references import extract_patterns
from constellation.joint import recognize_joint
from lab.cache import ROOT
from lab.synth import dataset

SEED = 2027          # fresh; 17 and 99 are historical
PATTERNS = None
CFG = dict(models=('affine',), gap=.03, score_mode='binom', tolerance=18.,
           cap=80000, top_k=8, quad_share=0., diag_top=48)


def _init():
    global PATTERNS
    PATTERNS = extract_patterns(ROOT / 'patterns')


def _one(scene):
    name, _, diag = recognize_joint(scene['alternatives'], PATTERNS, **CFG)
    rows = []
    for h in diag.get('hypotheses', []):
        if h.get('support', 0) < 4:
            continue
        rows.append({'name': h['name'], 'nodes': len(PATTERNS[h['name']]),
                     'support': h['support'], 'score': h['score'],
                     'coverage': h.get('coverage'), 'residual': h.get('mean_residual'),
                     'is_true': h['name'] == scene['name']})
    return {'true': scene['name'], 'picked': name, 'pool': diag.get('pool_size'),
            'rows': rows, 'n_figure': scene['n_figure']}


def main():
    _init()
    scenes = dataset(PATTERNS, 192, SEED, 7)
    with ProcessPoolExecutor(max_workers=8, initializer=_init) as pool:
        res = list(pool.map(_one, scenes, chunksize=1))
    acc = float(np.mean([r['picked'] == r['true'] for r in res]))
    print(f'identification on fresh seed {SEED}: {acc:.3f} '
          f'+-{np.sqrt(acc*(1-acc)/len(res)):.3f} (n={len(res)})')
    verified = [len(r['rows']) for r in res]
    print(f'verified classes per scene: median={np.median(verified):.0f} '
          f'max={max(verified)}  pool size median='
          f'{np.median([r["pool"] for r in res if r["pool"]]):.0f}')

    wrong = [h for r in res for h in r['rows'] if not h['is_true']]
    true = [h for r in res for h in r['rows'] if h['is_true']]
    print(f'\n{len(wrong)} wrong-class fits, {len(true)} true-class fits')

    print('\nIf the null were calibrated, wrong-class score would not track node count.')
    print(f"{'nodes':>10s} {'n':>6s} {'score med':>10s} {'support med':>12s} "
          f"{'coverage med':>13s}")
    buckets = defaultdict(list)
    for h in wrong:
        b = ('4-6' if h['nodes'] <= 6 else '7-9' if h['nodes'] <= 9
             else '10-13' if h['nodes'] <= 13 else '14-18' if h['nodes'] <= 18
             else '19+')
        buckets[b].append(h)
    for b in ('4-6', '7-9', '10-13', '14-18', '19+'):
        v = buckets.get(b, [])
        if not v:
            continue
        print(f'{b:>10s} {len(v):6d} {np.median([h["score"] for h in v]):10.2f} '
              f'{np.median([h["support"] for h in v]):12.1f} '
              f'{np.median([h["coverage"] for h in v]):13.3f}')
    sizes = np.array([h['nodes'] for h in wrong], float)
    scores = np.array([h['score'] for h in wrong], float)
    r = float(np.corrcoef(sizes, scores)[0, 1])
    print(f'\ncorrelation of wrong-class score with node count: {r:+.3f}')
    print('A positive value means large references are systematically rewarded and the '
          'size correction is too weak; negative means over-penalised.')

    print('\ntrue-class fits for comparison:')
    print(f"  score med={np.median([h['score'] for h in true]):.2f} "
          f"support med={np.median([h['support'] for h in true]):.1f} "
          f"coverage med={np.median([h['coverage'] for h in true]):.3f} "
          f"residual med={np.median([h['residual'] for h in true]):.2f}")
    print(f"  wrong: score med={np.median([h['score'] for h in wrong]):.2f} "
          f"support med={np.median([h['support'] for h in wrong]):.1f} "
          f"coverage med={np.median([h['coverage'] for h in wrong]):.3f} "
          f"residual med={np.median([h['residual'] for h in wrong]):.2f}")

    print('\nseparability of single features (AUC, true vs wrong within a scene):')
    for key in ('score', 'support', 'coverage', 'residual'):
        wins = ties = tot = 0
        for r_ in res:
            t = [h for h in r_['rows'] if h['is_true']]
            w = [h for h in r_['rows'] if not h['is_true']]
            if not t or not w:
                continue
            tv = t[0][key]
            for h in w:
                tot += 1
                sign = -1 if key == 'residual' else 1
                if sign * tv > sign * h[key]:
                    wins += 1
                elif tv == h[key]:
                    ties += 1
        if tot:
            print(f'  {key:9s} AUC={(wins + .5 * ties) / tot:.3f}')
    (ROOT / 'outputs/lab/null_diag.json').write_text(json.dumps(
        {'seed': SEED, 'accuracy': acc, 'size_score_correlation': r,
         'n_wrong': len(wrong), 'n_true': len(true)}, indent=2, default=float))


if __name__ == '__main__':
    main()

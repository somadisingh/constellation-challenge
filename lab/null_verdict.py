"""Paired test of the null corrections, plus real-scene corroboration.

Accuracy differences of a few points on n=192 are inside the sampling error, so the
comparison is done paired per scene (McNemar counts) over both synthetic seeds pooled,
and then checked on the three labelled scenes.
"""
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate, ScenePrediction
from constellation.references import extract_patterns
from constellation.joint import recognize_joint
from constellation.finalize import COARSE_CUTOFF
from constellation.slate import slates_from_lists
from lab.cache import load_train, TRAIN, ROOT
from lab.harness import aux_for
from lab.synth import dataset
from lab.null_sweep import BASE, MODES

PATTERNS = None


def _init():
    global PATTERNS
    PATTERNS = extract_patterns(ROOT / 'patterns')


def _one(args):
    scene, mode = args
    name, _, _ = recognize_joint(scene['alternatives'], PATTERNS, null_mode=mode, **BASE)
    return name == scene['name'], scene['template_nodes']


def paired():
    _init()
    scenes = []
    for seed in (2027, 4099):
        scenes += dataset(PATTERNS, 192, seed, 7)
    print(f'pooled synthetic scenes: n={len(scenes)}')
    correct = {}
    for mode in MODES:
        with ProcessPoolExecutor(max_workers=8, initializer=_init) as pool:
            correct[mode] = list(pool.map(_one, [(s, mode) for s in scenes],
                                          chunksize=1))
    base = [c for c, _ in correct['groups']]
    print(f"\n{'mode':22s} {'acc':>6s} {'gained':>7s} {'lost':>6s} "
          f"{'net':>5s} {'p(two-sided)':>13s}")
    out = {}
    for mode in MODES:
        cur = [c for c, _ in correct[mode]]
        gained = sum(1 for a, b in zip(base, cur) if not a and b)
        lost = sum(1 for a, b in zip(base, cur) if a and not b)
        n = gained + lost
        # Exact binomial two-sided p under H0: a flip is equally likely either way.
        from math import comb
        p = (1. if n == 0 else
             min(1., 2 * sum(comb(n, k) for k in range(min(gained, lost) + 1)) / 2 ** n))
        acc = float(np.mean(cur))
        out[mode] = {'accuracy': acc, 'gained': gained, 'lost': lost, 'p': p}
        print(f'{mode:22s} {acc:6.3f} {gained:7d} {lost:6d} '
              f'{gained-lost:+5d} {p:13.3f}')
    print(f"\ngroups baseline accuracy {np.mean(base):.3f}")
    return out


def real_scenes():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    cache = load_train()
    aux = {n: aux_for(n) for n in TRAIN}
    print(f"\n{'mode':22s} {'total':>7s} {'id':>5s}  classes")
    out = {}
    for mode in MODES:
        preds = {}
        for n in TRAIN:
            competing = []
            for label, lists, cutoff in [('refined', cache[n]['refined'], .72),
                                         ('coarse', cache[n]['coarse'], COARSE_CUTOFF)]:
                sl = slates_from_lists(lists)
                ids = [i for i, s in enumerate(sl) if len(s)
                       and s.presence_score >= cutoff]
                if len(ids) < 3:
                    continue
                name, chosen, d = recognize_joint([sl[i] for i in ids], patterns,
                                                  null_mode=mode, aux_weight=3.,
                                                  shear_penalty=2.,
                                                  auxiliary_map=aux[n], **BASE)
                best = d['hypotheses'][0] if d.get('hypotheses') else {'score': -1e9}
                competing.append((best.get('score', -1e9), name,
                                  {ids[k]: v for k, v in chosen.items()}, d, label))
            competing.sort(key=lambda x: (-x[0], x[1], x[4]))
            _, name, chosen, diag, _ = competing[0]
            nodes = (np.array(diag['hypotheses'][0].get('nodes', [])).reshape(-1, 2)
                     if diag.get('hypotheses') else np.empty((0, 2)))
            sl = slates_from_lists(cache[n]['refined'])
            patches = []
            for i, s in enumerate(sl):
                if not len(s) or s.presence_score < .72:
                    patches.append(None); continue
                x, y = chosen[i] if i in chosen else s.xy[s.report_index]
                member = int(len(nodes) > 0
                             and np.linalg.norm(nodes - [x, y], axis=1).min() < 18.)
                patches.append((float(x), float(y), member))
            preds[n] = ScenePrediction(patches, name)
        m = evaluate(preds, truth)
        out[mode] = {'mean': m['mean'], 'worst': m['worst_score'],
                     'classes': {n: preds[n].constellation for n in TRAIN}}
        print(f"{mode:22s} {m['mean']['score']:7.4f} "
              f"{m['mean']['identification']:5.3f}  "
              + ' '.join(preds[n].constellation for n in TRAIN))
    return out


def main():
    p = paired()
    r = real_scenes()
    print('\nreference: production total 0.72869 with null_mode=groups')
    print('Node counts of the three labelled references: pisces 18, scorpius 13, '
          'taurus 12 - all large. The decorrelation trades large-template accuracy for '
          'small, so the real class-size distribution decides its sign, and that '
          'distribution is unknown for the unlabelled scenes.')
    (ROOT / 'outputs/lab/null_verdict.json').write_text(
        json.dumps({'paired_synthetic': p, 'real_scenes': r}, indent=2, default=float))


if __name__ == '__main__':
    main()

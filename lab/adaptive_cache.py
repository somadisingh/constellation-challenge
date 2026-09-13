"""Load candidate lists produced by a chosen verification policy from a run output.

Keyed by the run directory, so caches never mix policies.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.cache import TRAIN, ROOT


def load_run(directory, scenes=TRAIN):
    out = {}
    for n in scenes:
        d = json.load(open(ROOT / directory / f'{n}.json'))
        qs = d['diagnostics']['queries']
        out[n] = {'refined': [q['candidates'] for q in qs],
                  'coarse': [q['coarse_candidates'] for q in qs]}
    return out


def score_summary(cache, label):
    top, gaps = [], []
    for n in TRAIN:
        for q in cache[n]['refined']:
            if not len(q):
                continue
            s = np.array([c[2] for c in q], float)
            top.append(s[0])
            if len(s) > 1:
                gaps.append(s[0] - s[1])
    top, gaps = np.array(top), np.array(gaps)
    print(f'{label:22s} top score: median={np.median(top):.3f} '
          f'p10={np.percentile(top,10):.3f} p90={np.percentile(top,90):.3f} | '
          f'top-two gap: median={np.median(gaps):.4f} '
          f'p25={np.percentile(gaps,25):.4f} p75={np.percentile(gaps,75):.4f}')
    return top, gaps


if __name__ == '__main__':
    from lab.cache import load_train
    score_summary(load_train(), 'baseline (fixed12)')
    score_summary(load_run('outputs/lab/adaptive_train'), 'adaptive radius')

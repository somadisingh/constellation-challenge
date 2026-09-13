"""Evaluate full-pool verification variants on all 116 labelled queries.

Selection is on retention and discrimination, not on the three-scene total, which
cannot resolve changes of this size (one identification flip is worth 34 single-query
localization repairs).
"""
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from lab.cache import TRAIN, ROOT
from lab.samples import load_scene_patches
from lab.proposals import build_train as build_proposals
from lab.verify_full import REPS, VARIANTS, score_pool, select


def evaluate(label, rep, policy, stride, channels, truth, props, patches, scenes):
    t0 = time.perf_counter()
    rows = []
    f = REPS[rep]
    for n in TRAIN:
        scene_rep = f(scenes[n])
        for i, prop in enumerate(props[n]):
            t = truth[n].patches[i]
            p = np.asarray(prop, float)
            s, _ = score_pool(scene_rep, f(patches[n][i].astype(np.float32)), p,
                              policy, stride, channels)
            rec = {'scene': n, 'query': i,
                   'cat': 'absent' if t is None else
                          ('figure' if t[2] == 1 else 'off-figure'),
                   'top': float(s.max())}
            if t is not None:
                d = np.linalg.norm(p[:, :2] - np.array(t[:2]), axis=1)
                good = np.where(d <= 12)[0]
                if len(good):
                    order = np.argsort(-s)
                    rec['rank'] = int(np.where(np.isin(order, good))[0][0])
                    rec['best_good'] = float(s[good].max())
                for keep in (20, 40):
                    ch = select(p, s, keep)
                    dd = np.linalg.norm(p[ch, :2] - np.array(t[:2]), axis=1)
                    rec[f'hit12@{keep}'] = bool((dd <= 12).any())
                    rec[f'hit4@{keep}'] = bool((dd <= 4).any())
            rows.append(rec)
    secs = time.perf_counter() - t0
    pres = [r for r in rows if r['cat'] != 'absent']
    absent = [r for r in rows if r['cat'] == 'absent']
    have = [r for r in pres if 'rank' in r]
    out = {'label': label, 'seconds': secs, 'n_present': len(pres),
           'in_pool': len(have),
           'recall12@20': float(np.mean([r['hit12@20'] for r in pres])),
           'recall12@40': float(np.mean([r['hit12@40'] for r in pres])),
           'recall4@20': float(np.mean([r['hit4@20'] for r in pres])),
           'median_rank': float(np.median([r['rank'] for r in have])),
           'rank_le20': float(np.mean([r['rank'] < 20 for r in have])),
           'absent_top_median': float(np.median([r['top'] for r in absent])),
           'present_top_median': float(np.median([r['top'] for r in pres])),
           }
    for cat in ('figure', 'off-figure'):
        sub = [r for r in pres if r['cat'] == cat]
        out[f'recall12@20 {cat}'] = float(np.mean([r['hit12@20'] for r in sub]))
    out['separation'] = out['present_top_median'] - out['absent_top_median']
    return out, rows


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    props = build_proposals()
    scenes = {n: cv2.imread(str(ROOT / 'train' / n / f'{n}_image.png'), 0).astype(np.float32)
              for n in TRAIN}
    patches = {n: load_scene_patches(n, 'train', len(props[n])) for n in TRAIN}

    print(f"{'variant':32s} {'r12@20':>7s} {'r12@40':>7s} {'r4@20':>6s} "
          f"{'medRank':>8s} {'rank<20':>8s} {'fig':>6s} {'off':>6s} "
          f"{'sep':>6s} {'sec':>5s}")
    results, allrows = [], {}
    for label, rep, policy, stride, channels in VARIANTS:
        out, rows = evaluate(label, rep, policy, stride, channels, truth, props,
                             patches, scenes)
        results.append(out); allrows[label] = rows
        print(f"{label:32s} {out['recall12@20']:7.3f} {out['recall12@40']:7.3f} "
              f"{out['recall4@20']:6.3f} {out['median_rank']:8.1f} "
              f"{out['rank_le20']:8.3f} {out['recall12@20 figure']:6.3f} "
              f"{out['recall12@20 off-figure']:6.3f} {out['separation']:6.3f} "
              f"{out['seconds']:5.0f}", flush=True)

    base = results[0]
    print(f"\nbaseline recall12@20 {base['recall12@20']:.3f}; "
          f"present queries {base['n_present']}, correct neighbourhood in pool "
          f"{base['in_pool']}")
    best = max(results, key=lambda r: r['recall12@20'])
    print(f"best retention: {best['label']} at {best['recall12@20']:.3f}")
    (ROOT / 'outputs/lab/verify_full.json').write_text(
        json.dumps({'summary': results}, indent=2, default=float))
    # Per-query ranks for the seven, under every variant, for targeted diagnosis.
    hard = [(r['scene'], r['query']) for r in allrows[VARIANTS[0][0]]
            if r['cat'] != 'absent' and not r.get('hit12@20', True)]
    print(f'\nranks for the {len(hard)} baseline failures:')
    for label in allrows:
        m = {(r['scene'], r['query']): r.get('rank') for r in allrows[label]}
        print(f'  {label:32s} {[m.get(h) for h in hard]}')
    (ROOT / 'outputs/lab/verify_full_hard.json').write_text(
        json.dumps({'hard': [list(h) for h in hard],
                    'ranks': {k: [next((r.get('rank') for r in v
                                        if (r['scene'], r['query']) == h), None)
                                  for h in hard] for k, v in allrows.items()}},
                   indent=2, default=float))


if __name__ == '__main__':
    main()

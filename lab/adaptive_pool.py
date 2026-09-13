"""Widen pool eligibility for the adaptive verifier.

Observed mismatch: with the adaptive verifier every figure query retains a candidate
within 12px (figure recall 1.000, rank<20 0.944), yet figure localization falls to
0.500-0.538 against the baseline's 0.577. Two facts explain it together:

  * the correct candidate is retained but not ranked first, and
  * better scores make queries *less* ambiguous, so the gap gate stops offering their
    alternatives to geometry, which is what previously relocated them.

So the baseline's high recovery depended partly on ambiguity that better appearance
evidence removes. The hypothesis tested here is that with the correct location now
reliably present in the retained set, pool eligibility should be governed less by the
score gap and more by simply admitting alternatives for geometry to arbitrate.

Large gaps were tested before and were harmful, but that was with the old candidate
sets where the correct location was often absent. Different regime, so retested.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate, ScenePrediction, reward
from constellation.references import extract_patterns
from lab.cache import load_train, TRAIN, ROOT
from lab.harness import run_stage, aux_for, DEFAULT
from lab.adaptive_cache import load_run

GRID = [(th, gp, tk)
        for th in (.70, .72, .74)
        for gp in (.05, .12, .25, 1.01)
        for tk in (8, 12)]


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    aux = {n: aux_for(n) for n in TRAIN}
    caches = {'adaptive': load_run('outputs/lab/adaptive_train'),
              'baseline': load_train()}

    print(f"{'cache':9s} {'th':>5s} {'gap':>5s} {'topk':>5s} {'total':>7s} {'worst':>6s} "
          f"{'pres':>6s} {'loc':>6s} {'fig':>6s} {'off':>6s} {'rec':>6s} {'id':>5s}")
    rows = {}
    for label, cache in caches.items():
        grid = GRID if label == 'adaptive' else [(.72, .03, 8), (.72, 1.01, 8)]
        for th, gp, tk in grid:
            cfg = dict(DEFAULT); cfg.update(threshold=th, gap=gp, top_k=tk)
            preds = {}
            for n in TRAIN:
                name, patches, _ = run_stage(cache[n], patterns, aux[n], cfg)
                preds[n] = ScenePrediction(patches, name)
            m = evaluate(preds, truth)
            cat = {}
            for c, want in (('figure', 1), ('off-figure', 0)):
                v = [0. if p is None else float(reward(np.linalg.norm(
                    np.array(t[:2]) - np.array(p[:2]))))
                    for n in TRAIN for p, t in zip(preds[n].patches, truth[n].patches)
                    if t is not None and t[2] == want]
                cat[c] = float(np.mean(v))
            rows[(label, th, gp, tk)] = (m, cat,
                                         {n: preds[n].constellation for n in TRAIN})
            print(f"{label:9s} {th:5.2f} {gp:5.2f} {tk:5d} {m['mean']['score']:7.4f} "
                  f"{m['worst_score']:6.3f} {m['mean']['presence']:6.3f} "
                  f"{m['mean']['localization']:6.3f} {cat['figure']:6.3f} "
                  f"{cat['off-figure']:6.3f} {m['mean']['recovery']:6.3f} "
                  f"{m['mean']['identification']:5.3f}", flush=True)

    ad = {k: v for k, v in rows.items() if k[0] == 'adaptive'}
    best = max(ad, key=lambda k: ad[k][0]['mean']['score'])
    print(f'\nbest adaptive: {best[1:]} total={ad[best][0]["mean"]["score"]:.4f} '
          f'classes={ad[best][2]}')
    scenes = {k: {n: v[0]["scenes"][n]["score"] for n in TRAIN} for k, v in ad.items()}
    held = {}
    for h in TRAIN:
        dev = [n for n in TRAIN if n != h]
        pick = max(ad, key=lambda k: np.mean([scenes[k][n] for n in dev]))
        held[h] = (pick[1:], scenes[pick][h])
    print(f'leave-one-scene-out over this grid: '
          f'{np.mean([v for _, v in held.values()]):.4f}  {held}')
    print('Baseline: in-sample 0.72869, threshold-only leave-one-out 0.70152.')
    (ROOT / 'outputs/lab/adaptive_pool.json').write_text(json.dumps(
        {f'{k}': {'mean': v[0]['mean'], 'worst': v[0]['worst_score'],
                  'category': v[1], 'classes': v[2]} for k, v in rows.items()},
        indent=2, default=float))


if __name__ == '__main__':
    main()

"""Push recovery on the adaptive + two-pass configuration.

Best so far: adaptive candidates with a geometry-guided second pass at node radius 18
and three leading hypotheses gives total .7185, presence .737, figure localization .615
(the best figure figure of any configuration, against the baseline's .577), but recovery
.822 against the baseline's .878. Recovery is the remaining deficit, and it responds to
how many alternatives reach the pool, so `top_k` and the gap are swept jointly with the
second pass active.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from constellation.references import extract_patterns
from lab.cache import TRAIN, ROOT
from lab.harness import aux_for
from lab.adaptive_cache import load_run
import lab.twopass_experiment as TP


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    aux = {n: aux_for(n) for n in TRAIN}
    cache = load_run('outputs/lab/adaptive_train')

    print(f"{'topk':>5s} {'gap':>5s} {'th':>5s} {'total':>7s} {'worst':>6s} "
          f"{'pres':>6s} {'loc':>6s} {'fig':>6s} {'off':>6s} {'rec':>6s} {'id':>5s}"
          f"  classes")
    out, scenes = {}, {}
    for tk in (8, 12, 16):
        for gap in (.03, .05):
            for th in (.72, .74):
                TP.KW['top_k'] = tk
                m, cat, preds, _ = TP.score(cache, patterns, aux, truth, th, gap,
                                            True, 18., 3)
                key = (tk, gap, th)
                out[key] = {'mean': m['mean'], 'worst': m['worst_score'],
                            'category': cat,
                            'classes': {n: preds[n].constellation for n in TRAIN}}
                scenes[key] = {n: m['scenes'][n]['score'] for n in TRAIN}
                print(f"{tk:5d} {gap:5.2f} {th:5.2f} {m['mean']['score']:7.4f} "
                      f"{m['worst_score']:6.3f} {m['mean']['presence']:6.3f} "
                      f"{m['mean']['localization']:6.3f} {cat['figure']:6.3f} "
                      f"{cat['off-figure']:6.3f} {m['mean']['recovery']:6.3f} "
                      f"{m['mean']['identification']:5.3f}  "
                      + ' '.join(preds[n].constellation[:9] for n in TRAIN), flush=True)
    TP.KW['top_k'] = 8

    best = max(out, key=lambda k: out[k]['mean']['score'])
    print(f"\nbest: top_k={best[0]} gap={best[1]} threshold={best[2]} "
          f"total={out[best]['mean']['score']:.4f} worst={out[best]['worst']:.3f}")
    held = {}
    for h in TRAIN:
        dev = [n for n in TRAIN if n != h]
        pick = max(out, key=lambda k: np.mean([scenes[k][n] for n in dev]))
        held[h] = (pick, scenes[pick][h])
    print(f"leave-one-scene-out over this grid: "
          f"{np.mean([v for _, v in held.values()]):.4f}")
    for h, (pick, sc) in held.items():
        print(f'  held out {h:9s} picked {pick} -> {sc:.4f}')
    print('Baseline: in-sample 0.72869, threshold-only leave-one-out 0.70152.')
    print('All three scenes influenced development; calibration checks only.')
    (ROOT / 'outputs/lab/twopass_push.json').write_text(json.dumps(
        {str(k): v for k, v in out.items()}, indent=2, default=float))


if __name__ == '__main__':
    main()

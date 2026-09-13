"""Recalibrate presence threshold and ambiguity gate for the adaptive verifier.

The upstream change was established on its own metric first (recall@12 after selection
0.901 -> 0.972, recall@4 0.873 -> 0.958, figure recall 1.000, present-minus-absent
separation 0.074 -> 0.117), so recalibrating downstream is warranted here rather than
being a way to rescue a change that did not stand on its own.

Reported in-sample and under leave-one-scene-out over the same grid. All three scenes
influenced development, so both are development/calibration checks.
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

THRESHOLDS = (.62, .66, .70, .72, .74, .78)
GAPS = (.02, .03, .05, .08)


def per_component(preds, truth):
    m = evaluate(preds, truth)
    cat = {}
    for c, want in (('figure', 1), ('off-figure', 0)):
        v = [0. if p is None else float(reward(np.linalg.norm(
            np.array(t[:2]) - np.array(p[:2]))))
            for n in TRAIN for p, t in zip(preds[n].patches, truth[n].patches)
            if t is not None and t[2] == want]
        cat[c] = float(np.mean(v))
    return m, cat


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    aux = {n: aux_for(n) for n in TRAIN}
    caches = {'baseline fixed12': load_train(),
              'adaptive radius': load_run('outputs/lab/adaptive_train')}

    print('per-scene components at the shipped calibration (0.72 / 0.03):')
    for label, cache in caches.items():
        preds = {}
        for n in TRAIN:
            cfg = dict(DEFAULT)
            name, patches, _ = run_stage(cache[n], patterns, aux[n], cfg)
            preds[n] = ScenePrediction(patches, name)
        m, cat = per_component(preds, truth)
        print(f'  {label:18s} total={m["mean"]["score"]:.4f}')
        for n in TRAIN:
            s = m['scenes'][n]
            print(f'     {n:9s} pres={s["presence"]:.3f} loc={s["localization"]:.3f} '
                  f'rec={s["recovery"]:.3f} id={s["identification"]:.0f} '
                  f'-> {s["score"]:.4f}  ({preds[n].constellation})')

    print(f'\nrecalibration grid for the adaptive verifier '
          f'({len(THRESHOLDS)}x{len(GAPS)} configurations)')
    print(f"{'th':>5s} {'gap':>5s} {'total':>7s} {'worst':>6s} {'pres':>6s} "
          f"{'loc':>6s} {'fig':>6s} {'off':>6s} {'rec':>6s} {'id':>5s}")
    grid, scenes_grid = {}, {}
    cache = caches['adaptive radius']
    for th in THRESHOLDS:
        for gp in GAPS:
            cfg = dict(DEFAULT); cfg.update(threshold=th, gap=gp)
            preds = {}
            for n in TRAIN:
                name, patches, _ = run_stage(cache[n], patterns, aux[n], cfg)
                preds[n] = ScenePrediction(patches, name)
            m, cat = per_component(preds, truth)
            grid[(th, gp)] = (m, cat)
            scenes_grid[(th, gp)] = {n: m['scenes'][n]['score'] for n in TRAIN}
            print(f"{th:5.2f} {gp:5.2f} {m['mean']['score']:7.4f} "
                  f"{m['worst_score']:6.3f} {m['mean']['presence']:6.3f} "
                  f"{m['mean']['localization']:6.3f} {cat['figure']:6.3f} "
                  f"{cat['off-figure']:6.3f} {m['mean']['recovery']:6.3f} "
                  f"{m['mean']['identification']:5.3f}", flush=True)

    best = max(grid, key=lambda k: grid[k][0]['mean']['score'])
    bm, bc = grid[best]
    print(f'\nbest in-sample: threshold={best[0]} gap={best[1]} '
          f'total={bm["mean"]["score"]:.4f} worst={bm["worst_score"]:.3f}')
    held = {}
    for h in TRAIN:
        dev = [n for n in TRAIN if n != h]
        pick = max(grid, key=lambda k: np.mean([scenes_grid[k][n] for n in dev]))
        held[h] = (pick, scenes_grid[pick][h])
    loso = float(np.mean([v for _, v in held.values()]))
    print(f'leave-one-scene-out over this grid: {loso:.4f}')
    for h, (pick, sc) in held.items():
        print(f'  held out {h:9s} picked {pick} -> {sc:.4f}')
    print('Baseline reference: in-sample 0.72869, threshold-only leave-one-out 0.70152.')
    print('All three scenes influenced development; these are calibration checks.')
    (ROOT / 'outputs/lab/adaptive_calibration.json').write_text(json.dumps(
        {'grid': {f'{k[0]}/{k[1]}': {'mean': v[0]['mean'], 'worst': v[0]['worst_score'],
                                     'category': v[1]} for k, v in grid.items()},
         'best': list(best), 'loso': loso,
         'held': {k: [list(v[0]), v[1]] for k, v in held.items()}},
        indent=2, default=float))


if __name__ == '__main__':
    main()

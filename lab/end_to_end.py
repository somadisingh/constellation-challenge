"""End-to-end effect of appearance re-ranking, with downstream recalibration.

Re-ranking changes the score scale, so the presence threshold and the ambiguity
gate cannot be assumed transferable. For each recipe both are swept, and the
selected pair is also reported under leave-one-scene-out so the headline number is
not the value its constants were chosen on.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate, ScenePrediction
from constellation.references import extract_patterns
from lab.cache import load_train, TRAIN, ROOT
from lab.samples import build_train
from lab.harness import run_stage, aux_for, DEFAULT
from lab.recache import RECIPES, component_cache, rescored_cache

# Staged to keep the budget finite: every recipe is screened on a narrow grid, then
# only the leaders get the wide grid. Each configuration costs ~30s for three scenes.
SCREEN_THRESHOLDS = (.68, .72, .76)
SCREEN_GAPS = (.03,)
WIDE_THRESHOLDS = (.60, .64, .68, .70, .72, .74, .76, .80)
WIDE_GAPS = (.02, .03, .05)
N_FINALISTS = 3


def score_config(rc, patterns, aux, cfg, names=TRAIN):
    preds = {}
    for n in names:
        name, patches, _ = run_stage(rc[n], patterns, aux[n], cfg)
        preds[n] = ScenePrediction(patches, name)
    return preds


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    cache = load_train()
    data = build_train(cache)
    parts = component_cache(cache, data)
    aux = {n: aux_for(n) for n in TRAIN}

    def run_grid(rc, thresholds, gaps, grid=None):
        grid = dict(grid or {})
        for th in thresholds:
            for gp in gaps:
                if (th, gp) in grid:
                    continue
                cfg = dict(DEFAULT); cfg.update(threshold=th, gap=gp)
                grid[(th, gp)] = evaluate(score_config(rc, patterns, aux, cfg), truth)
        return grid

    caches = {label: rescored_cache(cache, parts, w) for label, w in RECIPES.items()}
    print('-- stage 1: screen every recipe on a narrow grid --', flush=True)
    screen = {}
    for label, rc in caches.items():
        g = run_grid(rc, SCREEN_THRESHOLDS, SCREEN_GAPS)
        screen[label] = g
        b = max(g, key=lambda k: g[k]['mean']['score'])
        print(f'{label:28s} best th={b[0]:.2f} score={g[b]["mean"]["score"]:.4f}',
              flush=True)
    order = sorted(screen, key=lambda k: -max(v['mean']['score']
                                              for v in screen[k].values()))
    finalists = list(dict.fromkeys(['incumbent'] + order[:N_FINALISTS]))
    print(f'\n-- stage 2: wide grid for {finalists} --', flush=True)

    summary = {}
    for label in finalists:
        rc = caches[label]
        t0 = time.perf_counter()
        grid = run_grid(rc, WIDE_THRESHOLDS, WIDE_GAPS, screen[label])
        best = max(grid, key=lambda k: grid[k]['mean']['score'])
        m = grid[best]['mean']
        # Leave-one-scene-out over the same grid: pick (threshold, gap) on two
        # scenes, report the held-out scene.
        held = {}
        for h in TRAIN:
            dev = [n for n in TRAIN if n != h]
            pick = max(grid, key=lambda k: np.mean(
                [grid[k]['scenes'][n]['score'] for n in dev]))
            held[h] = grid[pick]['scenes'][h]
        loso = {k: float(np.mean([held[n][k] for n in TRAIN])) for k in held[TRAIN[0]]}
        summary[label] = dict(best=list(best), in_sample=m, weights=RECIPES[label],
                              worst=grid[best]['worst_score'], loso=loso,
                              grid={f'{k[0]}/{k[1]}': v['mean']['score']
                                    for k, v in grid.items()},
                              seconds=time.perf_counter() - t0)
        print(f'{label:28s} th={best[0]:.2f} gap={best[1]:.2f} '
              f'score={m["score"]:.4f} (loso {loso["score"]:.4f}) '
              f'pres={m["presence"]:.3f} loc={m["localization"]:.3f} '
              f'rec={m["recovery"]:.3f} id={m["identification"]:.3f} '
              f'worst={grid[best]["worst_score"]:.3f}', flush=True)

    out = ROOT / 'outputs/lab/end_to_end_rescore.json'
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f'\nwritten to {out}')
    base = summary['incumbent']
    print(f"incumbent in-sample {base['in_sample']['score']:.4f}, "
          f"loso {base['loso']['score']:.4f}")
    best = max(summary, key=lambda k: summary[k]['loso']['score'])
    print(f"best by leave-one-scene-out: {best} {summary[best]['loso']['score']:.4f}")


if __name__ == '__main__':
    main()

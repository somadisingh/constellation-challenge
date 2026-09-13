"""Oracle recovery ceiling for each candidate set.

Recovery greedily matches every predicted-present coordinate against the issued figure
stars. The ceiling asks: if each query could report whichever of its retained candidates
best serves that matching, what recovery is reachable? That separates a candidate-set
limitation from an assignment/ranking limitation.

Presence decisions are held at each set's own threshold so the comparison is like for
like. Ground-truth figure membership is used only to compute the ceiling; it never
selects inference behaviour.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from lab.cache import load_train, TRAIN, ROOT
from lab.adaptive_cache import load_run

THRESHOLD = .72


def greedy_recovery(points, figure):
    """Documented nearest-first greedy one-to-one reward, as contracts.evaluate does."""
    if not len(figure):
        return 1.
    if not len(points):
        return 0.
    d = np.linalg.norm(np.asarray(figure)[:, None, :] - np.asarray(points)[None, :, :],
                       axis=2)
    used_f, used_p, total = set(), set(), 0.
    for flat in np.argsort(d, axis=None, kind='stable'):
        i, j = np.unravel_index(flat, d.shape)
        if i in used_f or j in used_p:
            continue
        total += float(np.clip((36 - d[i, j]) / 24, 0, 1))
        used_f.add(i); used_p.add(j)
    return total / len(figure)


def ceiling_for(cache, truth, top_k=None):
    """Per-scene: reported-as-is recovery, and the ceiling over retained candidates."""
    rows = {}
    for n in TRAIN:
        fig = np.array([p[:2] for p in truth[n].patches
                        if p is not None and p[2] == 1]).reshape(-1, 2)
        live = [q for q in cache[n]['refined'] if len(q) and q[0][2] >= THRESHOLD]
        as_is = np.array([q[0][:2] for q in live]).reshape(-1, 2)
        # Ceiling: each present query offers its candidate nearest to any figure star.
        best = []
        for q in live:
            xy = np.array([c[:2] for c in q], float)
            if top_k:
                xy = xy[:top_k]
            if len(fig):
                j = int(np.argmin(np.linalg.norm(
                    fig[:, None, :] - xy[None, :, :], axis=2).min(axis=0)))
                best.append(xy[j])
            else:
                best.append(xy[0])
        rows[n] = {'n_present': len(live), 'n_figure': len(fig),
                   'as_is': greedy_recovery(as_is, fig),
                   'ceiling': greedy_recovery(np.array(best).reshape(-1, 2), fig)}
    return rows


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    sets = {'baseline fixed12': load_train(),
            'adaptive radius': load_run('outputs/lab/adaptive_train')}
    out = {}
    print(f"{'candidate set':18s} {'scene':9s} {'present':>8s} {'figure':>7s} "
          f"{'as-is':>7s} {'ceiling@20':>11s} {'ceiling@8':>10s}")
    for label, cache in sets.items():
        full = ceiling_for(cache, truth)
        eight = ceiling_for(cache, truth, top_k=8)
        out[label] = {'full': full, 'top8': eight}
        for n in TRAIN:
            print(f"{label:18s} {n:9s} {full[n]['n_present']:8d} "
                  f"{full[n]['n_figure']:7d} {full[n]['as_is']:7.3f} "
                  f"{full[n]['ceiling']:11.3f} {eight[n]['ceiling']:10.3f}")
        m = np.mean([full[n]['ceiling'] for n in TRAIN])
        m8 = np.mean([eight[n]['ceiling'] for n in TRAIN])
        a = np.mean([full[n]['as_is'] for n in TRAIN])
        print(f"{label:18s} {'MEAN':9s} {'':8s} {'':7s} {a:7.3f} {m:11.3f} {m8:10.3f}")
    print('\nproduction achieves 0.8778; adaptive with the geometry-guided second pass '
          'achieves 0.8222')
    (ROOT / 'outputs/lab/recovery_ceiling.json').write_text(
        json.dumps(out, indent=2, default=float))


if __name__ == '__main__':
    main()

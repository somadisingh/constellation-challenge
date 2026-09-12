"""Why do relocation guards lose recovery, and is there a strictness that wins?

Recovery is a greedy one-to-one match of *every* predicted-present coordinate against
the issued figure stars, with no membership test. So moving an off-figure query onto a
figure node earns recovery even though it destroys that query's own localization. The
pipeline is exploiting that, which is legitimate under the documented procedure but
worth stating explicitly, because it is the reason a guard that plainly improves
localization can still lose the weighted total.

This script quantifies the exchange and sweeps guard strictness with a bounded grid.
A negative margin means the guard rejects a move only when the independent evidence
prefers the original by at least that much.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate, ScenePrediction, reward
from constellation.references import extract_patterns
from constellation.reloc import IndependentAppearance, apply_guard
from lab.cache import load_train, TRAIN, ROOT
from lab.samples import build_train
from lab.recache import component_cache
from lab.harness import aux_for
from lab.reloc_experiment import geometry_for_scene, assemble, THRESHOLD

GRID = (-1.0, -.6, -.4, -.3, -.2, -.1, -.05, 0.)


def components(preds, truth):
    m = evaluate(preds, truth)
    cat = {}
    for c, want in (('figure', 1), ('off-figure', 0)):
        v = [0. if p is None else float(reward(np.linalg.norm(
            np.array(t[:2]) - np.array(p[:2]))))
            for n in TRAIN for p, t in zip(preds[n].patches, truth[n].patches)
            if t is not None and t[2] == want]
        cat[c] = float(np.mean(v))
    return m, cat


def recovery_contribution(preds, truth):
    """How many figure truth stars are matched within 12px, and by which category."""
    hit = {'by relocated point': 0, 'by own coordinate': 0, 'unmatched': 0}
    for n in TRAIN:
        fig = np.array([p[:2] for p in truth[n].patches
                        if p is not None and p[2] == 1]).reshape(-1, 2)
        pts = np.array([p[:2] for p in preds[n].patches if p is not None]).reshape(-1, 2)
        if not len(fig) or not len(pts):
            hit['unmatched'] += len(fig); continue
        d = np.linalg.norm(fig[:, None, :] - pts[None, :, :], axis=2)
        used_f, used_p = set(), set()
        for flat in np.argsort(d, axis=None, kind='stable'):
            i, j = np.unravel_index(flat, d.shape)
            if i in used_f or j in used_p:
                continue
            used_f.add(i); used_p.add(j)
            hit['by own coordinate' if d[i, j] <= 12 else 'unmatched'] += 1
    return hit


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    cache = load_train()
    data = build_train(cache)
    parts = component_cache(cache, data)
    aux = {n: aux_for(n) for n in TRAIN}
    geom = {n: geometry_for_scene(cache, n, patterns, aux[n]) for n in TRAIN}

    # Attribute the baseline's recovery to relocated versus untouched coordinates.
    print('baseline: what earns recovery')
    for n in TRAIN:
        slates, name, chosen, diag, _ = geom[n]
        fig = np.array([p[:2] for p in truth[n].patches
                        if p is not None and p[2] == 1]).reshape(-1, 2)
        moved = {q for q, xy in chosen.items()
                 if np.linalg.norm(np.asarray(xy) - slates[q].xy[slates[q].report_index]) > .5}
        near = 0
        for q in moved:
            xy = np.asarray(chosen[q], float)
            if len(fig) and np.linalg.norm(fig - xy, axis=1).min() <= 12:
                near += 1
        offfig_moved_onto = sum(
            1 for q in moved
            if truth[n].patches[q] is not None and truth[n].patches[q][2] == 0
            and len(fig) and np.linalg.norm(fig - np.asarray(chosen[q]), axis=1).min() <= 12)
        print(f'  {n:9s} moved={len(moved):2d} landing within 12px of a figure star='
              f'{near:2d}  of which truth-off-figure queries={offfig_moved_onto}')

    # Independent scores must cover both branches' candidate coordinates, because a
    # winning coarse branch proposes coarse locations.
    data_c = build_train(cache, 'coarse')
    parts_c = component_cache(cache, data_c, branch='coarse')
    scores, coords = {}, {}
    for n in TRAIN:
        scores[n], coords[n] = {}, {}
        for i in range(len(cache[n]['refined'])):
            scores[n][i] = np.concatenate([np.asarray(parts[n]['ca_min'][i], float),
                                           np.asarray(parts_c[n]['ca_min'][i], float)])
            coords[n][i] = np.vstack([
                np.array([c[:2] for c in cache[n]['refined'][i]], float),
                np.array([c[:2] for c in cache[n]['coarse'][i]], float)])

    print(f"\n{'margin':>7s} {'total':>7s} {'worst':>6s} {'loc':>6s} {'fig':>6s} "
          f"{'off':>6s} {'rec':>6s} {'moves':>6s}")
    out = {}
    per_scene = {}
    for margin in GRID:
        preds, kept_n = {}, 0
        for n in TRAIN:
            slates, name, chosen, diag, _ = geom[n]
            guard = IndependentAppearance(scores[n], coords[n], margin=margin)
            kept, _ = apply_guard(chosen, slates, guard)
            kept_n += sum(1 for q, xy in kept.items()
                          if np.linalg.norm(np.asarray(xy)
                                            - slates[q].xy[slates[q].report_index]) > .5)
            preds[n] = ScenePrediction(assemble(slates, kept, diag), name)
        m, cat = components(preds, truth)
        out[margin] = {'mean': m['mean'], 'worst': m['worst_score'],
                       'category': cat, 'moves': kept_n}
        per_scene[margin] = {n: m['scenes'][n]['score'] for n in TRAIN}
        print(f"{margin:7.2f} {m['mean']['score']:7.4f} {m['worst_score']:6.3f} "
              f"{m['mean']['localization']:6.3f} {cat['figure']:6.3f} "
              f"{cat['off-figure']:6.3f} {m['mean']['recovery']:6.3f} {kept_n:6d}",
              flush=True)

    best = max(out, key=lambda k: out[k]['mean']['score'])
    print(f"\nbest in-sample margin {best} at {out[best]['mean']['score']:.4f}")
    # Development-fold check of the selection procedure itself.
    held = {}
    for h in TRAIN:
        dev = [n for n in TRAIN if n != h]
        pick = max(GRID, key=lambda k: np.mean([per_scene[k][n] for n in dev]))
        held[h] = (pick, per_scene[pick][h])
    loso = float(np.mean([v for _, v in held.values()]))
    print(f'leave-one-scene-out over this margin grid: {loso:.4f}  {held}')
    print('All three scenes informed development; this is a calibration check, '
          'not an untouched generalization estimate.')
    (ROOT / 'outputs/lab/reloc_tradeoff.json').write_text(
        json.dumps({'grid': out, 'loso': loso,
                    'held': {k: list(v) for k, v in held.items()}},
                   indent=2, default=float))


if __name__ == '__main__':
    main()

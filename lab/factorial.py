"""Factorial isolation of appearance ranking from geometric seed selection.

Earlier re-ranking experiments changed six things at once. With `QuerySlate` the
roles are separable, so the four cells the brief asks for can be run on identical
cached candidates:

  A  existing seeds + existing ranking      (production control)
  B  existing seeds + improved ranking      (better reported coordinate only)
  C  improved seeds + existing ranking      (better seed anchors only)
  D  improved seeds + improved ranking

Held fixed in every cell, and asserted rather than assumed:
  presence decisions      calibration score is always the incumbent
  ambiguity membership    derived from the incumbent calibration gap
  pool margin             applied to the incumbent calibration
  pool membership         `pool_by='calib'` fixes which candidates may enter, so
                          only their order can change
  hypothesis budget, tolerance, grouping radius, auxiliary weight, threshold

Not held fixed, and logged as such:
  the order of candidates inside the pool (that is the intervention in B and D)
  the grouping anchors, because `consolidate` runs on seed coordinates (C and D)
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate, ScenePrediction, reward
from constellation.references import extract_patterns
from constellation.slate import slates_from_lists, with_scores
from constellation.joint import recognize_joint
from constellation.finalize import auxiliary_map, COARSE_CUTOFF
from lab.cache import load_train, TRAIN, ROOT
from lab.samples import build_train
from lab.recache import component_cache, COMPONENTS
from lab.harness import aux_for

THRESHOLD = .72
RANKER = 'ca_min'      # centre/annulus minimum: best figure-query ranker measured


def build_slates(cache, parts, scene, ranker, use_rank, use_seed):
    """Slates with the incumbent calibration and selectable rank / seed roles."""
    out = []
    for i, q in enumerate(cache[scene]['refined']):
        base = slates_from_lists([q])[0]
        if not len(base):
            out.append(base); continue
        new = np.asarray(parts[scene][ranker][i], float)
        rank = new if use_rank else None
        seed = int(np.argmax(new)) if use_seed else None
        out.append(with_scores(base, rank=rank, seed=seed))
    return out


def run_scene(slates_refined, coarse_lists, patterns, aux, pool_by='calib'):
    """One scene through the two competing branches, mirroring finalize_joint."""
    competing = []
    for label, sl, cutoff in [('refined', slates_refined, THRESHOLD),
                              ('coarse', slates_from_lists(coarse_lists), COARSE_CUTOFF)]:
        ids = [i for i, s in enumerate(sl) if len(s) and s.presence_score >= cutoff]
        if len(ids) < 3:
            continue
        name, chosen, d = recognize_joint([sl[i] for i in ids], patterns,
                                          tolerance=18., gap=.03, top_k=8, cap=80000,
                                          quad_share=0., aux_weight=3.,
                                          models=('affine',), shear_penalty=2.,
                                          auxiliary_map=aux, pool_by=pool_by)
        best = d['hypotheses'][0] if d.get('hypotheses') else {'score': -1e9}
        competing.append((best.get('score', -1e9), name,
                          {ids[k]: v for k, v in chosen.items()}, d, label))
    if not competing:
        return 'unknown', [None] * len(slates_refined), {}
    competing.sort(key=lambda x: (-x[0], x[1], x[4]))
    _, name, chosen, diag, stage = competing[0]
    nodes = (np.array(diag['hypotheses'][0].get('nodes', [])).reshape(-1, 2)
             if diag.get('hypotheses') else np.empty((0, 2)))
    patches = []
    for i, s in enumerate(slates_refined):
        if not len(s):
            patches.append(None); continue
        if s.presence_score < THRESHOLD:
            patches.append(None); continue
        x, y = (chosen[i] if i in chosen else s.xy[s.report_index])
        member = int(len(nodes) > 0
                     and np.linalg.norm(nodes - [x, y], axis=1).min() < 18.)
        patches.append((float(x), float(y), member))
    return name, patches, {'stage': stage, 'relocated': sorted(chosen),
                           'true_class_rank': diag, 'geometry': diag}


def category_localization(preds, truth):
    out = {}
    for cat, want in (('figure', 1), ('off-figure', 0)):
        vals = []
        for n in TRAIN:
            for p, t in zip(preds[n].patches, truth[n].patches):
                if t is None or t[2] != want:
                    continue
                vals.append(0. if p is None else float(
                    reward(np.linalg.norm(np.array(t[:2]) - np.array(p[:2])))))
        out[cat] = float(np.mean(vals))
    return out


def true_class_rank(diag, true_name):
    hyps = [h['name'] for h in diag.get('geometry', {}).get('hypotheses', [])
            if h.get('support', 0) >= 4]
    return hyps.index(true_name) if true_name in hyps else None


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    cache = load_train()
    data = build_train(cache)
    parts = component_cache(cache, data)
    aux = {n: aux_for(n) for n in TRAIN}

    cells = {'A existing seed + existing rank': (False, False),
             'B existing seed + improved rank': (True, False),
             'C improved seed + existing rank': (False, True),
             'D improved seed + improved rank': (True, True)}

    print(f'ranker = {RANKER}; pool membership frozen by calibration '
          f'(pool_by="calib"); presence/ambiguity/margin frozen\n')
    print(f"{'cell':34s} {'total':>7s} {'worst':>6s} {'pres':>6s} {'loc':>6s} "
          f"{'fig':>6s} {'off':>6s} {'rec':>6s} {'id':>5s}  classes")
    results = {}
    for label, (use_rank, use_seed) in cells.items():
        preds, diags = {}, {}
        for n in TRAIN:
            sl = build_slates(cache, parts, n, RANKER, use_rank, use_seed)
            name, patches, d = run_scene(sl, cache[n]['coarse'], patterns, aux[n])
            preds[n] = ScenePrediction(patches, name); diags[n] = d
        m = evaluate(preds, truth)
        cat = category_localization(preds, truth)
        ranks = {n: true_class_rank(diags[n], truth[n].constellation) for n in TRAIN}
        results[label] = {'mean': m['mean'], 'worst': m['worst_score'],
                          'category_localization': cat,
                          'true_class_rank': ranks,
                          'classes': {n: preds[n].constellation for n in TRAIN},
                          'relocated': {n: len(diags[n].get('relocated', []))
                                        for n in TRAIN}}
        print(f"{label:34s} {m['mean']['score']:7.4f} {m['worst_score']:6.3f} "
              f"{m['mean']['presence']:6.3f} {m['mean']['localization']:6.3f} "
              f"{cat['figure']:6.3f} {cat['off-figure']:6.3f} "
              f"{m['mean']['recovery']:6.3f} {m['mean']['identification']:5.3f}  "
              + ' '.join(preds[n].constellation for n in TRAIN), flush=True)

    base = results['A existing seed + existing rank']['mean']['presence']
    frozen = all(abs(v['mean']['presence'] - base) < 1e-12 for v in results.values())
    print(f'\npresence identical across all cells (freeze held): {frozen}')
    print('true-class rank among verified hypotheses (0 = correct, None = no fit):')
    for label, v in results.items():
        print(f"  {label:34s} {v['true_class_rank']}  relocated={v['relocated']}")
    (ROOT / 'outputs/lab/factorial.json').write_text(
        json.dumps(results, indent=2, default=float))


if __name__ == '__main__':
    main()

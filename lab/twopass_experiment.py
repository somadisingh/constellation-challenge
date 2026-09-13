"""Does geometry-guided pool eligibility convert the retention gain into recovery?

Run on both candidate sets, because the mechanism is motivated by the adaptive verifier
but should be tested against the frozen one too:
  baseline  = frozen fixed-radius candidates (production)
  adaptive  = pose-admissible candidates (retention .972, recovery loss)

Bounded grid: node radius and the number of leading hypotheses whose nodes are used.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate, ScenePrediction, reward
from constellation.references import extract_patterns
from constellation.slate import slates_from_lists
from constellation.joint import recognize_joint
from constellation.twopass import recognize_two_pass
from constellation.finalize import COARSE_CUTOFF
from lab.cache import load_train, TRAIN, ROOT
from lab.harness import aux_for
from lab.adaptive_cache import load_run

KW = dict(tolerance=18., top_k=8, cap=80000, quad_share=0., aux_weight=3.,
          models=('affine',), shear_penalty=2.)


def run_scene(cache_scene, patterns, aux, threshold, gap, two_pass, radius, nhyp):
    """Branch competition mirroring finalize_joint; refined coordinates are reported."""
    refined = slates_from_lists(cache_scene['refined'])
    competing = []
    for label, sl, cutoff in [('refined', refined, threshold),
                              ('coarse', slates_from_lists(cache_scene['coarse']),
                               COARSE_CUTOFF)]:
        ids = [i for i, s in enumerate(sl) if len(s) and s.presence_score >= cutoff]
        if len(ids) < 3:
            continue
        sub = [sl[i] for i in ids]
        fn = (lambda *a, **k: recognize_two_pass(*a, node_radius=radius,
                                                 n_hypotheses=nhyp, **k)) \
            if two_pass else recognize_joint
        name, chosen, d = fn(sub, patterns, gap=gap, auxiliary_map=aux, **KW)
        best = d['hypotheses'][0] if d.get('hypotheses') else {'score': -1e9}
        competing.append((best.get('score', -1e9), name,
                          {ids[k]: v for k, v in chosen.items()}, d, label))
    if not competing:
        return 'unknown', [None] * len(refined), {}
    competing.sort(key=lambda x: (-x[0], x[1], x[4]))
    _, name, chosen, diag, stage = competing[0]
    nodes = (np.array(diag['hypotheses'][0].get('nodes', [])).reshape(-1, 2)
             if diag.get('hypotheses') else np.empty((0, 2)))
    patches = []
    for i, s in enumerate(refined):
        if not len(s) or s.presence_score < threshold:
            patches.append(None); continue
        x, y = chosen[i] if i in chosen else s.xy[s.report_index]
        member = int(len(nodes) > 0
                     and np.linalg.norm(nodes - [x, y], axis=1).min() < 18.)
        patches.append((float(x), float(y), member))
    return name, patches, diag


def score(cache, patterns, aux, truth, threshold, gap, two_pass, radius, nhyp):
    preds, diags = {}, {}
    for n in TRAIN:
        name, patches, d = run_scene(cache[n], patterns, aux[n], threshold, gap,
                                     two_pass, radius, nhyp)
        preds[n] = ScenePrediction(patches, name); diags[n] = d
    m = evaluate(preds, truth)
    cat = {}
    for c, want in (('figure', 1), ('off-figure', 0)):
        v = [0. if p is None else float(reward(np.linalg.norm(
            np.array(t[:2]) - np.array(p[:2]))))
            for n in TRAIN for p, t in zip(preds[n].patches, truth[n].patches)
            if t is not None and t[2] == want]
        cat[c] = float(np.mean(v))
    return m, cat, preds, diags


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    aux = {n: aux_for(n) for n in TRAIN}
    caches = {'baseline': (load_train(), .72),
              'adaptive': (load_run('outputs/lab/adaptive_train'), .72)}

    print(f"{'cache':9s} {'two-pass':>9s} {'rad':>4s} {'nh':>3s} {'gap':>5s} "
          f"{'total':>7s} {'worst':>6s} {'pres':>6s} {'loc':>6s} {'fig':>6s} "
          f"{'off':>6s} {'rec':>6s} {'id':>5s}  classes")
    out = {}
    for label, (cache, th) in caches.items():
        for two_pass, radius, nhyp, gap in [(False, 0, 0, .03),
                                            (True, 12., 3, .03),
                                            (True, 18., 3, .03),
                                            (True, 25., 3, .03),
                                            (True, 18., 1, .03),
                                            (True, 18., 8, .03),
                                            (True, 18., 3, .015)]:
            t0 = time.perf_counter()
            m, cat, preds, diags = score(cache, patterns, aux, truth, th, gap,
                                         two_pass, radius, nhyp)
            key = f'{label}|{two_pass}|{radius}|{nhyp}|{gap}'
            out[key] = {'mean': m['mean'], 'worst': m['worst_score'], 'category': cat,
                        'classes': {n: preds[n].constellation for n in TRAIN},
                        'seconds': time.perf_counter() - t0}
            print(f"{label:9s} {str(two_pass):>9s} {radius:4.0f} {nhyp:3d} {gap:5.3f} "
                  f"{m['mean']['score']:7.4f} {m['worst_score']:6.3f} "
                  f"{m['mean']['presence']:6.3f} {m['mean']['localization']:6.3f} "
                  f"{cat['figure']:6.3f} {cat['off-figure']:6.3f} "
                  f"{m['mean']['recovery']:6.3f} {m['mean']['identification']:5.3f}  "
                  + ' '.join(preds[n].constellation[:9] for n in TRAIN), flush=True)
    (ROOT / 'outputs/lab/twopass.json').write_text(json.dumps(out, indent=2,
                                                             default=float))
    print('\nbaseline production reference: total 0.72869, recovery 0.8778')


if __name__ == '__main__':
    main()

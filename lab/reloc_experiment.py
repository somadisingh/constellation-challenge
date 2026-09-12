"""Relocation safeguards, measured with the class hypothesis left alone.

The winning class and the geometric assignment are computed exactly as production
does. The only intervention is whether each proposed coordinate is adopted. Presence
is untouched by construction: a rejected relocation keeps its own coordinate and
stays present.

Reported per guard: fixes and regressions split by figure / off-figure, the size of
incorrect moves, all four competition components, and the equal-scene mean.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate, ScenePrediction, reward
from constellation.references import extract_patterns
from constellation.slate import slates_from_lists
from constellation.joint import recognize_joint, fit_affine
from constellation.finalize import COARSE_CUTOFF
from constellation.reloc import (AcceptAll, IndependentAppearance,
                                 GroupHeldOutSupport, AllOf, apply_guard)
from lab.cache import load_train, TRAIN, ROOT
from lab.samples import build_train
from lab.recache import component_cache
from lab.harness import aux_for

THRESHOLD = .72


def geometry_for_scene(cache, scene, patterns, aux):
    """Production branch competition; returns slates, chosen map and diagnostics."""
    competing = []
    branches = {'refined': (slates_from_lists(cache[scene]['refined']), THRESHOLD),
                'coarse': (slates_from_lists(cache[scene]['coarse']), COARSE_CUTOFF)}
    for label, (sl, cutoff) in branches.items():
        ids = [i for i, s in enumerate(sl) if len(s) and s.presence_score >= cutoff]
        if len(ids) < 3:
            continue
        name, chosen, d = recognize_joint([sl[i] for i in ids], patterns,
                                          tolerance=18., gap=.03, top_k=8, cap=80000,
                                          quad_share=0., aux_weight=3.,
                                          models=('affine',), shear_penalty=2.,
                                          auxiliary_map=aux)
        best = d['hypotheses'][0] if d.get('hypotheses') else {'score': -1e9}
        competing.append((best.get('score', -1e9), name,
                          {ids[k]: v for k, v in chosen.items()}, d, label, ids))
    competing.sort(key=lambda x: (-x[0], x[1], x[4]))
    score, name, chosen, diag, label, ids = competing[0]
    # Production always reports the refined coordinate and applies the refined
    # threshold, even when the coarse branch wins the geometry competition.
    return branches['refined'][0], name, chosen, diag, label


def contexts_from_diag(diag):
    """Per-query context for the group-held-out guard."""
    r = diag.get('relocation') or {}
    if not r.get('template'):
        return {}
    template = np.asarray(r['template'], float)
    pool = np.asarray(r['pool'], float)
    tags = np.asarray(r['tags'], int)
    pairs = [tuple(p) for p in r['pairs']]
    node_by_tag = {int(k): v for k, v in r['node_by_tag'].items()}
    out = {}
    for q, g in r['group_of_query'].items():
        out[int(q)] = {'template': template, 'pool': pool, 'pairs': pairs,
                       'tags': tags, 'group_of_query': int(g),
                       'node_of_query': node_by_tag.get(int(g)),
                       'fit': fit_affine}
    return out


def assemble(slates, chosen, diag, nodes_radius=18.):
    nodes = (np.array(diag['hypotheses'][0].get('nodes', [])).reshape(-1, 2)
             if diag.get('hypotheses') else np.empty((0, 2)))
    patches = []
    for i, s in enumerate(slates):
        if not len(s) or s.presence_score < THRESHOLD:
            patches.append(None); continue
        x, y = chosen[i] if i in chosen else s.xy[s.report_index]
        member = int(len(nodes) > 0
                     and np.linalg.norm(nodes - [x, y], axis=1).min() < nodes_radius)
        patches.append((float(x), float(y), member))
    return patches


def move_audit(slates, base_chosen, kept, truth_patches):
    """Fixes and regressions by category, relative to no relocation at all."""
    stats = {c: {'fix': 0, 'regress': 0, 'neutral': 0, 'big_wrong': 0}
             for c in ('figure', 'off-figure')}
    for q, xy in base_chosen.items():
        t = truth_patches[q]
        if t is None:
            continue
        s = slates[q]
        orig = s.xy[s.report_index]
        if np.linalg.norm(np.asarray(xy) - orig) < .5:
            continue
        final = np.asarray(kept.get(q, orig), float)
        cat = 'figure' if t[2] == 1 else 'off-figure'
        was = np.linalg.norm(orig - np.array(t[:2])) <= 12
        now = np.linalg.norm(final - np.array(t[:2])) <= 12
        if now and not was:
            stats[cat]['fix'] += 1
        elif was and not now:
            stats[cat]['regress'] += 1
            if np.linalg.norm(final - orig) > 100:
                stats[cat]['big_wrong'] += 1
        else:
            stats[cat]['neutral'] += 1
    return stats


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    cache = load_train()
    data = build_train(cache)
    parts = component_cache(cache, data)
    aux = {n: aux_for(n) for n in TRAIN}

    # Geometry is computed once; guards only filter its output.
    geom = {}
    for n in TRAIN:
        geom[n] = geometry_for_scene(cache, n, patterns, aux[n])
        print(f'  {n:9s} class={geom[n][1]:16s} branch={geom[n][4]:8s} '
              f'proposed relocations={len(geom[n][2])}', flush=True)

    def independent(key, margin=0.):
        scores = {n: {i: np.asarray(parts[n][key][i], float)
                      for i in range(len(cache[n]['refined']))} for n in TRAIN}
        coords = {n: {i: np.array([c[:2] for c in cache[n]['refined'][i]], float)
                      for i in range(len(cache[n]['refined']))} for n in TRAIN}
        return scores, coords, margin

    guards = {'baseline (accept all)': lambda n: AcceptAll(),
              'independent full32': lambda n: IndependentAppearance(
                  *[d[n] for d in independent('full32')[:2]], margin=0.,
                  name='full32'),
              'independent annulus': lambda n: IndependentAppearance(
                  *[d[n] for d in independent('annulus')[:2]], margin=0.,
                  name='annulus'),
              'independent ca_min': lambda n: IndependentAppearance(
                  *[d[n] for d in independent('ca_min')[:2]], margin=0.,
                  name='ca_min'),
              'group-held-out support': lambda n: GroupHeldOutSupport(18.),
              'full32 + held-out': lambda n: AllOf(
                  IndependentAppearance(*[d[n] for d in independent('full32')[:2]],
                                        name='full32'),
                  GroupHeldOutSupport(18.)),
              }

    print(f"\n{'guard':24s} {'total':>7s} {'worst':>6s} {'loc':>6s} {'fig':>6s} "
          f"{'off':>6s} {'rec':>6s} {'id':>5s} {'kept':>5s}  fig fix/reg  off fix/reg")
    out = {}
    for label, make in guards.items():
        preds, audit, kept_total = {}, {}, 0
        for n in TRAIN:
            slates, name, chosen, diag, _ = geom[n]
            kept, _ = apply_guard(chosen, slates, make(n), contexts_from_diag(diag))
            kept_total += sum(1 for q, xy in kept.items()
                              if np.linalg.norm(np.asarray(xy)
                                                - slates[q].xy[slates[q].report_index]) > .5)
            preds[n] = ScenePrediction(assemble(slates, kept, diag), name)
            audit[n] = move_audit(slates, chosen, kept, truth[n].patches)
        m = evaluate(preds, truth)
        cat = {}
        for c, want in (('figure', 1), ('off-figure', 0)):
            v = [0. if p is None else float(reward(np.linalg.norm(
                np.array(t[:2]) - np.array(p[:2]))))
                for n in TRAIN for p, t in zip(preds[n].patches, truth[n].patches)
                if t is not None and t[2] == want]
            cat[c] = float(np.mean(v))
        agg = {c: {k: sum(audit[n][c][k] for n in TRAIN)
                   for k in ('fix', 'regress', 'big_wrong')}
               for c in ('figure', 'off-figure')}
        print(f"{label:24s} {m['mean']['score']:7.4f} {m['worst_score']:6.3f} "
              f"{m['mean']['localization']:6.3f} {cat['figure']:6.3f} "
              f"{cat['off-figure']:6.3f} {m['mean']['recovery']:6.3f} "
              f"{m['mean']['identification']:5.3f} {kept_total:5d}  "
              f"{agg['figure']['fix']}/{agg['figure']['regress']}"
              f"({agg['figure']['big_wrong']} big)  "
              f"{agg['off-figure']['fix']}/{agg['off-figure']['regress']}"
              f"({agg['off-figure']['big_wrong']} big)", flush=True)
        out[label] = {'mean': m['mean'], 'worst': m['worst_score'],
                      'category_localization': cat, 'audit': agg,
                      'moves_kept': kept_total}
    (ROOT / 'outputs/lab/reloc_guards.json').write_text(
        json.dumps(out, indent=2, default=float))
    print('\npresence is identical for every guard by construction '
          '(a rejected move keeps its own coordinate and stays present)')


if __name__ == '__main__':
    main()

"""Verify the localization headroom claim: denominator, presence assumptions, and
aggregation. The earlier lab/headroom.py substituted true class and true membership,
which makes its 'score' column incomparable to production; only its localization
column was meaningful. This recomputes the ceiling against the production baseline.

Three reference points, all under identical presence decisions (rank-0 appearance
score >= 0.72, exactly as production):
  rank-0        report the top appearance alternative, no geometric relocation
  production    the shipped joint stage, relocation included
  oracle        report whichever of the 20 retained alternatives is nearest truth
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate, ScenePrediction, reward
from lab.cache import load_train, TRAIN, ROOT


def localization_terms(patches, truth_patches):
    """Per-query localization reward exactly as contracts.evaluate computes it."""
    out = []
    for p, t in zip(patches, truth_patches):
        if t is None:
            continue
        out.append(0. if p is None
                   else float(reward(np.linalg.norm(np.array(t[:2]) - np.array(p[:2])))))
    return out


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    cache = load_train()
    prod = {n: json.load(open(ROOT / f'outputs/joint_train/{n}.json')) for n in TRAIN}

    variants = {}
    for label in ('rank-0', 'production', 'oracle'):
        preds = {}
        for n in TRAIN:
            tp = truth[n].patches
            patches = []
            for i, q in enumerate(cache[n]['refined']):
                present = len(q) and q[0][2] >= .72
                if not present:
                    patches.append(None); continue
                if label == 'production':
                    patches.append(tuple(prod[n]['patches'][i]))
                    continue
                if label == 'oracle' and tp[i] is not None:
                    pool = np.array([c[:2] for c in q])
                    j = int(np.argmin(np.linalg.norm(pool - np.array(tp[i][:2]), axis=1)))
                else:
                    j = 0
                patches.append((float(q[j][0]), float(q[j][1]), 0))
            # Class and membership are held at the production values so only the
            # coordinate differs between variants.
            preds[n] = ScenePrediction(
                [None if p is None else (p[0], p[1],
                 prod[n]['patches'][i][2] if prod[n]['patches'][i] else 0)
                 for i, p in enumerate(patches)], prod[n]['constellation'])
        variants[label] = preds

    print('presence decisions are identical across variants by construction')
    n_pres = {n: sum(p is not None for p in variants['rank-0'][n].patches) for n in TRAIN}
    print(f'  reported present per scene: {n_pres}')

    print(f"\n{'variant':12s} {'loc(equal-scene)':>17s} {'loc(pooled)':>12s} "
          f"{'recovery':>9s} {'total':>7s}")
    for label, preds in variants.items():
        m = evaluate(preds, truth)
        pooled = [v for n in TRAIN
                  for v in localization_terms(preds[n].patches, truth[n].patches)]
        print(f'{label:12s} {m["mean"]["localization"]:17.4f} {np.mean(pooled):12.4f} '
              f'{m["mean"]["recovery"]:9.4f} {m["mean"]["score"]:7.4f}')

    print('\nlocalization denominator per scene (truth-present queries):')
    for n in TRAIN:
        tot = sum(p is not None for p in truth[n].patches)
        zero = sum(1 for p, t in zip(variants['production'][n].patches, truth[n].patches)
                   if t is not None and p is None)
        print(f'  {n:9s} truth-present={tot:2d}  of which reported absent={zero:2d} '
              f'(each contributes 0)')

    print('\nper-category localization reward (production vs oracle):')
    for cat, want in (('figure', 1), ('off-figure', 0)):
        for label in ('rank-0', 'production', 'oracle'):
            vals = []
            for n in TRAIN:
                for p, t in zip(variants[label][n].patches, truth[n].patches):
                    if t is None or t[2] != want:
                        continue
                    vals.append(0. if p is None else float(
                        reward(np.linalg.norm(np.array(t[:2]) - np.array(p[:2])))))
            print(f'  {cat:11s} {label:11s} n={len(vals):3d} mean={np.mean(vals):.4f}')


if __name__ == '__main__':
    main()

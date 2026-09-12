"""Attribute the present queries that have no retained candidate near truth.

Seven of 71 labelled present queries have no candidate within 12px. Before changing
strides or budgets, establish which stage lost each one. Stages, in pipeline order:

  border          truth lies inside the 20px margin excluded from the index
  index           no index point near truth (stride/border), so retrieval cannot rank it
  retrieval       the nearest index point exists but falls outside the top-2000 shortlist
  dense           the exhaustive half-resolution fallback also missed it
  verify          truth was among the proposals but not in the 20 retained by `verify`
  refine          truth survived `verify` but the ECC stage moved or dropped it
  scale           the pose needed lies outside the 0.75-1.33 search bounds
  present         a candidate is in fact within 12px (should not occur here)

Only the affected queries are recomputed, so this does not repeat the global search.
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from constellation.retrieval import (build_harmonic_index, retrieve_harmonic, verify,
                                     harmonic_features)
from constellation.dense import dense_candidates
from constellation.refine import refine_candidates
from lab.cache import load_train, TRAIN, ROOT

RADIUS = 12.
MARGIN = 20.


def near(points, xy, radius=RADIUS):
    if points is None or not len(points):
        return False
    p = np.asarray(points, float).reshape(-1, np.shape(points)[-1])[:, :2]
    return bool((np.linalg.norm(p - np.asarray(xy, float), axis=1) <= radius).any())


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    cache = load_train()
    report = []
    for n in TRAIN:
        failures = []
        for i, q in enumerate(cache[n]['refined']):
            t = truth[n].patches[i]
            if t is None:
                continue
            if not near(q, t[:2]):
                failures.append(i)
        if not failures:
            continue
        image = cv2.imread(str(ROOT / 'train' / n / f'{n}_image.png'), 0)
        print(f'{n}: {len(failures)} uncovered present queries -> {failures}', flush=True)
        points, desc, stars = build_harmonic_index(image)
        raw = cv2.GaussianBlur(image.astype(np.float32), (0, 0), .6)
        blurred_scene = cv2.GaussianBlur(image.astype(np.float32), (0, 0), .6)
        for i in failures:
            patch = cv2.imread(str(ROOT / 'train' / n / 'patches' /
                                   f'patch_{i+1:02}.png'), 0)
            xy = np.array(truth[n].patches[i][:2], float)
            entry = {'scene': n, 'query': i + 1, 'truth': xy.tolist(),
                     'coarse_had_it': near(cache[n]['coarse'][i], xy)}
            h, w = image.shape
            entry['border'] = bool(xy.min() < MARGIN or xy[0] > w - MARGIN
                                   or xy[1] > h - MARGIN)
            d_idx = np.linalg.norm(points - xy, axis=1)
            entry['nearest_index_point_px'] = float(d_idx.min())
            # Rank of the best index point near truth within the full descriptor scan.
            ds = np.concatenate([harmonic_features(
                cv2.GaussianBlur(patch.astype(np.float32), (0, 0), .6),
                np.array([[15, 15], [16, 16], [15, 16], [16, 15]]), s)
                for s in (.75, .87, 1., 1.15, 1.33)])
            scores = np.max(desc @ ds.T, axis=1)
            local = np.where(d_idx <= RADIUS)[0]
            if len(local):
                best_local = float(scores[local].max())
                entry['retrieval_rank_of_truth'] = int((scores > best_local).sum())
            else:
                entry['retrieval_rank_of_truth'] = None
            shortlist = retrieve_harmonic(patch, points, desc)
            entry['in_shortlist'] = near(shortlist, xy)
            dc = dense_candidates(image, patch)
            entry['in_dense'] = near(dc, xy)
            proposed = np.unique(np.vstack([shortlist, dc]), axis=0)
            entry['in_proposals'] = near(proposed, xy)
            kept = verify(blurred_scene,
                          cv2.GaussianBlur(patch.astype(np.float32), (0, 0), .6),
                          proposed, 20)
            entry['in_verified'] = near(kept, xy)
            if entry['in_verified']:
                pose = [k for k in kept
                        if np.linalg.norm(np.array(k[:2]) - xy) <= RADIUS]
                entry['verified_pose'] = [round(float(pose[0][3]), 1),
                                          round(float(pose[0][4]), 3)]
                ref = refine_candidates(image, patch, kept)
                entry['in_refined_recomputed'] = near(ref, xy)
            entry['stage'] = classify(entry)
            print('   ', json.dumps(entry), flush=True)
            report.append(entry)
    out = ROOT / 'outputs/lab/missing_attribution.json'
    out.write_text(json.dumps(report, indent=2))
    from collections import Counter
    print('\nattribution:', Counter(e['stage'] for e in report).most_common())
    print(f'written to {out}')


def classify(e):
    if e['border']:
        return 'border'
    if e['nearest_index_point_px'] > RADIUS:
        return 'index'
    if not e['in_proposals']:
        return 'retrieval+dense' if not e['in_dense'] else 'retrieval'
    if not e['in_verified']:
        return 'verify'
    if not e.get('in_refined_recomputed', True):
        return 'refine'
    return 'present-on-recompute'


if __name__ == '__main__':
    main()

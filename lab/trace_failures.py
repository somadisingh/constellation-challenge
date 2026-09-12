"""Per-query trace of the seven verification failures, mirroring production exactly.

The earlier attribution established that all seven reach the proposal set, and
`lab/verify_variants.py` reported a *median* pre-selection rank over all present
queries, which says nothing about these seven individually. This reproduces
`retrieval.verify` step by step, including its ordering and tie handling, then runs
the real `refine.refine_candidates`, and records for each failure:

  1. proposals in the correct neighbourhood, with their verification scores
  2. rank before suppression
  3. whether suppression removed it, and which candidate suppressed it
  4. whether truncation at `keep` removed it
  5. whether local pose refinement moved it away
  6. whether ECC preserved it

Correctness is tracked at both 4px and 12px.
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from constellation.retrieval import normalize
from constellation.refine import refine_candidates
from lab.cache import load_train, TRAIN, ROOT
from lab.samples import load_scene_patches
from lab.proposals import build_train as build_proposals

KEEP, NMS = 20, 8.


def verify_traced(image, patch, candidates, keep=KEEP, truth=None):
    """Byte-faithful reproduction of retrieval.verify, with a trace."""
    yy, xx = np.mgrid[-12:13:2, -12:13:2].astype(np.float32)
    mask = (xx * xx + yy * yy <= 144); xx = xx[mask]; yy = yy[mask]
    xmap = candidates[:, 0, None].astype(np.float32) + xx
    ymap = candidates[:, 1, None].astype(np.float32) + yy
    samples = normalize(cv2.remap(image, xmap, ymap, cv2.INTER_LINEAR))
    templates, transforms = [], []
    for scale in (.75, .87, 1., 1.15, 1.33):
        for angle in np.arange(0, 360, 15):
            a = np.deg2rad(angle)
            mx = (15.5 + (np.cos(a) * xx - np.sin(a) * yy) / scale).astype(np.float32)[None, :]
            my = (15.5 + (np.sin(a) * xx + np.cos(a) * yy) / scale).astype(np.float32)[None, :]
            v = (mx >= 0) & (mx <= 31) & (my >= 0) & (my <= 31)
            if not v.all():
                continue
            templates.append(cv2.remap(patch, mx, my, cv2.INTER_LINEAR).ravel())
            transforms.append((float(angle), float(scale)))
    templates = normalize(np.array(templates))
    scores = samples @ templates.T
    best = scores.max(axis=1)
    order = np.argsort(best)[::-1]          # production ordering, ties included

    d_truth = (np.linalg.norm(candidates[:, :2] - np.asarray(truth, float), axis=1)
               if truth is not None else None)
    good = set(np.where(d_truth <= 12)[0].tolist()) if truth is not None else set()
    trace = {'n_proposals': int(len(candidates)),
             'good_proposals': sorted(good),
             'good_scores': sorted((round(float(best[i]), 4) for i in good),
                                   reverse=True),
             'best_score_overall': round(float(best.max()), 4)}
    if good:
        pos = {int(i): int(np.where(order == i)[0][0]) for i in good}
        trace['pre_selection_rank'] = min(pos.values())
        trace['pre_selection_rank_all'] = sorted(pos.values())
        trace['top_score_in_neighbourhood'] = round(
            float(max(best[i] for i in good)), 4)

    selected, suppressed_by, truncated = [], None, False
    reached = False
    for idx in order:
        pt = candidates[idx]
        blocker = next((s for s in selected
                        if np.linalg.norm(pt[:2] - np.array(s[:2])) < NMS), None)
        if idx in good and not reached:
            reached = True
            if blocker is not None:
                suppressed_by = {'blocker_xy': [round(float(v), 2) for v in blocker[:2]],
                                 'blocker_score': round(float(blocker[2]), 4),
                                 'distance': round(float(np.linalg.norm(
                                     pt[:2] - np.array(blocker[:2]))), 2)}
            elif len(selected) >= keep:
                truncated = True
        if blocker is not None:
            continue
        angle, scale = transforms[int(scores[idx].argmax())]
        selected.append((float(pt[0]), float(pt[1]), float(best[idx]), angle, scale))
        if len(selected) >= keep:
            break
    trace['suppressed'] = suppressed_by
    trace['truncated_at_keep'] = truncated
    sel = np.array([s[:2] for s in selected]).reshape(-1, 2)
    if truth is not None and len(sel):
        d = np.linalg.norm(sel - np.asarray(truth, float), axis=1)
        trace['after_selection_min_px'] = round(float(d.min()), 2)
        trace['after_selection_hit12'] = bool((d <= 12).any())
        trace['after_selection_hit4'] = bool((d <= 4).any())

    # Production applies its own local pose refinement inside verify, then ECC.
    refined = []
    for x, y, _, angle, scale in selected:
        bestv = (-2, None)
        for da in (-7.5, 0, 7.5):
            a = np.deg2rad(angle + da)
            for ds in (.94, 1, 1.06):
                mx = (15.5 + (np.cos(a) * xx - np.sin(a) * yy) / (scale * ds)).astype(np.float32)[None, :]
                my = (15.5 + (np.sin(a) * xx + np.cos(a) * yy) / (scale * ds)).astype(np.float32)[None, :]
                if mx.min() < 0 or mx.max() > 31 or my.min() < 0 or my.max() > 31:
                    continue
                q = normalize(cv2.remap(patch, mx, my, cv2.INTER_LINEAR))
                offs = np.array([(dx, dy) for dx in (-2, -1, 0, 1, 2)
                                 for dy in (-2, -1, 0, 1, 2)], np.float32)
                sm = cv2.remap(image, (x + offs[:, 0, None] + xx).astype(np.float32),
                               (y + offs[:, 1, None] + yy).astype(np.float32),
                               cv2.INTER_LINEAR)
                corr = normalize(sm) @ q.ravel(); k = int(corr.argmax())
                if corr[k] > bestv[0]:
                    bestv = (float(corr[k]), (x + float(offs[k, 0]), y + float(offs[k, 1]),
                                              float(corr[k]), angle + da, scale * ds))
        if bestv[1] is not None:
            refined.append(bestv[1])
    refined = sorted(refined, key=lambda v: -v[2])
    if truth is not None and refined:
        d = np.linalg.norm(np.array([r[:2] for r in refined]) - np.asarray(truth, float), axis=1)
        trace['after_local_refine_min_px'] = round(float(d.min()), 2)
        trace['after_local_refine_hit12'] = bool((d <= 12).any())
    return selected, refined, trace


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    cache = load_train()
    props = build_proposals()
    report = []
    for n in TRAIN:
        failures = [i for i, q in enumerate(cache[n]['refined'])
                    if truth[n].patches[i] is not None
                    and np.linalg.norm(np.array([c[:2] for c in q])
                                       - np.array(truth[n].patches[i][:2]),
                                       axis=1).min() > 12]
        if not failures:
            continue
        image = cv2.imread(str(ROOT / 'train' / n / f'{n}_image.png'), 0)
        blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), .6)
        patches = load_scene_patches(n, 'train', len(cache[n]['refined']))
        for i in failures:
            t = np.array(truth[n].patches[i][:2], float)
            p = np.asarray(props[n][i], float)
            pb = cv2.GaussianBlur(patches[i].astype(np.float32), (0, 0), .6)
            selected, refined, tr = verify_traced(blurred, pb, p, KEEP, t)
            ecc = refine_candidates(image, patches[i], refined)
            if ecc:
                d = np.linalg.norm(np.array([c[:2] for c in ecc]) - t, axis=1)
                tr['after_ecc_min_px'] = round(float(d.min()), 2)
                tr['after_ecc_hit12'] = bool((d <= 12).any())
                tr['after_ecc_hit4'] = bool((d <= 4).any())
            tr.update(scene=n, query=i + 1, truth=t.tolist(),
                      figure=bool(truth[n].patches[i][2] == 1))
            report.append(tr)
            print(json.dumps(tr), flush=True)
    out = ROOT / 'outputs/lab/failure_traces.json'
    out.write_text(json.dumps(report, indent=2))

    print('\nsummary of the seven:')
    for k, label in (('suppressed', 'removed by 8px suppression'),
                     ('truncated_at_keep', 'removed by truncation at 20')):
        hits = [f"{r['scene']}#{r['query']}" for r in report if r.get(k)]
        print(f'  {label:34s} {len(hits)}  {hits}')
    ranks = [r.get('pre_selection_rank') for r in report]
    print(f'  pre-suppression rank per failure  {ranks}')
    surv = [f"{r['scene']}#{r['query']}" for r in report
            if r.get('after_selection_hit12') and not r.get('after_ecc_hit12')]
    print(f'  survived selection, lost later    {len(surv)}  {surv}')
    print(f'  written to {out}')


if __name__ == '__main__':
    main()

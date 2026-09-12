"""Verification variants, measured on identical cached proposals.

All seven uncovered present queries reach the proposal set and are lost inside
`verify`. `verify` scores each proposal by normalized correlation over a stride-2
disc of radius 12, using only a 0.6px blur -- no high-pass -- then applies an 8px
non-maximum suppression and keeps 20. The rank study on retained candidates showed
raw correlation is markedly worse than background-subtracted or DoG correlation, so
representation is the first suspect.

Reported per variant:
  recall@12   present queries with a retained candidate within 12px (the quantity
              that bounds everything downstream)
  pre-rank    rank of the truth-nearest proposal among all proposals, before
              suppression and truncation. Separates a scoring failure (rank deep)
              from a selection failure (rank shallow but discarded)
  post        whether it survives suppression and truncation
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from constellation.retrieval import normalize
from lab.cache import TRAIN, ROOT
from lab.samples import load_scene_patches
from lab.proposals import build_train

SCALES = (.75, .87, 1., 1.15, 1.33)
ANGLES = np.arange(0, 360, 15)


def pose_templates(patch, xx, yy):
    """Patch sampled at scene-aligned offsets for every admissible pose."""
    templates, transforms = [], []
    for scale in SCALES:
        for angle in ANGLES:
            a = np.deg2rad(angle)
            mx = (15.5 + (np.cos(a) * xx - np.sin(a) * yy) / scale).astype(np.float32)[None, :]
            my = (15.5 + (np.sin(a) * xx + np.cos(a) * yy) / scale).astype(np.float32)[None, :]
            if not ((mx >= 0) & (mx <= 31) & (my >= 0) & (my <= 31)).all():
                continue
            templates.append(cv2.remap(patch, mx, my, cv2.INTER_LINEAR).ravel())
            transforms.append((float(angle), float(scale)))
    return normalize(np.array(templates)), transforms


def disc(step, radius=12):
    yy, xx = np.mgrid[-radius:radius + 1:step, -radius:radius + 1:step].astype(np.float32)
    m = (xx * xx + yy * yy) <= radius * radius
    return xx[m], yy[m]


def score_proposals(image, patch, proposals, step=2, radius=12):
    xx, yy = disc(step, radius)
    xmap = proposals[:, 0, None].astype(np.float32) + xx
    ymap = proposals[:, 1, None].astype(np.float32) + yy
    samples = normalize(cv2.remap(image, xmap, ymap, cv2.INTER_LINEAR))
    templates, transforms = pose_templates(patch, xx, yy)
    if not len(templates):
        return np.full(len(proposals), -1.), None
    s = samples @ templates.T
    return s.max(axis=1), [transforms[i] for i in s.argmax(axis=1)]


def select(proposals, scores, keep=20, nms=8.):
    order = np.argsort(-scores)
    chosen = []
    for idx in order:
        p = proposals[idx, :2]
        if any(np.linalg.norm(p - proposals[j, :2]) < nms for j in chosen):
            continue
        chosen.append(int(idx))
        if len(chosen) >= keep:
            break
    return chosen


REPS = {
    'baseline blur0.6': lambda x: cv2.GaussianBlur(x, (0, 0), .6),
    'bgsub s3': lambda x: (cv2.GaussianBlur(x, (0, 0), .6)
                           - cv2.GaussianBlur(x, (0, 0), 3.)),
    'dog .8/2.5': lambda x: (cv2.GaussianBlur(x, (0, 0), .8)
                             - cv2.GaussianBlur(x, (0, 0), 2.5)),
    'dog .8/4': lambda x: (cv2.GaussianBlur(x, (0, 0), .8)
                           - cv2.GaussianBlur(x, (0, 0), 4.)),
}

VARIANTS = [
    ('baseline (production)', 'baseline blur0.6', 2, 12, 20, 8.),
    ('bgsub s3', 'bgsub s3', 2, 12, 20, 8.),
    ('dog .8/2.5', 'dog .8/2.5', 2, 12, 20, 8.),
    ('dog .8/4', 'dog .8/4', 2, 12, 20, 8.),
    ('dog .8/2.5, step1', 'dog .8/2.5', 1, 12, 20, 8.),
    ('dog .8/2.5, r15', 'dog .8/2.5', 2, 15, 20, 8.),
    ('baseline, keep 40', 'baseline blur0.6', 2, 12, 40, 8.),
    ('baseline, nms 4', 'baseline blur0.6', 2, 12, 20, 4.),
    ('dog .8/2.5, keep 40', 'dog .8/2.5', 2, 12, 40, 8.),
]


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    props = build_train()
    images = {n: cv2.imread(str(ROOT / 'train' / n / f'{n}_image.png'), 0).astype(np.float32)
              for n in TRAIN}
    patches = {n: load_scene_patches(n, 'train', len(props[n])) for n in TRAIN}

    print(f"{'variant':24s} {'recall@12':>10s} {'recall@4':>9s} "
          f"{'pre-rank med':>13s} {'lost after sel':>15s} {'sec':>5s}")
    for label, rep, step, radius, keep, nms in VARIANTS:
        f = REPS[rep]
        hit12 = hit4 = n_present = lost = 0
        pre_ranks = []
        t0 = time.perf_counter()
        for n in TRAIN:
            img = f(images[n])
            for i, prop in enumerate(props[n]):
                t = truth[n].patches[i]
                if t is None:
                    continue
                n_present += 1
                p = np.asarray(prop, float)
                d = np.linalg.norm(p[:, :2] - np.array(t[:2]), axis=1)
                if d.min() > 12:
                    continue                      # not in proposals at all
                sc, _ = score_proposals(img, f(patches[n][i].astype(np.float32)),
                                        p, step, radius)
                good = np.where(d <= 12)[0]
                order = np.argsort(-sc)
                pre = int(np.where(np.isin(order, good))[0][0])
                pre_ranks.append(pre)
                chosen = select(p, sc, keep, nms)
                cd = np.linalg.norm(p[chosen, :2] - np.array(t[:2]), axis=1)
                if (cd <= 12).any():
                    hit12 += 1
                    if (cd <= 4).any():
                        hit4 += 1
                else:
                    lost += 1
        print(f'{label:24s} {hit12/n_present:10.3f} {hit4/n_present:9.3f} '
              f'{np.median(pre_ranks):13.0f} {lost:15d} '
              f'{time.perf_counter()-t0:5.0f}', flush=True)


if __name__ == '__main__':
    main()

"""Verification variants scored over the *whole* proposal pool (~2200 per query).

The seven uncovered present queries reach verification and are ranked 23-247 there, so
any re-ranker that only sees the 20 retained candidates cannot recover them. These
variants therefore score every cached proposal.

Production samples the scene on a fixed radius-12 disc and compares against the patch
resampled at `15.5 + R(angle) . offset / scale`, skipping any pose whose patch
coordinates leave [0, 31]. Two consequences, both measured:

  * at scale 0.75 only 12 of 24 angles are admissible, so half that scale is unusable;
  * at scale 1.33 the admissible radius is 20.6px but only 12px is sampled, discarding
    about two thirds of the valid area.

`adaptive` sets the disc radius per scale to `floor(15.2 * scale)`, which keeps every
sample inside the patch by construction (asserted) while using the evidence the pose
actually admits. Sampling direction, the 15.5 even-patch centre convention and the
valid-overlap requirement are all preserved; nothing samples outside the patch, so a
larger support cannot be rewarded for reading invalid borders.

Poses are searched afresh for every proposal; no cached pose metadata is reused.
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.retrieval import normalize
from lab.cache import TRAIN, ROOT

SCALES = (.75, .87, 1., 1.15, 1.33)
ANGLES = np.arange(0, 360, 15)
CENTRE = 15.5

REPS = {
    'blur0.6': lambda x: cv2.GaussianBlur(x, (0, 0), .6),
    'dog.8/2.5': lambda x: (cv2.GaussianBlur(x, (0, 0), .8)
                            - cv2.GaussianBlur(x, (0, 0), 2.5)),
    'dog.8/4': lambda x: (cv2.GaussianBlur(x, (0, 0), .8)
                          - cv2.GaussianBlur(x, (0, 0), 4.)),
    'bgsub3': lambda x: (cv2.GaussianBlur(x, (0, 0), .6)
                         - cv2.GaussianBlur(x, (0, 0), 3.)),
}


def disc(radius, stride):
    yy, xx = np.mgrid[-radius:radius + 1:stride,
                      -radius:radius + 1:stride].astype(np.float32)
    m = (xx * xx + yy * yy) <= radius * radius
    return xx[m], yy[m]


def radius_for(scale, policy):
    if policy == 'fixed12':
        return 12
    return max(6, int(np.floor(15.2 * scale)))


def score_pool(scene_rep, patch_rep, proposals, policy='adaptive', stride=2,
               channels='disc'):
    """Best-over-pose score for every proposal. Returns (scores, poses)."""
    n = len(proposals)
    best = np.full(n, -2., np.float32)
    best_pose = np.zeros((n, 2), np.float32)
    px = proposals[:, 0, None].astype(np.float32)
    py = proposals[:, 1, None].astype(np.float32)
    for scale in SCALES:
        r = radius_for(scale, policy)
        xx, yy = disc(r, stride)
        if len(xx) < 12:
            continue
        samples = cv2.remap(scene_rep, px + xx, py + yy, cv2.INTER_LINEAR)
        if channels == 'centre_annulus':
            rad = np.hypot(xx, yy)
            inner, outer = rad <= r * .45, rad > r * .45
            if inner.sum() < 6 or outer.sum() < 6:
                continue
            parts = [normalize(samples[:, inner]), normalize(samples[:, outer])]
        else:
            parts = [normalize(samples)]
        tmpl = [[] for _ in parts]
        kept = []
        for angle in ANGLES:
            a = np.deg2rad(angle)
            mx = (CENTRE + (np.cos(a) * xx - np.sin(a) * yy) / scale).astype(np.float32)
            my = (CENTRE + (np.sin(a) * xx + np.cos(a) * yy) / scale).astype(np.float32)
            if not ((mx >= 0) & (mx <= 31) & (my >= 0) & (my <= 31)).all():
                continue                      # valid-overlap requirement, preserved
            v = cv2.remap(patch_rep, mx[None, :], my[None, :], cv2.INTER_LINEAR).ravel()
            if channels == 'centre_annulus':
                tmpl[0].append(v[inner]); tmpl[1].append(v[outer])
            else:
                tmpl[0].append(v)
            kept.append((float(angle), float(scale)))
        if not kept:
            continue
        chan = [p @ normalize(np.array(t)).T for p, t in zip(parts, tmpl)]
        s = chan[0] if len(chan) == 1 else np.minimum(chan[0], chan[1])
        k = s.argmax(axis=1)
        v = s[np.arange(n), k]
        upd = v > best
        best[upd] = v[upd]
        best_pose[upd] = np.array(kept, np.float32)[k[upd]]
    return best, best_pose


def select(proposals, scores, keep=20, nms=8.):
    """Production spatial selection: descending score, 8px suppression, truncate."""
    order = np.argsort(scores)[::-1]
    chosen = []
    for idx in order:
        p = proposals[idx, :2]
        if any(np.linalg.norm(p - proposals[j, :2]) < nms for j in chosen):
            continue
        chosen.append(int(idx))
        if len(chosen) >= keep:
            break
    return chosen


VARIANTS = [
    ('production (fixed12, blur)', 'blur0.6', 'fixed12', 2, 'disc'),
    ('adaptive radius, blur', 'blur0.6', 'adaptive', 2, 'disc'),
    ('adaptive, dog.8/2.5', 'dog.8/2.5', 'adaptive', 2, 'disc'),
    ('adaptive, dog.8/4', 'dog.8/4', 'adaptive', 2, 'disc'),
    ('adaptive, bgsub3', 'bgsub3', 'adaptive', 2, 'disc'),
    ('fixed12, dog.8/2.5', 'dog.8/2.5', 'fixed12', 2, 'disc'),
    ('adaptive, blur, stride1', 'blur0.6', 'adaptive', 1, 'disc'),
    ('adaptive, dog, centre+annulus', 'dog.8/2.5', 'adaptive', 2, 'centre_annulus'),
    ('adaptive, blur, centre+annulus', 'blur0.6', 'adaptive', 2, 'centre_annulus'),
]

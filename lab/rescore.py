"""Classical appearance re-ranking of the existing 20 candidates per query.

Candidate locations are held fixed, so any gain here is attributable to ranking
alone and needs no new global search. The incumbent is the cached ECC-refined
background-subtracted correlation stored as candidate[2].

No classifier is trained. Every variant is a fixed classical rule; the only fitted
quantities are the small number of combination weights swept in lab/sweep_rescore.py,
and those are reported under leave-one-scene-out.
"""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.cache import TRAIN
from lab.samples import SIZE, CENTRE

_V, _U = np.mgrid[:SIZE, :SIZE].astype(np.float32)
_R = np.hypot(_U - CENTRE, _V - CENTRE)
MASKS = {
    'full': np.ones_like(_R, bool),
    'disc14': _R <= 14,
    'centre': _R <= 6,
    'annulus': (_R > 6) & (_R <= 14),
}


# ---------------------------------------------------------------- representations
def rep_raw(x):
    return x


def rep_bgsub(x, sigma=3.):
    return x - _blur(x, sigma)


def rep_dog(x, lo=.8, hi=2.5):
    return _blur(x, lo) - _blur(x, hi)


def rep_lcn(x, sigma=3., eps=1e-3):
    """Local contrast normalization: removes smooth illumination and local gain."""
    h = x - _blur(x, sigma)
    energy = np.sqrt(np.maximum(_blur(h * h, sigma), 0.)) + eps
    return h / energy


def rep_gradmag(x, sigma=1.):
    b = _blur(x, sigma)
    gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _blur(x, sigma):
    if x.ndim == 2:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    return np.stack([cv2.GaussianBlur(v, (0, 0), sigma) for v in x])


def _gradients(x, sigma=1.):
    b = _blur(x, sigma)
    if b.ndim == 2:
        gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
        return gx, gy
    gx = np.stack([cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3) for v in b])
    gy = np.stack([cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3) for v in b])
    return gx, gy


# ------------------------------------------------------------------------ scores
def ncc(patch, samples, rep=rep_bgsub, mask='disc14'):
    """Masked normalized cross-correlation between patch and each sample."""
    m = MASKS[mask]
    p = rep(patch)[m]
    s = rep(samples)[:, m]
    p = p - p.mean()
    s = s - s.mean(axis=1, keepdims=True)
    den = np.linalg.norm(p) * np.linalg.norm(s, axis=1)
    return (s @ p) / np.maximum(den, 1e-9)


def orientation_agreement(patch, samples, mask='disc14', sigma=1.):
    """Magnitude-weighted agreement of gradient orientation, modulo pi.

    Insensitive to contrast sign and to any positive per-image gain, so it does
    not reward a candidate merely for being bright.
    """
    m = MASKS[mask]
    pgx, pgy = _gradients(patch, sigma)
    sgx, sgy = _gradients(samples, sigma)
    pgx, pgy = pgx[m], pgy[m]
    sgx, sgy = sgx[:, m], sgy[:, m]
    pmag = np.hypot(pgx, pgy)
    smag = np.hypot(sgx, sgy)
    # cos(2 d_theta) via double-angle vectors, avoids explicit arctan2.
    pc, ps = (pgx * pgx - pgy * pgy), (2 * pgx * pgy)
    sc, ss = (sgx * sgx - sgy * sgy), (2 * sgx * sgy)
    pn = np.maximum(pmag ** 2, 1e-9)
    sn = np.maximum(smag ** 2, 1e-9)
    cos2 = (pc / pn) * (sc / sn) + (ps / pn) * (ss / sn)
    w = pmag * smag
    return (w * cos2).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)


def robust_photometric(patch, samples, rep=rep_bgsub, mask='disc14', trim=.2):
    """Fit sample -> patch with gain and offset, score by a trimmed residual.

    Handles the measured brightness difference between patches (mean ~110) and
    scene background (~21-35) without letting a few bright pixels dominate.
    """
    m = MASKS[mask]
    p = rep(patch)[m]
    s = rep(samples)[:, m]
    pm = p.mean()
    sm = s.mean(axis=1, keepdims=True)
    pc, sc = p - pm, s - sm
    gain = (sc @ pc) / np.maximum((sc * sc).sum(axis=1), 1e-9)
    resid = np.abs(gain[:, None] * sc - pc[None, :])
    keep = int(round(resid.shape[1] * (1 - trim)))
    resid = np.sort(resid, axis=1)[:, :keep]
    scale = np.maximum(np.abs(pc).mean(), 1e-6)
    return 1. - resid.mean(axis=1) / scale


def multiblur_ncc(patch, samples, rep=rep_bgsub, mask='disc14',
                  sigmas=(0., .7, 1.4)):
    """Best correlation over a small set of blur hypotheses.

    Model flexibility is bounded to three levels so a wrong candidate cannot be
    rescued by unlimited smoothing.
    """
    best = None
    for sg in sigmas:
        s = samples if sg == 0 else _blur(samples, sg)
        v = ncc(patch, s, rep, mask)
        best = v if best is None else np.maximum(best, v)
    return best


def centre_annulus(patch, samples, rep=rep_bgsub, mode='min'):
    """Combine central-disc and surrounding-annulus correlation.

    A generic bright central star gives a high central correlation almost
    anywhere; requiring the annulus to agree suppresses that.
    """
    c = ncc(patch, samples, rep, 'centre')
    a = ncc(patch, samples, rep, 'annulus')
    if mode == 'min':
        return np.minimum(c, a)
    if mode == 'annulus':
        return a
    if mode == 'mean':
        return .5 * (c + a)
    raise ValueError(mode)


# ----------------------------------------------------------------------- registry
VARIANTS = {
    'incumbent (cached ECC)': None,                       # uses candidate[2]
    'raw ncc': lambda p, s: ncc(p, s, rep_raw, 'disc14'),
    'bgsub ncc disc14': lambda p, s: ncc(p, s, rep_bgsub, 'disc14'),
    'bgsub ncc full32': lambda p, s: ncc(p, s, rep_bgsub, 'full'),
    'dog ncc': lambda p, s: ncc(p, s, rep_dog, 'disc14'),
    'lcn ncc': lambda p, s: ncc(p, s, rep_lcn, 'disc14'),
    'gradmag ncc': lambda p, s: ncc(p, s, rep_gradmag, 'disc14'),
    'orientation agreement': lambda p, s: orientation_agreement(p, s, 'disc14'),
    'robust photometric': lambda p, s: robust_photometric(p, s, rep_bgsub, 'disc14'),
    'multiblur ncc': lambda p, s: multiblur_ncc(p, s, rep_bgsub, 'disc14'),
    'centre+annulus min': lambda p, s: centre_annulus(p, s, rep_bgsub, 'min'),
    'annulus only': lambda p, s: centre_annulus(p, s, rep_bgsub, 'annulus'),
    'lcn full32': lambda p, s: ncc(p, s, rep_lcn, 'full'),
    'dog full32': lambda p, s: ncc(p, s, rep_dog, 'full'),
}


def score_scene(fn, patches, samples, counts, cached):
    """Scores per query; falls back to the cached score when fn is None."""
    out = []
    for i in range(len(patches)):
        k = int(counts[i])
        if fn is None:
            out.append(np.array([c[2] for c in cached[i]], float))
        else:
            out.append(np.asarray(fn(patches[i], samples[i, :k]), float))
    return out


def category(truth_patch):
    if truth_patch is None:
        return 'absent'
    return 'figure' if truth_patch[2] == 1 else 'off-figure'

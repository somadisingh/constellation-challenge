"""Candidate/query alignment, derived from the classical sampling convention (plan §6).

`constellation.retrieval.verify` reads the QUERY at scene-aligned offsets:

    q = 15.5 + R(a) @ offset / scale,      a = deg2rad(angle),
    R(a) = [[cos a, -sin a], [sin a, cos a]]

so the inverse map, from query-frame coordinates to source pixels, is

    offset = scale * R(-a) @ (q - 15.5),   source = candidate_xy + offset

`constellation.refine.refine_candidates` builds exactly that matrix
(`[[c, s, ...], [-s, c, ...]]` with `c = scale*cos a`, `s = scale*sin a`, linear
part `scale*R(-a)`) and applies it with `WARP_INVERSE_MAP`, which confirms the
sign, the inverse-scale and the 15.5 centre.

Two consequences the plan calls out:

* `refine_candidates` returns the ORIGINAL `angle, scale` and only updates the
  centre from the ECC warp. The refined affine matrix is not stored anywhere, so
  the cached pose fields cannot reproduce the final ECC warp. We therefore
  re-estimate the local pose around each candidate with its CENTRE FIXED.
* Warping source -> query frame has no in-patch admissibility limit (unlike the
  classical direction, where patch coordinates could leave [0,31]). A pose is
  rejected only when it would read outside the image, and rejected poses produce
  no evidence rather than black pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

from .data import PATCH, PATCH_MID

# Plan §6 default local pose check: stored classical angle +/-10 deg, scale x .9/1/1.1.
ANGLE_TRIALS = (-10.0, 0.0, 10.0)
SCALE_TRIALS = (0.9, 1.0, 1.1)
N_POSE_TRIALS = len(ANGLE_TRIALS) * len(SCALE_TRIALS)

# Circular support used by the masked classical criterion. 15.5 is the inscribed
# radius of the 32x32 query frame about its (15.5, 15.5) midpoint.
SUPPORT_RADIUS = 15.5
# Antialias prefilter strength when a pose downsamples the source (scale > 1).
AA_FACTOR = 0.5
CLASSICAL_BLUR = 0.6      # matches pipeline VERIFY_REPS['blur']


def rotation(angle_deg: float) -> np.ndarray:
    """R(a) as used by the classical convention."""
    a = np.deg2rad(angle_deg)
    return np.array([[np.cos(a), -np.sin(a)],
                     [np.sin(a), np.cos(a)]], dtype=np.float64)


def source_to_query(offset, angle_deg: float, scale: float) -> np.ndarray:
    """Scene-frame offset from the candidate centre -> query-patch coordinate.

    This is the classical formula verbatim; used to check the round trip.
    """
    off = np.asarray(offset, float).reshape(-1, 2)
    a = np.deg2rad(angle_deg)
    qx = PATCH_MID + (np.cos(a) * off[:, 0] - np.sin(a) * off[:, 1]) / scale
    qy = PATCH_MID + (np.sin(a) * off[:, 0] + np.cos(a) * off[:, 1]) / scale
    return np.c_[qx, qy]


def query_to_source(q, centre, angle_deg: float, scale: float) -> np.ndarray:
    """Query-patch coordinate -> absolute source pixel coordinate."""
    qq = np.asarray(q, float).reshape(-1, 2)
    u = qq[:, 0] - PATCH_MID
    v = qq[:, 1] - PATCH_MID
    a = np.deg2rad(angle_deg)
    dx = scale * (np.cos(a) * u + np.sin(a) * v)
    dy = scale * (-np.sin(a) * u + np.cos(a) * v)
    cx, cy = float(centre[0]), float(centre[1])
    return np.c_[cx + dx, cy + dy]


def query_grid(size: int = PATCH) -> np.ndarray:
    """Query-frame coordinates of an oversampled grid centred on (15.5, 15.5).

    For `size == PATCH` these are exactly 0..31. For larger `size` the grid keeps
    the same pixel pitch and extends symmetrically, so the centre `PATCH` block is
    the query itself (offset `(size - PATCH) // 2`).
    """
    half = (size - 1) / 2.0
    axis = PATCH_MID + np.arange(size, dtype=np.float64) - half
    gx, gy = np.meshgrid(axis, axis)
    return np.stack([gx, gy], axis=-1)


def _prefilter(image: np.ndarray, scale: float, aa_factor: float) -> np.ndarray:
    """Antialias before a downsampling read, so scale > 1 does not alias."""
    if aa_factor <= 0 or scale <= 1.0:
        return image
    sigma = aa_factor * np.sqrt(max(scale * scale - 1.0, 0.0))
    if sigma < 0.05:
        return image
    return cv2.GaussianBlur(image, (0, 0), float(sigma))


def sample_query_frame(image: np.ndarray, centre, angle_deg: float, scale: float,
                       size: int = PATCH, aa_factor: float = AA_FACTOR):
    """Resample `image` into the query coordinate frame of one pose.

    Returns `(values, mask)`; `mask` marks samples that lie inside the image.
    `image` must already be float32 (raw or a chosen representation).
    """
    grid = query_grid(size)
    src = query_to_source(grid.reshape(-1, 2), centre, angle_deg, scale)
    h, w = image.shape[:2]
    inside = ((src[:, 0] >= 0) & (src[:, 0] <= w - 1) &
              (src[:, 1] >= 0) & (src[:, 1] <= h - 1))
    prepared = _prefilter(image, scale, aa_factor)
    mx = src[:, 0].reshape(size, size).astype(np.float32)
    my = src[:, 1].reshape(size, size).astype(np.float32)
    values = cv2.remap(prepared, mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return values, inside.reshape(size, size)


def aligned_candidate(image: np.ndarray, candidate_xy, pose,
                      aa_factor: float = AA_FACTOR):
    """Warp the candidate neighbourhood into the query frame (plan §6 adapter).

    `pose` is `(angle_deg, scale)`. Returns `(crop, mask)` with `crop` float32
    32x32. `mask.all()` false means the pose reads outside the image and must be
    rejected by the caller rather than treated as evidence.
    """
    angle, scale = float(pose[0]), float(pose[1])
    img = image if image.dtype == np.float32 else image.astype(np.float32)
    return sample_query_frame(img, candidate_xy, angle, scale, PATCH, aa_factor)


def support_mask(radius: float = SUPPORT_RADIUS) -> np.ndarray:
    grid = query_grid(PATCH)
    r2 = (grid[..., 0] - PATCH_MID) ** 2 + (grid[..., 1] - PATCH_MID) ** 2
    return r2 <= radius * radius


_SUPPORT = support_mask()


def masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Zero-mean normalised correlation over a mask.

    The single documented classical criterion (plan §6/§8): it selects the pose for
    every model, and it is control C1's score on the same aligned crops.
    """
    m = _SUPPORT if mask is None else (_SUPPORT & mask)
    if m.sum() < 12:
        return float('nan')
    x = a[m].astype(np.float64)
    y = b[m].astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom < 1e-9:
        return 0.0
    return float(x @ y / denom)


def pose_trials(angle_deg: float, scale: float) -> list:
    """The 9 admissible-pose trials around a stored classical pose."""
    return [(float(angle_deg + da), float(scale * ds))
            for da in ANGLE_TRIALS for ds in SCALE_TRIALS]


class SceneReps:
    """Cached float views of one sky: raw for descriptors, blur for the criterion."""

    def __init__(self, image: np.ndarray):
        self.raw = image.astype(np.float32)
        self.blur = cv2.GaussianBlur(self.raw, (0, 0), CLASSICAL_BLUR)

    @property
    def shape(self):
        return self.raw.shape


def prepare_query(patch: np.ndarray):
    """(raw float32, blur float32) views of a 32x32 query, matching SceneReps."""
    raw = np.asarray(patch, np.float32)
    if raw.max() > 1.5:
        raw = raw / 255.0
    return raw, cv2.GaussianBlur(raw, (0, 0), CLASSICAL_BLUR)


def select_pose(reps: SceneReps, query_blur: np.ndarray, candidate_xy,
                base_pose, aa_factor: float = AA_FACTOR) -> dict:
    """Image-estimated local pose with the candidate CENTRE HELD FIXED.

    Chooses among `pose_trials` by `masked_ncc` on the blur representation. Poses
    that read outside the image are rejected outright. The result is model
    independent, so it is cached once and reused by every arm (plan §6/§8).
    """
    best = None
    rejected = 0
    for angle, scale in pose_trials(base_pose[0], base_pose[1]):
        crop, mask = sample_query_frame(reps.blur, candidate_xy, angle, scale,
                                        PATCH, aa_factor)
        if not mask.all():
            rejected += 1
            continue
        score = masked_ncc(query_blur, crop, mask)
        if not np.isfinite(score):
            rejected += 1
            continue
        if best is None or score > best['ncc']:
            best = {'angle': angle, 'scale': scale, 'ncc': float(score)}
    return {
        'pose': None if best is None else (best['angle'], best['scale']),
        'ncc': None if best is None else best['ncc'],
        'trials': N_POSE_TRIALS,
        'rejected': rejected,
        'alignment_failed': best is None,
    }


def oracle_pose_trials(true_angle: float, true_scale: float) -> list:
    """Diagnostic-only pose set centred on a known synthetic pose (plan §6)."""
    return [(float(true_angle), float(true_scale))]

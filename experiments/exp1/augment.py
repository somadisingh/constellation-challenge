"""Real-pixel query synthesis (plan §6).

A synthetic query is a real source neighbourhood resampled through an explicit
query-to-source transform, then degraded. Source morphology is never generated.

Only the QUERY view is degraded. The candidate view stays a plain resample of the
real source image, so no identical noise or JPEG is shared between the views.

The starter ranges in `Ranges` are engineering choices, not recovered generator
parameters. One law is used for positives and negatives, figure and off-figure
categories, and same- and cross-sky donors.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

from .data import PATCH, PATCH_MID
from .pose import query_to_source, sample_query_frame, AA_FACTOR

OVERSAMPLE = 1.5      # synthesis support before the centre crop
GRAY = 255.0


@dataclass
class Ranges:
    """Main distribution (plan §6 table). `stress` swaps in the diagnostic column."""
    rotation: tuple = (0.0, 360.0)
    scale_log: tuple = (0.75, 1.33)
    blur_sigma: tuple = (0.3, 1.2)
    gain: tuple = (0.75, 1.25)
    offset: tuple = (-12.0, 12.0)
    drift: tuple = (0.0, 10.0)
    read_noise: tuple = (0.5, 5.0)
    shot_alpha: tuple = (0.0, 0.05)
    jpeg_none_prob: float = 0.5
    jpeg_quality: tuple = (80, 100)

    @staticmethod
    def stress() -> 'Ranges':
        return Ranges(
            rotation=(0.0, 360.0),
            scale_log=(0.65, 1.50),
            blur_sigma=(1.2, 1.8),
            gain=(0.5, 1.5),
            offset=(-20.0, 20.0),
            drift=(10.0, 20.0),
            read_noise=(5.0, 8.0),
            shot_alpha=(0.0, 0.05),
            jpeg_none_prob=0.5,
            jpeg_quality=(65, 80),
        )

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = Ranges()


def sample_pose(generator, ranges: Ranges = DEFAULT) -> dict:
    lo, hi = ranges.scale_log
    return {
        'angle': float(generator.uniform(*ranges.rotation)),
        'scale': float(np.exp(generator.uniform(np.log(lo), np.log(hi)))),
    }


def sample_degradation(generator, ranges: Ranges = DEFAULT) -> dict:
    jpeg = None
    if generator.random() >= ranges.jpeg_none_prob:
        jpeg = int(generator.integers(ranges.jpeg_quality[0],
                                      ranges.jpeg_quality[1] + 1))
    return {
        'blur_sigma': float(generator.uniform(*ranges.blur_sigma)),
        'gain': float(generator.uniform(*ranges.gain)),
        'offset': float(generator.uniform(*ranges.offset)),
        'drift': float(generator.uniform(*ranges.drift)),
        'drift_angle': float(generator.uniform(0.0, 360.0)),
        'read_noise': float(generator.uniform(*ranges.read_noise)),
        'shot_alpha': float(generator.uniform(*ranges.shot_alpha)),
        'jpeg_quality': jpeg,
        'noise_seed': int(generator.integers(0, 2 ** 31 - 1)),
    }


def synthesize_query(image: np.ndarray, centre, pose: dict, degradation: dict,
                     aa_factor: float = AA_FACTOR) -> dict:
    """Generate one 32x32 query from a real source neighbourhood.

    Pixel midpoint (15.5, 15.5) maps to `centre` by construction: the same
    `query_to_source` map the inference adapter uses, with no extra translation.
    """
    size = int(round(OVERSAMPLE * PATCH))
    if size % 2 != PATCH % 2:
        size += 1
    img = image if image.dtype == np.float32 else image.astype(np.float32)
    wide, mask = sample_query_frame(img, centre, pose['angle'], pose['scale'],
                                    size, aa_factor)
    if not mask.all():
        return {'ok': False, 'reason': 'sampling left the image'}

    # Optical blur at the larger support, then the centre crop.
    sigma = degradation['blur_sigma']
    if sigma > 0.02:
        wide = cv2.GaussianBlur(wide, (0, 0), float(sigma))
    off = (size - PATCH) // 2
    view = wide[off:off + PATCH, off:off + PATCH].astype(np.float64)

    # Radiometry on the 0-255 scale.
    view = view * degradation['gain'] + degradation['offset']

    if degradation['drift'] > 0:
        a = np.deg2rad(degradation['drift_angle'])
        gx, gy = np.meshgrid(np.arange(PATCH) - PATCH_MID, np.arange(PATCH) - PATCH_MID)
        ramp = (np.cos(a) * gx + np.sin(a) * gy) / (PATCH_MID * np.sqrt(2.0))
        view = view + degradation['drift'] * ramp

    # Shot-style variance sigma_read^2 + alpha*max(I,0), all in gray levels.
    gen = np.random.default_rng(degradation['noise_seed'])
    var = degradation['read_noise'] ** 2 + degradation['shot_alpha'] * np.maximum(view, 0.0)
    view = view + gen.normal(0.0, 1.0, view.shape) * np.sqrt(np.maximum(var, 0.0))

    clipped_low = float((view < 0).mean())
    clipped_high = float((view > 255).mean())
    out = np.clip(view, 0, 255).astype(np.uint8)

    if degradation['jpeg_quality'] is not None:
        ok, buf = cv2.imencode('.jpg', out,
                               [int(cv2.IMWRITE_JPEG_QUALITY),
                                int(degradation['jpeg_quality'])])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    std = float(out.std())
    return {
        'ok': True,
        'query': out,                       # uint8, like a real query patch
        'pose': (pose['angle'], pose['scale']),
        'centre': (float(centre[0]), float(centre[1])),
        'clipped_low': clipped_low,
        'clipped_high': clipped_high,
        'std': std,
        'constant': std < 1.0,
        'degenerate': std < 2.0 or (clipped_low + clipped_high) > 0.25,
    }


def perturbed_pose(generator, pose: dict, angle_jitter: float = 10.0,
                   scale_jitter: float = 0.10, centre_jitter: float = 2.0) -> dict:
    """Warmup residual perturbation of a KNOWN synthetic pose (plan §6)."""
    return {
        'angle': float(pose['angle'] + generator.uniform(-angle_jitter, angle_jitter)),
        'scale': float(pose['scale'] * (1.0 + generator.uniform(-scale_jitter, scale_jitter))),
        'dx': float(generator.uniform(-centre_jitter, centre_jitter)),
        'dy': float(generator.uniform(-centre_jitter, centre_jitter)),
    }


def degradation_stats(samples: list) -> dict:
    """Clipping/constant/degenerate rates, reported as a diagnostic set."""
    ok = [s for s in samples if s.get('ok')]
    if not ok:
        return {'n': 0}
    return {
        'n': len(ok),
        'n_failed': len(samples) - len(ok),
        'constant_rate': float(np.mean([s['constant'] for s in ok])),
        'degenerate_rate': float(np.mean([s['degenerate'] for s in ok])),
        'clipped_low_mean': float(np.mean([s['clipped_low'] for s in ok])),
        'clipped_high_mean': float(np.mean([s['clipped_high'] for s in ok])),
        'std_mean': float(np.mean([s['std'] for s in ok])),
        'std_p05': float(np.quantile([s['std'] for s in ok], 0.05)),
        'std_p95': float(np.quantile([s['std'] for s in ok], 0.95)),
    }

"""Split protocol and source banks (plan §5).

Outer folds hold out one sky at a time. Within each allowed sky a 3x3 grid of
cells, inset 96px per side, partitions space: row-major cells 0-5 fit, 6-7 inner
validation, 8 synthetic calibration.

The partition guarantee this module enforces is stronger than "the centre is in
the cell": the whole sampling footprint of a centre, including the oversampled
synthesis support, the antialias prefilter and the blur halo, must lie inside the
centre's own cell. `MARGIN` is that bound, and `max_sampling_radius` derives it
from the augmentation ranges so the two cannot drift apart.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from . import SCENES
from .data import PATCH, PATCH_MID, load_scene
from .env import SEED_SAMPLING, rng, sha256_json

INSET = 96                 # image border excluded entirely (plan §5)
GRID = 3
FIT_CELLS = (0, 1, 2, 3, 4, 5)
VAL_CELLS = (6, 7)
SYNTH_CELLS = (8,)
PARTITIONS = ('fit', 'val', 'synthcal')

# Source neighbourhood the plan declares (96x96) and the partition margin that
# actually keeps every sample inside a cell. See max_sampling_radius().
FOOTPRINT = 96
MARGIN = 72
MIN_SPACING = 8.0

TARGET = {'fit': 20000, 'val': 2000, 'synthcal': 1000}
# Plan §5 mixture: 50% multiscale local peaks, 25% uniform, 25% high-background.
MIXTURE = (('peak', 0.50), ('uniform', 0.25), ('background', 0.25))
BRIGHT_BINS = 4
CONTRAST_BINS = 3


def max_sampling_radius(max_scale: float = 1.50, oversample: float = 1.5,
                        max_blur_sigma: float = 1.8) -> float:
    """Largest source-pixel radius any generated view can touch around a centre.

    A 32x32 query is synthesised on an oversampled grid of side
    `oversample*PATCH`; its corner sits `oversample*PATCH_MID*sqrt(2)` from the
    centre in query units, and query units scale to source pixels by `scale`.
    Blur and interpolation add their own halo.
    """
    corner = oversample * PATCH_MID * math.sqrt(2.0)
    halo = 3.0 * max_blur_sigma + 3.0   # blur support + antialias prefilter
    return max_scale * corner + halo + 2.0   # +2 bilinear interpolation


def cell_box(index: int, height: int, width: int) -> tuple:
    """(x0, y0, x1, y1) half-open pixel box of a row-major 3x3 cell."""
    span_y = (height - 2 * INSET) // GRID
    span_x = (width - 2 * INSET) // GRID
    r, c = divmod(index, GRID)
    x0 = INSET + c * span_x
    y0 = INSET + r * span_y
    return (x0, y0, x0 + span_x, y0 + span_y)


def centre_box(index: int, height: int, width: int, margin: int = MARGIN) -> tuple:
    x0, y0, x1, y1 = cell_box(index, height, width)
    return (x0 + margin, y0 + margin, x1 - margin, y1 - margin)


def partition_of(cell: int) -> str:
    if cell in FIT_CELLS:
        return 'fit'
    if cell in VAL_CELLS:
        return 'val'
    return 'synthcal'


def fold_skies(fold: str) -> tuple:
    """(held_out, allowed_pair) for an outer fold."""
    if fold not in SCENES:
        raise ValueError(f'unknown fold {fold!r}; expected one of {SCENES}')
    return fold, tuple(s for s in SCENES if s != fold)


# --- candidate position generation ------------------------------------------------
def _peak_positions(image: np.ndarray) -> np.ndarray:
    """Multiscale DoG local maxima, the 50% 'local peak' class."""
    raw = image.astype(np.float32)
    out = []
    for lo, hi, thresh in ((0.8, 2.5, 1.5), (1.4, 4.0, 1.2), (2.5, 7.0, 1.0)):
        dog = cv2.GaussianBlur(raw, (0, 0), lo) - cv2.GaussianBlur(raw, (0, 0), hi)
        peak = (dog == cv2.dilate(dog, np.ones((5, 5), np.uint8))) & (dog > thresh)
        y, x = np.where(peak)
        out.append(np.c_[x, y])
    return np.unique(np.vstack(out), axis=0).astype(np.float64)


def _background_positions(image: np.ndarray, generator, count: int) -> np.ndarray:
    """High-background / extended-structure positions: bright at coarse scale but
    not a sharp peak. The 25% 'background' class."""
    raw = image.astype(np.float32)
    coarse = cv2.GaussianBlur(raw, (0, 0), 8.0)
    dog = cv2.GaussianBlur(raw, (0, 0), 0.8) - cv2.GaussianBlur(raw, (0, 0), 2.5)
    peaky = cv2.dilate(dog, np.ones((9, 9), np.uint8))
    score = coarse - 2.0 * np.maximum(peaky, 0)
    flat = score.ravel()
    take = min(flat.size, max(count * 40, 200000))
    idx = np.argpartition(flat, -take)[-take:]
    pick = generator.choice(idx, size=min(count * 6, take), replace=False)
    y, x = np.unravel_index(pick, score.shape)
    return np.c_[x, y].astype(np.float64)


def _stats_at(image: np.ndarray, xy: np.ndarray) -> tuple:
    """Local brightness and contrast at integer positions, for strata balancing."""
    raw = image.astype(np.float32)
    mean = cv2.blur(raw, (17, 17))
    sq = cv2.blur(raw * raw, (17, 17))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, raw.shape[1] - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, raw.shape[0] - 1)
    return mean[yi, xi], std[yi, xi]


def _inside(xy: np.ndarray, box: tuple) -> np.ndarray:
    x0, y0, x1, y1 = box
    return (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)


def _greedy_spaced(xy: np.ndarray, order: np.ndarray, limit: int,
                   spacing: float = MIN_SPACING) -> list:
    """Accept positions in `order`, rejecting any within `spacing` of an accepted one."""
    cell = max(spacing, 1.0)
    grid: dict = {}
    kept = []
    r2 = spacing * spacing
    for i in order:
        if len(kept) >= limit:
            break
        px, py = xy[i, 0], xy[i, 1]
        gx, gy = int(px // cell), int(py // cell)
        clash = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if (xy[j, 0] - px) ** 2 + (xy[j, 1] - py) ** 2 < r2:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if clash:
            continue
        grid.setdefault((gx, gy), []).append(i)
        kept.append(int(i))
    return kept


def _strata(bright: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    """Quantile bins over the candidate pool, so balancing is data-relative."""
    def binned(v, n):
        if len(v) == 0:
            return np.zeros(0, int)
        edges = np.quantile(v, np.linspace(0, 1, n + 1)[1:-1])
        return np.searchsorted(edges, v, side='right')
    return binned(bright, BRIGHT_BINS) * CONTRAST_BINS + binned(contrast, CONTRAST_BINS)


_BANK_CACHE: dict = {}


def build_source_bank(scene_name: str, data: str | None = None,
                      targets: dict | None = None) -> dict:
    """Memoised wrapper: each sky appears in two folds and the bank is fold-free."""
    key = (scene_name, str(data), tuple(sorted((targets or TARGET).items())))
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = _build_source_bank(scene_name, data, targets)
    return _BANK_CACHE[key]


def _build_source_bank(scene_name: str, data: str | None = None,
                       targets: dict | None = None) -> dict:
    """Deterministic per-partition source banks for one sky.

    Selection is stratified round-robin over (mixture class, brightness/contrast
    stratum) buckets, then greedily thinned to `MIN_SPACING`. Figure membership is
    never consulted.
    """
    targets = dict(TARGET if targets is None else targets)
    scene = load_scene(scene_name, data)
    h, w = scene.image.shape
    gen = rng(SEED_SAMPLING, 'bank', scene_name)

    peaks = _peak_positions(scene.image)
    n_uniform = sum(targets.values()) * 12
    uni = np.c_[gen.uniform(INSET, w - INSET, n_uniform),
                gen.uniform(INSET, h - INSET, n_uniform)]
    bg = _background_positions(scene.image, gen, sum(targets.values()))

    pool = np.vstack([peaks, uni, bg])
    klass = np.concatenate([np.zeros(len(peaks), int), np.ones(len(uni), int),
                            np.full(len(bg), 2, int)])
    bright, contrast = _stats_at(scene.image, pool)
    stratum = _strata(bright, contrast)

    cells = {}
    for cell in range(GRID * GRID):
        box = centre_box(cell, h, w)
        sel = np.where(_inside(pool, box))[0]
        part = partition_of(cell)
        n_cells = len({c for c in range(GRID * GRID) if partition_of(c) == part})
        want = int(math.ceil(targets[part] / n_cells))

        buckets: dict = {}
        for i in sel:
            buckets.setdefault((int(klass[i]), int(stratum[i])), []).append(int(i))
        for key in buckets:
            arr = np.array(buckets[key])
            buckets[key] = list(arr[gen.permutation(len(arr))])

        # Round-robin across strata inside each mixture class, then interleave the
        # classes at the plan's 50/25/25 proportions.
        per_class = {}
        for ci, (_, share) in enumerate(MIXTURE):
            keys = sorted(k for k in buckets if k[0] == ci)
            queue, cursor = [], {k: 0 for k in keys}
            while keys:
                progressed = False
                for k in list(keys):
                    if cursor[k] < len(buckets[k]):
                        queue.append(buckets[k][cursor[k]])
                        cursor[k] += 1
                        progressed = True
                    else:
                        keys.remove(k)
                if not progressed:
                    break
            per_class[ci] = queue

        order, cursors = [], {ci: 0 for ci in per_class}
        quota = {ci: share for ci, (_, share) in enumerate(MIXTURE)}
        while True:
            added = False
            for ci in sorted(per_class):
                take = max(1, int(round(quota[ci] * 4)))
                for _ in range(take):
                    if cursors[ci] < len(per_class[ci]):
                        order.append(per_class[ci][cursors[ci]])
                        cursors[ci] += 1
                        added = True
            if not added:
                break

        kept = _greedy_spaced(pool, np.array(order, int), want)
        cells[cell] = {
            'cell': cell,
            'partition': part,
            'cell_box': [int(v) for v in cell_box(cell, h, w)],
            'centre_box': [int(v) for v in box],
            'requested': want,
            'n_centres': len(kept),
            'centres': [[float(pool[i, 0]), float(pool[i, 1])] for i in kept],
            'source_class': [['peak', 'uniform', 'background'][int(klass[i])] for i in kept],
            'brightness': [float(bright[i]) for i in kept],
            'contrast': [float(contrast[i]) for i in kept],
            'stratum': [int(stratum[i]) for i in kept],
        }
    return {
        'scene': scene_name,
        'height': int(h), 'width': int(w),
        'inset': INSET, 'grid': GRID, 'margin': MARGIN,
        'footprint': FOOTPRINT,
        'min_spacing': MIN_SPACING,
        'max_sampling_radius': max_sampling_radius(),
        'targets': targets,
        'mixture': {k: v for k, v in MIXTURE},
        'seed': SEED_SAMPLING,
        'cells': cells,
        'totals': {p: sum(c['n_centres'] for c in cells.values()
                          if c['partition'] == p) for p in PARTITIONS},
    }


def partition_centres(bank: dict, partition: str) -> np.ndarray:
    out = [c['centres'] for c in bank['cells'].values() if c['partition'] == partition]
    return np.array([p for group in out for p in group], float).reshape(-1, 2)


def partition_records(bank: dict, partition: str) -> list:
    """Flat centre records carrying the physical-source id used for grouping."""
    recs = []
    for cell in sorted(bank['cells'], key=int):
        c = bank['cells'][cell]
        if c['partition'] != partition:
            continue
        for k, xy in enumerate(c['centres']):
            recs.append({
                'source_id': f"{bank['scene']}:c{c['cell']}:{k:05d}",
                'scene': bank['scene'],
                'cell': c['cell'],
                'partition': partition,
                'xy': (float(xy[0]), float(xy[1])),
                'source_class': c['source_class'][k],
                'stratum': c['stratum'][k],
                'brightness': c['brightness'][k],
                'contrast': c['contrast'][k],
            })
    return recs


def fold_manifest(fold: str, data: str | None = None,
                  targets: dict | None = None) -> dict:
    held_out, allowed = fold_skies(fold)
    banks = {s: build_source_bank(s, data, targets) for s in allowed}
    manifest = {
        'fold': fold,
        'held_out': held_out,
        'allowed': list(allowed),
        'protocol': {
            'inset': INSET, 'grid': GRID,
            'fit_cells': list(FIT_CELLS), 'val_cells': list(VAL_CELLS),
            'synthcal_cells': list(SYNTH_CELLS),
            'margin': MARGIN, 'footprint': FOOTPRINT,
            'min_spacing': MIN_SPACING,
            'max_sampling_radius': max_sampling_radius(),
            'seeds': {'sampling': SEED_SAMPLING},
        },
        'banks': banks,
        'unique_sources': {s: banks[s]['totals'] for s in allowed},
    }
    manifest['manifest_sha256'] = sha256_json(manifest)
    return manifest


# --- validity checks used by both the CLI and tests -------------------------------
def check_partition_disjoint(height: int, width: int,
                             radius: float | None = None) -> dict:
    """No footprint of any admissible centre can leave its own cell, and no two
    cells' footprint regions intersect."""
    radius = max_sampling_radius() if radius is None else radius
    problems = []
    boxes = {}
    for cell in range(GRID * GRID):
        cx0, cy0, cx1, cy1 = centre_box(cell, height, width)
        bx0, by0, bx1, by1 = cell_box(cell, height, width)
        fx0, fy0 = cx0 - radius, cy0 - radius
        fx1, fy1 = cx1 + radius, cy1 + radius
        boxes[cell] = (fx0, fy0, fx1, fy1)
        if fx0 < bx0 or fy0 < by0 or fx1 > bx1 or fy1 > by1:
            problems.append(f'cell {cell} footprint leaves its cell box')
        if cx1 <= cx0 or cy1 <= cy0:
            problems.append(f'cell {cell} has an empty centre box')
    for a in range(GRID * GRID):
        for b in range(a + 1, GRID * GRID):
            ax0, ay0, ax1, ay1 = boxes[a]
            bx0, by0, bx1, by1 = boxes[b]
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                problems.append(f'cells {a} and {b} have overlapping footprints')
    return {'radius': radius, 'ok': not problems, 'problems': problems,
            'footprint_boxes': {str(k): [float(v) for v in b] for k, b in boxes.items()}}


def check_bank(bank: dict) -> dict:
    """Spacing, containment and cross-partition separation of a realised bank."""
    problems = []
    radius = bank['max_sampling_radius']
    for cell, c in bank['cells'].items():
        xy = np.array(c['centres'], float).reshape(-1, 2)
        if not len(xy):
            continue
        bx0, by0, bx1, by1 = c['cell_box']
        if (xy[:, 0].min() - radius < bx0 or xy[:, 1].min() - radius < by0
                or xy[:, 0].max() + radius > bx1 or xy[:, 1].max() + radius > by1):
            problems.append(f'cell {cell}: a footprint leaves the cell box')
        if len(xy) > 1:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(xy).query(xy, k=2)
            if d[:, 1].min() < bank['min_spacing'] - 1e-9:
                problems.append(f'cell {cell}: spacing {d[:, 1].min():.3f} '
                                f'< {bank["min_spacing"]}')
    groups = {p: partition_centres(bank, p) for p in PARTITIONS}
    from scipy.spatial import cKDTree
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            if not len(groups[a]) or not len(groups[b]):
                continue
            d, _ = cKDTree(groups[b]).query(groups[a], k=1)
            if d.min() < 2 * radius:
                problems.append(f'{a} and {b} centres are {d.min():.1f}px apart, '
                                f'closer than 2x the {radius:.1f}px footprint')
    return {'scene': bank['scene'], 'ok': not problems, 'problems': problems,
            'totals': bank['totals'],
            'class_counts': _class_counts(bank),
            'stratum_counts': _stratum_counts(bank)}


def _class_counts(bank: dict) -> dict:
    out: dict = {}
    for c in bank['cells'].values():
        d = out.setdefault(c['partition'], {})
        for k in c['source_class']:
            d[k] = d.get(k, 0) + 1
    return out


def _stratum_counts(bank: dict) -> dict:
    out: dict = {}
    for c in bank['cells'].values():
        d = out.setdefault(c['partition'], {})
        for k in c['stratum']:
            d[str(k)] = d.get(str(k), 0) + 1
    return out

"""Partition-restricted classical search (plan §5, §7).

Mining and inner validation both need candidate locations for SYNTHETIC queries,
produced by the existing proposal/verification machinery but confined to one
partition. Retrieval must never index a cell outside the requested partition, so
each cell is cropped, indexed and searched independently and its coordinates are
translated back afterwards.

The per-cell index depends only on the sky and the cell box, never on a fold or a
label, so it is cached and reused. Mined banks are likewise per SCENE, not per
fold: a scene's fit cells are the same in both folds where that scene is allowed,
and the held-out sky is never searched at all.
"""
from __future__ import annotations

import numpy as np

from .data import load_scene
from .splits import GRID, cell_box, partition_of

# Retrieval's own unusable border inside any indexed crop.
INDEX_BORDER = 20


class CellIndex:
    """Harmonic proposal index for one cell of one sky."""

    def __init__(self, scene: str, cell: int, data: str | None = None):
        from constellation.retrieval import build_harmonic_index
        scene_obj = load_scene(scene, data)
        h, w = scene_obj.image.shape
        x0, y0, x1, y1 = cell_box(cell, h, w)
        self.scene = scene
        self.cell = cell
        self.box = (x0, y0, x1, y1)
        self.crop = np.ascontiguousarray(scene_obj.image[y0:y1, x0:x1])
        points, descriptors, stars = build_harmonic_index(self.crop)
        self.points = points
        self.descriptors = descriptors
        self.stars = stars

    def to_absolute(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, float) + np.array([self.box[0], self.box[1]], float)

    def contains_absolute(self, xy) -> bool:
        x0, y0, x1, y1 = self.box
        return x0 <= float(xy[0]) < x1 and y0 <= float(xy[1]) < y1


_INDEX_CACHE: dict = {}


def cell_index(scene: str, cell: int, data: str | None = None) -> CellIndex:
    key = (scene, cell, str(data))
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = CellIndex(scene, cell, data)
    return _INDEX_CACHE[key]


def clear_index_cache() -> None:
    _INDEX_CACHE.clear()


def cells_of(partition: str) -> tuple:
    return tuple(c for c in range(GRID * GRID) if partition_of(c) == partition)


def regional_candidates(scene: str, partition: str, patch: np.ndarray,
                        keep: int = 20, budget: int = 2000,
                        data: str | None = None,
                        verify_radius: str = 'adaptive') -> list:
    """Top candidates for `patch`, searched ONLY inside `partition`.

    Returns a list of `(x, y, score, angle, scale)` in ABSOLUTE sky coordinates,
    best first. Uses the same harmonic retrieval and pose-admissible verification
    the production pipeline uses, so mined negatives are the confusers the
    classical system actually produces.
    """
    from constellation.pipeline import VERIFY_REPS
    from constellation.retrieval import retrieve_harmonic, verify_adaptive
    from constellation.dense import dense_candidates

    rep = VERIFY_REPS['blur']
    patch_rep = rep(patch.astype(np.float32))
    merged = []
    for cell in cells_of(partition):
        index = cell_index(scene, cell, data)
        proposed = retrieve_harmonic(patch, index.points, index.descriptors,
                                     budget=min(budget, len(index.points)))
        try:
            dense = dense_candidates(index.crop, patch)
            proposed = np.unique(np.vstack([proposed, dense]), axis=0)
        except Exception:
            pass
        crop_rep = rep(index.crop.astype(np.float32))
        found = verify_adaptive(crop_rep, patch_rep, proposed, keep, verify_radius)
        for x, y, score, angle, scale in found:
            ax, ay = index.to_absolute((x, y))
            merged.append((float(ax), float(ay), float(score), float(angle),
                           float(scale), cell))
    merged.sort(key=lambda c: -c[2])
    return [(c[0], c[1], c[2], c[3], c[4]) for c in merged[:keep]]


def assert_partition(scene: str, partition: str, candidates, data: str | None = None):
    """Every returned location must lie inside the searched partition."""
    allowed = cells_of(partition)
    scene_obj = load_scene(scene, data)
    h, w = scene_obj.image.shape
    boxes = [cell_box(c, h, w) for c in allowed]
    bad = []
    for cand in candidates:
        x, y = float(cand[0]), float(cand[1])
        if not any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in boxes):
            bad.append((x, y))
    return {'ok': not bad, 'outside': bad, 'partition': partition,
            'cells': list(allowed)}

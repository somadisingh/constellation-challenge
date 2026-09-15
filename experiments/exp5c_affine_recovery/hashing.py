"""Affine-invariant quadruple hashing and nearest-neighbour retrieval."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def compute_quad_descriptors(points: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized signed-area affine invariant descriptors for given 4-point ids."""
    ids = np.asarray(ids, dtype=np.int32).reshape(-1, 4)
    if not len(ids):
        return ids, np.empty((0, 4), dtype=float)
    points = np.asarray(points, dtype=float)
    q = points[ids]
    areas = []
    for i in range(4):
        z = np.delete(q, i, axis=1)
        u = z[:, 1] - z[:, 0]
        v = z[:, 2] - z[:, 0]
        areas.append((u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) * (-1) ** i)
    a = np.stack(areas, axis=1)
    largest = np.argmax(abs(a), axis=1)
    a *= np.where(a[np.arange(len(a)), largest] >= 0, 1.0, -1.0)[:, None]
    a /= np.maximum(abs(a).sum(axis=1, keepdims=True), 1e-12)
    order = np.argsort(a, axis=1, kind='stable')
    a = np.take_along_axis(a, order, axis=1)
    ids = np.take_along_axis(ids, order, axis=1)
    keep = (np.min(abs(a), axis=1) > 0.015) & (np.min(np.diff(a, axis=1), axis=1) > 0.002)
    return ids[keep], a[keep]


class KDTreeMatcher:
    """Nearest-neighbour matcher using cKDTree in descriptor space."""
    def __init__(self, descriptors: np.ndarray):
        self.descriptors = np.asarray(descriptors, dtype=float)
        self.tree = cKDTree(self.descriptors) if len(self.descriptors) else None

    def query(self, query_desc: np.ndarray, k: int = 32, max_dist: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        if self.tree is None or not len(query_desc):
            return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=np.int32)
        k = min(k, len(self.descriptors))
        dist, ind = self.tree.query(query_desc, k=k, distance_upper_bound=max_dist)
        dist = np.atleast_2d(dist)
        ind = np.atleast_2d(ind)
        if dist.shape[0] != len(query_desc):
            dist, ind = dist.T, ind.T
        return dist, ind


class QuantizedBinIndex:
    """Multi-probe hash grid for fast descriptor retrieval."""
    def __init__(self, descriptors: np.ndarray, bin_size: float = 0.03):
        self.descriptors = np.asarray(descriptors, dtype=float)
        self.bin_size = bin_size
        self.bins: dict[tuple[int, int, int], list[int]] = {}
        if len(self.descriptors):
            grid = np.floor(self.descriptors[:, :3] / bin_size).astype(int)
            for idx, key in enumerate(map(tuple, grid)):
                if key not in self.bins:
                    self.bins[key] = []
                self.bins[key].append(idx)

    def query(self, query_desc: np.ndarray, k: int = 32, max_dist: float = 0.05) -> list[list[tuple[float, int]]]:
        results = []
        shifts = [-1, 0, 1]
        for q in query_desc:
            base_key = tuple(np.floor(q[:3] / self.bin_size).astype(int))
            candidates = []
            for dx in shifts:
                for dy in shifts:
                    for dz in shifts:
                        nbr = (base_key[0] + dx, base_key[1] + dy, base_key[2] + dz)
                        if nbr in self.bins:
                            candidates.extend(self.bins[nbr])
            if not candidates:
                results.append([])
                continue
            cand_unique = np.unique(candidates)
            cand_desc = self.descriptors[cand_unique]
            dists = np.linalg.norm(cand_desc - q, axis=1)
            valid = dists <= max_dist
            valid_cand = cand_unique[valid]
            valid_dists = dists[valid]
            sorted_idx = np.argsort(valid_dists)[:k]
            results.append([(float(valid_dists[i]), int(valid_cand[i])) for i in sorted_idx])
        return results


def measure_descriptor_recall(
    n_trials: int = 200,
    span: float = 1500.0,
    noise_sigma: float = 4.0,
    seed: int = 42
) -> dict[str, float]:
    """Measure raw descriptor recall independently from downstream scoring under realistic scale."""
    rng = np.random.default_rng(seed)
    success_k1, success_k5, success_k32 = 0, 0, 0
    kdtree_dists = []

    for _ in range(n_trials):
        # Generate 4 random non-collinear points with realistic figure span (~1500px)
        pts = rng.uniform(500.0, 500.0 + span, size=(4, 2))
        while True:
            _, d = compute_quad_descriptors(pts, [[0, 1, 2, 3]])
            if len(d) > 0:
                break
            pts = rng.uniform(500.0, 500.0 + span, size=(4, 2))

        # Apply random affine transform (rotation, scale, shear, reflection)
        theta = rng.uniform(0, 2 * np.pi)
        scale_x = rng.uniform(0.75, 1.33)
        scale_y = rng.uniform(0.75, 1.33)
        shear = rng.uniform(-0.15, 0.15)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        S = np.array([[scale_x, shear], [0, scale_y]])
        A = R @ S
        if rng.random() < 0.5:
            A[0, :] *= -1  # reflection
        t = rng.uniform(200, 800, size=(1, 2))
        transformed = pts @ A.T + t + rng.normal(0, noise_sigma, size=(4, 2))

        # Recompute descriptor
        _, d_trans = compute_quad_descriptors(transformed, [[0, 1, 2, 3]])
        if not len(d_trans):
            continue

        # Build bank with 500 random distractor quads
        distractors = rng.uniform(500.0, 500.0 + span, size=(500, 4, 2))
        d_bank = []
        for dq in distractors:
            _, dd = compute_quad_descriptors(dq, [[0, 1, 2, 3]])
            if len(dd) > 0:
                d_bank.append(dd[0])
        d_bank.append(d[0])
        d_bank = np.array(d_bank)
        true_idx = len(d_bank) - 1

        matcher = KDTreeMatcher(d_bank)
        dist, ind = matcher.query(d_trans, k=32, max_dist=1.0)
        found_rank = np.where(ind[0] == true_idx)[0]
        if len(found_rank) > 0:
            rank = found_rank[0]
            if rank == 0:
                success_k1 += 1
            if rank < 5:
                success_k5 += 1
            if rank < 32:
                success_k32 += 1
            kdtree_dists.append(float(dist[0, rank]))

    return {
        'n_trials': n_trials,
        'noise_sigma_px': noise_sigma,
        'recall_at_1': success_k1 / n_trials,
        'recall_at_5': success_k5 / n_trials,
        'recall_at_32': success_k32 / n_trials,
        'median_correct_descriptor_dist': float(np.median(kdtree_dists)) if kdtree_dists else None,
    }


if __name__ == '__main__':
    metrics = measure_descriptor_recall(n_trials=200, noise_sigma=0.5)
    print("Descriptor recall independent evaluation:", metrics)

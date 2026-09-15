"""Pattern graph extraction and template quadruple indexing."""
from __future__ import annotations

from pathlib import Path
import pickle
import cv2
import numpy as np
from itertools import combinations

from experiments.exp5c_affine_recovery import ROOT, OUT, PATTERNS_DIR
from constellation.quad import quads


def extract_pattern_graph(path: Path) -> dict:
    """Extract both star nodes and the supplied green line adjacency."""
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        raise ValueError(f"Invalid pattern image: {path}")
    white = ((rgba[:, :, :3].min(axis=2) > 190) & (rgba[:, :, 3] > 100)).astype(np.uint8)
    n, labels, stats, centres = cv2.connectedComponentsWithStats(white)
    keep = np.where(stats[1:, cv2.CC_STAT_AREA] >= 2)[0] + 1
    nodes = centres[keep].astype(float)
    b, g, r = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2]
    green = ((g > 120) & (g > r * 1.25) & (g > b * 1.25) & (rgba[:, :, 3] > 40)).astype(np.uint8)
    dilated = cv2.dilate(green, np.ones((7, 7), np.uint8))
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, c = nodes[i], nodes[j]
            v = c - a
            length = float(np.linalg.norm(v))
            if length < 12:
                continue
            intermediate = False
            for k, p in enumerate(nodes):
                if k in (i, j):
                    continue
                t = float(np.dot(p - a, v) / max(np.dot(v, v), 1e-9))
                if 0.08 < t < 0.92 and np.linalg.norm(p - (a + t * v)) < 8:
                    intermediate = True
                    break
            if intermediate:
                continue
            ts = np.linspace(min(10 / length, 0.2), max(1 - 10 / length, 0.8), max(20, int(length / 2)))
            xy = a[None, :] + ts[:, None] * v[None, :]
            xx = np.clip(np.rint(xy[:, 0]).astype(int), 0, dilated.shape[1] - 1)
            yy = np.clip(np.rint(xy[:, 1]).astype(int), 0, dilated.shape[0] - 1)
            coverage = float(dilated[yy, xx].mean())
            if coverage >= 0.70:
                edges.append((int(i), int(j)))
    return {
        'name': path.stem.removesuffix('_pattern'),
        'nodes': nodes,
        'edges': edges,
        'green_pixels': int(green.sum()),
        'shape': rgba.shape[:2],
    }


def extract_all(folder: Path = PATTERNS_DIR) -> dict[str, dict]:
    return {
        p.stem.removesuffix('_pattern'): extract_pattern_graph(p)
        for p in sorted(folder.glob('*_pattern.png'))
    }


def compute_template_quads(nodes: np.ndarray, max_quads: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    """Compute template quadruples and affine descriptors with long-baseline ordering."""
    if len(nodes) < 4:
        return np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=float)
    p = (nodes - nodes.mean(0)) / np.maximum(np.ptp(nodes, axis=0), 1e-6)
    ti, td = quads(p)
    if len(ti) <= max_quads:
        return ti, td
    # Rank quads by baseline span (minimum pairwise distance across the 4 nodes)
    q_pts = p[ti]  # (N, 4, 2)
    # Minimum pairwise distance
    min_dist = np.min([
        np.linalg.norm(q_pts[:, i] - q_pts[:, j], axis=1)
        for i in range(4) for j in range(i + 1, 4)
    ], axis=0)
    order = np.argsort(-min_dist)
    return ti[order[:max_quads]], td[order[:max_quads]]


class PatternIndex:
    def __init__(self, graphs: dict[str, dict] | None = None):
        if graphs is None:
            graphs = extract_all()
        self.graphs = graphs
        self.template_quads: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.usable_classes: list[str] = []
        self.small_classes: list[str] = []
        for name, g in sorted(self.graphs.items()):
            nodes = g['nodes']
            if len(nodes) >= 4:
                ti, td = compute_template_quads(nodes)
                self.template_quads[name] = (ti, td)
                self.usable_classes.append(name)
            else:
                self.small_classes.append(name)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'graphs': self.graphs,
                'template_quads': self.template_quads,
                'usable_classes': self.usable_classes,
                'small_classes': self.small_classes,
            }, f)

    @classmethod
    def load(cls, path: Path) -> PatternIndex:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        idx = cls.__new__(cls)
        idx.graphs = data['graphs']
        idx.template_quads = data['template_quads']
        idx.usable_classes = data['usable_classes']
        idx.small_classes = data['small_classes']
        return idx


def get_or_create_index(cache_path: Path | None = None) -> PatternIndex:
    if cache_path is None:
        cache_path = OUT / 'template_quad_index.pkl'
    if cache_path.exists():
        try:
            return PatternIndex.load(cache_path)
        except Exception:
            pass
    idx = PatternIndex()
    idx.save(cache_path)
    return idx


if __name__ == '__main__':
    idx = get_or_create_index()
    print(f"PatternIndex created: {len(idx.usable_classes)} usable classes (>=4 nodes), "
          f"{len(idx.small_classes)} small classes (<4 nodes).")

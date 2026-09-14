"""Scene, query and label access for the three labelled skies (plan §2, §5).

Read-only. This module never writes into `train/`, `outputs/joint_train/` or any
other production location.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import SCENES
from .env import ROOT, sha256_file

PATCH = 32
PATCH_MID = 15.5          # pixel-midpoint of an even 32-pixel array
MEMBER_RADIUS = 18.0      # constellation/finalize.py finalize_joint member_radius


@dataclass
class Scene:
    name: str
    image: np.ndarray          # (H, W) uint8 sky
    patches: list              # list of (32,32) uint8 query crops
    image_path: Path
    patch_paths: list

    @property
    def shape(self):
        return self.image.shape

    def __len__(self):
        return len(self.patches)


def data_root(data: str | Path | None = None) -> Path:
    """Accept a directory holding patterns/train/validation, or its parent."""
    base = Path(data or ROOT).resolve()
    for cand in (base, base / 'participant'):
        if (cand / 'train').is_dir() and (cand / 'patterns').is_dir():
            return cand
    raise FileNotFoundError(f'no train/ + patterns/ under {base}')


@functools.lru_cache(maxsize=8)
def load_scene(name: str, data: str | None = None) -> Scene:
    root = data_root(data)
    image_path = root / 'train' / name / f'{name}_image.png'
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    patch_paths = sorted((root / 'train' / name / 'patches').glob('patch_*.png'))
    patches = []
    for p in patch_paths:
        q = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if q is None or q.shape != (PATCH, PATCH):
            raise ValueError(f'unexpected query patch {p}: {None if q is None else q.shape}')
        patches.append(q)
    return Scene(name, image, patches, image_path, patch_paths)


@functools.lru_cache(maxsize=1)
def load_truth(data: str | None = None) -> dict:
    from constellation.contracts import read_truth
    return read_truth(data_root(data) / 'train_ground_truth.csv')


@functools.lru_cache(maxsize=1)
def patterns_dir(data: str | None = None) -> Path:
    return data_root(data) / 'patterns'


def query_records(name: str, data: str | None = None) -> list:
    """Per-query truth for one scene: present flag, centre, figure membership.

    `figure` is the dataset's on-figure flag from the labels. It is used only for
    stratified REPORTING, never to select training crops (plan §5).
    """
    scene = load_scene(name, data)
    truth = load_truth(data)[name]
    if len(truth.patches) != len(scene):
        raise ValueError(f'{name}: {len(truth.patches)} labels vs {len(scene)} patches')
    out = []
    for i, t in enumerate(truth.patches):
        out.append({
            'scene': name,
            'index': i,
            'query_id': f'{name}:{i:02d}',
            'present': t is not None,
            'xy': (float(t[0]), float(t[1])) if t is not None else None,
            'figure': int(t[2]) if t is not None else None,
            'stratum': ('absent' if t is None else ('figure' if t[2] == 1 else 'offfigure')),
        })
    return out


def all_query_records(data: str | None = None) -> list:
    return [r for n in SCENES for r in query_records(n, data)]


def data_provenance(data: str | None = None) -> dict:
    """Hashes and dimensions of every input this experiment reads."""
    root = data_root(data)
    out = {'root': str(root), 'scenes': {}}
    for name in SCENES:
        scene = load_scene(name, data)
        out['scenes'][name] = {
            'image': str(scene.image_path.relative_to(root)),
            'image_sha256': sha256_file(scene.image_path),
            'height': int(scene.image.shape[0]),
            'width': int(scene.image.shape[1]),
            'n_queries': len(scene),
            'patch_sha256': [sha256_file(p) for p in scene.patch_paths],
        }
    gt = root / 'train_ground_truth.csv'
    out['train_ground_truth_sha256'] = sha256_file(gt)
    out['patterns'] = sorted(p.name for p in patterns_dir(data).glob('*.png'))
    out['n_patterns'] = len(out['patterns'])
    return out


def to_float(patch: np.ndarray) -> np.ndarray:
    """uint8 -> float32 in [0,1]. Native preprocessing only (plan §6)."""
    return np.asarray(patch, dtype=np.float32) / 255.0

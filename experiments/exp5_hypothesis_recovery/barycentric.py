"""Affine-invariant barycentric-coordinate utilities.

No barycentric primitive exists anywhere in this repository (confirmed by a
whole-tree search before writing this module). Barycentric coordinates of a
point X relative to a non-degenerate triangle (A, B, C) satisfy
    X = l1*A + l2*B + l3*C,  l1 + l2 + l3 = 1
and are PRESERVED EXACTLY under any affine map T applied to the whole plane:
    T(X) = l1*T(A) + l2*T(B) + l3*T(C).
This is what makes them useful for validating a fourth pattern node against a
fitted affine transform without assuming any edge length is physically exact
(only the affine class of transforms -- rotation, reflection, anisotropic
scale, shear -- is assumed, matching the task's stated schematic distortion
model).
"""
from __future__ import annotations

import numpy as np


def to_barycentric(a: np.ndarray, b: np.ndarray, c: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """Barycentric coordinates (l1, l2, l3) of point(s) `x` w.r.t. triangle
    (a, b, c). `x` may be a single (2,) point or an (n, 2) array. Returns
    None if the triangle is degenerate (near-zero area)."""
    x = np.atleast_2d(x)
    T = np.array([[a[0] - c[0], b[0] - c[0]],
                 [a[1] - c[1], b[1] - c[1]]], float)
    det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
    if abs(det) < 1e-9:
        return None
    inv = np.array([[T[1, 1], -T[0, 1]], [-T[1, 0], T[0, 0]]], float) / det
    diff = x - c[None, :]
    l12 = diff @ inv.T
    l3 = 1.0 - l12[:, 0] - l12[:, 1]
    return np.c_[l12, l3]


def from_barycentric(a: np.ndarray, b: np.ndarray, c: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """Inverse of `to_barycentric`: reconstruct Cartesian point(s) from
    barycentric coordinates `bary` (n, 3) and triangle (a, b, c)."""
    bary = np.atleast_2d(bary)
    return bary @ np.stack([a, b, c], axis=0)


def barycentric_agreement(template_bary: np.ndarray, candidate_bary: np.ndarray) -> float:
    """L1 distance between two barycentric-coordinate vectors -- 0.0 for an
    exact affine-consistent fourth point, growing with schematic distortion or
    a wrong correspondence. Both inputs already sum to 1 by construction, so
    this is bounded in [0, 4]."""
    return float(np.sum(np.abs(np.asarray(template_bary) - np.asarray(candidate_bary))))

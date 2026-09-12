"""Explicit separation of the roles a candidate list plays.

In the shipped pipeline every query is a list of alternatives sorted by one
appearance score, and index 0 of that list simultaneously determines six things:

  1. the presence decision                (score at index 0 vs the threshold)
  2. the ambiguity decision               (index 0 minus index 1)
  3. the pool margin                      (scores within `margin` of index 0)
  4. the physical-star grouping anchor     (coordinate at index 0)
  5. the geometric hypothesis seed         (coordinate at index 0)
  6. the reported coordinate when geometry does not relocate the query

So replacing the appearance score changes all six at once, which is why earlier
re-ranking experiments could not be attributed. A `QuerySlate` keeps candidate
identity fixed and carries three independently settable roles:

  calib  presence, ambiguity and pool margin
  rank   pool ordering and the reported coordinate
  seed   grouping anchor and hypothesis seed

Building a slate from a score-sorted candidate list reproduces the shipped
behaviour exactly: calib and rank are that same score and seed is index 0.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class QuerySlate:
    """One query's candidates plus the three roles their ordering used to conflate."""
    xy: np.ndarray        # (k, 2) candidate coordinates; identity, never reordered
    pose: np.ndarray      # (k, 2) angle in degrees, scale
    calib: np.ndarray     # (k,) calibration score
    rank: np.ndarray      # (k,) ranking score
    seed: int             # index of the grouping/seed anchor

    def __len__(self):
        return len(self.xy)

    @property
    def report_index(self):
        """Candidate reported when geometry does not relocate this query."""
        return int(np.argmax(self.rank))

    @property
    def calib_index(self):
        return int(np.argmax(self.calib))

    @property
    def presence_score(self):
        return float(self.calib[self.calib_index])

    @property
    def ambiguity_gap(self):
        """Top-two calibration difference; large means one location clearly won."""
        if len(self.calib) < 2:
            return float('inf')
        top2 = np.sort(self.calib)[::-1][:2]
        return float(top2[0] - top2[1])

    def order_by_rank(self):
        """Candidate indices, best ranking score first, ties by original order."""
        return np.argsort(-self.rank, kind='stable')

    def order_by_calib(self):
        return np.argsort(-self.calib, kind='stable')

    def seed_xy(self):
        return self.xy[self.seed]


def slate_from_candidates(candidates):
    """Build a slate from a (x, y, score, angle, scale) list, preserving behaviour.

    The list is assumed sorted by score descending, as `retrieval.verify` and
    `refine.refine_candidates` emit it. calib and rank are that score and the seed
    is index 0, so every downstream decision matches the shipped pipeline.
    """
    c = [tuple(map(float, q)) for q in candidates]
    if not c:
        return QuerySlate(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0),
                          np.zeros(0), 0)
    arr = np.asarray(c, dtype=float)
    pose = arr[:, 3:5] if arr.shape[1] >= 5 else np.zeros((len(arr), 2))
    return QuerySlate(xy=arr[:, :2].copy(), pose=pose.copy(),
                      calib=arr[:, 2].copy(), rank=arr[:, 2].copy(), seed=0)


def slates_from_lists(alternatives):
    return [slate_from_candidates(q) for q in alternatives]


def with_scores(slate, rank=None, calib=None, seed=None):
    """Copy of `slate` with selected roles replaced. Candidate identity is kept."""
    return QuerySlate(
        xy=slate.xy, pose=slate.pose,
        calib=slate.calib if calib is None else np.asarray(calib, float),
        rank=slate.rank if rank is None else np.asarray(rank, float),
        seed=slate.seed if seed is None else int(seed))

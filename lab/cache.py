"""Shared cached-candidate loading for fast final-stage experiments.

Candidate generation (retrieval + dense + verify + ECC) is the expensive part and
is already cached for all 19 scenes. Everything downstream is cheap to iterate.
"""
import json
from pathlib import Path
import numpy as np

TRAIN = ['pisces', 'scorpius', 'taurus']
ROOT = Path(__file__).resolve().parent.parent


def load_train():
    """Return {scene: {'refined': [[cand,...],...], 'coarse': [...]}}."""
    out = {}
    for n in TRAIN:
        refined = json.load(open(ROOT / f'outputs/hybrid_ecc/{n}.json'))
        coarse = json.load(open(ROOT / f'outputs/hybrid/{n}.json'))
        out[n] = {
            'refined': [q['candidates'] for q in refined['diagnostics']['queries']],
            'coarse': [q['candidates'] for q in coarse['diagnostics']['queries']],
        }
    return out


def load_validation():
    out = {}
    for p in sorted((ROOT / 'outputs/final_submission').glob('constellation_*.json')):
        d = json.load(open(p))
        qs = d['diagnostics']['queries']
        out[p.stem] = {
            'refined': [q['candidates'] for q in qs],
            'coarse': [q['coarse_candidates'] for q in qs],
        }
    return out


def truth_points(scene, truth):
    """(present_xy, figure_xy, present_mask) for a labelled scene."""
    t = truth[scene].patches
    mask = np.array([p is not None for p in t])
    pres = np.array([p[:2] for p in t if p is not None]).reshape(-1, 2)
    fig = np.array([p[:2] for p in t if p is not None and p[2] == 1]).reshape(-1, 2)
    return pres, fig, mask

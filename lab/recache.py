"""Produce rescored candidate lists: identical locations and poses, new ranking.

Additive combinations are divided by (1 + sum of weights) so the resulting score
occupies roughly the incumbent's range. This is not a substitute for recalibration
-- the presence threshold and ambiguity gate are still swept -- but it keeps the
starting point comparable and makes the sweep interpretable.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.cache import TRAIN
from lab.rescore import (ncc, rep_bgsub, rep_lcn, rep_dog, centre_annulus,
                         multiblur_ncc, score_scene)

COMPONENTS = {
    'dog32': lambda p, s: ncc(p, s, rep_dog, 'full'),
    'full32': lambda p, s: ncc(p, s, rep_bgsub, 'full'),
    'lcn': lambda p, s: ncc(p, s, rep_lcn, 'disc14'),
    'annulus': lambda p, s: centre_annulus(p, s, rep_bgsub, 'annulus'),
    'ca_min': lambda p, s: centre_annulus(p, s, rep_bgsub, 'min'),
    'multiblur': lambda p, s: multiblur_ncc(p, s, rep_bgsub, 'disc14'),
}

# Carried forward from lab/combo_rank.py. Each reached top1 >= 0.828 with figure
# top1 >= 0.696, against the incumbent's 0.797 / 0.609.
RECIPES = {
    'incumbent': {},
    'dog32 w0.25': {'dog32': .25},
    'dog32 w0.5': {'dog32': .5},
    'full32 w0.25': {'full32': .25},
    'multiblur w1.0': {'multiblur': 1.},
    'ca_min w2.0': {'ca_min': 2.},
    'dog32 w0.5 + annulus w0.5': {'dog32': .5, 'annulus': .5},
}


def component_cache(cache, data, keys=None, branch='refined'):
    """Component scores per scene, computed once."""
    keys = keys or list(COMPONENTS)
    out = {}
    for n in TRAIN:
        samples, patches, counts = data[n]
        d = {'incumbent': score_scene(None, patches, samples, counts,
                                      cache[n][branch])}
        for k in keys:
            d[k] = score_scene(COMPONENTS[k], patches, samples, counts,
                               cache[n][branch])
        out[n] = d
    return out


def combined(parts_for_query, weights):
    s = np.array(parts_for_query['incumbent'], float)
    for k, w in weights.items():
        s = s + w * np.asarray(parts_for_query[k], float)
    return s / (1. + sum(weights.values()))


def rescored_cache(cache, parts, weights):
    """New cache with the same locations/poses, rescored and re-sorted."""
    out = {}
    for n in TRAIN:
        refined = []
        for i, q in enumerate(cache[n]['refined']):
            s = combined({k: v[i] for k, v in parts[n].items()}, weights)
            order = np.argsort(-s, kind='stable')
            refined.append([(float(q[j][0]), float(q[j][1]), float(s[j]),
                             float(q[j][3]), float(q[j][4])) for j in order])
        out[n] = {'refined': refined, 'coarse': cache[n]['coarse']}
    return out

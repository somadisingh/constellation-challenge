"""Synthetic geometry benchmark.

Identification is 30% of the competition score but there are only three labelled
scenes, so measuring it there gives a three-valued signal that flips on trivial
parameter changes. This module synthesises the *candidate lists* the recognizer
receives, so identification accuracy can be measured over hundreds of scenes.

Every noise parameter is taken from measurements on the real labelled scenes
(lab/figurestats.py, lab/ambiguity.py, lab/signals.py):

  correct alternative position error   median 0.78px, p90 1.33px
  wrong alternative distance           median 1256px (effectively uniform)
  figure query rank-0 correct          0.61
  all present rank-0 correct           0.72
  correct in top-20                    0.90
  rank-0 score, correct                median 0.925, p10 0.806
  rank-0 minus rank-1 gap, correct     median 0.136, p10 0.027
  rank-0 minus rank-1 gap, mislocated  median 0.009, p90 0.036
  issued figure nodes / template nodes 0.50 - 0.77
  figure span                          ~1000-1900 px inside a 3000px field
  off-figure present queries           12-17 per scene
  absent queries passing the threshold ~0.44
  affine template-to-sky residual      median ~5px, p90 8-16px (lab/geomerror.py)

The last figure matters most. The reference diagrams are schematics, so their
extracted node centroids are not an exact affine image of the sky; an oracle fit
to ground-truth figure points still leaves ~5px median residual. Modelling nodes
as landing within ~1px made the benchmark prefer a tolerance of 8px, which does
not transfer to the real scenes at all.

This tests geometry in isolation; it is not evidence about the appearance stage.
"""
import numpy as np

FIELD = 3000.
N_ALT = 20
# Rayleigh median for sigma=4.1 is 4.8px and p90 is 8.8px, matching the measured
# affine template-to-sky residual on all three labelled scenes.
MODEL_ERROR = 4.1


def random_transform(rng, template):
    """Similarity-plus-mild-anisotropy placement of a template into the field."""
    p = (template - template.mean(axis=0)) / np.maximum(np.ptp(template, axis=0), 1e-6)
    span = rng.uniform(900., 2000.)
    angle = rng.uniform(0, 2 * np.pi)
    aniso = rng.uniform(1.0, 1.20)
    reflect = rng.random() < .5
    c, s = np.cos(angle), np.sin(angle)
    m = np.array([[c, s], [-s, c]]) @ np.diag([span * (aniso if reflect else 1.), span])
    if reflect:
        m = np.array([[-1., 0.], [0., 1.]]) @ m
    shear = rng.uniform(-.06, .06)
    m = np.array([[1., shear], [0., 1.]]) @ m
    pts = p @ m
    pts -= pts.mean(0)
    # Rotation grows the axis-aligned extent, so rescale to keep the figure inside.
    margin = 120.
    extent = np.ptp(pts, axis=0).max()
    budget = (FIELD - 2 * margin) * .98
    if extent > budget:
        pts *= budget / extent
    lo, hi = pts.min(0), pts.max(0)
    centre = np.array([rng.uniform(margin - lo[k], max(margin - lo[k],
                                                       FIELD - margin - hi[k]))
                       for k in (0, 1)])
    return pts + centre


def _scores(rng, correct_rank, n=N_ALT):
    """Appearance scores whose rank-0/rank-1 gap matches the measured split."""
    if correct_rank == 0:
        top = np.clip(rng.normal(.925, .06), .5, .999)
        gap = max(.035, rng.lognormal(np.log(.13), .6))
    else:
        top = np.clip(rng.normal(.86, .08), .4, .999)
        gap = min(.030, abs(rng.normal(.009, .008)))
    rest = top - gap - np.abs(rng.normal(0, .02, n - 2)).cumsum()
    return np.concatenate([[top, top - gap], rest])[:n]


def make_scene(rng, patterns, class_name=None, fragment=True):
    """Return (alternatives, truth) for one synthetic scene."""
    names = sorted(patterns)
    name = class_name or names[rng.integers(len(names))]
    template = patterns[name]
    if len(template) < 4:
        return None
    placed = random_transform(rng, template)

    # The schematic does not map exactly onto the sky; displace each node.
    placed = placed + rng.normal(0, MODEL_ERROR, placed.shape)

    n_issued = max(4, int(round(len(placed) * rng.uniform(.50, .80))))
    issued = rng.choice(len(placed), min(n_issued, len(placed)), replace=False)
    figure_pts = placed[issued]

    n_off = int(rng.integers(12, 18))
    off_pts = rng.uniform(60., FIELD - 60., size=(n_off, 2))
    # A fragment of a different constellation, per the distractor requirement.
    if fragment:
        other = names[rng.integers(len(names))]
        if other != name and len(patterns[other]) >= 3:
            frag = random_transform(rng, patterns[other])
            take = rng.choice(len(frag), min(rng.integers(2, 5), len(frag)), replace=False)
            off_pts = np.vstack([off_pts, frag[take]])

    n_absent = int(rng.integers(12, 19))

    alternatives, truth = [], []
    def emit(true_xy, hit_prob):
        """One query's ranked alternative list."""
        if true_xy is None:
            sc = _scores(rng, 1)
            pts = rng.uniform(0., FIELD, size=(N_ALT, 2))
            alternatives.append([(float(x), float(y), float(s), 0., 1.)
                                 for (x, y), s in zip(pts, sc)])
            truth.append(None)
            return
        r = rng.random()
        if r < hit_prob:
            rank = 0
        elif r < .90:
            rank = int(rng.integers(1, 9))
        else:
            rank = -1                      # correct location absent from the pool
        sc = _scores(rng, 0 if rank == 0 else 1)
        pts = rng.uniform(0., FIELD, size=(N_ALT, 2))
        if rank >= 0:
            pts[rank] = true_xy + rng.normal(0, .8, 2)
        alternatives.append([(float(x), float(y), float(s), 0., 1.)
                             for (x, y), s in zip(pts, sc)])
        truth.append(tuple(true_xy))

    for xy in figure_pts:
        emit(xy, .61)
    for xy in off_pts:
        emit(xy, .80)
    for _ in range(n_absent):
        if rng.random() < .44:             # false positive that survives threshold
            emit(None, 0.)
    return dict(name=name, alternatives=alternatives, truth=truth,
                n_figure=len(figure_pts), template_nodes=len(template))


def dataset(patterns, n_scenes=200, seed=17, min_nodes=4):
    """Sweep over all classes with at least `min_nodes` reference nodes.

    `min_nodes=4` includes a large population of scenes where only four figure
    queries are ever issued; measured accuracy there is 0/42, because three or
    four points cannot distinguish a class from clutter under a free affine
    transform. The three labelled scenes all use 12-18 node templates, so
    `min_nodes=7` is the regime that actually reflects the competition data.
    """
    rng = np.random.default_rng(seed)
    usable = [n for n, v in sorted(patterns.items()) if len(v) >= min_nodes]
    out = []
    for i in range(n_scenes):
        s = make_scene(rng, patterns, usable[i % len(usable)])
        if s:
            out.append(s)
    return out

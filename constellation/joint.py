"""Joint localization and recognition.

The frozen recognizer consumed one point per query (the top appearance
alternative) and returned only a class plus membership. Measured on the labelled
scenes, the correct location is present in the 20 retained alternatives for 90%
of present queries but ranks first for only 61% of *figure* queries, with a
median appearance gap of 0.031. Appearance alone cannot separate them.

This module verifies hypotheses against the whole alternative pool under
one-to-one constraints per query group, and reports which pool point each query
matched so the caller can adopt the geometrically supported coordinate. A
restricted similarity/reflection branch runs beside the affine branch so a
four-degree-of-freedom explanation is preferred when it fits comparably well.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binom
from .geometry import consolidate, triangles
from .slate import QuerySlate, slate_from_candidates
from .quad import quads

# Rejected transform envelope: scene pixels per normalized template unit.
MIN_SV, MAX_SV, MAX_ANISO = 50., 6000., 8.
SIMILARITY_BONUS = .35   # reward for explaining the data with fewer parameters
GEN_LIMIT = 56           # quad generation is O(n^4); cap the seeding point set


def build_pool(slates, groups, top_k=8, margin=.15, gap=.03, pool_by='rank'):
    """Flatten per-query candidates into a verification pool tagged by group.

    Only *ambiguous* queries contribute more than their top candidate. On the
    labelled scenes the top-two calibration gap separates correctly located queries
    (median 0.136) from mislocated ones (median 0.009), so a query whose best
    location wins clearly is not offered up for geometric relocation.

    `pool_by` selects which role decides *membership* when only `top_k` of the
    candidates may enter:
      'rank'   membership and order both follow the ranking score (shipped
               behaviour, since rank equals calib there)
      'calib'  membership is fixed by the calibration score and only the order
               follows the ranking score, which isolates re-ordering from a change
               of pool membership
    Eligibility, ambiguity and the margin always use calibration, so they stay
    fixed when only the ranking score changes.
    """
    pts, tags, scores, ranks = [], [], [], []
    for i, s in enumerate(slates):
        if not len(s):
            continue
        best = s.presence_score
        limit = top_k if s.ambiguity_gap < gap else 1
        if pool_by == 'calib':
            member = list(s.order_by_calib()[:limit])
            chosen = [j for j in s.order_by_rank() if j in member]
        else:
            chosen = list(s.order_by_rank()[:limit])
        for r, j in enumerate(chosen):
            # The margin bounds how far *additional* alternatives may fall below the
            # best calibration. A query's primary contribution is always admitted, so
            # a query can never end up unrepresented in the pool. With rank equal to
            # calibration this is unreachable and behaviour is unchanged.
            if r > 0 and s.calib[j] < best - margin:
                continue
            pts.append(s.xy[j]); tags.append(groups[i])
            scores.append(s.calib[j]); ranks.append(r)
    return (np.asarray(pts, dtype=float).reshape(-1, 2), np.asarray(tags, dtype=int),
            np.asarray(scores, dtype=float), np.asarray(ranks, dtype=int))


def assignment(transformed, points, tolerance, tags):
    """Greedy nearest-first one-to-one match; at most one point per query group."""
    if not len(points):
        return [], []
    d = np.linalg.norm(transformed[:, None, :] - points[None, :, :], axis=2)
    candidates = np.argwhere(d < tolerance)
    if not len(candidates):
        return [], []
    order = np.argsort(d[candidates[:, 0], candidates[:, 1]], kind='stable')
    used_node, used_tag, pairs, res = set(), set(), [], []
    for k in order:
        i, j = candidates[k]
        tag = int(tags[j])
        if i not in used_node and tag not in used_tag:
            used_node.add(i); used_tag.add(tag)
            pairs.append((int(i), int(j))); res.append(float(d[i, j]))
    return pairs, res


def fit_similarity(src, dst, reflect=False):
    """Least-squares similarity (optionally reflected) returned as a 3x2 matrix."""
    s = src * np.array([-1., 1.]) if reflect else src
    sm, dm = s.mean(0), dst.mean(0)
    a, b = s - sm, dst - dm
    den = float(np.sum(a * a))
    if den < 1e-12:
        return None
    c = float(np.sum(a * b)) / den
    d = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])) / den
    r = np.array([[c, d], [-d, c]])
    if reflect:
        r = np.array([[-1., 0.], [0., 1.]]) @ r
    matrix = np.zeros((3, 2))
    matrix[:2] = r
    matrix[2] = dm - src.mean(0) @ r
    return matrix


def fit_affine(src, dst):
    try:
        return np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None


def valid(matrix):
    if not np.isfinite(matrix).all():
        return False
    sv = np.linalg.svd(matrix[:2], compute_uv=False)
    return not (sv[-1] < MIN_SV or sv[0] > MAX_SV or sv[0] / max(sv[-1], 1e-9) > MAX_ANISO)


def generation_points(slates, groups, limit=GEN_LIMIT):
    """Hypothesis seeding set: each query's seed anchor, capped by calibration.

    The cap exists because quad enumeration is O(n^4); triangle-only seeding no
    longer needs it, but it is retained so the shipped configuration is unchanged.
    Which candidate is a query's seed is a separate decision from which candidate
    the query reports, and is carried by `QuerySlate.seed`.
    """
    items = [(s.presence_score, s.seed_xy()) for s in slates if len(s)]
    if len(items) > limit:
        keep = sorted(range(len(items)), key=lambda i: -items[i][0])[:limit]
        keep.sort()
        items = [items[i] for i in keep]
    return np.asarray([xy for _, xy in items], dtype=float).reshape(-1, 2)


class SceneIndex:
    """Scene-side invariants and KD-trees, built once and reused for all classes."""

    def __init__(self, gen_pts, use_quads=True):
        self.tri_ids, tri_desc = triangles(gen_pts)
        self.tri_tree = cKDTree(tri_desc) if len(tri_desc) else None
        self.quad_ids, self.quad_tree = np.empty((0, 4), int), None
        if use_quads and len(gen_pts) >= 4:
            qi, qd = quads(gen_pts)
            if len(qd):
                self.quad_ids, self.quad_tree = qi, cKDTree(qd)

    @staticmethod
    def _query(tree, ids, desc, k):
        if tree is None or not len(desc):
            return []
        dist, idx = tree.query(desc, k=min(k, tree.n))
        dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
        if dist.shape[0] != len(desc):
            dist, idx = dist.T, idx.T
        out = [(float(d), i, ids[j])
               for i in range(len(desc)) for j, d in zip(idx[i], dist[i])]
        out.sort(key=lambda x: x[0])
        return out

    def seeds(self, template, budget, quad_share=.6):
        """Quad and triangle seeds are budgeted separately.

        Quad invariants are 4-dimensional and triangle invariants 2-dimensional,
        so their nearest-neighbour distances are not on a common scale. Merging
        and sorting them lets triangles crowd out the more discriminative quads.
        """
        quad_seeds = []
        if self.quad_tree is not None and len(template) >= 4:
            ti, td = quads(template)
            quad_seeds = [(d, ti[i], sid)
                          for d, i, sid in self._query(self.quad_tree, self.quad_ids, td, 4)]
        tri_seeds = []
        ti3, td3 = triangles(template)
        tri_seeds = [(d, ti3[i], sid)
                     for d, i, sid in self._query(self.tri_tree, self.tri_ids, td3, 20)]
        n_quad = min(len(quad_seeds), int(budget * quad_share))
        n_tri = min(len(tri_seeds), budget - n_quad)
        n_quad = min(len(quad_seeds), budget - n_tri)
        return quad_seeds[:n_quad] + tri_seeds[:n_tri]


def recognize_joint(alternatives, patterns, seed=6643, cap=80000, tolerance=18.,
                    top_k=8, margin=.15, shear_penalty=2., auxiliary_map=None,
                    models=('affine',), min_support=4,
                    appearance_weight=0., rank_weight=0., gap=.03, diag_top=8,
                    score_mode='binom', sigma=5., size_penalty=0., quad_share=0.,
                    aux_weight=3., pool_by='rank'):
    """Return (name, {query index: (x, y)}, diagnostics).

    `alternatives` is either a list of (x, y, score, angle, scale) candidate lists,
    in which case ranking, calibration and seeding all follow that single score as
    shipped, or a list of `QuerySlate` carrying those three roles separately.

    Defaults match what `finalize.finalize_joint` passes in production, so calling
    this directly reproduces the shipped configuration. The similarity branch and
    quad seeding remain selectable but measured worse; see FINDINGS.md.
    """
    slates = [s if isinstance(s, QuerySlate) else slate_from_candidates(s)
              for s in alternatives]
    anchors = np.array([s.seed_xy() for s in slates if len(s)]).reshape(-1, 2)
    if len(anchors) < 3:
        return 'unknown', {}, {'reason': 'fewer than three points', 'hypotheses': []}
    # Grouping consolidates seed anchors, so the seed policy also decides which
    # queries are treated as one physical star.
    _, groups = consolidate(anchors)
    live = [i for i, s in enumerate(slates) if len(s)]
    group_of = {q: groups[k] for k, q in enumerate(live)}
    live_slates = [slates[i] for i in live]
    pool, tags, pool_scores, pool_ranks = build_pool(
        live_slates, groups, top_k, margin, gap, pool_by)
    gen_pts = generation_points(live_slates, groups)
    if len(gen_pts) < 3 or not len(pool):
        return 'unknown', {}, {'reason': 'insufficient points', 'hypotheses': []}

    index = SceneIndex(gen_pts)
    n_groups = len(set(tags.tolist()))
    hypotheses = []
    per_class = cap // max(len(patterns), 1)
    for name, template in sorted(patterns.items()):
        if len(template) < 3:
            hypotheses.append({'name': name, 'score': -1e9, 'support': 0,
                               'reason': 'pair-only ambiguity'})
            continue
        # Normalize schematic axis lengths; supplied canvas aspect is arbitrary.
        p = (template - template.mean(axis=0)) / np.maximum(np.ptp(template, axis=0), 1e-6)
        # Seed selection is deterministic and class-local, so it is invariant to
        # catalog order; `seed` is retained only for reporting.
        seeds = index.seeds(p, per_class, quad_share)
        if not seeds:
            hypotheses.append({'name': name, 'score': -1e9, 'support': 0, 'reason': 'no seeds'})
            continue
        best = {'name': name, 'score': -1e9, 'support': 0}
        accepted = 0
        for _, ti, si in seeds:
            src, dst = p[np.asarray(ti)], gen_pts[np.asarray(si)]
            for model in models:
                if model == 'affine':
                    if len(src) == 3:
                        try:
                            matrix = np.linalg.solve(np.c_[src, np.ones(3)], dst)
                        except np.linalg.LinAlgError:
                            continue
                    else:
                        matrix = fit_affine(src, dst)
                    bonus = 0.
                else:
                    matrix = fit_similarity(src, dst)
                    mirror = fit_similarity(src, dst, reflect=True)
                    if matrix is not None and mirror is not None:
                        a = np.c_[src, np.ones(len(src))]
                        if np.linalg.norm(a @ mirror - dst) < np.linalg.norm(a @ matrix - dst):
                            matrix = mirror
                    bonus = SIMILARITY_BONUS
                if matrix is None or not valid(matrix):
                    continue
                accepted += 1
                mapped = np.c_[p, np.ones(len(p))] @ matrix
                pairs, res = assignment(mapped, pool, max(tolerance, 24.), tags)
                if len(pairs) >= min_support:
                    for _ in range(3):
                        ii, jj = np.array(pairs).T
                        refit = (fit_affine(p[ii], pool[jj]) if model == 'affine'
                                 else fit_similarity(p[ii], pool[jj]))
                        if refit is None or not valid(refit):
                            break
                        matrix = refit
                        mapped = np.c_[p, np.ones(len(p))] @ matrix
                        pairs, res = assignment(mapped, pool, tolerance, tags)
                        if len(pairs) < min_support:
                            break
                support = len(pairs)
                if support < min_support:
                    continue
                shear = abs(matrix[0] @ matrix[1]) / max(
                    np.linalg.norm(matrix[0]) * np.linalg.norm(matrix[1]), 1e-9)
                jj = [j for _, j in pairs]
                appearance = float(np.mean(pool_scores[jj]))
                rank_cost = float(np.mean(pool_ranks[jj]))
                if score_mode == 'likelihood':
                    # Nearly every template can collect four chance matches from a
                    # cluttered pool, so support alone does not discriminate. A
                    # genuine correspondence lands within ~1px (measured: median
                    # 0.78px), whereas a chance match is spread over the whole
                    # tolerance disc. Score each matched node by the log ratio of
                    # a Gaussian inlier density to the uniform chance density, and
                    # charge a fixed cost per template node offered up for matching.
                    r = np.asarray(res)
                    gain = (np.log(tolerance * tolerance / (2 * sigma * sigma))
                            - r * r / (2 * sigma * sigma))
                    score = float(gain.sum()) - size_penalty * len(p)
                else:
                    fraction = min(.8, n_groups * np.pi * tolerance * tolerance / 9e6)
                    surprise = -float(binom.logsf(support - min_support,
                                                  max(len(p) - 3, 1), fraction)) / np.log(10)
                    score = surprise - .5 * np.mean(res) / tolerance
                score += (bonus - shear_penalty * shear
                          + appearance_weight * appearance - rank_weight * rank_cost)
                auxiliary = .5
                if auxiliary_map is not None:
                    h, w = auxiliary_map.shape
                    inside = ((mapped[:, 0] >= 0) & (mapped[:, 1] >= 0)
                              & (mapped[:, 0] < w) & (mapped[:, 1] < h))
                    supported = {i for i, _ in pairs}
                    vals = [float(auxiliary_map[int(pt[1]), int(pt[0])]) if inside[k] else 0.
                            for k, pt in enumerate(mapped) if k not in supported]
                    auxiliary = float(np.mean(vals)) if vals else .5
                    score += aux_weight * (auxiliary - .5)
                if score > best['score']:
                    best = {'name': name, 'template': p.tolist(),
                            'score': float(score), 'support': support,
                            'model': model, 'coverage': support / len(p),
                            'mean_residual': float(np.mean(res)), 'shear': float(shear),
                            'auxiliary': auxiliary, 'appearance': appearance,
                            'rank_cost': rank_cost, 'matrix': matrix.tolist(),
                            'nodes': mapped.tolist(),
                            'pairs': [(int(i), int(j)) for i, j in pairs]}
        best.update(attempted=len(seeds), accepted=accepted, cap_hit=len(seeds) >= per_class)
        hypotheses.append(best)

    hypotheses.sort(key=lambda h: (-h['score'], h['name']))
    if not hypotheses or hypotheses[0]['support'] < min_support:
        return 'unknown', {}, {'reason': 'no verified fit',
                               'hypotheses': hypotheses[:diag_top], 'groups': groups}
    winner = hypotheses[0]
    by_tag = {int(tags[j]): pool[j] for _, j in winner['pairs']}
    chosen = {i: (float(by_tag[group_of[i]][0]), float(by_tag[group_of[i]][1]))
              for i in live if group_of[i] in by_tag}
    # Enough of the winning fit for a relocation guard to re-derive it while holding
    # out a query's own physical-star group. Lists, so diagnostics stay serializable.
    node_by_tag = {int(tags[j]): int(i) for i, j in winner['pairs']}
    relocation = {'class': winner['name'], 'template': winner.get('template'),
                  'pool': pool.tolist(), 'tags': tags.tolist(),
                  'pairs': [[int(i), int(j)] for i, j in winner['pairs']],
                  'node_by_tag': {str(k): v for k, v in node_by_tag.items()},
                  'group_of_query': {str(i): int(group_of[i]) for i in live}}
    diag_extra = {'pool_by': pool_by, 'relocation': relocation,
                  'seed_moved': sum(1 for s in slates if len(s) and s.seed != 0),
                  'report_differs_from_seed':
                      sum(1 for s in slates if len(s) and s.report_index != s.seed)}
    diag = {'hypotheses': hypotheses[:diag_top], 'groups': groups,
            'pool_size': int(len(pool)), **diag_extra,
            'score_gap': (winner['score'] - hypotheses[1]['score']
                          if len(hypotheses) > 1 else None)}
    return winner['name'], chosen, diag

"""Safeguards on geometric relocation.

Relocation is worth +0.234 recovery but costs -0.067 off-figure localization, and
off-figure queries are 45 of the 71-query localization denominator. The failures are
large: coordinates already correct to about 1px are moved roughly 1000px onto a
template node.

A guard decides, per query, whether to accept the geometric coordinate. It receives
evidence that is *independent of the score which selected the candidates*, so it is
not simply re-asking the question that produced the proposal.

Guards must never change a presence decision. A well-localized off-figure star stays
present and simply keeps its own coordinate.
"""
import numpy as np


class AcceptAll:
    """Baseline behaviour: adopt every geometric assignment."""

    name = 'accept-all'

    def __call__(self, query, original, proposed, context=None):
        return True, 'baseline'


class IndependentAppearance:
    """Accept only if an independent representation prefers the proposed location.

    `scores` maps a query index to a per-candidate score array computed from a
    representation not used to rank the candidates (full 32x32 support, or the
    surrounding annulus). Candidates are matched back by coordinate, so the guard
    works on the same fixed candidate identities.
    """

    def __init__(self, scores, coords, margin=0., name='independent-appearance'):
        self.scores, self.coords, self.margin, self.name = scores, coords, margin, name

    def _lookup(self, query, xy):
        c = self.coords.get(query)
        if c is None or not len(c):
            return None
        d = np.linalg.norm(np.asarray(c, float) - np.asarray(xy, float), axis=1)
        j = int(np.argmin(d))
        return float(self.scores[query][j]) if d[j] <= 1.5 else None

    def __call__(self, query, original, proposed, context=None):
        a = self._lookup(query, original)
        b = self._lookup(query, proposed)
        if a is None or b is None:
            return False, 'unmatched-candidate'
        if b >= a + self.margin:
            return True, f'independent {b:.3f} >= {a:.3f}'
        return False, f'independent {b:.3f} < {a:.3f}'


class GroupHeldOutSupport:
    """Accept only if the fit still predicts the location without this query's group.

    The transform is refitted from the winning correspondences after removing every
    correspondence belonging to the query's own physical-star group, then the
    proposed location is required to stay within `tolerance` of the node it was
    matched to. Removing the whole group matters: two queries that are repeated
    views of one star would otherwise supply each other's evidence.
    """

    name = 'group-held-out'

    def __init__(self, tolerance=18., min_remaining=4):
        self.tolerance, self.min_remaining = tolerance, min_remaining

    def __call__(self, query, original, proposed, context=None):
        if context is None:
            return True, 'no-context'
        template, pool, pairs, tags, group, fit = (
            context['template'], context['pool'], context['pairs'],
            context['tags'], context['group_of_query'], context['fit'])
        keep = [(i, j) for i, j in pairs if int(tags[j]) != group]
        if len(keep) < self.min_remaining:
            return False, f'only {len(keep)} independent correspondences'
        ii = np.array([i for i, _ in keep]); jj = np.array([j for _, j in keep])
        m = fit(template[ii], pool[jj])
        if m is None:
            return False, 'held-out refit failed'
        node = context['node_of_query']
        if node is None:
            return False, 'no matched node'
        predicted = np.r_[template[node], 1.] @ m
        d = float(np.linalg.norm(predicted - np.asarray(proposed, float)))
        if d <= self.tolerance:
            return True, f'held-out prediction {d:.1f}px'
        return False, f'held-out prediction {d:.1f}px'


class AllOf:
    """Accept only if every guard accepts."""

    def __init__(self, *guards):
        self.guards = guards
        self.name = '+'.join(g.name for g in guards)

    def __call__(self, query, original, proposed, context=None):
        for g in self.guards:
            ok, why = g(query, original, proposed, context)
            if not ok:
                return False, f'{g.name}: {why}'
        return True, 'all guards accept'


def apply_guard(chosen, slates, guard, contexts=None):
    """Filter a {query: (x, y)} relocation map. Returns (kept, rejected_reasons)."""
    kept, rejected = {}, {}
    for q, xy in chosen.items():
        s = slates[q]
        original = s.xy[s.report_index]
        if np.linalg.norm(np.asarray(xy, float) - original) < .5:
            kept[q] = xy                     # no move; nothing to guard
            continue
        ok, why = guard(q, original, xy, (contexts or {}).get(q))
        if ok:
            kept[q] = xy
        else:
            rejected[q] = why
    return kept, rejected

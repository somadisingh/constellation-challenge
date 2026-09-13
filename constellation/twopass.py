"""One geometry-guided pass over the retained appearance candidates, then one refit.

Measured problem this addresses: pool eligibility is decided by the top-two appearance
gap, so improving the appearance score makes queries *less* ambiguous, supplies fewer
alternatives to the verification pool, reduces relocation, and lowers recovery. With the
pose-admissible verifier every figure query retains a candidate within 12px (retention
1.000) yet figure localization falls, because the retained correct candidate is neither
ranked first nor offered to geometry.

So eligibility is decoupled from the appearance gap. A first pass produces competing
hypotheses; the node positions they predict then admit any *already existing* candidate
lying near them, whatever its appearance gap, and a second pass re-verifies. No location
is manufactured from a template prediction: only candidates the appearance stage already
produced can be admitted, so every pooled point retains image support.

Bounded by construction: one extra pass, node positions taken from a fixed number of
leading hypotheses, and the per-query pool cap still applies.
"""
import numpy as np
from .joint import recognize_joint
from .slate import slate_from_candidates, mark_eligible, with_scores


def predicted_nodes(diag, n_hypotheses=3):
    """Node positions predicted by the leading verified hypotheses."""
    out = []
    for h in (diag.get('hypotheses') or [])[:n_hypotheses]:
        nodes = h.get('nodes')
        if nodes and h.get('support', 0) >= 4:
            out.append(np.asarray(nodes, float).reshape(-1, 2))
    if not out:
        return np.empty((0, 2))
    return np.vstack(out)


def recognize_two_pass(alternatives, patterns, node_radius=18., n_hypotheses=3,
                       **kwargs):
    """Two-pass recognition. Returns (name, chosen, diagnostics) like `recognize_joint`.

    The diagnostics carry both passes so the contribution is attributable.
    """
    slates = [s if hasattr(s, 'xy') else slate_from_candidates(s) for s in alternatives]
    name1, chosen1, diag1 = recognize_joint(slates, patterns, **kwargs)
    nodes = predicted_nodes(diag1, n_hypotheses)
    if not len(nodes):
        diag1['two_pass'] = {'applied': False, 'reason': 'no verified hypothesis'}
        return name1, chosen1, diag1

    enriched = [mark_eligible(s, nodes, node_radius) if len(s) else s for s in slates]
    added = sum(len(s.eligible_indices()) for s in enriched if len(s))
    gap = kwargs.get('gap', .03)
    newly = 0
    for s in enriched:
        if not len(s):
            continue
        already = 1 if s.ambiguity_gap >= gap else min(
            kwargs.get('top_k', 8), len(s))
        newly += max(0, len(s.eligible_indices()) - already)
    if not added:
        diag1['two_pass'] = {'applied': False, 'reason': 'no candidate near a node'}
        return name1, chosen1, diag1

    name2, chosen2, diag2 = recognize_joint(enriched, patterns, **kwargs)
    diag2['two_pass'] = {'applied': True, 'node_radius': node_radius,
                         'n_hypotheses': n_hypotheses,
                         'eligible_candidates': int(added),
                         'newly_admitted': int(newly),
                         'first_pass_class': name1,
                         'first_pass_relocations': len(chosen1),
                         'class_changed': name1 != name2}
    return name2, chosen2, diag2

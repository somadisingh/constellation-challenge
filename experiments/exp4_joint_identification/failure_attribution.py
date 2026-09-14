"""Phase 2 (continued): synthesize `headroom_oracles.json` into an explicit
failure-mode attribution per scene, distinguishing:

  - retrieval failure           (level 4 loses vs level 2/3: no candidate near
                                  truth exists in the frozen bank at all)
  - candidate-ranking failure    (level 5 loses vs level 4: the correct
                                  candidate exists but appearance ranks it low)
  - insufficient figure coverage (level 1 issues too few figure points to reach
                                  min_support=4 after consolidation)
  - hypothesis-generation failure (the TRUE class never appears with support>=4
                                  in ANY level's hypothesis list, i.e. no seed
                                  triple/quad ever produced a valid affine fit
                                  that geometry could verify)
  - correct-class-wrong-placement (true class appears and has support, but its
                                  matched nodes are NOT close to true figure
                                  coordinates -- a fit exists but is geometrically
                                  wrong)
  - incorrect-scoring-of-correct-placement (true class's fit IS geometrically
                                  correct near truth, has support, but a
                                  DIFFERENT class scores higher -- a pure
                                  scoring/null-model problem, not a fitting one)
  - clutter/null-model failure   (winner has high support but low, cluttered
                                  coverage; consistent with lab/LEDGER.md's
                                  documented +0.660 score-vs-node-count
                                  correlation for wrong classes)

This module makes NO new geometry calls; it reads only `headroom_oracles.json`
plus, for the correct-placement check, the raw hypothesis lists that
`headroom_oracles.py` already discarded (`_rank_and_gap` only recorded top-5
names) -- so the placement check re-runs `recognize_joint` ONCE per scene at
level 4 (oracle_best_candidate) specifically to recover the true class's own
matched-node coordinates when it is not the winner. This is the SAME oracle
level already computed, re-run only to keep its full hypothesis list instead of
discarding it; it does not consult any additional information.
"""
from __future__ import annotations

import numpy as np

from . import OUT, SCENES
from constellation.joint import recognize_joint
from experiments.exp1.data import query_records
from experiments.exp1.env import write_json
from experiments.exp1.evaluation import _aligned_set, load_real_aligned
from .headroom_oracles import (_old_paths, _patterns, _true_constellation,
                               _single_point_slates)


def _full_hypotheses_level4(scene: str) -> tuple:
    records = query_records(scene)
    node = load_real_aligned(_old_paths(), scene)
    coords = []
    for r in records:
        if not r['present']:
            coords.append(None)
            continue
        aligned = _aligned_set(node, r['index'])
        if not len(aligned):
            coords.append(None)
            continue
        d = np.hypot(aligned.xy[:, 0] - r['xy'][0], aligned.xy[:, 1] - r['xy'][1])
        best = int(np.argmin(d))
        coords.append((float(aligned.xy[best, 0]), float(aligned.xy[best, 1]))
                      if d[best] <= 12.0 else None)
    patterns = _patterns()
    name, chosen, diag = recognize_joint(_single_point_slates(coords), patterns, diag_top=48)
    return name, diag, records


def attribute_scene(scene: str, headroom: dict) -> dict:
    levels = headroom['levels']
    true_name = headroom['true_constellation']
    l1, l2, l4, l5 = (levels['1_oracle_figure_only'], levels['2_oracle_all_present'],
                     levels['4_oracle_best_candidate'], levels['5_c0_rank1'])

    reasons = []

    # Retrieval failure: oracle_best_candidate (perfect selection among REAL
    # retrieved candidates) already fails where oracle_all_present (perfect
    # coordinates) succeeds -> the true source has no admissible candidate
    # within 12px anywhere in the frozen bank for enough figure queries.
    retrieval_failure = (l2['diagnostics_summary']['correct_class_wins']
                         and not l4['diagnostics_summary']['correct_class_wins'])
    if retrieval_failure:
        reasons.append('retrieval_failure: perfect-coordinate oracle (level 2) '
                       'identifies correctly, but perfect-SELECTION-among-real-'
                       'candidates (level 4) does not -- some figure-star queries '
                       'have no admissible candidate within 12px of truth anywhere '
                       f'in the frozen bank (n_pool_missing={l4.get("n_pool_missing")}).')

    # Candidate-ranking failure: the correct candidate EXISTS (level 4 succeeds)
    # but the classical system's own top-1 appearance choice (level 5) does not.
    ranking_failure = (l4['diagnostics_summary']['correct_class_wins']
                       and not l5['diagnostics_summary']['correct_class_wins'])
    if ranking_failure:
        reasons.append('candidate_ranking_failure: the correct candidate is '
                       'present in the bank (level 4 succeeds) but classical '
                       'appearance ranking does not put it first (level 5 fails) '
                       'for enough queries to break the geometric fit.')

    # Insufficient figure coverage: fewer than 2x min_support(=4) figure points
    # issued at level 1, i.e. even a perfect coordinate oracle has thin support.
    thin_coverage = l1['n_issued'] < 8
    if thin_coverage:
        reasons.append(f'insufficient_figure_coverage: only {l1["n_issued"]} figure '
                       f'points exist for this constellation in the labelled scene, '
                       f'close to the min_support=4 floor.')

    # Hypothesis-generation vs scoring vs placement failure, via a full re-run at
    # level 4 that keeps the entire hypothesis list.
    name4, diag4, records = _full_hypotheses_level4(scene)
    hyps = diag4.get('hypotheses', [])
    true_hyp = next((h for h in hyps if h.get('name') == true_name), None)
    hypothesis_generation_failure = (true_hyp is None
                                     or true_hyp.get('support', 0) < 4)
    placement = None
    if true_hyp is not None and true_hyp.get('support', 0) >= 4 and 'nodes' in true_hyp:
        truth_xy = np.array([r['xy'] for r in records if r['present']], float)
        nodes = np.array(true_hyp['nodes'], float).reshape(-1, 2)
        if len(truth_xy) and len(nodes):
            d = np.linalg.norm(nodes[:, None, :] - truth_xy[None, :, :], axis=2)
            nearest = d.min(axis=1)
            placement = {'fraction_nodes_near_truth_12px': float((nearest <= 12.0).mean())}

    correct_placement_wrong_scoring = False
    wrong_placement = False
    if true_hyp is not None and true_hyp.get('support', 0) >= 4:
        if placement and placement['fraction_nodes_near_truth_12px'] >= 0.5:
            if name4 != true_name:
                correct_placement_wrong_scoring = True
                reasons.append('incorrect_scoring_of_correct_placement: the TRUE '
                               "class's own fit lands near the true figure-star "
                               f'coordinates ({placement["fraction_nodes_near_truth_12px"]:.0%} '
                               'of nodes within 12px) and has support>=4, but a '
                               f'DIFFERENT class ({name4}) scores higher -- a null-'
                               'model / scoring problem, not a fitting problem.')
        else:
            wrong_placement = True
            reasons.append('correct_class_wrong_placement: the TRUE class reaches '
                           'support>=4 somewhere, but its matched nodes are not '
                           'close to the true figure-star coordinates -- the '
                           'geometric fit itself is wrong even though the class '
                           'label eventually would be right if scored.')
    elif hypothesis_generation_failure:
        reasons.append('hypothesis_generation_failure: no seed triple/quad for the '
                       'TRUE class ever produced a verified affine fit with '
                       'support>=4 at this oracle level -- the seeding/RANSAC '
                       'stage itself never reaches the true correspondence, not a '
                       'scoring problem downstream of it.')

    # Clutter/null-model check on the actual WINNER (not the true class): does
    # the winner's support/coverage/node-count pattern match the LEDGER's
    # documented wrong-class-score-vs-size correlation?
    winner_hyp = hyps[0] if hyps else None
    clutter_signature = None
    if winner_hyp is not None and winner_hyp.get('name') != true_name:
        clutter_signature = {
            'winner_name': winner_hyp.get('name'),
            'winner_support': winner_hyp.get('support'),
            'winner_coverage': winner_hyp.get('coverage'),
            'winner_nodes_count': winner_hyp.get('nodes_count'),
        }
        if (winner_hyp.get('coverage') or 1.0) < 0.5:
            reasons.append(f'clutter_null_model_signature: the winning wrong class '
                           f'({winner_hyp.get("name")}) matches only '
                           f'{winner_hyp.get("coverage", 0):.0%} of its own template '
                           f'nodes -- consistent with lab/LEDGER.md\'s documented '
                           f'wrong-class score correlating with reference size '
                           f'rather than genuine correspondence density.')

    if not reasons:
        reasons.append('no single dominant failure mode isolated by this ladder; '
                       'see the full oracle table for this scene')

    return {
        'scene': scene, 'true_constellation': true_name,
        'retrieval_failure': retrieval_failure,
        'candidate_ranking_failure': ranking_failure,
        'insufficient_figure_coverage': thin_coverage,
        'hypothesis_generation_failure': hypothesis_generation_failure,
        'correct_class_wrong_placement': wrong_placement,
        'incorrect_scoring_of_correct_placement': correct_placement_wrong_scoring,
        'winner_clutter_signature': clutter_signature,
        'true_class_placement': placement,
        'reasons': reasons,
    }


def run(headroom_doc: dict, say=print) -> dict:
    out = {}
    for scene in SCENES:
        att = attribute_scene(scene, headroom_doc[scene])
        out[scene] = att
        say(f'\n=== {scene} failure attribution ===')
        for r in att['reasons']:
            say(f'  - {r}')
    return out


if __name__ == '__main__':
    import json
    doc = json.loads((OUT / 'headroom_oracles.json').read_text())
    result = run(doc)
    write_json(OUT / 'failure_attribution.json', result)
    print('\nwrote failure_attribution.json')

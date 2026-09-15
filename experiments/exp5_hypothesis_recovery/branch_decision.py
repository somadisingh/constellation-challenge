"""Predeclared, mechanical branch decision (Branch G vs Branch C), computed
from `oracle_audit.json` evidence ONLY. This decision is made BEFORE any new
recovery method is designed or implemented, and is never revisited after
seeing final held-out results (task rule: "Do not override this decision
after reading final results.").

Fixed rule (verbatim from the task):
  Branch G (geometric recovery) is prioritized when EVERY scene has:
    - at least 4 distinct figure-star candidates within 36px in the full bank; AND
    - at least 90% of the maximum recoverable figure queries represented in the bank.
  Branch C (candidate recovery) is added when either condition fails for any scene.
  If failure is mixed: implement Branch G first, then add Branch C only for the
  specific queries/scenes where candidate recall fails.
"""
from __future__ import annotations

import json

from . import OUT
from experiments.exp1.env import write_json

MIN_DISTINCT_SOURCES = 4
MIN_COVERAGE_FRACTION = 0.90


def evaluate(oracle_audit: dict, say=print) -> dict:
    recall = oracle_audit['candidate_recall']
    per_scene = {}
    all_pass_g = True
    for scene, node in recall.items():
        n_distinct = node['n_distinct_correct_physical_sources']
        fig_summary = node['per_stratum_summary']['figure']
        # "maximum recoverable figure queries represented in the bank" = figure
        # queries with a correct candidate ANYWHERE in the full bank at 36px
        # (the widest radius/coverage the task allows for this check).
        coverage = fig_summary.get('recall_r36_kall')
        distinct_ok = n_distinct >= MIN_DISTINCT_SOURCES
        coverage_ok = (coverage is not None) and (coverage >= MIN_COVERAGE_FRACTION)
        scene_pass_g = distinct_ok and coverage_ok
        per_scene[scene] = {
            'n_distinct_correct_physical_sources': n_distinct,
            'distinct_sources_meets_4': distinct_ok,
            'figure_recall_36px_full_bank': coverage,
            'coverage_meets_90pct': coverage_ok,
            'branch_g_condition_met': scene_pass_g,
        }
        if not scene_pass_g:
            all_pass_g = False
        say(f'  {scene}: distinct_sources={n_distinct} (>=4: {distinct_ok}), '
           f'coverage@36px={coverage} (>=90%: {coverage_ok}) -> '
           f'branch_g_condition_met={scene_pass_g}')

    if all_pass_g:
        decision = 'G'
        rationale = (
            'Every scene meets BOTH Branch-G preconditions (>=4 distinct correct '
            'physical figure-star sources in the full candidate bank, and >=90% of '
            'figure queries recoverable within 36px anywhere in the bank). '
            'Candidate retrieval is not the bottleneck. Branch G (robust '
            'multi-candidate geometric hypothesis recovery) is selected as the sole '
            'implemented branch; Branch C (scene-adaptive candidate recovery) is '
            'NOT implemented in this experiment.'
        )
    elif any(v['branch_g_condition_met'] for v in per_scene.values()):
        decision = 'mixed'
        rationale = (
            'Some but not all scenes meet the Branch-G preconditions. Per the '
            'task\'s fixed rule: implement Branch G first for every scene, then add '
            'Branch C only for the specific scenes/queries where candidate recall '
            'fails.'
        )
    else:
        decision = 'C'
        rationale = (
            'No scene meets the Branch-G preconditions. Candidate retrieval itself '
            'is the bottleneck; Branch C (scene-adaptive self-supervised candidate '
            'recovery) must be implemented before geometric recovery can be '
            'evaluated meaningfully.'
        )

    say(f'DECISION: {decision}')
    return {
        'rule': {
            'branch_g_requires_min_distinct_sources': MIN_DISTINCT_SOURCES,
            'branch_g_requires_min_coverage_fraction': MIN_COVERAGE_FRACTION,
            'coverage_definition': 'fraction of figure queries with a correct '
                                  '(<=36px) candidate anywhere in the full stored bank',
        },
        'per_scene': per_scene,
        'decision': decision,
        'rationale': rationale,
        'scenes_requiring_branch_c': sorted(s for s, v in per_scene.items()
                                            if not v['branch_g_condition_met']),
        'note': 'Computed mechanically from oracle_audit.json before Stage 2 was '
               'designed; not revisited after seeing held-out results.',
    }


def run(say=print) -> dict:
    oracle_audit = json.loads((OUT / 'oracle_audit.json').read_text())
    result = evaluate(oracle_audit, say=say)
    write_json(OUT / 'branch_decision.json', result)
    return result


if __name__ == '__main__':
    run()

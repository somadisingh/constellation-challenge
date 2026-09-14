"""Deployment decision for Experiment 4B (task's "Only if ... clears all
required promotion gates, generate ..." clause).

`gates.py` reports 8/14 gates pass, `overall_pass=False`. Six gates fail on
genuine measured evidence, not missing implementation:
  - gate 3: template-size normalization regresses independent geometric support
  - gate 6: the complete joint solver regresses a previously-correct scene
    (scorpius) on both seeds
  - gate 7: the complete joint solver fixes none of the previously-wrong scenes
  - gate 9/10: the full pipeline (Phase 1 rank-fusion integrated end to end)
    is flat on the primary seed and REGRESSES on the repeat seed vs the
    matching fixed-policy baseline
  - gate 11: three scene/seed combinations regress by more than the 0.02 floor

Because overall_pass is False, this module does NOT select a policy to deploy
and NOT generate a submission candidate. It records the decision and the exact
evidence behind it, mirroring `exp3_pairwise/deployment.py`'s structure but
recording a `not_promoted` verdict instead of a selected arm.
"""
from __future__ import annotations

import json

from . import OUT
from experiments.exp1.env import write_json


def build_deployment_policy(say=print) -> dict:
    gates = json.loads((OUT / 'gates.json').read_text())

    failing = [name for name, node in gates['checks'].items() if not node['pass']]
    policy = {
        'policy_type': 'not_promoted',
        'promotion_gate_result': {
            'n_pass': gates['n_pass'], 'n_gates': gates['n_gates'],
            'overall_pass': gates['overall_pass'],
            'failing_gates': failing,
        },
        'decision': (
            'No new identification policy from Experiment 4B is deployed. The task '
            'requires generating a submission candidate "only if the identification '
            'method clears all required promotion gates" -- it clears 8 of 14, not '
            'all, so none is generated.'
        ),
        'what_was_implemented_and_measured': (
            'Every phase (rank fidelity, independent geometric support, unqueried-star '
            'evidence + null model, joint multi-candidate beam solver, full whole-sky '
            'evaluation, synthetic screen) has executable code and real recorded '
            'results -- see metrics.json. Implementation completeness and performance-'
            'gate success are reported as separate fields (task rule 20); this file '
            'concerns only the latter.'
        ),
        'production_system_unchanged': (
            'The currently deployed system remains Experiment 3\'s verifier + '
            'Experiment 2\'s geometry integration, per '
            'outputs/exp3_pairwise/deployment_policy.json. This experiment does not '
            'modify, supersede, or replace that policy.'
        ),
        'reusable_positive_findings_not_gating_this_decision': [
            'Phase 1: linear score fusion of classical NCC + Exp3 pair logit genuinely '
            'improves isolated candidate-rank fidelity (mean top1_reward 0.7863->0.8267) '
            '-- this held on its own leave-one-sky-out evaluation (gate 1 passes) but did '
            'not survive full-pipeline integration (gates 9/10/11 fail).',
            'Phase 2: held_out_stability tie-break improves independent geometric support '
            'rank on both previously-poor scenes (pisces 15->8, taurus 18->11) without '
            'regressing scorpius (gate 2 passes).',
            'Phase 3: the sqrt(n) multiple-testing correction is necessary to prevent '
            'uncorrected unqueried-star evidence from breaking an already-correct scene '
            '(gate 4 passes) -- a real methodological finding worth keeping even though '
            'the complete joint solver did not end up using it successfully (gate 6 fails).',
        ],
        'never_uses': ['scene filename', 'scene identity', 'hidden labels',
                      'held-out results', 'Kaggle score'],
    }
    write_json(OUT / 'deployment_policy.json', policy)
    say(f"Deployment policy: not_promoted ({gates['n_pass']}/{gates['n_gates']} gates pass). "
       f"Failing gates: {failing}")
    return policy


if __name__ == '__main__':
    build_deployment_policy()

"""Required correction to Experiment 4B's gate semantics (Exp4C task's own
"Required correction to Experiment 4B gate semantics" section).

Experiment 4B's gate 8
(`8_joint_solver_direction_repeats_under_second_seed`) counted the fact that
its scorpius regression reproduces on both seeds as a PASSED promotion gate.
Reproducibility of a regression is useful diagnostic evidence (it tells you
the failure is systematic, not seed noise) -- but it is not positive evidence
FOR deployment, and must not be counted toward "gates that support promoting
this method." This module recomputes Exp4B's own `gates.json` with gate 8
reclassified as a diagnostic/informational check, never a pass, and records
the corrected descriptive count.

This does NOT rewrite or delete `outputs/exp4b_joint_solver/gates.json` --
that file is a protected, frozen record of Exp4B's own run. This module reads
it read-only and writes a NEW, separate correction record.
"""
from __future__ import annotations

import json

from . import OUT, ROOT
from experiments.exp1.env import write_json

EXP4B_GATES_PATH = ROOT / 'outputs' / 'exp4b_joint_solver' / 'gates.json'

# Gate 8 is the one being reclassified: it measured whether the scorpius
# regression reproduces across seeds, not whether the method is fit for
# deployment.
DIAGNOSTIC_ONLY_GATES = ('8_joint_solver_direction_repeats_under_second_seed',)


def run(say=print) -> dict:
    exp4b_gates = json.loads(EXP4B_GATES_PATH.read_text())
    checks = exp4b_gates['checks']

    recount = {'positive': [], 'failed': [], 'diagnostic_only': []}
    for name, node in checks.items():
        if name in DIAGNOSTIC_ONLY_GATES:
            recount['diagnostic_only'].append(name)
        elif node.get('pass'):
            recount['positive'].append(name)
        else:
            recount['failed'].append(name)

    corrected_summary = (
        f"Experiment 4B's own gates.json reports {exp4b_gates['n_pass']}/"
        f"{exp4b_gates['n_gates']} gates passing, including gate 8 ("
        f"'{DIAGNOSTIC_ONLY_GATES[0]}') as a pass. Gate 8 measured only "
        f"whether the scorpius regression reproduces across both seeds -- "
        f"repeatability of a FAILURE is diagnostic evidence that the failure "
        f"is systematic (not seed noise), never positive evidence for "
        f"deployment. Recounted correctly: "
        f"{len(recount['positive'])} positive gates, "
        f"{len(recount['failed'])} failed gates, and "
        f"{len(recount['diagnostic_only'])} diagnostic-only check "
        f"(excluded from both the positive and failed counts, since it does "
        f"not bear on promotion either way). Experiment 4B implemented every "
        f"requested phase; its resulting method failed performance "
        f"promotion; this correction does not change that verdict "
        f"(overall_pass was already False), it only corrects the descriptive "
        f"gate tally."
    )
    say(corrected_summary)

    result = {
        'source': str(EXP4B_GATES_PATH.relative_to(ROOT)),
        'exp4b_reported_n_pass': exp4b_gates['n_pass'],
        'exp4b_reported_n_gates': exp4b_gates['n_gates'],
        'exp4b_reported_overall_pass': exp4b_gates['overall_pass'],
        'diagnostic_only_gates': list(DIAGNOSTIC_ONLY_GATES),
        'diagnostic_only_gate_reason': (
            'This gate tests reproducibility of the scorpius regression across '
            'seeds, not fitness for deployment. A repeatable regression cannot '
            'count as a positive promotion signal.'
        ),
        'recount': {
            'n_positive': len(recount['positive']),
            'n_failed': len(recount['failed']),
            'n_diagnostic_only': len(recount['diagnostic_only']),
            'positive_gates': sorted(recount['positive']),
            'failed_gates': sorted(recount['failed']),
            'diagnostic_only_gates_list': sorted(recount['diagnostic_only']),
        },
        'corrected_summary': corrected_summary,
        'overall_pass_unchanged': exp4b_gates['overall_pass'] is False,
        'note': (
            'This file does not modify outputs/exp4b_joint_solver/gates.json, '
            'which remains the frozen, protected record of what Experiment 4B '
            'actually computed and reported at the time. This is a dated, '
            'separate correction record, exactly like '
            'prior_claim_corrections.json corrected an earlier false claim '
            'without deleting the original text it corrected.'
        ),
    }
    write_json(OUT / 'prior_gate_corrections.json', result)
    say(f"n_positive={result['recount']['n_positive']} "
       f"n_failed={result['recount']['n_failed']} "
       f"n_diagnostic_only={result['recount']['n_diagnostic_only']}")
    return result


if __name__ == '__main__':
    run()

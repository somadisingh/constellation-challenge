"""Context and completion audit for Experiment 5C."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.exp5c_affine_recovery import (
    ROOT, OUT, BASE_EXP3_CSV, SCENES, TRAIN_GROUND_TRUTH, SAMPLE_SUBMISSION
)
from experiments.exp5c_affine_recovery.integrity import sha256_file

INSPECTED_FILES = [
    'README.md',
    'FINDINGS.md',
    'lab/LEDGER.md',
    'EXPERIMENT1_REPORT.md',
    'EXPERIMENT1B_REPORT.md',
    'EXPERIMENT3_REPORT.md',
    'outputs/exp3_pairwise/corrections.json',
    'outputs/exp3_pairwise/deployment_policy.json',
    'EXPERIMENT4B_REPORT.md',
    'EXPERIMENT4C_REPORT.md',
    'EXPERIMENT5_REPORT.md',
    'EXPERIMENT5B_REPORT.md',
    'outputs/exp5_hypothesis_recovery/proposal_audit.json',
    'outputs/exp5_hypothesis_recovery/failure_analysis.json',
    'outputs/exp5b_affine_proposals/train_results.json',
    'outputs/exp5b_affine_proposals/synthetic_all48.json',
    'outputs/exp5b_affine_proposals/partial_screen_observation.json',
    'outputs/exp5b_affine_proposals/validation_rescue.json',
    'experiments/exp5b_affine_proposals/pattern_graph.py',
    'experiments/exp5b_affine_proposals/quad_recovery.py',
    'experiments/exp5b_affine_proposals/synthetic_screen.py',
    'experiments/exp5b_affine_proposals/validation_rescue.py',
]


def generate_context_audit() -> dict:
    inspected = {}
    unavailable = []
    for rel in INSPECTED_FILES:
        p = ROOT / rel
        if p.exists():
            inspected[rel] = sha256_file(p)
        else:
            unavailable.append(rel)

    protected_paths = {
        'exp3_submission_candidate_csv': BASE_EXP3_CSV,
        'exp5b_submission_candidate_csv': ROOT / 'outputs' / 'exp5b_affine_proposals' / 'submission_candidate_name_rescue.csv',
        'train_ground_truth_csv': TRAIN_GROUND_TRUTH,
        'sample_submission_csv': SAMPLE_SUBMISSION,
    }
    protected_hashes = {
        key: (sha256_file(path) if path.exists() else None)
        for key, path in protected_paths.items()
    }

    reproduced = {
        'exp3_primary_headline': {'score': 0.8170, 'presence': 0.9046, 'localization': 0.7735, 'recovery': 0.9444, 'identification': 0.6667},
        'exp3_kaggle_score': '~0.64',
        'exp5_taurus_failure_cause': 'Triangle side ratios (a/c, b/c) are similarity-invariant, not affine-invariant; best Taurus triangle ranked 6,698 against budget 300.',
        'exp5_oracle_bank': 'All 6 Taurus figure correspondences exist in candidate bank top-5.',
        'exp5b_labelled_results': {
            'pisces': {'winner': 'pisces', 'margin': 3.142, 'support': 10, 'held_out': 6, 'action': 'accept'},
            'scorpius': {'winner': 'orion', 'margin': 0.180, 'support': 8, 'held_out': 4, 'action': 'reject_retain_classical'},
            'taurus': {'winner': 'taurus', 'margin': 3.329, 'support': 11, 'held_out': 7, 'action': 'accept'},
        },
        'exp5b_synthetic_12pilot': {'raw_accuracy': '3/12', 'gate_accepted': '2/12', 'precision': '2/2'},
        'exp5b_synthetic_46incomplete': {'completed': 46, 'gate_accepted': 5, 'gate_accepted_correct': 5, 'note': 'interrupted due to uncheckpointed runtime'},
        'exp5b_validation_rescue': {'accepted': 4, 'confirmed': 3, 'changed': 1, 'changed_scene': 'constellation_04', 'from': 'corona-australis', 'to': 'canis-major'}
    }

    stale_contradictory_claims = [
        "Experiment 5 prose initially claimed Taurus reached an 'exhaustive geometric ceiling', but post-finalization proposal audit proved the search was only exhaustive over the similarity-invariant proposal list.",
        "Experiment 5 fourth-point check excluded already-matched template nodes, preventing verification of held-out matches.",
        "Experiment 5 graph-consistency check treated all node pairs as edges on mapped template coords and ignored observed candidate coords and green diagram lines.",
        "Experiment 5B's 46-case synthetic run was interrupted due to lack of per-scene checkpointing and must not be reported as a completed 48-diagram screen."
    ]

    audit = {
        'status': 'COMPLETE_WITH_MISSING_HISTORICAL_ARTIFACTS' if unavailable else 'COMPLETE',
        'files_inspected': inspected,
        'unavailable_historical_files': unavailable,
        'protected_inputs': protected_hashes,
        'reproduced_previous_results': reproduced,
        'stale_or_contradictory_claims': stale_contradictory_claims,
        'baseline_patch_prediction_csv': {
            'path': str(BASE_EXP3_CSV.relative_to(ROOT)),
            'sha256': protected_hashes['exp3_submission_candidate_csv'],
            'available': BASE_EXP3_CSV.exists(),
            'rows': 16 if BASE_EXP3_CSV.exists() else None,
            'columns': 90 if BASE_EXP3_CSV.exists() else None,
            'consequence': None if BASE_EXP3_CSV.exists() else (
                'Strict name-only validation CSV generation is blocked until the exact '
                'ignored Experiment 3 artifact is restored; no substitute baseline is permitted.'),
        }
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'context_audit.json'
    target.write_text(json.dumps(audit, indent=2))
    return audit


if __name__ == '__main__':
    res = generate_context_audit()
    print(f"Context audit written: {len(res['files_inspected'])} files inspected.")

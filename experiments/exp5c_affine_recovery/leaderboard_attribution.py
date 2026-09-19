"""Record public-leaderboard ablations without treating them as validation."""
from __future__ import annotations

import csv
from pathlib import Path

from experiments.exp1.env import sha256_file
from experiments.exp5c_affine_recovery import BASE_EXP3_CSV, OUT
from experiments.exp5c_affine_recovery.synthetic import atomic_json
from experiments.exp5c_affine_recovery.validation import mutate_names_only


PUBLIC_SCORES = {
    'experiment3_baseline': 0.64695,
    'constellation_04_only': 0.64695,
    'constellation_08_only': 0.69695,
    'combined_04_and_08': 0.69695,
}


def _rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline='') as handle:
        return {row['Id']: row for row in csv.DictReader(handle)}


def _differences(base: Path, candidate: Path) -> list[dict[str, str]]:
    before = _rows(base)
    after = _rows(candidate)
    if before.keys() != after.keys():
        raise AssertionError('candidate scene rows differ from Experiment 3 baseline')
    changes = []
    for scene in before:
        for column, old_value in before[scene].items():
            new_value = after[scene][column]
            if old_value != new_value:
                changes.append({
                    'scene': scene,
                    'column': column,
                    'before': old_value,
                    'after': new_value,
                })
    return changes


def run() -> dict:
    if not BASE_EXP3_CSV.exists():
        raise FileNotFoundError(f'exact Experiment 3 baseline missing: {BASE_EXP3_CSV}')

    recommended = OUT / 'submission_recommended_constellation_08_only.csv'
    csv_record = mutate_names_only(
        BASE_EXP3_CSV,
        recommended,
        {'constellation_08': 'orion'},
    )
    changes = _differences(BASE_EXP3_CSV, recommended)
    expected = [{
        'scene': 'constellation_08',
        'column': 'constellation',
        'before': 'eridanus',
        'after': 'orion',
    }]
    if changes != expected:
        raise AssertionError(f'unexpected recommended-candidate changes: {changes}')

    record = {
        'status': 'COMPLETE',
        'evidence_type': 'user_reported_kaggle_public_leaderboard',
        'not_validation_evidence': True,
        'public_scores': PUBLIC_SCORES,
        'attribution': {
            'constellation_08_to_orion_gain': 0.05000,
            'constellation_04_to_canis_major_measurable_gain': 0.0,
            'combined_increment_over_constellation_08_only': 0.0,
        },
        'recommended_candidate': {
            **csv_record,
            'base_path': str(BASE_EXP3_CSV),
            'base_sha256': sha256_file(BASE_EXP3_CSV),
            'changes': changes,
            'reason': (
                'The constellation_08-only submission matches the best reported public '
                'score while changing fewer cells than the combined submission.'
            ),
        },
        'rejected_for_deployment': {
            'constellation_04': {
                'proposed_name': 'canis-major',
                'reason': 'No measurable public-score gain in the one-change ablation.',
            },
        },
        'taurus_policy': {
            'role': 'adversarial_diagnostic_only',
            'excluded_from_validation': True,
            'excluded_from_confidence_calibration': True,
            'excluded_from_model_selection': True,
            'excluded_from_promotion_gates': True,
            'reason': 'User states Taurus was intentionally constructed to skew results.',
        },
        'no_kaggle_upload_performed_by_experiment_code': True,
    }
    atomic_json(OUT / 'leaderboard_attribution.json', record)
    return record


if __name__ == '__main__':
    import json
    print(json.dumps(run(), indent=2))

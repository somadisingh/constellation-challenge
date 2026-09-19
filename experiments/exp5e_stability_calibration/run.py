"""Calibrate winner stability using only known synthetic truth."""
from __future__ import annotations

import json
import time

from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from experiments.exp5c_affine_recovery.solver import solve
from experiments.exp5c_affine_recovery.synthetic import _affine_case, atomic_json
from experiments.exp5d_stability_audit.run import perturb
from experiments.exp5e_stability_calibration import (
    CALIBRATION_GATE, FINAL_SEEDS, INCORRECT_PER_SEED, OUT,
    PERTURBATION_FAMILIES, STABLE_WINS_MIN,
)


SCREEN_CONFIG = {
    'max_scene_quads': 3000,
    'proposal_budget': 2000,
    'progressive_budgets': [1000, 2000],
}
SCREEN_ARMS = ('rank1', 'one_alt_top5', 'adaptive')


def select_cases() -> list[dict]:
    selected = []
    for seed_label, path in FINAL_SEEDS:
        data = json.loads(path.read_text())
        rows = [row for name, row in data['completed'].items() if name != 'taurus']
        correct = sorted((row for row in rows if row['correct']), key=lambda row: row['pattern'])
        incorrect = sorted(
            (row for row in rows if not row['correct']),
            key=lambda row: (-float(row['decisions']['confidence']), row['pattern']),
        )[:INCORRECT_PER_SEED]
        for group, group_rows in [('all_correct', correct),
                                  ('highest_confidence_incorrect', incorrect)]:
            for row in group_rows:
                selected.append({
                    'seed_label': seed_label,
                    'selection_group': group,
                    'pattern': row['pattern'],
                    'generation_seed': int(row['generation_seed']),
                    'baseline_winner': row['winner'],
                    'baseline_correct': bool(row['correct']),
                    'baseline_confidence': float(row['decisions']['confidence']),
                    'baseline_margin': float(row['margin']),
                })
    return selected


def evaluate(cases: list[dict], trials: list[dict]) -> dict:
    stability = []
    for case in cases:
        matches = [row for row in trials
                   if row['seed_label'] == case['seed_label']
                   and row['pattern'] == case['pattern']]
        wins = sum(row['winner_matches_baseline'] for row in matches)
        stability.append({**case, 'stable_wins': wins, 'trials': len(matches),
                          'stable': wins >= STABLE_WINS_MIN})
    stable = [row for row in stability if row['stable']]
    stable_correct = [row for row in stable if row['baseline_correct']]
    stable_incorrect = [row for row in stable if not row['baseline_correct']]
    all_incorrect = [row for row in stability if not row['baseline_correct']]
    precision = len(stable_correct) / len(stable) if stable else None
    incorrect_rate = len(stable_incorrect) / max(1, len(all_incorrect))
    metrics = {
        'selected_cases': len(stability),
        'selected_correct': sum(row['baseline_correct'] for row in stability),
        'selected_incorrect': len(all_incorrect),
        'stable_cases': len(stable),
        'stable_correct': len(stable_correct),
        'stable_incorrect': len(stable_incorrect),
        'stable_precision': precision,
        'stable_incorrect_rate': incorrect_rate,
    }
    checks = {
        'stable_precision': precision is not None and precision >= CALIBRATION_GATE['stable_precision_min'],
        'stable_correct_count': len(stable_correct) >= CALIBRATION_GATE['stable_correct_count_min'],
        'stable_incorrect_rate': incorrect_rate <= CALIBRATION_GATE['stable_incorrect_rate_max'],
        'taurus_excluded': all(row['pattern'] != 'taurus' for row in stability),
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'metrics': metrics, 'case_stability': stability}


def run(say=print) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = select_cases()
    if any(row['pattern'] == 'taurus' for row in cases):
        raise AssertionError('Taurus must not enter stability calibration')
    atomic_json(OUT / 'selection.json', {
        'status': 'FROZEN_BEFORE_PERTURBATION_RUN',
        'selection_rule': (
            'All correct non-Taurus winners plus the five highest-confidence '
            'incorrect non-Taurus winners from each untouched final seed.'),
        'families': list(PERTURBATION_FAMILIES),
        'stable_wins_min': STABLE_WINS_MIN,
        'calibration_gate': CALIBRATION_GATE,
        'cases': cases,
    })
    index = get_or_create_index()
    trials = []
    started = time.perf_counter()
    for case_pos, selected in enumerate(cases):
        graph = index.graphs[selected['pattern']]
        synthetic = _affine_case(
            selected['pattern'], graph['nodes'], selected['generation_seed'])
        for family_pos, family in enumerate(PERTURBATION_FAMILIES):
            trial_seed = selected['generation_seed'] + 9001 + family_pos
            alternatives = perturb(synthetic['alternatives'], family, trial_seed)
            result = solve(alternatives, index, arms=SCREEN_ARMS,
                           config=SCREEN_CONFIG, seed=trial_seed)
            top = result['ranked'][0] if result['ranked'] else {}
            trial = {
                'seed_label': selected['seed_label'],
                'pattern': selected['pattern'],
                'baseline_correct': selected['baseline_correct'],
                'baseline_winner': selected['baseline_winner'],
                'family': family,
                'winner': result['winner'],
                'winner_matches_baseline': result['winner'] == selected['baseline_winner'],
                'margin': result['margin'],
                'support': int(top.get('support', 0)),
                'held_out_support': int(top.get('held_out_support', 0)),
                'runtime_seconds': result['runtime_seconds'],
            }
            trials.append(trial)
            atomic_json(OUT / 'checkpoint.json', {
                'status': 'RUNNING', 'completed': len(trials),
                'total': len(cases) * len(PERTURBATION_FAMILIES),
                'trials': trials,
            })
            say(f"{case_pos + 1}/{len(cases)} {selected['seed_label']} "
                f"{selected['pattern']} {family}: {result['winner']}")

    decision = evaluate(cases, trials)
    output = {
        'status': 'COMPLETE',
        'screen_config': SCREEN_CONFIG,
        'screen_arms': list(SCREEN_ARMS),
        'selection': cases,
        'trials': trials,
        'decision': decision,
        'validation_scan_authorized': decision['passed'],
        'validation_scan_performed': False,
        'submission_candidate': None,
        'taurus_read_or_used': False,
        'kaggle_score_used_for_thresholds': False,
        'runtime_seconds': time.perf_counter() - started,
        'no_kaggle_upload': True,
    }
    atomic_json(OUT / 'stability_calibration.json', output)
    atomic_json(OUT / 'checkpoint.json', {
        'status': 'COMPLETE', 'completed': len(trials),
        'total': len(trials), 'trials': trials,
    })
    return output


if __name__ == '__main__':
    result = run()
    print(json.dumps({
        'decision': result['decision'],
        'validation_scan_authorized': result['validation_scan_authorized'],
        'runtime_seconds': result['runtime_seconds'],
    }, indent=2))

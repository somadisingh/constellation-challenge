"""Run a frozen perturbation audit before considering a constellation_07 CSV."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from experiments.exp1.env import sha256_file
from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from experiments.exp5c_affine_recovery.solver import alternatives_from_prediction_json, solve
from experiments.exp5c_affine_recovery.synthetic import atomic_json
from experiments.exp5c_affine_recovery.validation import mutate_names_only
from experiments.exp5d_stability_audit import (
    BASE_EXP3_CSV, CANDIDATE_NAME, EXISTING_NAME, OUT, PASS_RULE, SCENE, SEEDS, SOURCE,
)


FAMILIES = ('full_top5', 'top3', 'top1', 'jitter_2px', 'query_dropout_10pct')


def _depth(alternatives: list, depth: int) -> list:
    return [list(candidates[:depth]) for candidates in alternatives]


def _jitter(alternatives: list, seed: int, sigma: float = 2.0) -> list:
    rng = np.random.default_rng(seed)
    perturbed = []
    for candidates in alternatives:
        row = []
        for candidate in candidates:
            values = list(candidate)
            values[0] += float(rng.normal(0.0, sigma))
            values[1] += float(rng.normal(0.0, sigma))
            row.append(tuple(values))
        perturbed.append(row)
    return perturbed


def _drop_queries(alternatives: list, seed: int, fraction: float = 0.10) -> list:
    rng = np.random.default_rng(seed)
    keep_count = max(4, int(round(len(alternatives) * (1.0 - fraction))))
    keep = set(map(int, rng.choice(len(alternatives), keep_count, replace=False)))
    return [row for index, row in enumerate(alternatives) if index in keep]


def perturb(alternatives: list, family: str, seed: int) -> list:
    if family == 'full_top5':
        return _depth(alternatives, 5)
    if family == 'top3':
        return _depth(alternatives, 3)
    if family == 'top1':
        return _depth(alternatives, 1)
    if family == 'jitter_2px':
        return _jitter(_depth(alternatives, 5), seed)
    if family == 'query_dropout_10pct':
        return _drop_queries(_depth(alternatives, 5), seed)
    raise ValueError(f'unknown perturbation family: {family}')


def evaluate(records: list[dict]) -> dict:
    candidate_rows = [row for row in records if row['winner'] == CANDIDATE_NAME]
    by_family = defaultdict(list)
    for row in records:
        by_family[row['family']].append(row)
    family_wins = {
        family: sum(row['winner'] == CANDIDATE_NAME for row in rows)
        for family, rows in sorted(by_family.items())
    }
    margins = [row['margin'] for row in candidate_rows]
    supports = [row['support'] for row in candidate_rows]
    held_out = [row['held_out_support'] for row in candidate_rows]
    metrics = {
        'trials': len(records),
        'candidate_wins': len(candidate_rows),
        'candidate_rate': len(candidate_rows) / max(1, len(records)),
        'winner_counts': dict(Counter(row['winner'] for row in records)),
        'candidate_wins_by_family': family_wins,
        'candidate_median_margin': float(np.median(margins)) if margins else 0.0,
        'candidate_median_support': float(np.median(supports)) if supports else 0.0,
        'candidate_median_held_out_support': float(np.median(held_out)) if held_out else 0.0,
    }
    checks = {
        'overall_candidate_rate': metrics['candidate_rate'] >= PASS_RULE['overall_candidate_rate_min'],
        'each_family_candidate_wins': all(
            wins >= PASS_RULE['candidate_wins_per_family_min']
            for wins in family_wins.values()),
        'median_margin': metrics['candidate_median_margin'] >= PASS_RULE['median_margin_min'],
        'median_support': metrics['candidate_median_support'] >= PASS_RULE['median_support_min'],
        'median_held_out_support': (
            metrics['candidate_median_held_out_support'] >=
            PASS_RULE['median_held_out_support_min']),
    }
    return {'passed': all(checks.values()), 'checks': checks, 'metrics': metrics}


def run(say=print) -> dict:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    doc = json.loads(SOURCE.read_text())
    if doc.get('scene_id', SCENE) != SCENE and doc.get('Id', SCENE) != SCENE:
        raise AssertionError('unexpected validation scene source')
    alternatives = alternatives_from_prediction_json(doc)
    index = get_or_create_index()
    records = []
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for family in FAMILIES:
        for seed in SEEDS:
            trial_alternatives = perturb(alternatives, family, seed)
            result = solve(trial_alternatives, index, seed=seed)
            top = result['ranked'][0] if result['ranked'] else {}
            row = {
                'family': family,
                'seed': seed,
                'winner': result['winner'],
                'margin': result['margin'],
                'support': int(top.get('support', 0)),
                'held_out_support': int(top.get('held_out_support', 0)),
                'supported_edges': int(top.get('graph', {}).get('supported_edges', 0)),
                'largest_component': int(top.get('graph', {}).get('largest_component', 0)),
                'stream_agreement': int(top.get('stream_agreement', 0)),
                'proposal_stream': top.get('proposal_stream'),
                'proposals_evaluated': result['proposals_evaluated'],
                'runtime_seconds': result['runtime_seconds'],
            }
            records.append(row)
            atomic_json(OUT / 'checkpoint.json', {
                'status': 'RUNNING', 'completed': len(records),
                'total': len(FAMILIES) * len(SEEDS), 'records': records,
            })
            say(f"{family} seed={seed}: {row['winner']} margin={row['margin']:.3f}")

    decision = evaluate(records)
    csv_record = None
    if decision['passed']:
        if not BASE_EXP3_CSV.exists():
            raise FileNotFoundError(BASE_EXP3_CSV)
        csv_record = mutate_names_only(
            BASE_EXP3_CSV,
            OUT / 'submission_constellation_07_perseus.csv',
            {SCENE: CANDIDATE_NAME},
        )
    output = {
        'status': 'COMPLETE',
        'scene': SCENE,
        'existing_name': EXISTING_NAME,
        'candidate_name': CANDIDATE_NAME,
        'source_path': str(SOURCE),
        'source_sha256': sha256_file(SOURCE),
        'frozen_pass_rule': PASS_RULE,
        'families': list(FAMILIES),
        'seeds': list(SEEDS),
        'records': records,
        'decision': decision,
        'submission_candidate': csv_record,
        'taurus_read_or_used': False,
        'kaggle_score_used_for_thresholds': False,
        'runtime_seconds': time.perf_counter() - started,
        'no_kaggle_upload': True,
    }
    atomic_json(OUT / 'stability_results.json', output)
    atomic_json(OUT / 'checkpoint.json', {
        'status': 'COMPLETE', 'completed': len(records),
        'total': len(records), 'records': records,
    })
    return output


if __name__ == '__main__':
    result = run()
    print(json.dumps({
        'status': result['status'],
        'decision': result['decision'],
        'submission_candidate': result['submission_candidate'],
        'runtime_seconds': result['runtime_seconds'],
    }, indent=2))

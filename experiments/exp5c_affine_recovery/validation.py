"""Frozen validation inference with a strict Experiment-3 name-only contract."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.exp1.env import sha256_file
from experiments.exp5c_affine_recovery import OUT, BASE_EXP3_CSV, SAMPLE_SUBMISSION
from experiments.exp5c_affine_recovery.confidence import gate_decisions
from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from experiments.exp5c_affine_recovery.solver import alternatives_from_prediction_json, solve
from experiments.exp5c_affine_recovery.synthetic import atomic_json


SOURCE = OUT / 'reconstructed_validation'


def mutate_names_only(base: Path, target: Path, names: dict[str, str]) -> dict:
    with base.open(newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    before = [{k: v for k, v in row.items()} for row in rows]
    for row in rows:
        if row['Id'] in names: row['constellation'] = names[row['Id']]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    immutable = [c for c in fields if c != 'constellation']
    identical = all(a[c] == b[c] for a, b in zip(before, rows) for c in immutable)
    if not identical: raise AssertionError('non-constellation CSV cell changed')
    return {'path': str(target), 'sha256': sha256_file(target),
            'non_constellation_cells_identical': True}


def _load_base():
    if not BASE_EXP3_CSV.exists(): return None
    with BASE_EXP3_CSV.open(newline='') as f: return {r['Id']: r for r in csv.DictReader(f)}


def run(say=print) -> dict:
    index = get_or_create_index()
    policy = json.loads((OUT / 'deployment_policy.json').read_text())
    base = _load_base()
    with SAMPLE_SUBMISSION.open(newline='') as f:
        scene_ids = [r['Id'] for r in csv.DictReader(f)]
    records = {}
    for pos, scene in enumerate(scene_ids, 1):
        source = SOURCE / f'{scene}.json'
        doc = json.loads(source.read_text())
        result = solve(alternatives_from_prediction_json(doc), index, seed=42001)
        decisions = gate_decisions(result, policy.get('model'), policy)
        top = result['ranked'][0] if result['ranked'] else {}
        existing = base[scene]['constellation'] if base else None
        records[scene] = {
            'existing_name': existing, 'affine_winner': result['winner'],
            'final_strict': result['winner'] if decisions['strict'] and base else existing,
            'final_primary': result['winner'] if decisions['primary'] and base else existing,
            'final_exploratory': result['winner'] if decisions['exploratory'] and base else existing,
            'accepted': decisions, 'proposal_stream': top.get('proposal_stream'),
            'score': top.get('score'), 'margin': result['margin'],
            'support': top.get('support'), 'held_out_support': top.get('held_out_support'),
            'graph_support': top.get('graph'), 'calibration_confidence': decisions['confidence'],
            'runtime_seconds': result['runtime_seconds'],
            'peak_rss_platform_units': result['peak_rss_platform_units'],
            'candidate_source_sha256': sha256_file(source),
            'proposals_evaluated': result['proposals_evaluated'],
        }
        atomic_json(OUT / 'validation_predictions.checkpoint.json', {
            'status': 'RUNNING', 'completed': pos, 'total': len(scene_ids), 'records': records})
        say(f"{pos}/{len(scene_ids)} {scene}: affine={result['winner']} "
            f"strict={decisions['strict']} primary={decisions['primary']}")

    variants = {}
    if base:
        for variant, field in [('strict', 'final_strict'), ('primary', 'final_primary'),
                               ('exploratory', 'final_exploratory')]:
            names = {s: r[field] for s, r in records.items()}
            variants[variant] = mutate_names_only(
                BASE_EXP3_CSV, OUT / f'submission_{variant}_name_only.csv', names)
    status = 'COMPLETE' if base else 'BLOCKED_EXACT_EXP3_BASELINE_MISSING'
    summary = {}
    for variant, field in [('strict', 'final_strict'), ('primary', 'final_primary'),
                           ('exploratory', 'final_exploratory')]:
        accepted = [s for s, r in records.items() if r['accepted'][variant]]
        changed = [s for s in accepted if base and records[s][field] != records[s]['existing_name']]
        confirmed = [s for s in accepted if base and records[s][field] == records[s]['existing_name']]
        summary[variant] = {'accepted': len(accepted), 'changed': changed, 'confirmed': confirmed}
    out = {'status': status, 'exact_exp3_baseline_available': bool(base),
           'base_path': str(BASE_EXP3_CSV), 'records': records, 'summary': summary,
           'csvs': variants, 'no_kaggle_upload': True}
    atomic_json(OUT / 'validation_predictions.json', out)
    diff = {'status': status, 'base_sha256': sha256_file(BASE_EXP3_CSV) if base else None,
            'variants': summary, 'non_constellation_cells_identical': bool(base)}
    atomic_json(OUT / 'csv_diff.json', diff)
    return out


if __name__ == '__main__':
    run()

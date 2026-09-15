"""Pure aggregator: assembles `metrics.json` from every already-written
per-phase JSON. Computes nothing new -- every number here is read and
reshaped, never recomputed, matching the convention of every prior
experiment's own `metrics.py`.
"""
from __future__ import annotations

import json

from . import OUT, SCENES
from experiments.exp1.env import write_json


def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def run(say=print) -> dict:
    out = {
        'baseline_verification': _load('baseline_verification.json'),
        'oracle_audit_summary': {},
        'branch_decision': _load('branch_decision.json'),
        'geometry_ablations': _load('geometry_ablations.json'),
        'oof_primary': _load('oof_primary.json') if (OUT / 'oof_primary.json').exists() else None,
        'oof_repeat': _load('oof_repeat.json') if (OUT / 'oof_repeat.json').exists() else None,
        'synthetic_all48': _load('synthetic_all48.json') if (OUT / 'synthetic_all48.json').exists() else None,
        'runtime': _load('runtime.json') if (OUT / 'runtime.json').exists() else None,
    }
    oa = _load('oracle_audit.json')
    out['oracle_audit_summary'] = {
        scene: {
            'n_figure_with_correct_candidate_12px': node['n_figure_with_correct_candidate_12px'],
            'n_figure_queries': node['n_figure_queries'],
            'n_distinct_correct_physical_sources': node['n_distinct_correct_physical_sources'],
            'at_least_4_correct_figure_locations': node['at_least_4_correct_figure_locations'],
        } for scene, node in oa['candidate_recall'].items()
    }
    write_json(OUT / 'metrics.json', out)
    say('wrote metrics.json')
    return out


if __name__ == '__main__':
    run()

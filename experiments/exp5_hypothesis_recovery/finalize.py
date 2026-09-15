"""Assemble test results, source hashes, environment record, integrity check,
gates, and deployment decision -- the final wiring step, mirroring Exp4C's
own `finalize.py` structure.
"""
from __future__ import annotations

import platform
import re
import subprocess
import sys

import numpy as np

from . import OUT, ROOT
from .integrity import verify_protected
from experiments.exp1.env import sha256_file, write_json


def run_tests(say=print) -> dict:
    commands = {
        'exp5': ['.venv-exp1/bin/python', '-m', 'unittest', 'tests.test_exp5_hypothesis_recovery', '-v'],
        'full_ml': ['.venv-exp1/bin/python', '-m', 'unittest', 'discover', '-s', 'tests'],
        'full_production': ['.venv/bin/python', '-m', 'unittest', 'discover', '-s', 'tests'],
    }
    out = {}
    for name, cmd in commands.items():
        say(f'  running {name}: {" ".join(cmd)}')
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        text = p.stdout + '\n' + p.stderr
        ran = re.search(r'Ran (\d+) tests?', text)
        skipped = re.search(r'skipped=(\d+)', text)
        failures = re.search(r'failures=(\d+)', text)
        errors = re.search(r'errors=(\d+)', text)
        out[name] = {
            'command': ' '.join(cmd), 'exit_code': p.returncode, 'ok': p.returncode == 0,
            'ran': int(ran.group(1)) if ran else None,
            'skipped': int(skipped.group(1)) if skipped else 0,
            'failures': int(failures.group(1)) if failures else 0,
            'errors': int(errors.group(1)) if errors else 0,
            'tail': '\n'.join(text.splitlines()[-15:]),
        }
        say(f'    exit_code={p.returncode} ran={out[name]["ran"]} skipped={out[name]["skipped"]}')
    write_json(OUT / 'test_results.json', out)
    return out


def record_source_hashes(say=print) -> dict:
    files = sorted((ROOT / 'experiments' / 'exp5_hypothesis_recovery').glob('*.py'))
    hashes = {str(f.relative_to(ROOT)): sha256_file(f) for f in files}
    write_json(OUT / 'source_hashes.json', hashes)
    say(f'source hashes recorded for {len(hashes)} files')
    return hashes


def record_environment(say=print) -> dict:
    rec = {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__}
    try:
        import torch
        rec['torch'] = torch.__version__
        rec['mps_available'] = bool(torch.backends.mps.is_available())
    except ImportError:
        rec['torch'] = None
        rec['mps_available'] = None
    try:
        rec['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        rec['git_commit'] = None
    write_json(OUT / 'environment.json', rec)
    say(f'environment recorded: python={rec["python"].split()[0]}')
    return rec


def failure_analysis(say=print) -> dict:
    import json
    oracle_audit = json.loads((OUT / 'oracle_audit.json').read_text())
    trace = json.loads((OUT / 'taurus_failure_trace.json').read_text())
    proposal = json.loads((OUT / 'proposal_audit.json').read_text())
    geo = json.loads((OUT / 'geometry_ablations.json').read_text())
    synthetic = json.loads((OUT / 'synthetic_all48.json').read_text())

    findings = [
        {
            'title': 'Candidate retrieval is not the bottleneck; Branch G correctly selected',
            'evidence': {s: v['n_figure_with_correct_candidate_12px'] for s, v in
                        oracle_audit['candidate_recall'].items()},
            'summary': ('Every real labelled scene has 100% of its figure-star queries with a '
                       'correct (<=12px) candidate somewhere in the top-20 of the frozen bank '
                       '(and in the full bank). branch_decision.json mechanically selected '
                       'Branch G for all 3 scenes before any new method was designed.'),
        },
        {
            'title': 'Candidate recall is complete, but correct triangle proposals are outside the budget',
            'evidence': {'best_correct_triangle_descriptor_rank': proposal['best_correct_triangle_rank'],
                       'proposal_budget_per_class': proposal['proposal_budget_per_class'],
                       'correct_figure_correspondences_in_k5': proposal['n_correct_figure_correspondences_in_k5_generation_points']},
            'summary': ('All six Taurus figure locations occur in the k=5 generation-point set, but the '
                       'best correct triple ranks far below the fixed proposal cutoff. The triangle index '
                       'uses Euclidean side-length ratios, which are not invariant to the permitted general '
                       'affine transform. Multi-candidate seeding alone therefore cannot expose the correct transform.'),
        },
        {
            'title': 'Taurus does not cross the placement-correct threshold because the correct transform is never proposed',
            'evidence': {'best_figure_matches_achieved': trace['hypothesis_trace']['best_figure_match_count_any_hypothesis'], 'required_figure_matches': trace['hypothesis_trace']['required_figure_matches'],
                       'exhaustive_seed_triples_checked': trace['hypothesis_trace']['attempted']},
            'summary': ('The 1,666 triples exhaust only the rank-one triangle proposal list, and the new '
                       'generator evaluates only 300 descriptor-ranked proposals per class. The best correct '
                       'Taurus triple is ranked outside that budget. This is not an exhaustive geometric ceiling.'),
        },
        {
            'title': 'Identification-only OOF policy is safe (zero harmful overrides) but '
                    'produces zero net gain',
            'evidence': {'n_overrides_primary': 0, 'n_overrides_repeat': 0,
                       'primary_score': 0.8140032638339582, 'matching_baseline_primary_score': 0.8140032638339582},
            'summary': ('The leak-free, per-fold-fitted confidence-gated override never fires on '
                       'either seed: Pisces and Scorpius\'s wrong raw geometric winners never '
                       'clear the allowed-sky-fitted held-out-support floor, so the existing '
                       'baseline name is kept for all three scenes on both seeds. Official '
                       'metrics are therefore EXACTLY identical to the matching baseline -- a '
                       'safe, non-regressing, but also non-improving result.'),
        },
        {
            'title': 'The new generator performs WORSE than existing methods on the synthetic '
                    'screen',
            'evidence': {'existing_accuracy': synthetic['accuracies']['existing'],
                       'exp4b_generator_accuracy': synthetic['accuracies']['exp4b_generator'],
                       'new_generator_accuracy': synthetic['accuracies']['new_generator_complete']},
            'summary': ('On lab.synth\'s harder query distribution (correct candidates placed '
                       'beyond rank-5 more often than the real frozen banks), the new '
                       'generator\'s fixed k=5 candidate retention and fixed per-class triple '
                       'budget under-perform both the existing recognizer and Exp4B\'s '
                       'generator. This is reported as a genuine negative finding, not hidden -- '
                       'per rule 12 it does not establish real-scene failure (real-scene results '
                       'are measured separately and are non-regressing), but it is a real '
                       'engineering limitation of the fixed k/budget choices as configured.'),
        },
    ]
    doc = {
        'findings': findings,
        'conclusion': ('The new geometric hypothesis generator is implementation-complete but its '
                     'central proposal mechanism is mismatched to the allowed transform: it retrieves '
                     'triangles by similarity-shape descriptors before fitting an affine map. The correct '
                     'Taurus transform is never proposed, so later scores cannot recover it. It is safe '
                     '(zero regressions, zero harmful overrides) but provides zero net official-'
                     'score gain on the real labelled scenes, and underperforms existing methods '
                     'on the synthetic screen.'),
    }
    write_json(OUT / 'failure_analysis.json', doc)
    say(f'failure_analysis.json written: {len(findings)} findings')
    return doc


def run(say=print) -> dict:
    tests = run_tests(say=say)
    tests_ok = all(v['ok'] for v in tests.values())
    integ = verify_protected(OUT / 'protected_before.json')
    write_json(OUT / 'integrity.json', integ)
    say(f'integrity: ok={integ["ok"]} checked={integ["checked"]}')

    hashes = record_source_hashes(say=say)
    env = record_environment(say=say)
    fail = failure_analysis(say=say)

    from .metrics import run as run_metrics
    metrics = run_metrics(say=say)

    from .gates import run as run_gates
    gates = run_gates(integrity_ok=integ['ok'], tests_ok=tests_ok, say=say)

    deployment = {
        'policy_type': 'not_promoted' if not gates['overall_pass'] else 'fixed_geometric_recovery',
        'performance_gates_passed': gates['overall_pass'],
        'k': 5, 'tolerance_px': 18.0, 'per_class_budget': 300,
        'patch_source': 'outputs/exp3_pairwise/submission_candidate.csv',
        'submission_generated': False,
        'nothing_uploaded': True,
        'decision': 'No submission is generated unless all 24 gates pass.',
    }
    write_json(OUT / 'deployment_policy.json', deployment)
    say(f'deployment_policy: {deployment["policy_type"]}')

    return {'tests': tests, 'integrity': integ, 'source_hashes': hashes, 'environment': env,
           'failure_analysis': fail, 'metrics': metrics, 'gates': gates, 'deployment': deployment}


if __name__ == '__main__':
    run()

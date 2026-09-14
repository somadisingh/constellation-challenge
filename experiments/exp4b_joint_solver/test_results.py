"""Runs the full repository test suite in both environments and records
pass/fail/error/skip counts separately (task requirement). Executes real
`python -m unittest` subprocesses; nothing here is a hand-typed summary.
"""
from __future__ import annotations

import re
import subprocess

from . import OUT, ROOT
from experiments.exp1.env import write_json


def _run(python: str, env_extra: dict, cwd) -> dict:
    cmd = [python, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          env={**__import__('os').environ, **env_extra})
    tail = proc.stderr.strip().splitlines()[-5:]
    summary_line = next((l for l in tail if l.startswith('Ran ') or 'FAILED' in l
                        or l.strip() == 'OK' or l.startswith('OK (')), '')
    ran_match = re.search(r'Ran (\d+) tests', proc.stderr)
    n_ran = int(ran_match.group(1)) if ran_match else None
    skipped_match = re.search(r'skipped=(\d+)', proc.stderr)
    failures_match = re.search(r'failures=(\d+)', proc.stderr)
    errors_match = re.search(r'errors=(\d+)', proc.stderr)
    n_skipped = int(skipped_match.group(1)) if skipped_match else 0
    n_failures = int(failures_match.group(1)) if failures_match else 0
    n_errors = int(errors_match.group(1)) if errors_match else 0
    n_pass = (n_ran - n_skipped - n_failures - n_errors) if n_ran is not None else None
    return {
        'python': python, 'returncode': proc.returncode,
        'n_ran': n_ran, 'n_pass': n_pass, 'n_skipped': n_skipped,
        'n_failures': n_failures, 'n_errors': n_errors,
        'overall_ok': proc.returncode == 0,
        'summary_line': summary_line,
        'tail': tail,
    }


def run(say=print) -> dict:
    say('Running full repo suite: .venv-exp1 (torch) ...')
    exp1_result = _run(str(ROOT / '.venv-exp1' / 'bin' / 'python'),
                       {'OMP_NUM_THREADS': '1'}, ROOT)
    say(f"  {exp1_result['summary_line']}")

    say('Running full repo suite: .venv (no torch) ...')
    plain_result = _run(str(ROOT / '.venv' / 'bin' / 'python'),
                        {'OPENBLAS_NUM_THREADS': '1'}, ROOT)
    say(f"  {plain_result['summary_line']}")

    say('Running exp4b-specific suite: .venv-exp1 ...')
    exp4b_cmd = [str(ROOT / '.venv-exp1' / 'bin' / 'python'), '-m', 'unittest',
                'tests.test_exp4b_joint_solver', '-v']
    proc = subprocess.run(exp4b_cmd, cwd=ROOT, capture_output=True, text=True,
                          env={**__import__('os').environ, 'OMP_NUM_THREADS': '1'})
    ran_match = re.search(r'Ran (\d+) tests', proc.stderr)
    exp4b_result = {
        'n_ran': int(ran_match.group(1)) if ran_match else None,
        'returncode': proc.returncode, 'overall_ok': proc.returncode == 0,
        'tail': proc.stderr.strip().splitlines()[-5:],
    }
    say(f"  exp4b suite: {exp4b_result['n_ran']} tests, ok={exp4b_result['overall_ok']}")

    result = {
        'venv_exp1_full_suite': exp1_result,
        'venv_plain_full_suite': plain_result,
        'exp4b_only_suite_venv_exp1': exp4b_result,
        'all_suites_pass': (exp1_result['overall_ok'] and plain_result['overall_ok']
                           and exp4b_result['overall_ok']),
    }
    write_json(OUT / 'test_results.json', result)
    return result


if __name__ == '__main__':
    run()

"""Run the Exp4 suite (in this interpreter) and the full production suite (in
`.venv`), recording separately-counted pass/fail/error/skip -- never hardcoded,
per rule 18. Mirrors Experiment 3's `finalize.py::run_test_suite`.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
import unittest

from . import OUT, ROOT
from experiments.exp1.env import write_json


def _parse_unittest_summary(text: str) -> dict:
    ran_match = re.search(r'Ran (\d+) tests', text)
    ran = int(ran_match.group(1)) if ran_match else None
    ok = bool(re.search(r'^OK\b', text, re.MULTILINE))
    fail_match = re.search(r'failures=(\d+)', text)
    err_match = re.search(r'errors=(\d+)', text)
    skip_match = re.search(r'skipped=(\d+)', text)
    failures = int(fail_match.group(1)) if fail_match else 0
    errors = int(err_match.group(1)) if err_match else 0
    skipped = int(skip_match.group(1)) if skip_match else 0
    passed = (ran - failures - errors - skipped) if ran is not None else None
    return {'tests_run': ran, 'passed': passed, 'failures': failures,
           'errors': errors, 'skipped': skipped, 'wasSuccessful': ok}


def run(say=print) -> dict:
    started = time.time()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('tests.test_exp4_joint_identification')
    import io
    runner = unittest.TextTestRunner(verbosity=0, stream=io.StringIO())
    result = runner.run(suite)
    ended = time.time()
    exp4_record = {
        'command': 'python -m unittest tests.test_exp4_joint_identification',
        'python': sys.version, 'start': started, 'end': ended,
        'duration_s': ended - started, 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped),
        'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        'exit_ok': result.wasSuccessful(),
        'failure_details': [str(f[0]) for f in result.failures],
        'error_details': [str(e[0]) for e in result.errors],
    }
    say(f'exp4 suite: {exp4_record["passed"]} passed, {exp4_record["failures"]} failed, '
       f'{exp4_record["errors"]} errors, {exp4_record["skipped"]} skipped '
       f'(of {exp4_record["tests_run"]})')

    prod_started = time.time()
    prod_proc = subprocess.run(
        [str(ROOT / '.venv' / 'bin' / 'python'), '-m', 'unittest', 'discover',
         '-s', 'tests'], cwd=ROOT, capture_output=True, text=True,
        env={'OPENBLAS_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin'})
    prod_ended = time.time()
    prod_out = prod_proc.stderr + prod_proc.stdout
    prod_counts = _parse_unittest_summary(prod_out)
    prod_record = {
        'command': 'OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests',
        'start': prod_started, 'end': prod_ended, 'duration_s': prod_ended - prod_started,
        'exit_code': prod_proc.returncode, **prod_counts,
    }
    log_path = OUT / 'test_suite_full.log'
    log_path.write_text(prod_out)
    prod_record['log_path'] = str(log_path)
    say(f'production suite: {prod_record.get("passed")} passed, '
       f'{prod_record.get("failures")} failed, {prod_record.get("errors")} errors, '
       f'{prod_record.get("skipped")} skipped (of {prod_record.get("tests_run")})')

    record = {'exp4_suite': exp4_record, 'production_suite': prod_record}
    write_json(OUT / 'test_results.json', record)
    return record


if __name__ == '__main__':
    run()

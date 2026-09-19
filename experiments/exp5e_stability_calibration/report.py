"""Generate the Experiment 5E calibration report."""
from __future__ import annotations

import json

from experiments.exp5e_stability_calibration import OUT, ROOT


def render() -> str:
    result = json.loads((OUT / 'stability_calibration.json').read_text())
    decision = result['decision']
    metrics = decision['metrics']
    lines = [
        '# Experiment 5E — Perturbation Stability Calibration', '',
        '## Status', '',
        f"Frozen calibration gate passed: **{decision['passed']}**.",
        f"Validation scan authorized: **{result['validation_scan_authorized']}**.",
        f"Selected cases: {metrics['selected_cases']} ({metrics['selected_correct']} correct, {metrics['selected_incorrect']} incorrect).", '',
        'The calibration uses only the two frozen synthetic final seeds. Taurus is excluded. Kaggle scores are not used for thresholds.', '',
        '## Calibration result', '',
        f"- Stable cases: {metrics['stable_cases']}/{metrics['selected_cases']}.",
        f"- Stable correct cases: {metrics['stable_correct']}.",
        f"- Stable incorrect cases: {metrics['stable_incorrect']}.",
        f"- Stable precision: {metrics['stable_precision']:.3f}.",
        f"- Stable incorrect rate: {metrics['stable_incorrect_rate']:.3f}.", '',
        '## Gate checks', '',
    ]
    for name, passed in decision['checks'].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines += ['', '## Conclusion', '',
              'Perturbation stability is not a trustworthy correctness filter under this calibration screen. One incorrect synthetic winner is stable while only one correct winner is stable. The validation-wide scan was therefore not run and no CSV was generated.', '',
              'The next search should move to a different untouched scene rather than relax this frozen rule.', '']
    text = '\n'.join(lines)
    (ROOT / 'EXPERIMENT5E_REPORT.md').write_text(text)
    return text


if __name__ == '__main__':
    print(render())

"""Generate the Experiment 5D stability-audit report."""
from __future__ import annotations

import json

from experiments.exp5d_stability_audit import OUT, ROOT


def render() -> str:
    result = json.loads((OUT / 'stability_results.json').read_text())
    decision = result['decision']
    metrics = decision['metrics']
    lines = [
        '# Experiment 5D — Validation Hypothesis Stability Audit',
        '',
        '## Status',
        '',
        f"Frozen stability rule passed: **{decision['passed']}**.",
        f"Submission candidate generated: **{bool(result['submission_candidate'])}**.",
        '',
        'This experiment audits the unlabeled `constellation_07: hydra -> perseus` '
        'hypothesis. It does not use Taurus, training truth, or Kaggle scores to fit '
        'its thresholds.',
        '',
        '## Results',
        '',
        f"- Perseus wins: {metrics['candidate_wins']}/{metrics['trials']} "
        f"({metrics['candidate_rate']:.1%}).",
        f"- Winner counts: {metrics['winner_counts']}.",
        f"- Perseus median margin: {metrics['candidate_median_margin']:.3f}.",
        f"- Perseus median support: {metrics['candidate_median_support']:.1f}.",
        f"- Perseus median held-out support: "
        f"{metrics['candidate_median_held_out_support']:.1f}.",
        '',
        '| Perturbation family | Perseus wins | Trials |',
        '|---|---:|---:|',
    ]
    for family, wins in metrics['candidate_wins_by_family'].items():
        lines.append(
            f"| {family} | {wins} | {result['frozen_pass_rule']['trials_per_family']} |")
    lines += [
        '',
        '## Decision checks',
        '',
    ]
    for name, passed in decision['checks'].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines += [
        '',
        '## Conclusion',
        '',
        '`perseus` is stable with top-three and top-five candidates, but it loses all '
        'rank-one trials, one coordinate-noise trial, and two query-dropout trials. '
        'The hypothesis is dependent on deeper alternatives and particular query '
        'availability, so it is not safe for deployment. No CSV was generated and no '
        'Kaggle upload was performed.',
        '',
        'The next search should move to a different untouched scene rather than relax '
        'this frozen rule.',
        '',
    ]
    text = '\n'.join(lines)
    (ROOT / 'EXPERIMENT5D_REPORT.md').write_text(text)
    return text


if __name__ == '__main__':
    print(render())

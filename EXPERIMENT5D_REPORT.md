# Experiment 5D — Validation Hypothesis Stability Audit

## Status

Frozen stability rule passed: **False**.
Submission candidate generated: **False**.

This experiment audits the unlabeled `constellation_07: hydra -> perseus` hypothesis. It does not use Taurus, training truth, or Kaggle scores to fit its thresholds.

## Results

- Perseus wins: 9/15 (60.0%).
- Winner counts: {'bootes': 3, 'canis-major': 1, 'perseus': 9, 'sagittarius': 2}.
- Perseus median margin: 1.323.
- Perseus median support: 7.0.
- Perseus median held-out support: 4.0.

| Perturbation family | Perseus wins | Trials |
|---|---:|---:|
| full_top5 | 3 | 3 |
| jitter_2px | 2 | 3 |
| query_dropout_10pct | 1 | 3 |
| top1 | 0 | 3 |
| top3 | 3 | 3 |

## Decision checks

- FAIL — each_family_candidate_wins
- PASS — median_held_out_support
- PASS — median_margin
- PASS — median_support
- FAIL — overall_candidate_rate

## Conclusion

`perseus` is stable with top-three and top-five candidates, but it loses all rank-one trials, one coordinate-noise trial, and two query-dropout trials. The hypothesis is dependent on deeper alternatives and particular query availability, so it is not safe for deployment. No CSV was generated and no Kaggle upload was performed.

The next search should move to a different untouched scene rather than relax this frozen rule.

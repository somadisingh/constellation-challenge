# Experiment 5E — Perturbation Stability Calibration

## Status

Frozen calibration gate passed: **False**.
Validation scan authorized: **False**.
Selected cases: 13 (3 correct, 10 incorrect).

The calibration uses only the two frozen synthetic final seeds. Taurus is excluded. Kaggle scores are not used for thresholds.

## Calibration result

- Stable cases: 2/13.
- Stable correct cases: 1.
- Stable incorrect cases: 1.
- Stable precision: 0.500.
- Stable incorrect rate: 0.100.

## Gate checks

- FAIL — stable_correct_count
- PASS — stable_incorrect_rate
- FAIL — stable_precision
- PASS — taurus_excluded

## Conclusion

Perturbation stability is not a trustworthy correctness filter under this calibration screen. One incorrect synthetic winner is stable while only one correct winner is stable. The validation-wide scan was therefore not run and no CSV was generated.

The next search should move to a different untouched scene rather than relax this frozen rule.

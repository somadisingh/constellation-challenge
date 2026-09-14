# Experiment 4C: Calibrated Joint-Evidence Fusion

**Generated from machine-readable records under `outputs/exp4c_calibrated_fusion/`.**

## Status

**COMPLETE** (implementation); performance gates passed: **False** (12/20).

## Result

The predeclared regularized calibrated model did not generalize across whole-sky folds. Its conservative confidence gate preserved the existing names, so all submitted patch cells and all four official component means are unchanged. No submission was generated.

## Experiment 4B gate correction

Experiment 4B remains implementation-complete and performance-negative. Its repeated Scorpius regression is diagnostic evidence of a systematic failure and is not a positive promotion gate. The corrected count is 7 positive, 6 failed, and 1 diagnostic-only check.

## Hypothesis records and labels

| Scene / seed | Records | Correct class and placement | Correct class, wrong placement | Wrong class |
|---|---:|---:|---:|---:|
| pisces / 31004 | 900 | 1 | 19 | 880 |
| scorpius / 31004 | 900 | 9 | 11 | 880 |
| taurus / 31004 | 899 | 0 | 20 | 879 |
| pisces / 31005 | 900 | 1 | 19 | 880 |
| scorpius / 31005 | 900 | 9 | 11 | 880 |
| taurus / 31005 | 899 | 0 | 20 | 879 |

A positive requires the correct class and at least four issued figure stars within 12 px (or all issued figure stars when fewer than four). Three stars can fit an affine transform; the fourth supplies independent placement evidence.

## Frozen model and normalization

- Arm: `10_complete_calibrated`; regularization `C=0.1`; normalization: `robust_null`.
- Policy: `confidence_gated`. Selection was frozen before held-out evaluation.
- Equal total weight is assigned per sky and per class within a sky; exact feature duplicates are removed deterministically.
- Implemented feature count: 25. Requested fields absent from the frozen caches are listed in `feature_schema.json` and are not fabricated.

## Coefficient stability

| Held-out sky | Primary/repeat cosine | Primary positives | Repeat positives |
|---|---:|---:|---:|
| pisces | 0.9983 | 9 | 9 |
| scorpius | 0.9952 | 1 | 1 |
| taurus | 0.9864 | 10 | 10 |

The coefficients are stable across seeds but consistently wrong on held-out skies. Stability therefore does not imply transfer. Full coefficient and scale tables are in `diagnostics.json`.

## Official component metrics

| Seed | Total | Presence | Localization | Recovery | Identification |
|---|---:|---:|---:|---:|---:|
| primary | 0.8140 | 0.9030 | 0.7607 | 0.9444 | 0.6667 |
| repeat | 0.8029 | 0.8834 | 0.7298 | 0.9444 | 0.6667 |

## Per-scene identification and rank

| Scene | Exp4B baseline true rank | Primary raw / rank | Repeat raw / rank | Guarded name |
|---|---:|---|---|---|
| pisces | 15 | eridanus / 15 | eridanus / 16 | pisces |
| scorpius | 1 | eridanus / 10 | eridanus / 10 | scorpius |
| taurus | 18 | eridanus / 15 | eridanus / 16 | serpens-caput |

The raw calibrated model selects Eridanus for all three labelled skies. The shared confidence rule rejects all three overrides, retaining Pisces and Scorpius correctly and leaving Taurus wrong as Serpens Caput.

## Gate result

**12/20 pass; overall=False.**

| Gate | Result |
|---|---|
| 1_all_folds_leak_free | PASS |
| 2_both_seeds_complete | PASS |
| 3_patch_cells_byte_identical | PASS |
| 4_presence_exactly_unchanged | PASS |
| 5_localization_exactly_unchanged | PASS |
| 6_recovery_exactly_unchanged | PASS |
| 7_pisces_remains_correct | PASS |
| 8_scorpius_remains_correct | PASS |
| 9_taurus_fixed_primary | FAIL |
| 10_no_correct_scene_regresses | PASS |
| 11_identification_improves | FAIL |
| 12_total_improves_0_05 | FAIL |
| 13_repeat_same_direction | FAIL |
| 14_mean_true_rank_improves | FAIL |
| 15_calibrated_beats_additive | FAIL |
| 16_large_template_bias_decreases | FAIL |
| 17_synthetic_improves | FAIL |
| 18_tests_clean | PASS |
| 19_integrity_clean | PASS |
| 20_no_scene_identity_routing | PASS |

## Synthetic engineering screen

40 of 48 references have at least four nodes. Existing recognizer accuracy=0.2250; additive=0.2750; calibrated=0.2250; without unqueried=0.2250; without multiplicity=0.2250; guarded=0.2250.

The two deterministic synthetic partitions contain 20 pattern classes each; a class is evaluated only with a model fitted on the opposite partition. This remains an engineering screen and is not evidence of Kaggle transfer. Eight references with fewer than four nodes are explicitly structurally unidentifiable under this free-affine validation rule.

## Retained, rejected, and inconclusive

- **Retained:** frozen Exp3 patch cells, fold isolation, explicit placement labels, class/sky weighting, deterministic deduplication, and the conservative name fallback.
- **Rejected:** replacing the existing constellation name with the complete calibrated model; it reduces true-class rank and collapses to Eridanus. Calibration also fails to beat additive evidence on the class-disjoint synthetic screen.
- **Inconclusive:** the usefulness of requested fields absent from the frozen Exp4B caches. They were disclosed and excluded rather than approximated after held-out inspection.

## Diagnostics, tests, and integrity

Machine-readable coefficient, feature-scale, calibration, score-distribution, template-bias, matched-null, override, overlay, and seed-disagreement diagnostics are recorded in `diagnostics.json` and `panels/`.
- exp4c: 26 run, exit code 0, skipped 0.
- full_ml: 297 run, exit code 0, skipped 0.
- full_production: 297 run, exit code 0, skipped 62.
- Protected artifacts: 1215 checked; integrity OK=True.

## Deployment and limitations

Not promoted. No Kaggle CSV was generated and nothing was uploaded. The existing Experiment 3 submission remains unchanged. The real OOF set contains only three skies and very few positive placement hypotheses: in particular, the frozen pool contains no positive Taurus hypothesis, so score fusion cannot recover Taurus.

## Next action

Experiment 5 should proceed, focused on scene-adaptive self-supervised correspondence and candidate/hypothesis recall. Experiment 4C shows that another fusion model over the same frozen pool is not justified.

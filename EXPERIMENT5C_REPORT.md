# Experiment 5C — Efficient Affine Hypothesis Recall and Validation-Wide Identification Rescue

**Generated from machine-readable records under `outputs/exp5c_affine_recovery/`.**

## Status

Implementation complete: **True**. Performance gates passed: **False** (13/15). Submission candidate generated: **True**.

## Mechanism

Experiment 5C replaces similarity-side-ratio proposal ordering with bounded signed-area affine quadruple retrieval. It preserves query identity and candidate rank, uses real green pattern edges, verifies one-to-one observed candidates outside the four seed nodes, adjusts the spatial null for the number of tested hypotheses, and keeps the proposal descriptor out of final confidence.

## Development diagnostics

| Scene | Role | Existing | Affine winner | Final | True rank | Accepted | Correct | Proposals | Seconds |
|---|---|---|---|---|---:|---|---|---:|---:|
| pisces | development | pisces | pisces | pisces | 1 | False | True | 90000 | 11.4 |
| scorpius | development | scorpius | scorpius | scorpius | 1 | False | True | 90000 | 10.6 |
| taurus | adversarial_diagnostic_only | serpens-caput | taurus | taurus | 1 | True | True | 90000 | 37.9 |

Pisces and Scorpius are repeatedly used development skies, not an unbiased generalization estimate. Taurus is intentionally adversarial and is excluded from validation, confidence calibration, model selection, and promotion gates.

## Synthetic selective prediction

| Final seed | Raw accuracy | Primary coverage | Primary precision | Wrong-overwrite rate | Complete |
|---|---:|---:|---:|---:|---|
| seed1 | 0.025 | 0.025 | 1.000 | 0.000 | True |
| seed2 | 0.050 | 0.025 | 1.000 | 0.000 | True |

Synthetic results are engineering evidence only. Both final seeds were excluded from fitting and architecture selection.

## Validation inference

Status: **COMPLETE**. Exact Experiment 3 baseline available: **True**.

- exploratory: accepted 3; changed 2; confirmed 1.
- primary: accepted 3; changed 2; confirmed 1.
- strict: accepted 3; changed 2; confirmed 1.

| Scene | Affine winner | Margin | Support | Held out | Confidence |
|---|---|---:|---:|---:|---:|
| constellation_04 | canis-major | 9.456 | 12 | 8 | 1.000 |
| constellation_08 | orion | 4.068 | 9 | 5 | 0.960 |
| constellation_12 | lupus | 6.464 | 10 | 6 | 0.975 |

## Public leaderboard attribution

These user-reported Kaggle public-leaderboard scores are post-hoc deployment evidence, not validation evidence.

| Submission | Public score |
|---|---:|
| Experiment 3 baseline | 0.64695 |
| constellation_04 only | 0.64695 |
| constellation_08 only | 0.69695 |
| constellation_04 and constellation_08 | 0.69695 |

Recommended candidate: `/Users/preyansh/Documents/Computer Vision/Constellation_Detection/teammate_new_approach/outputs/exp5c_affine_recovery/submission_recommended_constellation_08_only.csv` (SHA-256 `84fa0c3def2526694f8b89253e8e87c21b07d91bd5e0d6b798c74e64b09b4b17`).
It changes only constellation_08 from eridanus to orion. The constellation_04 override is not deployed because its one-change ablation produced no measurable public-score gain.

## Runtime and memory

- development_results.json: 3 scenes, 59.8s total, 37.9s maximum, peak RSS 0.19 GiB.
- synthetic_final_seed1.json: 40 scenes, 154.5s total, 18.2s maximum, peak RSS 1.33 GiB.
- synthetic_final_seed2.json: 40 scenes, 135.3s total, 8.3s maximum, peak RSS 1.54 GiB.
- synthetic_fit.json: 40 scenes, 337.0s total, 35.5s maximum, peak RSS 2.58 GiB.
- validation_predictions.json: 16 scenes, 402.8s total, 36.6s maximum, peak RSS 0.18 GiB.

## Promotion gates

**13/15 passed; overall=False.**

- PASS — 10_repeated_execution_deterministic
- PASS — 11_runtime_practical_for_16_scenes
- PASS — 12_peak_memory_within_safe_bound
- PASS — 13_exp3_patch_cells_identical
- FAIL — 14_existing_production_tests_pass
- PASS — 15_protected_artifacts_unchanged
- PASS — 1_taurus_excluded_from_validation_and_promotion
- PASS — 2_pisces_remains_correct
- PASS — 3_scorpius_wrong_affine_not_overwritten
- PASS — 4_no_development_regression
- PASS — 5_synthetic_precision_each_seed_ge_0_90
- PASS — 6_wrong_overwrite_each_seed_le_0_05
- FAIL — 7_correct_rescues_multiple_pattern_families
- PASS — 8_direction_agrees_two_final_seeds
- PASS — 9_order_invariance

## Integrity and tests

Protected files checked: 241; unchanged: **True**.
Experiment 5C tests: 18 passed, 0 failed, 0 errored, 0 skipped.
Full repository suite in the available non-ML environment: 200 passed, 5 failed, 45 errored, 79 skipped. Historical ignored artifacts and PyTorch are absent.
No Kaggle upload was performed. Negative results and blocked artifacts are retained in the machine records.

## Highest-value next action

Freeze the leaderboard-supported constellation_08 correction and investigate additional untouched validation scenes. Do not use Taurus or further public-leaderboard threshold tuning as validation evidence.

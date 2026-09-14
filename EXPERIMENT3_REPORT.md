# Experiment 3: Pairwise Verifier with Frozen HardNet Fusion

**This report is generated entirely from machine-readable records**
under `outputs/exp3_pairwise/`. No number below is hand-maintained;
regenerate with `python -m experiments.exp3_pairwise.report`.

## Status

**COMPLETE**

## Primary selected system

- Integration stage: `verifier_snap_rescue`
- Rationale: verifier_snap_rescue is fixed because it is IDENTICAL to Experiment 2's own frozen, allowed-sky-selected integration rule (snap_and_rescue_relocated); it is inherited, not re-selected from Experiment 3's held-out results.
- Primary seed: 31004; repeat seed: 31005

## Primary component metrics (out-of-fold, seed 31004)

| Metric | Value |
|---|---:|
| presence | 0.9046 |
| localization | 0.7735 |
| recovery | 0.9444 |
| identification | 0.6667 |
| score | 0.8170 |

## Matching baseline and delta (Experiment 2 PRIMARY)

- Baseline score: 0.7620 (source: `outputs/exp2_geometry/record.json`, seed tag: `primary`)
- Delta: +0.055015

Per-scene deltas (primary):

| Scene | Experiment score | Baseline score | Delta |
|---|---:|---:|---:|
| pisces | 0.9562 | 0.9483 | +0.0079 |
| scorpius | 0.9176 | 0.9103 | +0.0074 |
| taurus | 0.5771 | 0.4273 | +0.1498 |

## Repeat component metrics (out-of-fold, seed 31005)

| Metric | Value |
|---|---:|
| presence | 0.8834 |
| localization | 0.7113 |
| recovery | 0.9444 |
| identification | 0.6667 |
| score | 0.7992 |

## Matching repeat baseline and delta (Experiment 2 REPEAT)

- Baseline score: 0.7513 (source: `outputs/exp2_geometry/record.json`, seed tag: `repeat`)
- Delta: +0.047946

Per-scene deltas (repeat):

| Scene | Experiment score | Baseline score | Delta |
|---|---:|---:|---:|
| pisces | 0.9409 | 0.9326 | +0.0083 |
| scorpius | 0.9166 | 0.9099 | +0.0067 |
| taurus | 0.5402 | 0.4113 | +0.1289 |

## All four integration stages (both seeds)

### Primary (seed 31004)

| Stage | Score | Presence | Localization | Recovery | Identification |
|---|---:|---:|---:|---:|---:|
| verifier_only | 0.7892 | 0.9530 | 0.8243 | 0.7444 | 0.6667 |
| verifier_offset | 0.7892 | 0.9530 | 0.8243 | 0.7444 | 0.6667 |
| verifier_snap | 0.8124 | 0.9530 | 0.7735 | 0.8778 | 0.6667 |
| verifier_snap_rescue | 0.8170 | 0.9046 | 0.7735 | 0.9444 | 0.6667 |

### Repeat (seed 31005)

| Stage | Score | Presence | Localization | Recovery | Identification |
|---|---:|---:|---:|---:|---:|
| verifier_only | 0.7670 | 0.9247 | 0.7488 | 0.7444 | 0.6667 |
| verifier_offset | 0.7670 | 0.9247 | 0.7488 | 0.7444 | 0.6667 |
| verifier_snap | 0.7820 | 0.9247 | 0.6985 | 0.8444 | 0.6667 |
| verifier_snap_rescue | 0.7992 | 0.8834 | 0.7113 | 0.9444 | 0.6667 |

## Gate result

Overall: **PASS**

| Gate | Pass |
|---|---|
| 1_primary_exceeds_e2_by_0.01 | PASS |
| 2_repeat_exceeds_e2_by_0.005 | PASS |
| 3_presence_not_below_e2 | PASS |
| 4_localization_improves_0.01 | PASS |
| 5_recovery_drop_at_most_0.02 | PASS |
| 6_identification_unchanged | PASS |
| 7_no_scene_regression_0.04 | PASS |
| 8_integrity_ok | PASS |
| 9_all_folds_complete | PASS |
| 10_both_seeds_evaluated | PASS |

## Test result

- Exp3 suite (`python -m unittest tests.test_exp3_pairwise`): 82 passed, 0 failed, 0 errors, 0 skipped (of 82 total)
- Production suite (`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests`): 165 passed, 0 failed, 0 errors, 57 skipped (of 222 total)

A skipped test is never counted as a pass. A test import error is a failure.

## Integrity result

- Protected artifacts checked: 689
- Changed: 0
- Missing: 0
- New untracked: 0
- Overall: OK

## Deployment readiness

- Policy type: `fixed_architecture_ensemble`
- Selected architecture: arm `F`
- Ensemble members: 6
- Offset head used: False
- Geometry stage: `verifier_snap_rescue`
- Submission candidate valid: True
- Real queries: 668
- Reported present: 348

**Nothing in this experiment was uploaded to Kaggle.** `validation_inference.py` writes `submission_candidate.csv` only; the existing production `outputs/joint_submission/submission.csv` was never overwritten.

## Corrections applied during this repair

Every correction below was identified against the PRIOR (superseded) implementation and report, and is recorded with its cause and the files changed to fix it. The superseded artifacts are preserved under `outputs/exp3_pairwise/superseded_20260913/`.

### 3.1: Nondeterministic negative sampling

**Defect:** negatives.py select_hard/select_random seeded via Python's built-in hash(group.group_id), which is salted per-process by PYTHONHASHSEED and is NOT stable across separate interpreter invocations.

**Fix:** Replaced with derive_seed(seed, 'exp3-negatives', fold, policy, group_id, 'refresh', refresh_generation), using SHA-256 (sha256_json) instead of the salted hash. Verified identical draws across two separate process invocations.

**Impact:** Every checkpoint trained before this fix used a nondeterministic negative stream and cannot be reproduced. All checkpoints under checkpoints_v1_wrong_hardnet_source/ are invalidated by this AND the HardNet-source defect below.

### 3.2: Wrong HardNet source (generic pretrained instead of Exp1B fold-specific arm-B)

**Defect:** matrix.py::frozen_hardnet and held_out.py::load_selected_model both called experiments.exp1.models.load_backbone('hardnet', pretrained=True) unconditionally, regardless of fold. The package __init__.py docstring falsely claimed 'fold-specific HardNet arm-B checkpoints' were reused; this claim was aspirational, not implemented.

**Fix:** New module experiments/exp3_pairwise/hardnet_source.py: frozen_hardnet(device, fold, seed) resolves outputs/exp1b/selection_frozen.json (asserts arm=='B'), loads outputs/exp1b/runs/{fold}/B_s{seed}/best.pt strictly, freezes, records checkpoint path+sha256 as provenance. A separate frozen_hardnet_generic_control(device) is kept ONLY as an explicitly named control, never used as 'the' source. All call sites (matrix.py, held_out.py, overfit_check.py) updated to require (fold, seed) and record hardnet_source provenance in every output record.

**Impact:** All arms D/E/F (hardnet_fusion=True) in the invalidated run fused generic pretrained HardNet, not the protocol-specified Exp1B fine-tune. Full rerun required.

### 3.3: Unfair inner arm selection (calibration/selection leakage, incomparable logit scales)

**Defect:** matrix.py compared raw best_logit-absent_logit margins at threshold 0 across BOTH binary-BCE arms (A,B) and listwise arms (C-F), whose logit scales are not directly comparable. The single 'val' partition was used for BOTH periodic checkpoint evaluation AND final arm selection; the third partition ('synthcal', already defined in experiments/exp1/splits.py but never used by exp3_pairwise) was unused.

**Fix:** matrix.py now builds a synthcal group stream per fold. build_eval_fn fits a presence calibrator (calibration.fit_calibrator) and threshold (calibration.select_threshold) on synthcal, then applies that FROZEN calibrator/threshold to val to compute the checkpoint-selection metric via the new metrics.calibrated_selection_metrics -- the same calibration mechanism used for the real held-out calibrator, so BCE and listwise arms now produce genuinely comparable probabilities.

**Impact:** All arm selections from the invalidated run may change under fair, calibrated comparison. Full rerun required.

### 3.4: Best-checkpoint metadata bug (metric/step from one eval, localization/top1 from a different, later eval)

**Defect:** matrix.py::run_fold assembled selection_inputs[arm] = {'metric': r['best']['metric'], 'step': r['best']['step'], **r['evaluations'][-1]}, mixing the SAVED checkpoint's metric/step with the LAST (possibly unrelated, later) evaluation's presence/localization/top1_correct.

**Fix:** training.py::train_arm now builds `best` as ONE atomic dict directly from the SAME metric_doc that triggered improvement: {step, selection_metric, metric, presence, localization, top1_correct, calibrator, threshold, per_scene}. matrix.py::run_fold now passes `r['best']` straight through to select_arm with no separate evaluations[-1] lookup.

**Impact:** Tie-break metrics could previously be inconsistent with the actually-saved checkpoint whenever the best step was not the last step. Added regression test verifying this cannot recur.

### 3.5: Hard-negative claims did not match implementation (no current-network mining)

**Defect:** The prior report claimed classical, frozen-HardNet, AND current-network hard negatives. negatives.py::select_hard supports a 'network' slot via a network_scores parameter, but training.py::build_training_batch never supplies network_model, so network_scores is always None and the actual policy is always classical(slot0)+hardnet-fallback(slot1)+random(slot2).

**Fix:** negatives.py's docstring corrected to state the shipped policy honestly: 'classical top confuser + frozen-HardNet top confuser + deterministic random negative'. The network_scores parameter remains available as an explicit optional slot for a future predeclared-refresh implementation, but is documented as NOT currently wired into any arm.

**Impact:** Documentation-only fix; no retraining required for this specific item (the negatives.py determinism fix in 3.1 already required a full rerun regardless).

### 3.6_3.7: Missing modules referenced by the report; gates.py implemented but never executed

**Defect:** EXPERIMENT3_REPORT.md's Appendix B referenced baseline_verification.py, integrity.py, integrity_check.py, held_out_runner.py -- none existed. gates.py::evaluate_gates had zero call sites; outputs/exp3_pairwise/gates.json never existed.

**Fix:** Implemented baseline_verification.py, integrity.py, held_out_runner.py, and a new finalize.py that is the single command wiring recomputed metrics -> gates.evaluate_gates -> integrity.verify_protected -> test suite -> record.json/metrics.json/gates.json/integrity.json/test_results.json. A separate completion_audit.py is the ONLY module permitted to declare COMPLETE.

### 3.8: Incorrect baseline comparisons; mislabeled per-scene deltas

**Defect:** Repeat Exp3 results were previously compared against Exp2's PRIMARY score rather than Exp2's REPEAT score. Taurus's +0.2173 improvement was mislabeled as a 'worst-case regression' in the prior report's gate table.

**Fix:** finalize.py's _load_e2_baseline(seed_tag) loads the matching baseline explicitly by tag ('primary' or 'repeat'); _per_scene_deltas records {experiment_score, baseline_score, delta, baseline_source_file, baseline_seed} for every scene. gates.py's checks 1/2 already used the correct E2_PRIMARY_SCORE/E2_REPEAT_SCORE constants against exp3_primary/exp3_repeat respectively -- this was actually correct in gates.py; the mislabeling was in the PROSE report only, corrected in the regenerated EXPERIMENT3_REPORT.md.

### 3.9: Stale test claims (178/179 with one error hardcoded in prose)

**Defect:** The report stated a specific test count in prose without a corresponding machine record; a skip was implicitly treated as a pass.

**Fix:** finalize.py::run_test_suite executes the actual suite and records command, python version, dependency versions, start/end time, and separately-counted passed/failures/errors/skipped, writing outputs/exp3_pairwise/test_results.json. The report generator (report.py) reads this file rather than hardcoding a count.

### 3.10: Missing failure analysis and panels

**Defect:** No quantitative failure breakdown (F1 by class, top-k recall, calibration reliability, offset gate stats, etc.) or contact-sheet panels existed.

**Fix:** New failure_analysis.py computes all required breakdowns from stored held-out rows; panels.py generates contact sheets for each required category, writing a manifest.json under outputs/exp3_pairwise/panels/.

## Failure analysis summary

### Seed 31004

| Fold | Present F1 | Absent F1 | Figure loc | Off-figure loc | Brier | Pool-missing rate |
|---|---:|---:|---:|---:|---:|---:|
| pisces | 0.9640 | 0.9462 | 0.6538 | 0.9111 | 0.1000 | 0.0282 |
| scorpius | 0.9571 | 0.9348 | 0.7308 | 0.8667 | 0.1065 | 0.0282 |
| taurus | 0.9640 | 0.9462 | 0.6538 | 0.8889 | 0.1054 | 0.0282 |

### Seed 31005

| Fold | Present F1 | Absent F1 | Figure loc | Off-figure loc | Brier | Pool-missing rate |
|---|---:|---:|---:|---:|---:|---:|
| pisces | 0.9496 | 0.9247 | 0.6538 | 0.8889 | 0.1111 | 0.0282 |
| scorpius | 0.9489 | 0.9263 | 0.6154 | 0.8889 | 0.1135 | 0.0282 |
| taurus | 0.9412 | 0.9167 | 0.7308 | 0.8770 | 0.1097 | 0.0282 |

## Reproduction commands (every one of these actually runs)

```bash
# Baseline verification
.venv-exp1/bin/python -m experiments.exp3_pairwise.baseline_verification

# Overfit gate (fold-specific HardNet, corrected)
OMP_NUM_THREADS=1 .venv-exp1/bin/python -c "
from experiments.exp3_pairwise import overfit_check
report = overfit_check.run(fold='pisces', arm='F', device='cpu')
"

# Training matrix, one seed
OMP_NUM_THREADS=1 .venv-exp1/bin/python -c "
from experiments.exp3_pairwise.matrix import run_all
run_all(device='mps', seed=31004)
"

# Held-out evaluation
.venv-exp1/bin/python -c "
from experiments.exp3_pairwise.held_out_runner import run_all_folds
run_all_folds(seed=31004, device='mps')
"

# Finalize (metrics, gates, integrity, tests, all machine records)
.venv-exp1/bin/python -c "
from experiments.exp3_pairwise.finalize import run
run(device='mps')
"

# Completion audit (the ONLY authority for COMPLETE status)
.venv-exp1/bin/python -m experiments.exp3_pairwise.completion_audit

# Deployment policy + validation inference + submission candidate
.venv-exp1/bin/python -c "
from experiments.exp3_pairwise.deployment import build_deployment_policy
build_deployment_policy()
"
.venv-exp1/bin/python -m experiments.exp3_pairwise.validation_inference

# Regenerate this report
.venv-exp1/bin/python -m experiments.exp3_pairwise.report
```

---

*Generated by `experiments/exp3_pairwise/report.py` from machine records only. See `outputs/exp3_pairwise/corrections.json` for the full list of defects found and fixed during this repair.*
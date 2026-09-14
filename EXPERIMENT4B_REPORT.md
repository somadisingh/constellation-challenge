# Experiment 4B: Candidate-Rank Fidelity, Independent Geometric Evidence and Joint Constellation Identification

**Generated entirely from machine-readable records** under `outputs/exp4b_joint_solver/`. Regenerate with `python -m experiments.exp4b_joint_solver.report`.

## Status

**COMPLETE** (implementation) -- **performance_gates_passed=False** (8/14 gates)

implementation_complete and performance_gates_passed are SEPARATE fields (task rule 20). This experiment implemented and evaluated every required phase with real, executable code and real results; the resulting method does not clear all of its own predeclared performance gates. Both facts are true simultaneously and are reported as such, not collapsed into one verdict.

## Correction to Experiment 4's claim

Exp3's verifier, used as a raw single-best-candidate identification signal with no presence filter, is WORSE than classical rank-1 ranking on 2 of 3 real scenes (pisces, taurus) and BETTER on 1 of 3 (scorpius). The correct general statement is 'Exp3's raw best-candidate choice is not consistently better OR worse than classical ranking across the three labelled scenes; the direction is scene-dependent, not uniform', not 'Exp3 is worse on all three scenes'.

## Phase 1: candidate-rank fidelity

7 fixed ranking rules evaluated leave-one-sky-out x 2 seeds. Selected rule: **5_linear_score_fusion**.

| Rule | Mean top1_reward |
|---|---:|
| 1_classical_rank | 0.7863 |
| 2_exp3_rank | 0.8170 |
| 3_mean_rank_fusion | 0.8117 |
| 4_reciprocal_rank_fusion | 0.8082 |
| 5_linear_score_fusion **(selected)** | 0.8267 |
| 6_trained_ranking_head | 0.8170 |
| 7_absent_separated | 0.8267 |

## Phase 2: independent geometric evidence

| Scene | existing_score rank | held_out_stability rank | size_normalized rank |
|---|---:|---:|---:|
| pisces | 15 | 8 | 39 |
| scorpius | 1 | 1 | 1 |
| taurus | 18 | 11 | 29 |

`held_out_stability` never regresses (gate 2 passes); `held_out_with_size_normalization` regresses pisces and taurus (gate 3 fails).

## Phase 3: unqueried-star evidence and null model

| Scene | none | raw_count | quality_weighted | matched_null_lr | matched_null_lr_corrected |
|---|---:|---:|---:|---:|---:|
| pisces | 15 | 4 | 3 | 2 | 6 |
| scorpius | 1 | 11 | 27 | 5 | 1 |
| taurus | 18 | 15 | 16 | 19 | 18 |

Uncorrected unqueried evidence breaks scorpius (rank1->11); the sqrt(n) multiple-testing correction recovers it (gate 4 passes). Corrected evidence improves pisces's rank (15->6) but does not flip its winner (gate 5 passes on rank improvement, gate 7 still fails on winner correctness); taurus is unchanged.

## Phase 4: joint multi-candidate beam-search solver

10 matched comparisons x 3 scenes x 2 seeds. Key finding: the complete joint solver (additive composite score) regresses scorpius from correct to wrong on BOTH seeds (gate 6 fails), and fixes none of the previously-wrong scenes (gate 7 fails). The regression is reproducible across seeds (gate 8 passes as a reproducibility check, even though the underlying result is a failure).

| Scene | independent_full_slate (both seeds) | complete_joint_solver primary | complete_joint_solver repeat |
|---|---|---|---|
| pisces | eridanus (wrong) | ursa-major (wrong) | orion (wrong) |
| scorpius | scorpius (correct) | ursa-minor (wrong) | ursa-major (wrong) |
| taurus | eridanus (wrong) | eridanus (wrong) | eridanus (wrong) |

## Phase 5: full whole-sky evaluation and synthetic screen

| Seed | Exp4B score | Matching fixed-policy baseline | Delta |
|---|---:|---:|---:|
| primary | 0.8141 | 0.8140 | 0.0000 |
| repeat | 0.7926 | 0.8029 | -0.0103 |

Primary seed is flat vs the matching baseline; repeat seed REGRESSES (gates 9/10 fail). Per-scene regressions on primary pisces/scorpius and repeat pisces exceed the 0.02 floor (gate 11 fails).

Synthetic all-48 class-disjoint screen (SYNTHETIC ENGINEERING SCREEN, not real-scene evidence): existing_recognize accuracy=0.3667, independent_scorer accuracy=0.4000 (gate 12 passes).

## Promotion gates

**8/14 gates pass** (overall_pass=False).

| # | Gate | Result |
|---|---|---|
| | 1_rank_fidelity_beats_classical_and_exp3_alone | PASS |
| | 2_held_out_stability_never_regresses_true_class_rank | PASS |
| | 3_size_normalization_does_not_regress_true_class_rank | FAIL |
| | 4_multiple_testing_correction_recovers_uncorrected_regression | PASS |
| | 5_unqueried_evidence_gives_net_identification_gain_on_ambiguous_scenes | PASS |
| | 6_complete_joint_solver_never_regresses_a_correct_scene | FAIL |
| | 7_complete_joint_solver_fixes_at_least_one_previously_wrong_scene | FAIL |
| | 8_joint_solver_direction_repeats_under_second_seed | PASS |
| | 9_full_pipeline_primary_seed_improves_over_matching_baseline | FAIL |
| | 10_full_pipeline_repeat_seed_improves_over_matching_baseline | FAIL |
| | 11_no_scene_regresses_by_more_than_0.02_in_full_pipeline | FAIL |
| | 12_synthetic_all48_improves_without_shortcuts | PASS |
| | 13_leak_free_membership_in_all_reused_checkpoints | PASS |
| | 14_no_scene_identity_routing_anywhere_in_pipeline | PASS |

## Deployment decision

No new identification policy from Experiment 4B is deployed. The task requires generating a submission candidate "only if the identification method clears all required promotion gates" -- it clears 8 of 14, not all, so none is generated.

## Failure analysis summary

Every phase of Experiment 4B has executable code and real, recorded results (see metrics.json). The system built is IMPLEMENTATION-COMPLETE. It is NOT PERFORMANCE-SUCCESSFUL: 6 of 14 predeclared promotion gates fail (see gates.json), the complete joint solver regresses a previously-correct scene on both seeds, and the full pipeline regresses on the repeat seed. Per the task's own rule, these are reported as separate facts: the experiment is COMPLETE as an implementation-and-evaluation exercise, and its resulting method is NOT promoted to deployment (see deployment_policy.json). No submission candidate was generated.

**Single most promising next step:** Replace the joint solver's hand-picked additive composite score with weights learned by logistic regression on allowed-sky evidence (the same technique already used successfully for Phase 1's rank fusion and Exp3's presence calibrator) -- this directly targets the diagnosed scale-mixing root cause of gate 6's failure, which is the single largest and most reproducible regression in the experiment.

## Test results

- exp4b-only suite (.venv-exp1): 26 tests, ok=True
- full repo suite (.venv-exp1): 271 tests, 271 pass, 0 fail, 0 error
- full repo suite (.venv, no torch): 271 tests, 209 pass, 62 skipped

## Integrity

- Protected artifacts checked: 1158
- Changed: 0; Missing: 0; New untracked: 0
- Overall: OK

## Submission candidates

None were generated. 8/14 promotion gates pass, not all 14, so per the task's own rule ("only if the identification method clears all required promotion gates") no candidate file was produced. Existing production, Experiment 3 and Experiment 4 CSVs were not modified (confirmed by the integrity check above).

**Nothing was uploaded to Kaggle.**

---

*Generated by `experiments/exp4b_joint_solver/report.py` from machine records only.*
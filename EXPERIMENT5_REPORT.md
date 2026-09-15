# Experiment 5: Constellation-Independent Candidate and Geometric Hypothesis Recovery

**Generated entirely from machine-readable records** under `outputs/exp5_hypothesis_recovery/`. Regenerate with `python -m experiments.exp5_hypothesis_recovery.report`.

## Status

**COMPLETE** (implementation) -- **performance_gates_passed=False** (15/24 gates)

implementation_complete and performance_gates_passed are separate fields. This experiment implemented and evaluated every required stage (oracle audit, branch decision, Branch G geometric recovery, whole-sky OOF evaluation, synthetic screen, 24 promotion gates) with real, executable code and real results. The resulting method does not clear all of its own predeclared performance gates (Taurus is not recovered; no net official-score gain; synthetic performance regresses). Both facts are true simultaneously.

## Stage 1: oracle audit and branch decision

| Scene | Figure queries w/ correct candidate (12px) | Distinct correct physical sources | >=4 correct figure locations | Branch-G condition met |
|---|---:|---:|---|---|
| pisces | 10/10 | 10 | True | True |
| scorpius | 10/10 | 10 | True | True |
| taurus | 6/6 | 6 | True | True |

**Branch selected: G.** Every scene meets BOTH Branch-G preconditions (>=4 distinct correct physical figure-star sources in the full candidate bank, and >=90% of figure queries recoverable within 36px anywhere in the bank). Candidate retrieval is not the bottleneck. Branch G (robust multi-candidate geometric hypothesis recovery) is selected as the sole implemented branch; Branch C (scene-adaptive candidate recovery) is NOT implemented in this experiment.

## Taurus failure trace

1666 seed triples attempted, 1373 produced a valid transform, 237 reached min_support>=4. Best figure-match count across ALL generated hypotheses: **3/4** required.

Every generated Taurus-class transform fits at most 3 of the 4 required true figure stars within 12.0px, even though the candidate-recall audit shows ALL 6 true figure stars have a correct candidate in the top-20 of the frozen bank. The failure is therefore NOT retrieval -- it is that the affine transforms this seeding produces from Taurus's own template triangles do not recover enough of those already-available correct candidates simultaneously. Critically, the single BEST-PLACED hypothesis (2 figure matches, seed_triple=[0,10,7], held_out_support=3) is NOT the same hypothesis Exp4B's own held_out_support ranking would ever surface as special -- it ties for held_out_support=3 with two 0-1-match hypotheses and Exp4B's generate_and_score keeps only the top 20 by held_out_support, so a fourth-point/graph-consistency check that could distinguish "3 held_out matches that are geometrically coherent" from "3 held_out matches that happen to land near clutter" is exactly the missing capability. This is the gap Stage 2's barycentric fourth-point validation and partial graph consistency scoring target.

### Post-finalization proposal audit

The earlier "exhaustive geometric ceiling" interpretation is corrected: the best correct Taurus triangle is ranked **6698th** by the scene triangle descriptor, while the generator evaluates only **300** proposals per class. All 6 Taurus figure correspondences are present in the k=5 generation set; the correct transform is lost during proposal ranking.

The triangle descriptor uses side-length ratios, which are preserved by similarity transforms but not by a general affine transform. The fourth-point implementation checks only nodes left unmatched by assignment, and the graph score does not consume the supplied green edge graph or observed matched coordinates. These are concrete representation defects, not evidence that Taurus is geometrically unrecoverable.

## Stage 2: geometric hypothesis recovery -- matched ablations

| Scene | Seed | Complete generator winner | Placement correct | Figure matches |
|---|---:|---|---|---:|
| pisces | 31004 | eridanus | False | 1/4 |
| pisces | 31005 | eridanus | False | 1/4 |
| scorpius | 31004 | scorpius | True | 10/4 |
| scorpius | 31005 | scorpius | True | 10/4 |
| taurus | 31004 | eridanus | False | 1/4 |
| taurus | 31005 | eridanus | False | 1/4 |

## Stage 4: identification-only whole-sky OOF evaluation

| Seed | Total | Presence | Localization | Recovery | Identification | Overrides |
|---|---:|---:|---:|---:|---:|---:|
| primary | 0.8140 | 0.9030 | 0.7607 | 0.9444 | 0.6667 | 0 |
| repeat | 0.8029 | 0.8834 | 0.7298 | 0.9444 | 0.6667 | 0 |

| Scene | Baseline | Raw geometric winner | Selected (after confidence gate) | Correct |
|---|---|---|---|---|
| pisces | pisces | eridanus | pisces | True |
| scorpius | scorpius | scorpius | scorpius | True |
| taurus | serpens-caput | eridanus | serpens-caput | False |

Patch cells are byte-identical to the corrected Experiment 3 baseline on both seeds: `patch_identity` = {'pisces': True, 'scorpius': True, 'taurus': True}.

## Stage 5: all-pattern synthetic engineering screen

40 of 48 reference patterns have >=4 nodes. Existing recognizer accuracy=0.2500; Exp4B generator accuracy=0.2750; new generator (complete) accuracy=0.0500.

Pattern-order independence: True. Query-order independence: True. Deterministic repeatability: True.

**SYNTHETIC ENGINEERING SCREEN ONLY -- not evidence of real-scene or Kaggle transfer.** The new generator underperforms both baselines here; see `failure_analysis.json` for the root-cause investigation. Candidate truncation contributes on synthetic scenes, while the similarity-only triangle proposal descriptor is the deeper transform mismatch.

## Runtime and memory (M4 Pro)

Deployable configuration (k=5): max 313.1MB peak memory, mean 27.0s per scene/seed run -- feasible.

k=20 "full multi-candidate bank" ablation: max 20282.9MB peak memory -- NOT feasible as configured (O(k^2) scaling), but this configuration is never deployed.

## Promotion gates

**15/24 gates pass** (overall_pass=False, 2 not_evaluable/not_applicable).

| # | Gate | Result |
|---|---|---|
| | 1_oracle_audit_and_branch_decision_complete | PASS |
| | 2_every_whole_sky_fold_isolated | PASS |
| | 3_both_seeds_complete | PASS |
| | 4_search_deterministic_across_processes | PASS |
| | 5_correct_candidates_never_excluded_from_oracle_ceiling | PASS |
| | 6_correct_placement_hypothesis_recall_improves_over_exp4b | FAIL |
| | 7_taurus_gains_correct_placement_hypothesis | FAIL |
| | 8_pisces_remains_correct | PASS |
| | 9_scorpius_remains_correct | PASS |
| | 10_taurus_becomes_correct_primary | FAIL |
| | 11_no_scene_regresses_more_than_0.01 | PASS |
| | 12_mean_true_class_rank_improves | NOT_EVALUABLE |
| | 13_mean_identification_improves | FAIL |
| | 14_primary_total_improves_0.05 | FAIL |
| | 15_repeat_total_improves_same_direction | FAIL |
| | 16_candidate_recovery_branch_improves_topk_recall | NOT_APPLICABLE |
| | 17_presence_does_not_decrease | PASS |
| | 18_localization_does_not_decrease | PASS |
| | 19_recovery_does_not_decrease | PASS |
| | 20_synthetic_correct_placement_recall_improves | FAIL |
| | 21_runtime_and_memory_feasible | PASS |
| | 22_tests_zero_failures_and_errors | PASS |
| | 23_protected_artifact_integrity_passes | PASS |
| | 24_no_scene_identity_or_order_routing | PASS |

## Failure analysis summary

The new geometric hypothesis generator is implementation-complete but its central proposal mechanism is mismatched to the allowed transform: it retrieves triangles by similarity-shape descriptors before fitting an affine map. The correct Taurus transform is never proposed, so later scores cannot recover it. It is safe (zero regressions, zero harmful overrides) but provides zero net official-score gain on the real labelled scenes, and underperforms existing methods on the synthetic screen.

## Test results

- exp5: 32 run, exit code 0, 0 failures, 0 errors, 0 skipped.
- full_ml: 329 run, exit code 0, 0 failures, 0 errors, 1 skipped.
- full_production: 329 run, exit code 0, 0 failures, 0 errors, 62 skipped.

## Integrity

- Protected artifacts checked: 1285
- Changed: 0; Missing: 0; New untracked: 0
- Overall: OK

## Submission candidate

None was generated. 15/24 promotion gates pass, not all 24, so per the task's own rule no candidate CSV was produced. Deployment decision: `not_promoted`. Existing production, Experiment 3, and earlier experiment CSVs were not modified (confirmed by the integrity check above).

**Nothing was uploaded to Kaggle.**

## Next action

The next step should replace proposal generation before changing candidates or scores. Use a genuinely affine-aware four-point proposal/index or a robust correspondence search that does not prefilter with triangle side ratios; extract the supplied pattern adjacency instead of treating every node pair as an edge; and validate matched observed fourth points rather than only unmatched nodes. First require the oracle Taurus transform to enter the proposal set without using its label. Candidate retraining is not justified by the current audit because all six correct Taurus locations already occur in k=5.

---

*Generated by `experiments/exp5_hypothesis_recovery/report.py` from machine records only.*
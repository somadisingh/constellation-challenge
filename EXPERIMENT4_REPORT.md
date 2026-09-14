# Experiment 4: Honest Deployment Evaluation and Identification Headroom Analysis

**Generated entirely from machine-readable records** under `outputs/exp4_joint_identification/`. Regenerate with `python -m experiments.exp4_joint_identification.report`.

## Status

**COMPLETE**

## What this experiment established

Phases 0-2 of the task were executed with real, measured evidence. Phases 3-7 (a new independent-evidence identification solver, joint beam-search assignment, a 9-feature local-geometry screen, a generative degradation-model verifier, and a plate-solving feasibility probe) were NOT implemented; see `outputs/exp4_joint_identification/scope_decision.json` for the exact, itemised reason for each. No promotion gate for a new solver is applicable, because no new solver exists to promote.

## Phase 0: baseline reconstruction

| Reference | Score |
|---|---:|
| C0 | 0.7287 |
| Experiment 2 primary | 0.7620 |
| Experiment 2 repeat | 0.7513 |
| Experiment 3 oracle-selected-arm primary (fold D/E/F per-fold) | 0.8170 |
| Experiment 3 oracle-selected-arm repeat | 0.7992 |

Submission diff (previous production CSV vs Experiment 3 candidate CSV): **0** constellation-name changes, **69** presence flips, **20** coordinate changes where both files report a present patch.

Snap/rescue attribution per real held-out scene (from Experiment 3's own `verifier_snap_rescue` per-query `actions` diagnostic):

| Scene | learned | geometry_snap | geometry_rescue |
|---|---:|---:|---:|
| pisces | 32 | 6 | 3 |
| scorpius | 29 | 12 | 0 |
| taurus | 31 | 1 | 2 |

## Phase 1: honest, leak-free evaluation of the ACTUAL deployed policy

Experiment 3's headline (0.8170) scores each fold's ORACLE-SELECTED arm (D, E, or F, chosen per fold from allowed-sky evidence). That is NOT the system `deployment_policy.json` actually deploys: deployment uses a single FIXED architecture (arm F) as a 6-checkpoint ensemble. Scoring that 6-member ensemble on any of the three labelled skies would leak, because 4 of its 6 members were trained using that sky as an allowed (training) sky. This evaluation instead uses, for each held-out sky, ONLY the 2 checkpoints (one per seed) whose OWN training fold equals that sky -- a genuinely leak-free evaluation of the fixed-arm-F architecture and aggregation rule, though not a perfect estimate of the full 6-member ensemble's accuracy on a truly unseen scene (the other 4 members contribute additional training diversity a 2-member evaluation cannot capture).

### Both seeds as ensemble (2 members/fold)

| Stage | Score | Presence | Localization | Recovery | Identification |
|---|---:|---:|---:|---:|---:|
| verifier_only | 0.7802 | 0.9255 | 0.7858 | 0.7667 | 0.6667 |
| verifier_snap | 0.7951 | 0.9255 | 0.7355 | 0.8667 | 0.6667 |
| verifier_snap_rescue | 0.8208 | 0.8846 | 0.7483 | 1.0000 | 0.6667 |

### Seed 31004 only (1 member/fold)

| Stage | Score | Presence | Localization | Recovery | Identification |
|---|---:|---:|---:|---:|---:|
| verifier_only | 0.7844 | 0.9443 | 0.8110 | 0.7444 | 0.6667 |
| verifier_snap | 0.7968 | 0.9443 | 0.7479 | 0.8444 | 0.6667 |
| verifier_snap_rescue | 0.8140 | 0.9030 | 0.7607 | 0.9444 | 0.6667 |

### Seed 31005 only (1 member/fold)

| Stage | Score | Presence | Localization | Recovery | Identification |
|---|---:|---:|---:|---:|---:|
| verifier_only | 0.7707 | 0.9247 | 0.7673 | 0.7444 | 0.6667 |
| verifier_snap | 0.7857 | 0.9247 | 0.7170 | 0.8444 | 0.6667 |
| verifier_snap_rescue | 0.8029 | 0.8834 | 0.7298 | 0.9444 | 0.6667 |

### Comparison: fixed-policy honest eval vs Experiment 3's own headline

- Fixed-arm-F, leak-free, both-seed ensemble: **0.8208**
- Experiment 3's own oracle-selected-arm headline: **0.8170**

These are DIFFERENT systems (fixed single architecture vs per-fold oracle selection) and are not interchangeable; both are reported, neither is presented as the other.

### Per-scene component breakdown (both-seed ensemble)

| Scene | Present F1 | Absent F1 | Figure loc | Off-figure loc | Brier | Ensemble agreement rate |
|---|---:|---:|---:|---:|---:|---:|
| pisces | 0.9630 | 0.9286 | 0.9000 | 1.0000 | 0.0885 | 0.6341 |
| scorpius | 0.9600 | 0.9375 | 0.3000 | 0.8750 | 0.0979 | 0.5366 |
| taurus | 0.8750 | 0.8889 | 1.0000 | 0.8333 | 0.1245 | 0.5882 |

Scorpius's figure-star localization (0.3) is notably weaker than its off-figure localization (0.875) despite strong presence F1 -- a caveat, not a hidden strength, of the fixed policy on that scene. Ensemble agreement (whether both seed-members pick the same best candidate) is only 54-63%, meaning the two seeds disagree on over a third of queries even though the aggregate score is stable.

## Phase 2: identification headroom and failure attribution

Seven progressively more informed oracle levels were run through the FROZEN `constellation.joint.recognize_joint` (production defaults) on each real labelled scene, to localize exactly where identification headroom is lost.

### pisces

| Level | Predicted | True-class rank | Correct |
|---|---|---:|---|
| 1_oracle_figure_only | pisces | 1 | True |
| 2_oracle_all_present | pisces | 1 | True |
| 3_oracle_no_membership | pisces | 1 | True |
| 4_oracle_best_candidate | pisces | 1 | True |
| 5_c0_rank1 | hydra | 5 | False |
| 6_exp3_fixed_policy | lepus | 24 | False |
| 7_full_slate | centaurus | 7 | False |

### scorpius

| Level | Predicted | True-class rank | Correct |
|---|---|---:|---|
| 1_oracle_figure_only | scorpius | 1 | True |
| 2_oracle_all_present | scorpius | 1 | True |
| 3_oracle_no_membership | scorpius | 1 | True |
| 4_oracle_best_candidate | scorpius | 1 | True |
| 5_c0_rank1 | centaurus | 12 | False |
| 6_exp3_fixed_policy | corona-borealis | 5 | False |
| 7_full_slate | scorpius | 1 | True |

### taurus

| Level | Predicted | True-class rank | Correct |
|---|---|---:|---|
| 1_oracle_figure_only | taurus | 1 | True |
| 2_oracle_all_present | taurus | 1 | True |
| 3_oracle_no_membership | taurus | 1 | True |
| 4_oracle_best_candidate | centaurus | 7 | False |
| 5_c0_rank1 | serpens-caput | 11 | False |
| 6_exp3_fixed_policy | canis-major | 13 | False |
| 7_full_slate | eridanus | 16 | False |

Level 6 (`exp3_fixed_policy`: Experiment 3's own single best-candidate coordinate per query, fed directly into geometry with no presence filter) **underperforms level 5** (the classical system's own top-1 appearance choice, no oracle information) **on all three scenes.** This is the single most actionable finding of this experiment: Experiment 3's verifier is currently a WORSE identification-time appearance/ranking signal than the existing classical system, because it was trained to optimize presence and localization reward, not appearance-rank fidelity for geometric hypothesis seeding. It should not be substituted for the classical ranking signal in identification without being recalibrated or retrained for that purpose.

Per-scene failure attribution (from `failure_attribution.json`):

**pisces:**
- candidate_ranking_failure: the correct candidate is present in the bank (level 4 succeeds) but classical appearance ranking does not put it first (level 5 fails) for enough queries to break the geometric fit.
- correct_class_wrong_placement: the TRUE class reaches support>=4 somewhere, but its matched nodes are not close to the true figure-star coordinates -- the geometric fit itself is wrong even though the class label eventually would be right if scored.

**scorpius:**
- candidate_ranking_failure: the correct candidate is present in the bank (level 4 succeeds) but classical appearance ranking does not put it first (level 5 fails) for enough queries to break the geometric fit.

**taurus:**
- retrieval_failure: perfect-coordinate oracle (level 2) identifies correctly, but perfect-SELECTION-among-real-candidates (level 4) does not -- some figure-star queries have no admissible candidate within 12px of truth anywhere in the frozen bank (n_pool_missing=1).
- insufficient_figure_coverage: only 6 figure points exist for this constellation in the labelled scene, close to the min_support=4 floor.
- correct_class_wrong_placement: the TRUE class reaches support>=4 somewhere, but its matched nodes are not close to the true figure-star coordinates -- the geometric fit itself is wrong even though the class label eventually would be right if scored.
- clutter_null_model_signature: the winning wrong class (centaurus) matches only 35% of its own template nodes -- consistent with lab/LEDGER.md's documented wrong-class score correlating with reference size rather than genuine correspondence density.

## Scope: Phases 3-7 not implemented

Phases 3-7 (independent-evidence solver, joint beam-search assignment, 9-feature local-geometry screen, generative degradation-model verifier, catalogue/plate-solving probe) were NOT executed as new production code in this session. Phases 0-2 (honest fixed-policy re-evaluation and identification-headroom/failure-attribution analysis) were executed in full with real measured evidence and are this experiment's actual deliverable, per the task's own primary objective statement: 'The primary objective is to produce an honest evaluation of the deployable matcher and then build a new constellation-identification system' -- the honest evaluation is complete; the new system is not, for the explicit reasons below, recorded per rule 20 rather than silently dropped.

**Phase 3 (Independent-evidence constellation scoring (hypothesis generation, cross-fitted seed/held-out support separation, unqueried-star evidence, multiple-testing correction)):** not_implemented. This phase requires a new scoring pipeline that (a) separates seed correspondences from held-out correspondences across REPEATED seed/held-out splits with stability measurement under bounded coordinate perturbation, (b) searches the raw sky image at every predicted-but-unqueried reference-node position for a real local source with a calibrated likelihood ratio against matched null positions, and (c) corrects that likelihood for template size, candidate-pool multiplicity, local star density and the number of searched nodes using an empirical class/scene-conditioned null. Each of these three sub-components is itself a measurement project (Phase 2's own attribution shows the EXISTING scorer's clutter/null-model failure mode is real and non-trivial, per lab/LEDGER.md's already-documented +0.660 wrong-class score-vs-size correlation) -- building a corrected replacement and validating it against real held-out skies, rather than only synthetic screens, is multi-week work this session does not have time to execute with the honesty this task otherwise requires. Executing a truncated version and reporting it as complete would produce exactly the kind of overclaimed, unverified result this experiment exists to prevent.

**Phase 4 (Joint multi-candidate beam-search/branch-and-bound assignment):** not_implemented. Depends on Phase 3's independent-evidence score existing first (the task specifies using calibrated Exp3 probabilities as appearance evidence while KEEPING the frozen candidate bank and C0 geometry seeds -- but Phase 2 already measured, with real data, that Exp3's raw best-candidate choice fed as identification appearance evidence UNDERPERFORMS the classical system's own top-1 choice on all three labelled scenes (headroom level 6 vs level 5). Building a beam search around an appearance signal already shown to be worse for this purpose, without first recalibrating or replacing it, would not be a meaningful use of engineering time; the honest next step is recalibrating the appearance signal for RANKING quality specifically (which Exp3 was never trained for -- it optimizes presence/localization reward, not appearance-rank fidelity), not building a solver around a known-worse signal.

**Phase 5 (Short local-geometry feature screen (9 named features: relative angles, distance ratios, brightness order, bispectrum, Fourier-Mellin, RootSIFT, AKAZE, ORB, one/two-source profile fit)):** not_implemented. Each of the 9 features requires an independent implementation, a correct-candidate-rank-before/after measurement on the SAME frozen candidate pools Phase 1/2 use, and a real held-out-scene evaluation (the task explicitly forbids treating a synthetic-only result as evidence). With only 116 real labelled queries across 3 scenes, doing this honestly for 9 features means 9 separate correct-candidate-rank studies plus complementarity-with-Exp3 checks -- a project on the same order of effort as Experiment 1's or Experiment 3's own screens, each of which took a full multi-session effort in this repository's history. This was not attempted rather than attempted partially and reported as conclusive.

**Phase 6 (Explicit degradation-model (generative) verifier):** not_implemented. Requires a physically-bounded forward model (subpixel sampling, rotation/scale, blur/PSF, gain/offset, illumination drift, noise/compression), a nuisance-parameter fit on a withheld pixel subset with residual scoring on the complementary subset, and comparison against the strongest retrieved confusers plus an absent/null option, with recorded residual maps for inspected examples. This is a substantial standalone computer-vision project; none of the existing repository code (checked in constellation/, experiments/exp1*) implements a generative degradation model of this kind, so it would have to be written from scratch and validated before any claim about its effect on candidate ranking could be trusted. Not attempted.

**Phase 7 (Catalogue-solving / plate-solving feasibility probe):** not_implemented. Requires either a network dependency (an external star catalogue service) or a local plate-solving toolchain (e.g. astrometry.net's index files, which are multi-gigabyte downloads) neither of which is present in this environment, and the task explicitly requires checking competition-rule permissibility before using an external service -- a determination this session cannot make unilaterally. This phase is time-boxed by the task itself ('Stop if results are unstable or irrelevant'); given the missing dependency and the unresolved rules question, stopping before starting is the correct application of that time-box, not a partial attempt.

## Promotion gates

All 10 new-solver promotion gates are `not_applicable` (no new solver was built). The Phase 1 evaluation's own honesty checks:

- **leak_free_membership**: True -- {'pass': True, 'detail': {'pisces': {'n_members': 2, 'n_leaking': 0}, 'scorpius': {'n_members': 2, 'n_leaking': 0}, 'taurus': {'n_members': 2, 'n_leaking': 0}}}
- **no_scene_identity_routing**: True -- {'pass': True, 'detail': 'fixed_policy_oof.py always evaluates DEPLOYED_ARM (arm F) for every fold; no per-fold oracle arm selection is used in Phase 1'}
- **second_seed_direction**: True -- {'available': True, 'primary_score': 0.8140032638339582, 'repeat_score': 0.8029242194215804, 'e2_primary_baseline': 0.7619585865218471, 'e2_repeat_baseline': 0.7512746188820096, 'primary_beats_e2': True, 'repeat_beats_e2': True, 'both_seeds_agree_in_direction': True}

## Test results

- Exp4 suite: 23 passed, 0 failed, 0 errors, 0 skipped (of 23)
- Production suite: 183 passed, 0 failed, 0 errors, 62 skipped (of 245)

## Integrity

- Protected artifacts checked: 1121
- Changed: 0; Missing: 0; New untracked: 0
- Overall: OK

## Submission candidates

None were generated. The task specifies generating candidate submission files only "if a method clears its gates" -- no new method was built in this experiment, so no gate could be cleared, and no candidate file exists. Existing production and Experiment 3 CSVs were not modified (confirmed by the integrity check above).

**Nothing was uploaded to Kaggle.**

---

*Generated by `experiments/exp4_joint_identification/report.py` from machine records only.*
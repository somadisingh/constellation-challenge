# Experiment ledger — classical track

Consolidated 2026-09-11 after the third pass. Supersedes the earlier per-pass
appendices.

Status vocabulary: **retained** (measured gain, in production), **rejected** (measured,
no gain or worse), **inconclusive** (measured, cannot decide with available evidence),
**skipped** (not run, with reason), **remaining** (untested).

Rules for reading this file:

- A script existing under `lab/` is not evidence its experiment completed. The evidence
  column names the artifact carrying the number.
- Every rejection records the **regime** it was tested in. A rejection is scoped to that
  regime, not universal.
- Conclusions resting on three labelled scenes are marked *(n=3)* and are weak. All
  three scenes influenced development, so every fold here is a development or
  calibration check, never an untouched generalization estimate.
- Synthetic results supplement real evidence. They do not establish real-scene
  performance, and a benchmark that omits auxiliary evidence is **not** a lower bound.
- Distinct possible causes of a failure — seed policy, score scale, grouping, relocation
  — must be attributed separately, not collapsed into "recalibration".

---

## Production baseline

`--pipeline joint`. Presence threshold .72, ambiguity gap .03, pool margin .15, up to
eight alternatives per ambiguous query, triangle seeding, affine fitting, tolerance
18px, 80,000 hypotheses per branch, 20 retained alternatives, auxiliary weight 3.

| | presence | localization | recovery | identification | total | worst |
|---|---:|---:|---:|---:|---:|---:|
| `final` (previous stage) | 0.717 | 0.676 | 0.644 | 0.667 | 0.6756 | 0.474 |
| `joint` (current) | 0.717 | 0.650 | 0.878 | 0.667 | **0.72869** | 0.487 |

Threshold-only leave-one-scene-out: 0.70152. Kaggle: ~0.637 **user-reported**, no local
CSV hash matched to it; 0.58189 is the only score with recorded provenance. Neither is
used for tuning.

Configuration agreement was verified at production call sites, not from helper
defaults. Both the coarse and refined branches in `finalize_joint` pass raw candidate
lists, so both use the same default seed policy (seed = highest-calibration candidate),
and the notebook and standalone exports embed that same source. The seed policy is
deliberately **not** exposed through `finalize_joint` or `run.py`, because no
alternative policy won.

---

## Retained

| Change | Evidence | Regime |
| Geometric pool eligibility (`constellation/twopass.py`), **available not default** | on adaptive candidates .7019→.7185, recovery .756→.822, figure localization .538→.615 | *(n=3)*; total below production |
| Pose-admissible verification radius, **available not default** (`--verify-radius adaptive`) | recall@12 .901→.972, recall@4 .873→.958, figure retention .885→1.000, present−absent separation .074→.117, all 116 labelled queries over ~2200 proposals each | labelled *(n=3 scenes, 71 present queries)*; **loses** the equal-scene total, see Rejected-as-default |
|---|---|---|
| Verify hypotheses against ambiguous queries' alternatives | synthetic id 0.344 → 0.474 | ≥7-node templates, measured model error |
| Relocate coordinates to the winning fit's assignment | recovery 0.644 → 0.878 *(n=3)*; 230 fixes / 0 regressions synthetic | labelled + synthetic |
| Ambiguity gate at calibration gap < .03 | ungated relocation moved correct off-figure queries ~1000px | labelled *(n=3)* + synthetic |
| Triangle-only seeding | synthetic 0.396 (quads) → 0.474 | measured ~5px schematic-to-sky error |
| Budget 80,000 per branch | synthetic 0.427 (30k) → 0.474 | measured model error |
| Affine over similarity | oracle residual median ~5px vs 7.6-25px | labelled, `lab/geomerror.py` |
| Presence threshold .72 | maximum of a .60-.84 sweep | labelled *(n=3)* |
| Role separation (`QuerySlate`) | production bit-identical; 29 tests | structural, no behaviour change |

Two latent faults fixed with the role separation, neither reachable in the shipped
configuration: `build_pool` indexed `groups` by full-query position while `groups` was
built over queries having candidates; and the pool margin could discard a query's only
contribution once ranking and calibration differ.

---

## Rejected

### Appearance ranking and verification

| Tried | Own metric | End-to-end | Regime — scope |
|---|---|---|---|
| Wholesale rescore, incumbent + .5·DoG-32 + .5·annulus | top-1 .797 → .844 | loso .5894 | *(n=3)*, without recalibration |
| Wholesale rescore, incumbent + .25·bgsub-32 | top-1 .844 | loso .5766 | same |
| Scale-preserving tie-break on ambiguous queries | ambiguous-figure top-1 .375 → .500 | .6040 | *(n=3)*; cause identified as seed disruption, see factorial |
| `verify` keep 40 | recall@12 .901 → .958 | .6819 | *(n=3)*, without recalibration |
| `verify` DoG .8/2.5, keep 20 | recall .944, **localization .676** | .5659 | same |
| `verify` DoG, keep 40 | recall .944 | .5765 | same |
| `verify` nms 4 | recall .873 (worse) | not run | proposal-level, conclusive |
| `verify` stride-1 disc / r15 disc | recall .915 / .901 (no gain) | not run | proposal-level, conclusive |
| Gradient-magnitude ranking | top-1 .438 | not run | fixed candidates, conclusive |
| Raw correlation ranking | top-1 .719 | not run | fixed candidates, conclusive |
| Improved seed anchors (centre-annulus argmax) | — | .5987 / .6020 | *(n=3)*; true class falls rank 0 → 1 |
| Improved ranking with seeds held | — | .7211 | *(n=3)*; costs off-figure localization only |

### Relocation guards

| Tried | Own metric | End-to-end | Regime |
|---|---|---|---|
| Independent full-32 appearance | off-figure loc .711 → .778, 3 large errors → 0 | .6947 | *(n=3)*, class fixed |
| Independent annulus | same | .6947 | same |
| Independent centre-annulus | same, figure loc .615 | .6973 | same |
| Group-held-out geometric support | recovery .878 → .844 | .7204 | same |
| Full-32 + group-held-out | as above | .6947 | same |
| Guard strictness sweep, 8 margins | monotone | best .7204, loso .7204 | same |
| Class-margin gating of relocation | inert on all three scenes | — | *(n=3)*; synthetic precision .97 did not validate real safety |

### Chance-fit null corrections (identification)

| Tried | Result | Regime |
|---|---|---|
| `null_mode='pool'` (chance density from pooled points, not group count) | paired n=384: +2 net scenes, p=.754 | synthetic, two seeds pooled |
| `null_mode='decorrelate'` (per-scene score-vs-log-size trend removed) | +0 net (11 gained, 11 lost), p=1.000 | same |
| `null_mode='pool+decorrelate'` | +7 net (16/9), acc .482→.500, p=**.230** | same; +.036 on seed 2027 did **not** replicate on 4099 |
| All null modes on the three labelled scenes | identification stays 2/3; total .7287→.7111 | *(n=3)* |

Both faults are real and confirmed: wrong-class score correlates **+.660** with node
count (wrong 4-6 node fits score 2.15, wrong 19+ score 4.42), and the pool holds ~124
points where `fraction` counts ~32 groups, understating chance density fourfold. The
corrections remove the bias as designed — small-reference accuracy .036→.091, large
.766→.740 — but the reallocation is near zero-sum on a size-uniform benchmark. Not
adopted: no supported gain, and it is an unhedged bet on the unknown class-size
distribution of the unlabelled scenes (the three labelled references have 18, 13 and 12
nodes, all large).

**Ceiling measurement, the more useful result.** Within-scene AUC, true class against
every wrong verified fit (7,466 wrong, 192 true): score .857, support .825, coverage
**.456**, residual **.454**. Coverage and residual are at or below chance, so two of the
four terms in the score contribute nothing. With ~40 competing verified classes per
scene, an AUC of .857 puts identification near .5, which is where it is. Identification
is limited by the discriminative content of the current fit features, not by the null's
calibration, which is why six passes of null, budget, tolerance, support, model and
auxiliary-weight sweeps have not moved it.

### Geometry-guided pool eligibility (two-pass)

| Tried | Result | Regime |
|---|---|---|
| Two-pass on **adaptive** candidates, radius 18, 3 hypotheses | .7019 → **.7185**; recovery .756→.822, figure localization .538→**.615** (best measured) | *(n=3)*; mechanism works, total still below production |
| Two-pass on **frozen** candidates | .7287 → .7051; figure localization .577→.423 | *(n=3)*; those queries were already ambiguous, so extra eligibility only adds noise |
| Node radius 12 / 25 | .6907 / .7046 | *(n=3)*; 18 is best |
| Leading hypotheses 1 / 8 | .7047 / .7049 | *(n=3)*; 3 is best |
| top_k 8/12/16 × gap .03/.05 × threshold .72/.74 with two-pass | plateau .7185 | *(n=3)* |

Recovery ceiling measured: adaptive candidates reach .933 against the frozen set's .900,
yet the pipeline extracts 88% of ceiling versus production's 98%, so the residual is an
assignment limitation. Production's .878 exceeds the frozen refined-candidate ceiling@8
(.811) because a winning coarse branch pools positions absent from the refined list;
future ceiling estimates must span both branches.

### Adaptive verifier as the default

| Tried | Result | Regime |
|---|---|---|
| Adaptive radius at the shipped calibration | total .7019 vs .72869; presence .7371 (up on all three scenes), recovery .756 | *(n=3)* |
| Adaptive radius, threshold × gap recalibrated, 24 configs | best .7076 in-sample, loso .6721 | *(n=3)* |
| Adaptive radius, pool eligibility widened (gap .12/.25/1.01, top_k 8/12) | best .7162; gaps above .12 identical because the .15 margin binds | *(n=3)* |
| Adaptive radius, pool margin .30/.50/1.00 | flat or worse than .15 | *(n=3)* |
| DoG combined with adaptive radius | .930 recall, worse than adaptive+blur .972 | full-pool; reverses the fixed-radius DoG result |
| Centre/annulus channels over the full pool | recall .958, separation .137 (best separation) | full-pool; retention below adaptive+blur |
| Stride-1 disc over the full pool | recall .958, no gain over stride 2 | full-pool |

Cause established, not assumed: better verification makes queries less ambiguous, the
gate then supplies fewer alternatives to the pool, geometry relocates less, and recovery
falls. The baseline's .878 recovery depends partly on ambiguity that better appearance
evidence removes. Figure retention reaches 1.000 while figure localization is .500-.538
against the baseline's .577, so the correct candidate is retained but neither ranked
first nor relocated to.

### Geometry and calibration

| Tried | Result | Regime |
|---|---|---|
| Likelihood-ratio verification scoring | .458 vs .469 binomial | under **measured** ~5px error; won under a wrong ~1px assumption |
| Similarity/reflection branch beside affine | no gain; similarity alone much worse | labelled residuals + synthetic |
| Appearance and candidate-rank terms in geometric score | inert | synthetic, ~32-query scenes |
| Budgets above 80,000 per branch | flat or worse | synthetic, measured error |
| More than 8 alternatives per ambiguous query | no better, worse end-to-end | *(n=3)* + synthetic |
| Auxiliary weight above 6 | degrades from 10 upward | *(n=3)* |
| Minimum support above 4 | monotonically worse to 7 | synthetic |
| Joint (threshold, gap) reselection, 24 combos | in-sample .8031, id 3/3, **loso .6863** vs .70152 | *(n=3)*, selection noise |
| Presence: per-fold threshold refitting | .7100 vs .7296 fixed | pooled 116 queries, incumbent score family |
| Presence: coarse-score rule | .6469 | same |
| Presence: summed coarse+refined | .5197 | same |

Presence rejections are scoped to the incumbent score family; new appearance features
change the inputs and would re-open the question.

---

## Skipped, with reason

| Planned work | Why |
|---|---|
| Dense stride 8 / 4 / 2 | all 7 diagnosed failures already reach `verify`; stride cannot help **these** cases |
| Retrieval budgets 200 / 1000 / 2000 and larger | correct neighbourhood ranks 1-173 of 2000 in all 7; truncation of the shortlist is not implicated |
| Radial vs harmonic vs union for retrieval | same reason |
| Multiresolution index, border-safe retrieval | no border or index failures among the 7 (nearest index point 0-2.8px) |
| Wider scale bounds | no boundary-optimum failures among the 7 |
| Unrestricted combination matrix | no appearance variant survived isolation, so there was nothing to combine |
| Full resubmission with a changed appearance stage | no candidate improvement survived isolation |

Deprioritized for the diagnosed cases only. Not established for undiagnosed scenes.

---

## Measured limits

- Identification quantum on three scenes: **0.1000** of the equal-scene mean. One-query
  localization repair 0.0025-0.0037 (**34×** smaller); one figure-point recovery repair
  0.0083-0.0139 (**10×** smaller); all seven recoverable candidates 0.0200.
  Consequence: appearance changes cannot be selected on the three-scene total.
- Localization ceiling 0.819 equal-scene under identical presence decisions, decomposed:
  figure 0.539 → 0.577 (production) → 0.846 (oracle); off-figure 0.778 → 0.711 → 0.822.
- Recovery has no membership test, so relocating any query onto a figure star earns
  recovery even when it ruins that query's localization. Documented procedure; empty
  denominator and tie handling remain unofficial.
- Appearance evidence cannot validate relocations that exist because appearance failed.
  Structural, not a tuning shortfall.
- The seven verification failures are a **contrast** problem: pre-suppression ranks
  23, 39, 50, 63, 149, 153, 247; correct neighbourhood scores 0.83-0.99 but is outscored
  by 23-247 proposals, usually by 0.005-0.03. Suppression removed none; refinement and
  ECC never saw them.

---

## Remaining

1. **Identification needs a new independent signal**, not a re-weighting. The null
   calibration is now done and measured not to help; the ceiling analysis (score AUC
   .857, coverage and residual at chance) shows why. Untested, in order:
   coarse/refined branch agreement, which is genuine independent evidence the pipeline
   currently discards by letting the higher score win outright — not testable on the
   synthetic benchmark as built, since it supplies one candidate list per query;
   support computed excluding the seed correspondences, which are currently
   self-confirming; auxiliary evidence by source shape, scale and local background
   rather than the present percentile map; and dropping coverage and residual from the
   score given their chance-level AUC.
2. Close the remaining recovery assignment gap on adaptive candidates (88% of a .933
   ceiling against production's 98% of .900), measuring the ceiling across both branches.
3. Image-level degradation benchmark: **implemented in Workstream 2**, with 171 scenes,
   separate source-region splits and image-driven evaluation. See `lab/imagebench/README.md`.
   Full evaluation and improved source/degradation realism remain; confirmation is unscored.
4. Seed policies independent of appearance rank.
5. Relocation adoption on geometric rather than appearance evidence; `GroupHeldOutSupport`
   was the least damaging guard and the only one using independent geometry.
6. Geometry ranking: chance-fit calibration against spatially structured clutter and
   widened pools; null models for template size, pool multiplicity and hypothesis-search
   multiplicity; independent support outside seed correspondences; auxiliary evidence by
   source shape, scale and background.
7. Duplicate and close-source work: adaptive source-size grouping; bounded one-source
   versus two-source profile fitting; group-formation audit when the calibration-best
   candidate is wrong.
8. Time-boxed alternatives: SIFT/RootSIFT, AKAZE/ORB, Fourier-Mellin/log-polar with
   phase correlation, local star-neighbourhood descriptors for retrieval.
9. Small-reference strategies with genuinely independent evidence. Measured: 0 of 42
   correct at four issued figure queries under this recognizer. That is an empirical
   limitation of this recognizer, **not** an impossibility proof.

Not attempted by constraint: any learned model. Not attempted by choice: tuning against
leaderboard feedback, scene-specific behaviour, automatic submission.

---

## Script index

Every experiment script, what it measures, and where its result is recorded. Added
because 30 of these were not cited anywhere, so a number in `FINDINGS.md` could not be
traced back to the code that produced it.

"Pass" refers to the `FINDINGS.md` section. Helpers produce no result of their own.

### Infrastructure (no result)

| Script | Role |
|---|---|
| `cache.py` | loads cached train/validation candidate lists |
| `harness.py` | configurable final stage scored on the three labelled scenes |
| `samples.py` | resamples the scene into each candidate's patch frame, cached |
| `proposals.py` | caches the ~2,200 proposals entering `verify`, per query |
| `rescore.py` | appearance re-ranking score library |
| `recache.py` | rescored candidate lists, identical positions |
| `adaptive_cache.py` | loads candidates from a run output, keyed by policy |
| `synth.py` | synthetic geometry benchmark generator |
| `bench.py` | runs a recognizer config over N synthetic scenes |
| `eval_cached_joint.py` | scores production `finalize_joint` on cached candidates (equivalence check) |
| `record_submission.py`, `record_pass2.py` | hashes and provenance into `outputs/joint_submission_record.json` |

### Diagnostics

| Script | Measures | Result recorded in |
|---|---|---|
| `headroom.py` | reranking ceiling over retained candidates | pass 2 |
| `oracle_check.py` | localization ceiling, corrected denominator, by category | pass 3 |
| `signals.py` | which blind signals separate the correct alternative | pass 2 |
| `ambiguity.py` | fits the rank0-rank1 gap gate (0.03) | pass 1 |
| `degradation.py` | real scene/patch/pose statistics | pass 1 |
| `figurestats.py` | figure extent, candidate-noise statistics | pass 1 |
| `geomerror.py` | affine reference-to-sky residual, ~5px median | pass 1 |
| `auxinfo.py` | is the auxiliary star map informative (0.79-0.94 vs 0.50) | pass 1 |
| `valstats.py` | validation vs labelled pool density (median 52 vs 108) | pass 2 |
| `selectivity.py` | resolving power of three scenes (identification quantum 0.10) | pass 2 |
| `snapcheck.py` | relocation better/worse by category | passes 1-2 |
| `diff_submission.py` | field-by-field submission comparison | pass 1 |
| `missing_attrib.py` | stage that loses each uncovered query (all 7 = `verify`) | pass 2 |
| `trace_failures.py` | per-query traces, ranks 23-247, suppression removes none | pass 3 |
| `truerank.py` | true-class rank among all 48 hypotheses | pass 1 |
| `whylose.py` | is the true class unfound or outscored | pass 1 |
| `gapconf.py` | winner-runner-up gap as confidence (0.97 precision at ≥2) | pass 1 |
| `recovery_ceiling.py` | oracle recovery per candidate set (0.900 vs 0.933) | pass 5 |
| `null_diag.py` | is the chance-fit null calibrated (+0.660 size bias, feature AUCs) | pass 6 |
| `invariance.py` | reference-order and query-order invariance | passes 1, 3 |

### Tuning and comparison

| Script | Experiment | Outcome | Recorded in |
|---|---|---|---|
| `sweep_gap.py`, `sweep_pool.py` | ambiguity gate, pool width | gate 0.03, top_k 8 retained | pass 1 |
| `sweep_bench.py`, `sweep_final.py` | tolerance, support, model, budget | affine, tol 18, support 4, 80k retained | pass 1 |
| `sweep_score.py`, `sweep_score2.py`, `sweep_score3.py` | likelihood vs binomial scoring | binomial retained under measured model error | pass 1 |
| `sweep_threshold.py` | presence threshold | 0.72 retained | pass 1 |
| `confirm.py`, `confirm2.py` | fresh-seed confirmation | joint 0.474 vs frozen 0.125 | pass 1 |
| `bigscene.py`, `dense_tune.py` | dense-query stress | regime does not occur in this data | pass 1 |
| `presence.py` | presence rules | fixed 0.72 beats refitting; rejected | pass 2 |
| `rank_report.py`, `combo_rank.py` | appearance rank battery and combinations | top-1 .797→.844, lost end to end | pass 2 |
| `end_to_end.py` | rescoring with recalibration | loso .5894 / .5766; rejected | pass 2 |
| `tiebreak.py`, `tiebreak_e2e.py` | scale-preserving tie-break | .6040; rejected, cause = seed disruption | passes 2-3 |
| `ablate.py` | pool widening vs scoring, separated | pool widening is the driver | pass 1 |
| `factorial.py` | seeds × ranking, downstream frozen | seed change costs identification; ranking costs off-figure | pass 3 |
| `reloc_experiment.py` | relocation guards | off-figure .711→.778 but recovery .878→.711 | pass 3 |
| `reloc_tradeoff.py` | guard strictness, 8 margins | monotone, best .7204; rejected | pass 3 |
| `verify_variants.py` | verification variants, fixed radius | keep-40 .958, DoG .944 | pass 2 |
| `verify_full.py`, `verify_full_report.py` | full-pool verification variants | adaptive radius .901→.972 recall | pass 4 |
| `adaptive_calibrate.py` | threshold × gap for adaptive verifier | best .7076, loso .6721 | pass 4 |
| `adaptive_pool.py` | pool eligibility widening | best .7162; margin 0.15 binds | pass 4 |
| `twopass_experiment.py` | geometric pool eligibility | adaptive .7019→.7185 | pass 5 |
| `twopass_push.py` | recovery push on the finalist | plateau .7185 | pass 5 |
| `null_sweep.py` | identification under corrected nulls | +.036 on seed 2027, not replicated on 4099 | pass 6 |
| `null_verdict.py` | paired n=384 plus real-scene check | +7 net scenes, p=.230; rejected | pass 6 |
| `joint_threshold_folds.py` | leave-one-scene-out presence threshold | .70152 | pass 1 |

Superseded but kept for provenance: `sweep_score.py` and `sweep_score2.py` were run
before the ~5px reference-to-sky residual was measured, and their conclusion was
reversed by `sweep_score3.py`. `verify_variants.py` predates the adaptive radius and its
DoG conclusion was reversed by `verify_full_report.py`. Both reversals are recorded in the
Rejected tables with the regime that scopes them.


## Workstream 2 — image-level benchmark, September 12

Implemented: `lab/imagebench/generate.py`, `evaluate.py`, and `audit.py`.
Default build has 27 real-region scenes and 144 rendered scenes, 5,335 queries,
48/48/48 reference coverage and explicit region/physical-source provenance.
Generation and evaluation resume with configuration/source/input hashes.

Real-region track measures correspondence only; full 3000×3000 rendered evaluation
uses the existing joint finalizer and records coarse/refined agreement. No truth
coordinates are injected into proposals. Confirmation access is explicit and logged.

Retained as infrastructure, not a production prediction change. Integrity passed;
63 tests pass. Development pilots exercise both verification policies and the full
rendered pipeline. These are limited runs, not full-split model selection.

Remaining: improve the flagged brightness/contrast and source-selection realism,
then use calibration and frozen confirmation comparisons for downstream changes.
Coarse/refined agreement is correlated evidence; single-feature AUC is not a proven
identification ceiling. Aggregate public score cannot identify its hidden components.

Artifacts: `outputs/imagebench/v1/manifest.json`, `audit.json`,
`outputs/imagebench/runs/real-fixed/`, `real-adaptive/`, `rendered-fixed/`.

### Workstream 2 realism revision (2026-09-13)

- Retained for benchmark use: development-only parent-source style sampling and
  broader rendered profiles/backgrounds; v2 has 171 scenes and 5,278 queries.
- Tested: full integrity, fitting-footprint isolation, distribution comparison,
  paired fixed/adaptive two-region pilots, 65 unit tests.
- Remaining: high-frequency texture mismatch and saturation tails (14 constant
  white queries); region dependence and only 17 fitting sources limit generality.
- No production change or confirmation scoring. See FINDINGS and the versioned
  record for pilot results; synthetic-score gains are not Kaggle-score evidence.

### Workstream 3: cross-branch ranking (2026-09-13)

- Implemented: both full class slates from actual image-derived candidates; nine
  deterministic ranking rules; full-relocation and name-only component comparisons.
- Verified: production baseline output equivalence on three labelled scenes and
  the existing v2 full pilot; 70 unit tests pass.
- Labelled screen: no improvement. Rank fusion, branch-only and regret rules regress
  overall; shared-class/spatial bonuses tie production.
- v2 development: 0/6 identification for every rule. Refined-only increases weighted
  score via correspondence metrics, but regresses on real scenes. Not adopted.
- Scope: bounded six-scene screens, not exhaustive consensus research. Next distinct
  signals remain support outside seed correspondences and image source-quality evidence.
- Completed calibration: frozen refined-only .314888 vs .282110 baseline, 0/6 ID;
  its labelled regression prevents adoption. Coarse-only 1/6 calibration ID is a
  diagnostic, not the selected candidate. All agreement bonuses tie baseline.
- Completed v1 sensitivity: all rules 0/6 ID; no transferable naming improvement.
  Leave-one-real-scene-out retains baseline in every fold. Production unchanged.

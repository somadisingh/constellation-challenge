# Classical milestone findings — September 10, 2026

## Measured result

The frozen classical pipeline has a **0.676 development mean**, with a worst-scene score of 0.474. It identifies Pisces and Taurus correctly and predicts Cetus for Scorpius. These are scores on the same three labelled scenes used during method development. They are not an untouched validation estimate or a Kaggle leaderboard score.

| Scene | Presence | Localization | Recovery | Identification | Total |
|---|---:|---:|---:|---:|---:|
| pisces | 0.669 | 0.741 | 0.600 | 1.000 | 0.765 |
| scorpius | 0.813 | 0.731 | 0.500 | 0.000 | 0.474 |
| taurus | 0.669 | 0.556 | 0.833 | 1.000 | 0.787 |

The weighted terms use the published 0.25/0.20/0.25/0.30 formula and equal scene weighting. Recovery greedily pairs all reported present points with issued figure-star truth. Membership flags do not filter recovery. Empty-denominator conventions are explicitly unofficial.

## Experiments actually executed

- Decoded all 851 supplied PNG files: 19 skies, 784 queries, 48 reference diagrams. No exact decoded-pixel duplicates.
- Extracted white reference nodes automatically and visually inspected the 48-node-overlay contact sheet. Eight templates contain fewer than four nodes; three contain only two.
- A0 translation-only baseline: 0.087 mean, zero localization reward.
- Radial retrieval plus transformed matching: 0.254 mean before geometric identification.
- Harmonic retrieval plus transformed matching: 0.279 mean before geometric identification; localization reward 0.426.
- Full-image rotation/scale proposals, harmonic retrieval, and local ECC alignment: threshold-held-out evaluation reached 0.454 mean before final geometry integration. Its mean localization was 0.650 and recovery 0.611.
- Final geometry compares coarse/refined point sets using affine quadruple hashes, one-to-one verification, refitting, a soft shear penalty, and bounded auxiliary image-star evidence. Final development mean: 0.676.
- Eleven automated correctness tests pass, including synthetic rotated/scaled center recovery, affine/reflection invariants, partial figures, distractor fragments, duplicate serialization, non-transitive grouping and reference-order invariance.

The radial-only top-2,000 candidate coverage within four pixels was 0.741 for Pisces, 0.385 for Scorpius and 0.778 for Taurus. This motivated the slower dense fallback. It is not a final-pipeline proposal-recall claim. The planned 95% proposal-recall engineering target has not been established.

## Oracle diagnostics, not blind results

Using only true figure points, the initial triangle recognizer identifies all three classes. Using all true present coordinates, it identifies Pisces and Scorpius but mistakes Taurus. These probes expose geometric clutter sensitivity; they do not earn blind localization credit. Subsequent variants did not eliminate this failure universally.

## Validation scope

The presence-threshold script performs three leave-one-scene-out fits, selecting a threshold on the other two scenes each time. Final threshold 0.72 is fitted using all three labelled scenes. Iterative method choices used all three scenes, and final geometric integration was not evaluated on untouched held-out scenes. Applying the two-scene thresholds to the frozen final method gives a mean of 0.654 and worst scene 0.418 (`outputs/final_threshold_folds.json`); this isolates threshold holdout, not method-selection holdout. Synthetic correctness fixtures are not evidence of generalization to all 48 real classes.

All final queries are processed without scene-name classification, memorized coordinates, count-based class priors, or manual validation answers. Reference class stems are permitted catalog identifiers. No validation labels or public leaderboard results were used for tuning.

## Material gaps against PLAN.md

This is the first classical submission milestone, **not completion of the full two-person plan**. The independent learned descriptor/HPC pipeline, the full proposed ablation matrix, physical one-source/two-source fitting, all requested distractor-oracle injections, and clean Colab cloud execution remain outstanding. No cluster execution is claimed; cluster account/partition/environment/storage settings were not supplied.

The recognizer currently needs at least four independently supported points. Two- and three-node references do not have a reliable classification path. Its two point-set branches each allow about 50,000 verifications, for about 100,000 combined per scene, exceeding the plan's initial combined budget. Grouping uses a conservative three-pixel anchor rule and preserves all submitted queries, but close-star separation is not backed by a two-source image fit.

The notebook was executed locally in smoke mode. The standalone script independently reran all labelled scenes and reproduced the complete training CSV byte for byte (`outputs/reproduction_check.json`). Execution status, source hashes, runtime and peak-memory records are written into the output manifests. Do not describe a prepared CSV as a successful Kaggle upload until the website confirms it.

## Files

- `outputs/audit/inventory.csv`: decoded image inventory and hashes.
- `outputs/audit/reference_nodes.jpg`: all 48 reference overlays.
- `outputs/proposal_diagnostics.json`: radial proposal diagnostic.
- `outputs/geometry_oracles.json`: initial labelled-coordinate oracles.
- `outputs/hybrid_ecc/calibration.json`: threshold-only holdouts before final geometric integration.
- `outputs/failure_analysis.csv`: query-level final development failure categories.
- `outputs/final_train_cached/metrics.json`: final development metrics from saved blind candidates.
- `outputs/reproduction_train/`: independent standalone-script run.
- `outputs/final_submission/`: frozen validation predictions and submission CSV when complete.

## Submission constraints verified in Safari

The enrolled Somadi account has accepted the competition rules. The rules page lists five daily submissions and two final selections. The overview and data descriptions disagree about the public split; no resolution is assumed. The supplied competition URL is https://www.kaggle.com/competitions/constellation-detection-cs-gy-6643.


## Confirmed Kaggle result

Submitted successfully through Safari as Somadi on September 10, 2026. Kaggle reports **Complete**, with public score **0.58189**. See `outputs/kaggle_submission.json` for source and CSV hashes. No final-selection setting was changed. The live submission UI permits one final selection, while the rules text says two; this discrepancy remains unresolved.

---

# Joint localization and recognition — September 11, 2026

## Measured result

The geometry stage was replaced. Candidate generation (harmonic retrieval, dense
exhaustive proposals, circular-support verification, ECC refinement) is byte-for-byte
unchanged, so every number below is attributable to the final stage alone.

| Basis | Frozen | Joint |
|---|---:|---:|
| Development mean, three labelled scenes | 0.676 | **0.729** |
| Worst development scene | 0.474 | 0.487 |
| Leave-one-scene-out **threshold** holdout mean | 0.654 | **0.702** |
| Synthetic geometry benchmark, identification | 0.125 | **0.474** |

| Scene | Presence | Localization | Recovery | Identification | Total |
|---|---:|---:|---:|---:|---:|
| pisces | 0.669 | 0.778 | 0.800 | 1.000 | 0.823 |
| scorpius | 0.813 | 0.615 | 1.000 | 1.000 | 0.876 |
| taurus | 0.669 | 0.556 | 0.833 | 0.000 | 0.487 |
| mean | 0.717 | 0.650 | 0.878 | 0.667 | **0.729** |

Recovery rose 0.644 to 0.878 and localization fell 0.676 to 0.650. Identification
remains two of three scenes but on different scenes: Scorpius is now correct, which
the frozen method got wrong, and Taurus is now wrong. With three labelled scenes
identification moves in steps of 0.333, so that exchange is not evidence either way;
the synthetic benchmark below is what the identification claim rests on.

Presence is unchanged by construction. It is decided from the same appearance scores
by the same threshold, and 0.72 remains the maximum of a 0.60-0.84 sweep.

## What changed, and why

Four changes, each measured separately:

1. **Verification against ambiguous queries' alternatives.** The frozen recognizer
   consumed one point per query. The correct location is among the 20 retained
   alternatives for 90% of present queries but ranks first for only 61% of *figure*
   queries, and the median appearance gap to the correct alternative is 0.031, so
   appearance cannot separate them. Hypotheses are now verified against a pool that
   includes the alternatives of appearance-ambiguous queries, under one-to-one
   constraints per query group.
2. **Relocation.** The winning fit's chosen alternative replaces the reported
   coordinate. This is what raised recovery.
3. **Triangle-only seeding.** Four-point quad invariants compound the template-to-sky
   model error and performed far worse than triangle invariants (0.344 versus 0.583
   on the benchmark). The frozen pipeline's quad reliance was costing accuracy.
4. **Hypothesis budget raised to 80,000 per point-set branch.** This only matters once
   realistic model error is present; under an idealised error model the budget looked
   irrelevant.

Which queries may be relocated is decided by the rank-0 minus rank-1 appearance gap.
That gap separates correctly located queries (median 0.136, p10 0.027) from mislocated
ones (median 0.009, p90 0.036). Gating at 0.03 was necessary: relocating every query
moved off-figure queries that were already correct to within 1px onto figure nodes
over 1000px away, and cost more localization than it gained in recovery.

## The synthetic geometry benchmark

Identification carries 30% of the score and there are three labelled scenes, so
measuring it there yields a three-valued signal that flipped between classes on
parameter changes of no consequence. `lab/synth.py` synthesises the *candidate lists*
the recognizer receives, which allows hundreds of scenes. Every noise parameter is
measured on the real scenes: the correct-alternative position error, the wrong
alternative distance distribution, per-category rank-0 hit rates, the score and gap
distributions, the issued-node fraction, figure extent, and query counts.

The single most important parameter is the affine template-to-sky residual.
`lab/geomerror.py` fits reference nodes to ground-truth figure points for the true
class and finds a **median residual of about 5px, p90 8-16px** — the reference diagrams
are schematics, not an exact affine image of the sky. The same measurement confirms
affine is the right model: similarity leaves 7.6-25px median residual and loses support
on two of three scenes.

An earlier version of the benchmark assumed roughly 1px residuals. Under that
assumption a likelihood-ratio score and an 8px tolerance looked clearly best
(identification 0.620) and transferred so badly to the real scenes that Pisces broke.
After the measured model error was introduced, the original binomial-surprise score at
tolerance 18 was better than the likelihood score (0.469 versus 0.458), and the tuned
result transferred. This is recorded because the wrong noise model, not the wrong
search, produced the misleading conclusion.

Confirmation on 192 scenes at a seed unused during tuning, templates of at least
seven nodes:

| Configuration | Identification |
|---|---:|
| Frozen recognizer | 0.125 ± 0.024 |
| Joint, tuned | **0.474 ± 0.036** |
| Joint without pool widening | 0.344 ± 0.034 |
| Joint with quad seeding, as frozen | 0.396 ± 0.035 |
| Joint at the frozen 30,000 budget | 0.427 ± 0.036 |

Relocation produced 230 fixes and 0 regressions across those scenes.

The benchmark supplies no auxiliary star map, so it is a lower bound on the deployed
configuration. It also tests geometry in isolation and says nothing about the
appearance stage.

## Diagnostics that redirected the work

- **Ranking, not hypothesis generation, is the limit.** The true class obtains a
  verified four-point fit in every scene — but so do about 40 of the 48 templates.
  A minimum support of four is not discriminative in a cluttered field, and raising it
  only lost accuracy. Effort went into ranking instead of into search.
- **Identification tracks the number of issued figure queries.** With four issued
  figure queries accuracy was 0 of 42; with nine or more it was 0.86-1.00. Small
  references are close to information-theoretically hopeless under a free affine
  transform, consistent with PLAN.md's caution. Reported benchmark figures therefore
  restrict to templates of seven or more nodes; over all templates of four or more the
  tuned recognizer scores 0.292 against 0.115 frozen.
- **Presence has no available headroom.** A fixed 0.72 threshold scores 0.7296 over
  all 116 labelled queries, per-fold refitting scores 0.7100, and the best single
  threshold in-sample reaches only 0.7517. Coarse-score and combined-score rules were
  worse. Presence was left alone rather than tuned into the noise.
- **The winner-to-runner-up score gap is a strong confidence signal.** On 192
  benchmark scenes a gap of at least 2.0 carries 0.97 identification precision and at
  least 3.0 carries 1.00, against 0.47 unconditionally. It is recorded in diagnostics.
  It is not used to gate relocation, because all three labelled scenes exceed the
  threshold including the misidentified one, so the gate is inert where it could be
  checked.
- **The auxiliary star map is informative but already correctly weighted.** Figure
  points score 0.79-0.94 against 0.50 for random locations, and only 10% of random
  locations exceed 0.9. Weights of 3 to 6 form a plateau; larger values degrade.
- **Validation scenes are not denser than the labelled ones.** Patch counts reach 87
  against a maximum of 41 in training, but what drives chance fits is the number of
  queries passing the presence threshold. That median is 21 for validation against 27
  for the labelled scenes, and the 87-patch scene yields a smaller pool than Pisces.
  A synthetic stress test with 77 high-scoring queries collapsed identification to
  0.052, but that regime does not occur in this data.

## Approaches tested and not retained

- Likelihood-ratio verification scoring: better under an idealised error model, worse
  under the measured one.
- A similarity/reflection model branch beside affine: no gain; similarity alone is much
  worse, consistent with the measured residuals.
- Appearance-score and candidate-rank terms in the hypothesis score: completely inert.
- Per-fold presence-threshold refitting, coarse-score presence, and summed coarse plus
  refined presence: all worse than the fixed threshold.
- Budgets above 80,000, more than 8 alternatives per ambiguous query, auxiliary weights
  above 6, and minimum support above 4: no gain or worse.
- Gating relocation on classification confidence: inert on the labelled scenes.

## Verification

- Eleven existing correctness tests pass unchanged.
- The production stage reproduces the fast experiment harness exactly (0.72869 both
  ways), so the tuned configuration is the one that shipped.
- An end-to-end rerun from images reproduces the cached-candidate score exactly,
  confirming candidate generation is unchanged.
- Reference-catalog order and query-processing order both leave the class and every
  relocated coordinate unchanged on all three labelled scenes.

## Remaining gaps

Unchanged from the previous milestone: no learned descriptor or HPC track, no
image-supported one-source/two-source fitting, no Colab cloud execution. Two- and
three-node references still have no reliable classification path, and the benchmark
now quantifies how weak small-reference identification is rather than resolving it.
Localization regressed slightly and its oracle ceiling over the existing candidates is
0.819 against 0.650 achieved, so better appearance ranking remains the largest
untouched opportunity. The combined hypothesis budget is now about 160,000 per scene
across two point-set branches, further above PLAN.md's initial 50,000 target.

---

# Appearance ranking and verification — September 11, 2026 (second pass)

**Outcome: no production change.** The `joint` baseline is retained unchanged at
0.72869 development mean. Four candidate improvements were measured and all four are
rejected end-to-end. The pass produced one structural diagnosis that redirects future
work, one quantified limit on what the labelled set can decide, and a corrected
reading of the localization headroom.

Behaviour is unchanged and verified: with default settings the current source
reproduces the previous development metrics bit for bit (0.7286878605463721) and the
same training CSV (`7cc9d19e...`). The source hash moved to `0b6b7a0b...` because
configuration fields were added and stale defaults corrected, not because predictions
changed.

## Baseline reconciliation

`recognize_joint`'s own default arguments were stale: `cap=60000`,
`models=('affine','similarity')`, `quad_share=0.6`. Production always passed explicit
values so no shipped behaviour was affected, but a direct caller would have silently
got a non-production configuration. Defaults now equal production. Module, harness,
standalone export and notebook were checked at their call sites and agree.

## Corrected reading of the localization headroom

The earlier 0.819 ceiling was computed by a script that also substituted the true
class and true membership, so its `score` column was not comparable to production.
Recomputed under identical presence decisions, with class and membership held at
production values (`lab/oracle_check.py`), the ceiling stands, and the decomposition
is the useful part:

| category | n | rank-0 | production | oracle |
|---|---:|---:|---:|---:|
| figure | 26 | 0.539 | 0.577 | 0.846 |
| off-figure | 45 | 0.778 | **0.711** | 0.822 |
| all (equal-scene) | 71 | 0.676 | 0.650 | 0.819 |

Relocation *helps* figure queries (+0.038) and *hurts* off-figure ones (-0.067). Since
off-figure queries are 45 of the 71-query localization denominator, the net effect is
the observed -0.026. Eight truth-present queries are reported absent and contribute
zero, a fixed 0.113 of the localization denominator.

## The proposal bottleneck is `verify`, not retrieval

Seven of 71 labelled present queries have no retained candidate within 12px.
Attributing each one stage by stage (`lab/missing_attrib.py`) gives a unanimous
answer. For all seven: the truth is not near a border, an index point lies within
0-2.8px of it, it appears in the top-2000 retrieval shortlist at rank 1-173, and it
appears in the proposal union. None survives `verify`'s retained 20.

This removes a large block of planned work. Dense stride 8/4/2, retrieval budgets
200/1000/2000, radial versus harmonic, multiresolution indexes and border-safe
retrieval cannot address these cases, because the correct location already reaches
`verify` with a shallow retrieval rank. Those sweeps are not run.

Measuring verification variants on identical cached proposals
(`lab/verify_variants.py`, 71 present queries) shows the correct proposal is typically
ranked 0-1 *before* suppression, so the loss is tail truncation plus representation:

| verification variant | recall@12 | recall@4 | lost |
|---|---:|---:|---:|
| baseline: 0.6px blur, stride-2 r12 disc, keep 20, nms 8 | 0.901 | 0.873 | 7 |
| DoG 0.8/2.5 | 0.944 | 0.944 | 4 |
| DoG 0.8/4 | 0.915 | 0.915 | 6 |
| keep 40 | **0.958** | 0.944 | 3 |
| DoG, keep 40 | 0.944 | 0.944 | 4 |
| nms 4 | 0.873 | 0.859 | 9 |
| stride-1 disc, r15 disc | 0.915 / 0.901 | | |

Tightening suppression is worse. A finer sample grid and a larger disc do nothing.

## Why the improvements do not transfer

| configuration | recall@12 | presence | localization | recovery | id | total |
|---|---:|---:|---:|---:|---:|---:|
| production | 0.901 | 0.717 | 0.650 | 0.878 | 0.667 | **0.7287** |
| keep 40 | 0.958 | 0.724 | 0.588 | 0.733 | 0.667 | 0.6819 |
| DoG, keep 20 | 0.944 | 0.700 | **0.676** | 0.622 | 0.333 | 0.5659 |
| DoG, keep 40 | 0.944 | 0.687 | 0.663 | 0.689 | 0.333 | 0.5765 |

The ambiguity gate (0.03), the pool margin (0.15) and the presence threshold (0.72)
are calibrated to the exact score distribution `verify` currently emits. Changing the
representation or the retained count changes that distribution, so more queries become
near-tied, more are relocated, and recovery falls even though candidate recall rose.
DoG at keep 20 does deliver the predicted localization gain (0.650 -> 0.676, the best
of any configuration tested) but loses it back through recovery and identification.

## What the labelled set can and cannot decide

Identification contributes 0.30 and takes four values over three scenes, so one scene
changing its class name moves the equal-scene mean by 0.1000. Against that
(`lab/selectivity.py`):

- one query fully repaired is worth 0.0025-0.0037 of localization, **34x smaller**;
- one figure point repaired is worth 0.0083-0.0139 of recovery, **10x smaller**;
- repairing all seven recoverable candidates is worth 0.0200, about **one fifth** of a
  single identification flip.

Three appearance experiments each improved their own target metric and lost on the
total because one scene's class changed. Those losses are therefore not evidence
against the mechanisms. The correct conclusion is that the three-scene total cannot
select appearance changes, and that any change to the appearance stage requires a
recalibration signal the labelled set does not provide.

Direct confirmation that recalibration overfits at this sample size: sweeping the
presence threshold and the ambiguity gate jointly over 24 combinations reaches an
in-sample 0.8031 with all three scenes identified correctly, at (0.74, 0.05) — but the
leave-one-scene-out value of that selection procedure is 0.6863, below the 0.70152
obtained when only the threshold is selected. The apparent optimum is selection noise
and was not adopted.

## Rejected in this pass

| Mechanism | Own-metric result | End-to-end |
|---|---|---|
| Wholesale rescoring, incumbent + 0.5*DoG-32 + 0.5*annulus | rank top1 0.797 -> 0.844, figure 0.609 -> 0.696 | loso 0.5894 |
| Wholesale rescoring, incumbent + 0.25*bgsub-32 | same top1 0.844 | loso 0.5766 |
| Scale-preserving tie-break on ambiguous queries | ambiguous-figure top1 0.375 -> 0.500 with off-figure held | 0.6040 |
| `verify` keep 40 | recall 0.901 -> 0.958 | 0.6819 |
| `verify` DoG | recall 0.901 -> 0.944, localization 0.650 -> 0.676 | 0.5659 |
| Joint (threshold, gap) reselection | in-sample 0.8031 | loso 0.6863 |
| Gradient-magnitude ranking | top1 0.438 | not run |
| Raw correlation ranking (no high-pass) | top1 0.719 | not run |

The tie-break failure is itself diagnostic: `generation_points` seeds hypotheses from
each query's rank-0 alternative, so reordering the 53 ambiguous queries changes the
seed anchor set and degrades hypothesis generation, even though the score multiset and
therefore every threshold decision was preserved exactly.

## Also established

- Full 32x32 correlation support beats the 12px disc the production verifier uses
  (rank top1 0.750 -> 0.781 in isolation). The disc exists so that rotated patch
  samples stay inside the 32x32 array; sampling the scene into the patch frame instead
  removes that constraint. Retained as a finding, not shipped, because it is part of
  the same uncalibratable change.
- The centre/annulus split is the strongest single figure-query representation
  (figure top1 0.696 alone, 0.739 combined) and separates absent queries far better
  (median top score 0.236 against 0.705). It is the natural component to revisit once
  a recalibration signal exists.
- Presence remains at 0.717 in every variant where the score multiset is preserved,
  confirming the mechanism behaved as designed.

## Resumable queue

Ordered. The first item is a prerequisite for most of the rest, because it is what
supplies the missing calibration signal.

1. **Image-level degradation benchmark.** Render synthetic skies with known star
   positions and generate 32x32 queries under measured rotation, scale, blur,
   illumination and noise, so appearance and verification changes can be recalibrated
   on hundreds of scenes instead of three. Keep candidate-coordinate error, schematic
   error, wrong-candidate structure and score distributions as separate knobs, and
   sweep ranges rather than fitting one simulator.
2. With that in place, revisit as a single recalibrated package: DoG or full-32
   verification, keep 20 versus 40, the centre/annulus component, and refitted
   threshold / gap / margin.
3. Relocation safety, still untested: independent pixel reverification of original
   versus proposed location; leave-one-query-out geometric support; stability under
   bounded perturbation; retaining the original coordinate when evidence is
   inconclusive. Target the -0.067 off-figure localization cost while keeping the
   +0.234 recovery gain.
4. Seed anchors independent of appearance rank, given the tie-break diagnosis.
5. Geometry ranking under realistic uncertainty: chance-fit calibration against
   spatially structured clutter and widened pools; null models accounting for template
   size, pool multiplicity and hypothesis-search multiplicity; independent support
   outside seed correspondences; auxiliary evidence by source shape/scale/background.
6. Duplicate and close-source work: adaptive source-size grouping, bounded one-source
   versus two-source profile fitting, and group-formation audit when rank-0 is wrong.
7. Time-boxed untested alternatives: SIFT/RootSIFT, AKAZE/ORB, Fourier-Mellin/log-polar
   plus phase correlation, local star-neighbourhood descriptors for retrieval.
8. Small-reference strategies with genuinely independent evidence. Earlier wording
   claiming information-theoretic impossibility was an overstatement and is withdrawn;
   what is measured is that four issued figure queries gave 0 of 42 correct under this
   recognizer, and two-point free-transform ambiguity is real.

Not attempted, by constraint: any learned model. Not attempted, by choice: tuning
against leaderboard feedback. External catalog plate solving remains conditional on
first establishing competition rule and resource compatibility.

---

# Role separation, relocation guards, failure traces — September 11, 2026 (third pass)

**Outcome: no production change.** The `joint` baseline is retained at 0.72869. Two
mechanisms were implemented and measured; neither improves the weighted objective. The
pass produced a definite causal attribution for the earlier re-ranking failures, a
correction to a previously stated conclusion, and a reusable separation of concerns in
the recognizer.

Predictions are unchanged and verified bit-identical on cached candidates
(0.7286878605463721) and from images.

## Ranking, calibration and seeding are now separate roles

Previously one score per candidate, sorted, drove six decisions at once: the presence
test, the ambiguity test, the pool margin, the physical-star grouping anchor, the
hypothesis seed, and the reported coordinate. That is why no earlier re-ranking result
could be attributed. `constellation/slate.py` introduces `QuerySlate`, which keeps
candidate identity fixed and carries three independently settable roles:

| role | governs |
|---|---|
| `calib` | presence, ambiguity, pool margin |
| `rank` | pool ordering, reported coordinate |
| `seed` | grouping anchor, hypothesis seed |

`recognize_joint` accepts either raw candidate lists (roles collapse onto the single
score, reproducing the shipped pipeline exactly) or slates. `build_pool` gained
`pool_by`: `'rank'` lets a ranking change alter pool membership, `'calib'` fixes
membership and lets only the order change, which is what isolates re-ordering.

Two latent faults were fixed while doing this, neither reachable in the shipped
configuration:
- `build_pool` indexed `groups` by position in the full query list while `groups` was
  built over queries that have candidates. Identical whenever every query has
  candidates, which is currently always.
- The pool margin could discard a query's only contribution once ranking and
  calibration differ, leaving the query invisible to geometry. The primary
  contribution is now always admitted.

## The earlier re-ranking failures were seed disruption

Factorial on identical cached candidates, presence / ambiguity / margin / pool
membership all frozen and asserted equal, ranker = centre-annulus minimum:

| cell | total | loc figure | loc off-figure | recovery | id | pisces true-class rank |
|---|---:|---:|---:|---:|---:|---:|
| A existing seed, existing rank | **0.7287** | 0.577 | 0.711 | 0.878 | 0.667 | 0 |
| B existing seed, improved rank | 0.7211 | 0.577 | 0.644 | 0.878 | 0.667 | 0 |
| C improved seed, existing rank | 0.5987 | 0.500 | 0.711 | 0.778 | 0.333 | 1 |
| D improved seed, improved rank | 0.6020 | 0.538 | 0.644 | 0.811 | 0.333 | 1 |

Changing the seed anchors costs identification: the true class for Pisces falls from
rank 0 to rank 1 among verified hypotheses in both C and D, and never in B. Changing
the ranking alone costs off-figure localization (0.711 to 0.644) and leaves figure
localization and identification untouched.

The reason figure localization does not respond to ranking is that geometry already
relocates most figure queries, so their reported coordinate does not come from the
appearance ranking at all. The ranking only decides coordinates for queries geometry
declines to relocate, and those are predominantly off-figure — precisely where the
incumbent score is the strongest ranker measured (off-figure top-1 0.902 against 0.805
for centre-annulus). Improving figure top-1 therefore cannot help through this path.

This supersedes the earlier suggestion that the tie-break failure was uniformly a
recalibration problem. For ranking it is a seed-policy problem, and for the reported
coordinate it is a genuine category mismatch.

## Relocation guards work as designed and still lose

`constellation/reloc.py` adds guards that decide, per query, whether to adopt the
geometric coordinate. Evidence is independent of the score that selected the
candidates. `GroupHeldOutSupport` refits the winning transform after removing every
correspondence belonging to the query's own physical-star group, so two repeated views
of one star cannot supply each other's evidence. The class hypothesis and the geometric
assignment are computed exactly as production does; only adoption is filtered.
Presence is untouched by construction: a rejected move keeps its own coordinate and
stays present.

| guard | total | loc off-figure | recovery | moves kept | off-figure fix/regress |
|---|---:|---:|---:|---:|---|
| baseline, accept all | **0.7287** | 0.711 | 0.878 | 26 | 0/3 (3 large) |
| independent full-32 | 0.6947 | 0.778 | 0.711 | 10 | 0/0 |
| independent annulus | 0.6947 | 0.778 | 0.711 | 9 | 0/0 |
| independent centre-annulus | 0.6973 | 0.778 | 0.711 | 9 | 0/0 |
| group-held-out support | 0.7204 | 0.711 | 0.844 | 22 | 0/3 (3 large) |
| full-32 + group-held-out | 0.6947 | 0.778 | 0.711 | 8 | 0/0 |

The guards do exactly what they were built for: off-figure localization returns to
0.778, which is the value with no relocation at all, and all three large incorrect
moves disappear. They still lose, because recovery falls further than localization
rises. Sweeping guard strictness over eight margins is monotone and never reaches the
baseline (best 0.7204 at the most permissive setting; leave-one-scene-out over that
grid also 0.7204).

Two reasons, measured:

1. **Most relocations are legitimate.** Of 26 proposed moves, 18 land within 12px of an
   issued figure star, and only 2 are truth-off-figure queries. The three damaging
   moves are a minority side effect, not the main behaviour.
2. **Appearance cannot validate relocations that exist because appearance failed.** A
   relocated figure query is relocated precisely because its appearance ranking was
   wrong, so requiring an independent appearance representation to prefer the geometric
   position discards the good moves along with the bad. This is a structural limit of
   appearance-based guards here, not a tuning shortfall.

Worth stating plainly, because it explains why a guard that clearly improves
localization can still lose: recovery is a greedy one-to-one match of *every*
predicted-present coordinate against the issued figure stars with no membership test,
so moving a query onto a figure node earns recovery even when it ruins that query's own
localization. That follows the documented procedure, and the empty-denominator and
tie-handling conventions remain unofficial.

## Corrected: the seven verification failures are a contrast problem

The previous pass reported that the correct proposal "typically ranks 0-1 before
suppression". That figure was a median over *all* present queries, dominated by
successes, and it does not describe the seven failures. Tracing each one through a
faithful reproduction of `retrieval.verify`, including its ordering and tie handling,
then the real ECC stage (`lab/trace_failures.py`):

| failure | figure? | pre-suppression rank | best in neighbourhood | best overall | proposals |
|---|:--:|---:|---:|---:|---:|
| pisces #2 | yes | 39 | 0.973 | 0.986 | 2138 |
| pisces #20 | yes | 23 | 0.968 | 0.991 | 2110 |
| scorpius #14 | no | 247 | 0.605 | 0.792 | 2357 |
| scorpius #23 | yes | 50 | 0.971 | 0.984 | 2124 |
| scorpius #25 | no | 153 | 0.830 | 0.888 | 2294 |
| taurus #14 | no | 63 | 0.985 | 0.991 | 2307 |
| taurus #33 | no | 149 | 0.937 | 0.965 | 2132 |

- 8px suppression removed **none** of the seven.
- Local pose refinement and ECC removed none; they never received the correct location.
- The correct neighbourhood scores high in absolute terms, 0.83-0.99 for six of seven,
  but is outscored by 23 to 247 other proposals, usually by 0.005 to 0.03.

So this is verification **discrimination**, not retrieval breadth, not suppression, and
not truncation as an independent cause. Truncation at 20 is the mechanical step that
drops them, but only because the score fails to separate the true location from dense
clutter. This also explains why keep-40 recovered exactly four of the seven: the four
with pre-suppression rank at or below 63, after suppression thins the ranks above them.

The remedy has to raise contrast rather than widen the shortlist. The components that
looked strongest in isolation are the ones aimed at contrast: full 32x32 support
(top-1 0.750 to 0.781) and the centre/annulus split (figure top-1 0.696, absent-query
median top score 0.236 against 0.705).

## Ledger corrections

- Full 32x32 support: promising in isolation, **not shipped**.
- Appearance re-ranking and verification attribution: **completed**, not in progress.
- Retrieval-side sweeps: **deprioritized for the diagnosed cases**, on the evidence that
  those seven reach `verify` with shortlist ranks 1-173. Not permanently unnecessary,
  and not established for undiagnosed scenes.
- Recalibration: a **plausible contributor**, and for ranking-versus-seed it is now
  superseded by the factorial. It is not a proven explanation for every failure.
- The earlier claim that the candidate-level benchmark is a lower bound on real
  performance is **withdrawn**: omitting auxiliary evidence does not make a benchmark a
  bound. Broad impossibility language about small references is likewise withdrawn; what
  is measured is 0 of 42 correct at four issued figure queries under this recognizer.

## Remaining limitations

Three labelled scenes, all of which informed development; every fold here is a
development or calibration check, never an untouched generalization estimate. One
identification flip moves the equal-scene mean by 0.10, which is 34 times a
single-query localization repair, so the labelled set still cannot select appearance
changes. Priority 4's image-level benchmark was not built in this pass and remains the
gating item.

## Next experiment, highest value first

1. **Verification contrast at fixed candidate positions.** The traces name the target
   precisely: raise the score of the true neighbourhood relative to 23-247 near-tied
   clutter locations. Score all ~2200 cached proposals per query under full-32 support,
   centre/annulus, and their combinations, and report the *rank of the correct
   neighbourhood* rather than end-to-end score. This needs no new simulator, no
   recalibration, and no downstream change, so it is measurable now and directly
   attacks the 7 of 71 hard floor. Only if a variant moves those ranks inside 20 does
   the recalibrated package of Priority 5 become worth assembling.
2. Image-level degradation benchmark (Priority 4), the gate for anything that changes
   the score distribution.
3. Seed policies that do not depend on appearance rank, given the factorial result.
4. Relocation adoption driven by geometric rather than appearance evidence, since
   `GroupHeldOutSupport` at 0.7204 was the least damaging guard and the only one using
   independent geometry.

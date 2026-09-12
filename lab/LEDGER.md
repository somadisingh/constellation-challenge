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

1. **Verification contrast at fixed candidate positions.** Score all ~2200 cached
   proposals per query under full-32 support, centre/annulus and combinations; report
   the rank of the correct neighbourhood, not end-to-end score. No simulator, no
   recalibration, no downstream change. Directly targets the 7 of 71 hard floor.
2. Image-level degradation benchmark (real crops with synthetic degradation; controlled
   rendered scenes). Gates anything that changes the score distribution. Fresh
   confirmation seeds; seed 99 is historical.
3. Seed policies independent of appearance rank.
4. Relocation adoption on geometric rather than appearance evidence; `GroupHeldOutSupport`
   was the least damaging guard and the only one using independent geometry.
5. Geometry ranking: chance-fit calibration against spatially structured clutter and
   widened pools; null models for template size, pool multiplicity and hypothesis-search
   multiplicity; independent support outside seed correspondences; auxiliary evidence by
   source shape, scale and background.
6. Duplicate and close-source work: adaptive source-size grouping; bounded one-source
   versus two-source profile fitting; group-formation audit when the calibration-best
   candidate is wrong.
7. Time-boxed alternatives: SIFT/RootSIFT, AKAZE/ORB, Fourier-Mellin/log-polar with
   phase correlation, local star-neighbourhood descriptors for retrieval.
8. Small-reference strategies with genuinely independent evidence. Measured: 0 of 42
   correct at four issued figure queries under this recognizer. That is an empirical
   limitation of this recognizer, **not** an impossibility proof.

Not attempted by constraint: any learned model. Not attempted by choice: tuning against
leaderboard feedback, scene-specific behaviour, automatic submission.

# Experiment 1B — corrected learned patch verification

**Verdict: no demonstrated deployable gain.**

The inner-selected recipe scored **0.7214** out of fold, -0.0073 versus C0 and +0.0360 versus C1. These are repeated exploratory evaluations on three development skies, not an untouched test or a guaranteed Kaggle gain.

## End-to-end result

| Arm | Total | vs C0 | Presence | Localization | Recovery | Identification |
|---|---:|---:|---:|---:|---:|---:|
| C0 production | 0.7287 | +0.0000 | 0.717 | 0.650 | 0.878 | 0.667 |
| C1 classical expanded bank | 0.6855 | -0.0432 | 0.794 | 0.698 | 0.589 | 0.667 |
| Historical learned SELECTED | 0.6551 | -0.0736 | 0.848 | 0.618 | 0.478 | 0.667 |
| A | 0.7023 | -0.0264 | 0.878 | 0.650 | 0.611 | 0.667 |
| B | 0.7214 | -0.0073 | 0.906 | 0.724 | 0.600 | 0.667 |
| C | 0.7213 | -0.0073 | 0.871 | 0.699 | 0.656 | 0.667 |
| D | 0.7118 | -0.0168 | 0.853 | 0.743 | 0.600 | 0.667 |
| E | 0.7108 | -0.0179 | 0.872 | 0.672 | 0.633 | 0.667 |
| Inner-selected recipe | 0.7214 | -0.0073 | 0.906 | 0.724 | 0.600 | 0.667 |
| Selected, seed 31005 | 0.7196 | -0.0091 | 0.906 | 0.743 | 0.578 | 0.667 |

### Per-scene selected result

| Held-out sky | Arm | Step | Total | Presence | Localization | Recovery | Identification |
|---|---|---:|---:|---:|---:|---:|---:|
| pisces | B | 1000 | 0.9607 | 0.972 | 0.963 | 0.900 | 1.000 |
| scorpius | B | 500 | 0.7618 | 0.924 | 0.654 | 0.400 | 1.000 |
| taurus | B | 500 | 0.4418 | 0.823 | 0.556 | 0.500 | 0.000 |

### Every arm by held-out scene

| Scene | Arm | Total | Presence | Localization | Recovery | Identification |
|---|---|---:|---:|---:|---:|---:|
| pisces | C0 | 0.8229 | 0.669 | 0.778 | 0.800 | 1.000 |
| pisces | C1 | 0.8027 | 0.718 | 0.741 | 0.700 | 1.000 |
| pisces | A | 0.9321 | 0.917 | 0.889 | 0.900 | 1.000 |
| pisces | B | 0.9607 | 0.972 | 0.963 | 0.900 | 1.000 |
| pisces | C | 0.9386 | 0.914 | 0.926 | 0.900 | 1.000 |
| pisces | D | 0.9535 | 0.944 | 0.963 | 0.900 | 1.000 |
| pisces | E | 0.9321 | 0.917 | 0.889 | 0.900 | 1.000 |
| scorpius | C0 | 0.8764 | 0.813 | 0.615 | 1.000 | 1.000 |
| scorpius | C1 | 0.7269 | 0.846 | 0.577 | 0.400 | 1.000 |
| scorpius | A | 0.8041 | 0.924 | 0.615 | 0.600 | 1.000 |
| scorpius | B | 0.7618 | 0.924 | 0.654 | 0.400 | 1.000 |
| scorpius | C | 0.7419 | 0.875 | 0.615 | 0.400 | 1.000 |
| scorpius | D | 0.7436 | 0.851 | 0.654 | 0.400 | 1.000 |
| scorpius | E | 0.7362 | 0.875 | 0.462 | 0.500 | 1.000 |
| taurus | C0 | 0.4868 | 0.669 | 0.556 | 0.833 | 0.000 |
| taurus | C1 | 0.5267 | 0.818 | 0.778 | 0.667 | 0.000 |
| taurus | A | 0.3707 | 0.794 | 0.444 | 0.333 | 0.000 |
| taurus | B | 0.4418 | 0.823 | 0.556 | 0.500 | 0.000 |
| taurus | C | 0.4835 | 0.823 | 0.556 | 0.667 | 0.000 |
| taurus | D | 0.4384 | 0.765 | 0.611 | 0.500 | 0.000 |
| taurus | E | 0.4641 | 0.823 | 0.667 | 0.500 | 0.000 |

## What was corrected and measured

- 200 fixed tensor queries per sky, 400 allowed per fold; source manifests held about20k but gradients used only cached queries.
- Legacy sampler regeneration: none. Query degradation, positive and warmup positive tensors were fixed.
- Legacy negative refresh only re-scores existing candidate tensors.
- Legacy hard=False uniformly chooses cached candidates (including mined candidates), while triplet_loss still mines hardest row/column in batch.
- alignment_failed_all means no usable finite score among candidates; it does not certify correct-candidate pose quality.
- I>200 is bright-pixel fraction, not clipping. True clipping is I==255.
- Aggregate contrast/brightness means do not establish per-query bimodality.
- Legacy HardNet screened .6412 is about.0443 below C1; selected .6551 is about.0304 below C1.

The exact legacy reproduction again reached zero active loss by step 500 and kept it through step 2,000. A–E use an explicit policy without hidden in-batch hardest-negative selection. The historical reproduction remains separate because A also fixes the discovered 0–255 versus 0–1 cached-positive input inconsistency.

### Actual training coverage

| Fold | Arm | Unique centres | Unique generated queries | Replay | Nonzero-loss queries | Pool new locations |
|---|---|---:|---:|---:|---:|---:|
| pisces | A | 400 | 400 | 0.000 | 370 | 2004 |
| pisces | B | 32825 | 96224 | 0.248 | 18780 | 2004 |
| pisces | C | 400 | 400 | 0.000 | 328 | 2004 |
| pisces | D | 7712 | 96230 | 0.248 | 10206 | 2004 |
| pisces | E | 7712 | 96230 | 0.248 | 872 | 2004 |
| scorpius | A | 400 | 400 | 0.000 | 374 | 2004 |
| scorpius | B | 32538 | 96243 | 0.248 | 15723 | 2004 |
| scorpius | C | 400 | 400 | 0.000 | 363 | 2004 |
| scorpius | D | 8340 | 96254 | 0.248 | 10209 | 2004 |
| scorpius | E | 8340 | 96254 | 0.248 | 635 | 2004 |
| taurus | A | 400 | 400 | 0.000 | 374 | 2002 |
| taurus | B | 32928 | 96238 | 0.248 | 15472 | 2002 |
| taurus | C | 400 | 400 | 0.000 | 363 | 2002 |
| taurus | D | 10195 | 96254 | 0.248 | 11473 | 2002 |
| taurus | E | 10195 | 96254 | 0.248 | 841 | 2002 |

## Realism audit

Source weighting changed which real FIT locations were sampled; it did not edit query histograms. The original degradation laws remained unchanged because paired affine residuals contain alignment and clipping error and did not justify an arbitrary brightness law.

| Fold | Original query std | Corrected query std | Real query std | Original mean | Corrected mean | Real mean |
|---|---:|---:|---:|---:|---:|---:|
| pisces | 15.69 | 30.15 | 33.15 | 65.59 | 116.18 | 125.18 |
| scorpius | 18.04 | 30.66 | 33.31 | 59.21 | 88.49 | 104.89 |
| taurus | 17.95 | 29.83 | 31.71 | 75.52 | 95.19 | 103.27 |

Each fold has fixed-scale contact sheets at `outputs/exp1b/folds/<fold>/synthesis_contact.png`. Bright fraction (`I > 200`) and clipping (`I == 255`) are stored separately in `appearance_audit.json`.

## Hypothesis decisions

- **Fixed Data Exhaustion:** supported.
- **Appearance Mismatch:** correction narrowed measured contrast gap.
- **Hard Negative Value:** hard policy helped.

## Alignment audit

The post-freeze audit identifies the nearest retrieved candidate for each truly present held-out query, records whether it is within 12px and has a valid pose, its NCC and fitted residual, then tries a finer diagnostic pose grid. No diagnostic coordinate or pose changes a scored prediction. `alignment_failed_all` only means every bank score was invalid; it never proved the true candidate was aligned accurately.

- pisces: correct candidate 27/27; valid pose 27/27; finer pose changed descriptor rank for 6 auditable queries.
- scorpius: correct candidate 25/26; valid pose 25/26; finer pose changed descriptor rank for 11 auditable queries.
- taurus: correct candidate 17/18; valid pose 17/18; finer pose changed descriptor rank for 8 auditable queries.

## Selected-arm error changes and strata

| Scene | vs C0 presence fixes/regressions | vs C0 localization fixes/regressions | Figure reward | Off-figure reward | Absent count |
|---|---:|---:|---:|---:|---:|
| pisces | 9/0 | 5/0 | 0.900 | 1.000 | 14 |
| scorpius | 6/2 | 5/4 | 0.300 | 0.938 | 15 |
| taurus | 7/2 | 2/2 | 0.500 | 0.833 | 16 |

Exact query IDs, coordinates, probabilities, fixed/regressed labels and rewards are stored in each fold's `evaluation.json`; the report does not truncate that machine-readable evidence.

## Promotion decision

Selected gate pass: **False**. Checks: total_gain_at_least_0.01=False, beats_C1=True, components_not_below_C0=False, no_scene_regression_above_0.02=False, second_seed_stable=False.

Production remains unchanged. No validation submission or Kaggle upload was produced. Better-than-C1 performance without passing the C0 gates would be matcher evidence only.

## Environment and runtime

Python 3.12.13, PyTorch 2.14.0, Kornia 0.8.3, device MPS available=True. The 18 completed matrix/repeat runs recorded 1.66 aggregate training wall-hours; selected-arm real-query inference recorded 0.76s across the three fold evaluations. Every run records wall time, generation/mining/gradient time, MPS memory, config/source hashes and immutable checkpoints. Only one training process ran.

HardNet initialization: `cnn/hardnet-hf/checkpoint_liberty_with_aug.pth`, SHA256 `1e9a41b19f1dc93c986e91df9aaf5696d1a777ac1d67498492856a65d6f49c16`, strict load verified by Experiment 1 (digest match=True). Upstream source and license are preserved in `outputs/exp1/models.json`; Experiment 1B downloaded no weights.

Integrity checks: all required arms complete; D/E anchor streams byte-identical in every fold; all common-step checkpoints present; 330 protected production/Experiment 1 files unchanged.

Learning curves: `outputs/exp1b/panels/inner_learning_curves.png`.

## Reproduction

See `experiments/exp1b/README.md` for the fixed protocol and commands. Tests cover fresh generation, source coverage, fold isolation, footprint validity, physical negative separation, refresh turnover, genuinely random selection/loss, pose parity, alignment semantics, unit input range and complete-fold aggregation.

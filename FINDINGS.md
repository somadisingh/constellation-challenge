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

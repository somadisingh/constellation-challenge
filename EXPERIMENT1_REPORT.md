# Experiment 1 — learned verification of real sky patches

**Verdict: no demonstrated gain.**

Reproduced C0 (production, `outputs/joint_train/metrics.json`) total **0.7286879**, unchanged. C1, the same classical masked NCC on the expanded bank under the same integration, scores **0.6855**. The selected learned recipe scores **0.6551**.

Three things carry the result, and they separate cleanly.

1. **The integration itself costs recovery, before any learning.** C1 contains no learned component, yet it loses 0.0432 to C0 and its recovery falls from 0.878 to 0.589. Exp1 deliberately lets the matcher pick the coordinate and forbids snapping back to the classical geometry, which removes the relocation step that produced C0's recovery. That cost is charged to every arm equally and is not evidence about descriptors.
2. **Learning is real but does not reach the classical criterion.** Fine-tuning lifts HardNet from 0.5561 (pretrained) to 0.6412, a large gain over off-the-shelf descriptors. It still ends 0.0304 below C1. Presence improves markedly (0.848 against C0's 0.717); localization and recovery do not.
3. **Inner-validation selection did not transfer.** The learned arm beat C1 on inner validation in two folds but on the held-out sky in only one. The realism audit below shows why: the generated queries are far lower contrast than the real ones, so the selection set is not the test distribution.

No arm passed the predeclared gates, so production is retained and no submission candidate was produced. Per plan §12 a failed experiment is a valid outcome; three scenes could not have established a hidden-set gain even had the gates passed.

Out-of-fold over all three held-out skies (116 queries). Each scene is scored by the fold that held it out.

| arm | mean total | gain vs C0 | gain vs C1 | worst scene | presence | localization | recovery | ident | gates |
|---|---|---|---|---|---|---|---|---|---|
| **C0 (production)** | 0.7287 | +0.0000 | +0.0432 | 0.4868 | 0.717 | 0.650 | 0.878 | 0.667 | reference |
| C1 | 0.6855 | -0.0432 | +0.0000 | 0.5267 | 0.794 | 0.698 | 0.589 | 0.667 | fail |
| SELECTED | 0.6551 | -0.0736 | -0.0304 | 0.3963 | 0.848 | 0.618 | 0.478 | 0.667 | fail |
| T:hardnet_screen_s31004 | 0.6412 | -0.0875 | -0.0443 | 0.3707 | 0.807 | 0.600 | 0.478 | 0.667 | fail |
| T:hynet_screen_s31004 | 0.6364 | -0.0923 | -0.0491 | 0.3903 | 0.858 | 0.596 | 0.411 | 0.667 | fail |
| P-S | 0.5670 | -0.1617 | -0.1185 | 0.2500 | 0.712 | 0.486 | 0.367 | 0.667 | fail |
| P-H | 0.5561 | -0.1726 | -0.1294 | 0.2500 | 0.754 | 0.462 | 0.300 | 0.667 | fail |
| P-Y | 0.5256 | -0.2031 | -0.1599 | 0.2421 | 0.776 | 0.450 | 0.167 | 0.667 | fail |

The following arms exist on a scene SUBSET only, because the selected backbone and whether the continuation ran are both fold decisions. Their means are NOT comparable with the table above and no gate is applied to them: C0 per-scene totals span 0.487 to 0.876, so covering only the easy sky inflates a subset mean.

| arm | folds present | subset mean | note |
|---|---|---|---|
| T:hardnet_continue_s31004 | pisces | 0.8858 | not evaluable |
| H:hardnet_continue_s31004 | pisces | 0.8210 | not evaluable |
| T:hardnet_randomneg_s31004 | pisces, scorpius | 0.8022 | not evaluable |
| T:hardnet_scratch_s31004 | pisces, scorpius | 0.7739 | not evaluable |
| H:hardnet_screen_s31004 | scorpius | 0.6832 | not evaluable |
| T:hynet_scratch_s31004 | taurus | 0.5918 | not evaluable |
| T:hynet_continue_s31004 | taurus | 0.3963 | not evaluable |
| H:hynet_continue_s31004 | taurus | 0.3852 | not evaluable |
| T:hynet_randomneg_s31004 | taurus | 0.3707 | not evaluable |

## 1. Environment

- Apple M4 Pro, arm64, macOS 27.0, 24.0 GiB RAM, 309 GiB free
- Python 3.12.13, torch 2.14.0, kornia 0.8.3, numpy 2.5.3, scipy 1.18.1, opencv 5.0.0
- MPS built True, available True; package lock `requirements-exp1-lock.txt`
- hardnet: CPU/MPS informative cosine 1.000000, max abs 9.83e-07, rank agreement 1.00, 44.3 steps/s at microbatch 64
- hynet: CPU/MPS informative cosine 1.000000, max abs 1.36e-06, rank agreement 1.00, 15.1 steps/s at microbatch 64
- sosnet: CPU/MPS informative cosine 1.000000, max abs 1.04e-06, rank agreement 1.00, 42.0 steps/s at microbatch 64
- CPU fallback: float64 gradient-norm accumulation (MPS does not implement float64); diagnostic readout only; training math stays FP32 on device

## 2. Models and what actually trained

- **hardnet**: `cnn/hardnet-hf/checkpoint_liberty_with_aug.pth`, sha256 `1e9a41b19f1dc93c986e91df9aaf5696d1a777ac1d67498492856a65d6f49c16`, source https://github.com/DagnyT/hardnet (mirror: https://huggingface.co/kornia/hardnet), licence HardNet author repository: MIT licence
- **hynet**: `cnn/hynet-hf/HyNet_LIB.pth`, sha256 `d92559e90b8b228e7ed121935f64d2629a015c2bf9d7d7ee14b43798c5a2af13`, source https://github.com/yuruntian/HyNet (mirror: https://huggingface.co/kornia/hynet), licence HyNet author repository: MIT licence
- **sosnet**: `cnn/sosnet-local/sosnet_32x32_liberty.pth`, sha256 `baa575c55a5d1b7b140206ece4af56b735395409f2bb3c924069faf6a7ededa0`, source https://github.com/yuruntian/SOSNet/raw/master/sosnet-weights/sosnet_32x32_liberty.pth, licence SOSNet author repository: BSD-3-Clause
- Loss is the Exp1 symmetric hard-negative triplet, not the original HardNet/HyNet/SOSNet recipe. Results are labelled "backbone fine-tuned with the Exp1 loss".

## 3. Folds, sources and split tests

- Partition geometry check: ok=True, footprint radius 59.7px inside a 72px margin
- fold **pisces** (held out pisces): scorpius: fit 20004, val 2000, synthcal 1000, taurus: fit 20004, val 2000, synthcal 1000
- fold **scorpius** (held out scorpius): pisces: fit 20004, val 2000, synthcal 1000, taurus: fit 20004, val 2000, synthcal 1000
- fold **taurus** (held out taurus): pisces: fit 20004, val 2000, synthcal 1000, scorpius: fit 20004, val 2000, synthcal 1000
- mined pisces: 200 queries, recall@12 0.925, excluded ambiguity 0.0073, degenerate 0.080
- mined scorpius: 200 queries, recall@12 0.835, excluded ambiguity 0.0045, degenerate 0.085
- mined taurus: 200 queries, recall@12 0.860, excluded ambiguity 0.0050, degenerate 0.110

## 4. Run matrix

Full ledger in `runs.jsonl`; learning curves in `folds/<fold>/checkpoints/*.json` and `panels/curves_*.png`. Selection used each fold's inner validation only; outer labels were opened after the whole recipe was frozen.

**Seed 31004** (equal-source-sky top1 localization reward on inner validation):

| fold | C1 | P-H | P-Y | P-S | screen T-H | screen T-Y | selected | beats C1 | continued | random-neg | scratch | pair head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pisces | 0.8125 | 0.7881 | 0.8281 | 0.7600 | 0.8484 | 0.8406 | hardnet | yes | 0.8562@500 | 0.8562 | 0.8203 | keep descriptor distance |
| scorpius | 0.8906 | 0.8423 | 0.8594 | 0.8111 | 0.8906 | 0.8906 | hardnet | no | skipped (gate) | 0.8828 | 0.8516 | adopt pair head |
| taurus | 0.8125 | 0.7812 | 0.8010 | 0.7734 | 0.8203 | 0.8401 | hynet | yes | 0.8635@2000 | 0.8401 | 0.8516 | keep descriptor distance |

**Seed 31005** (equal-source-sky top1 localization reward on inner validation):

| fold | C1 | P-H | P-Y | P-S | screen T-H | screen T-Y | selected | beats C1 | continued | random-neg | scratch | pair head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pisces | 0.8125 | 0.7881 | 0.8281 | 0.7600 | 0.8484 | 0.8406 | hardnet | yes | 0.8484@2000 | 0.8484 | 0.8281 | adopt pair head |
| scorpius | 0.8906 | 0.8423 | 0.8594 | 0.8111 | 0.8906 | 0.8906 | hardnet | no | skipped (gate) | 0.8906 | 0.8670 | keep descriptor distance |
| taurus | 0.8125 | 0.7812 | 0.8010 | 0.7734 | 0.8281 | 0.8401 | hynet | yes | 0.8594@2000 | 0.8323 | 0.8438 | keep descriptor distance |

Ablation readings, all on inner validation:

- **Pretrained vs fine-tuned**: fine-tuning improves every backbone (out-of-fold P-H 0.5561 to T-H 0.6412), so the training loop does learn something real.
- **Hard vs random negatives at an equal 2,000-step budget**: the random-negative control matches the hard-negative run in every fold. The mined classical confusers add no measurable value at this budget.
- **Saturation**: HardNet drives the fraction of active triplets to 0.000 within ~750 steps (d_pos 0.48, d_neg 1.13, margin 0.5), so later steps produce no gradient; the continuation early-stopped at 3,500. HyNet keeps 0.39-0.80 of triplets active and keeps improving, which is why it was selected in fold taurus.
- **Scratch initialisation** at the same budget reaches 0.8203-0.8516, beating screened HardNet in fold taurus, so natural-image pretraining is helpful but not decisive here.
- **Estimated vs oracle pose** (diagnostic): the image-estimated pose never loses to the known synthetic pose (0.9732/0.9643, 0.9828/0.9828, 0.9910/0.9730). Pose estimation is not a bottleneck.
- **Pair head vs descriptor distance**: adopted in 1 of 3 folds; the plan's fallback to descriptor distance applied in the other two.

## 5. Metrics

Inner-validation controls (equal-source-sky top1 localization reward):

| fold | C1 | P-H | P-S | P-Y |
|---|---|---|---|---|
| pisces | 0.8125 | 0.7881 | 0.7600 | 0.8281 |
| scorpius | 0.8906 | 0.8423 | 0.8111 | 0.8594 |
| taurus | 0.8125 | 0.7812 | 0.7734 | 0.8010 |

Out-of-fold per-scene totals are in `evaluate_s*.json` under `oof`.

### Inner-validation to out-of-fold transfer

Selection used inner validation only, as required. This table shows whether that selection predicted the held-out outcome.

| fold | held out | inner: SELECTED | inner: C1 | inner says | outer: SELECTED | outer: C1 | outer says | agrees |
|---|---|---|---|---|---|---|---|---|
| pisces | pisces | 0.8562 | 0.8125 | learned better | 0.8858 | 0.8027 | learned better | yes |
| scorpius | scorpius | 0.8906 | 0.8906 | C1 at least equal | 0.6832 | 0.7269 | C1 better | yes |
| taurus | taurus | 0.8635 | 0.8125 | learned better | 0.3963 | 0.5267 | C1 better | **no** |

Inner validation predicted the held-out direction in 2 of 3 folds.

### Realism audit of the generated queries

Plan §6 requires a contact sheet and warns that brightness matching alone is insufficient evidence. Measured over all 600 generated and all 116 real queries:

| set | n | mean | within-patch std | saturated frac (>200) | peak minus background |
|---|---|---|---|---|---|
| REAL queries (116) | 116 | 110.9 | 32.7 | 0.177 | 117.6 |
| GENERATED all | 600 | 66.8 | 17.2 | 0.021 | 74.8 |
| GENERATED background | 126 | 106.2 | 19.6 | 0.037 | 67.2 |
| GENERATED peak | 252 | 57.0 | 16.3 | 0.016 | 75.3 |
| GENERATED uniform | 222 | 55.5 | 17.0 | 0.016 | 78.7 |

The generated queries are darker on average AND markedly lower contrast: within-patch standard deviation 17.2 against 32.7, saturated fraction 0.021 against 0.177, and peak-minus-background 74.8 against 117.6. Real queries are strongly bimodal, a near-black background with a saturated stellar core; no source class in the plan's 50/25/25 mixture reproduces that contrast. This is the leading explanation for the transfer failure above, and it is a property of the augmentation law and source mixture, not of the backbones.

### Candidate bank ceiling

A reranker cannot recover a location absent from its bank, so this bounds every arm above.

| scene | bank size (mean) | union @4 | union @12 | fixed_coarse @12 | fixed_ecc @12 | adaptive @12 |
|---|---|---|---|---|---|---|
| pisces | 47.3 | 1.000 | 1.000 | 0.926 | 0.926 | 1.000 |
| scorpius | 45.8 | 0.962 | 0.962 | 0.885 | 0.885 | 0.962 |
| taurus | 45.2 | 0.889 | 0.944 | 0.889 | 0.889 | 0.944 |

The union equals the adaptive branch in every scene: the two fixed branches contribute candidates but no additional truth hits. Bank entries trace entirely to the label-free branch caches (provenance-completeness check), so no true position was injected.

### Where the learned arm loses, by stratum

Held-out sky only, SELECTED against C1 on identical banks and poses.

| held-out sky | stratum | n | C1 top1 | SELECTED top1 | pool missing | alignment failures |
|---|---|---|---|---|---|---|
| pisces | figure | 10 | 0.700 | 0.800 | 0 | 0 |
| pisces | offfigure | 17 | 1.000 | 0.882 | 0 | 0 |
| pisces | absent | 14 | - | - | 0 | 0 |
| scorpius | figure | 10 | 0.300 | 0.200 | 0 | 0 |
| scorpius | offfigure | 16 | 0.938 | 0.812 | 1 | 0 |
| scorpius | absent | 15 | - | - | 0 | 0 |
| taurus | figure | 6 | 0.667 | 0.333 | 0 | 0 |
| taurus | offfigure | 12 | 0.833 | 0.750 | 1 | 0 |
| taurus | absent | 16 | - | - | 0 | 0 |

Figure queries drive recovery, which carries 0.25 of the weighted score. The learned arm ranks figure queries worse than the classical criterion in two of three skies, and off-figure top1 is lower everywhere. There were ZERO alignment failures across all 116 queries and at most one pool-missing query per scene, so the loss is ranking, not alignment and not bank coverage.

## 6. Diagnostics

- 459 real-query panels across 29 fold/arm combinations, covering fixes and regressions against C1, false present, false absent, pool-missing and alignment-failure cases: `panels/panels_*.png`
- Contact sheets comparing generated with ALLOWED real queries (held-out queries never shown): `panels/contact_sheet_*.png`
- Learning curves and inner-metric traces: `panels/curves_*.png`
- Per-query records, strata breakdowns, pool-missing and alignment failures: `folds/<fold>/evaluate_s*.json`
- Panels are post-evaluation diagnostics and were never used to override a prediction, revise a threshold or reselect a checkpoint.

## 7. Decision and reproduction

- Predeclared gates: mean gain >= 0.01 over C0, positive over C1, presence/localization/recovery not below C0, no per-scene regression beyond 0.02, stable direction across seeds.
- Passing arms: none

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-exp1/bin/python -m experiments.exp1 preflight --output outputs/exp1
.venv-exp1/bin/python -m experiments.exp1 download-models --output outputs/exp1
caffeinate -i .venv-exp1/bin/python -m experiments.exp1 run-all \
  --data . --output outputs/exp1 --device mps --resume
```

Code hash `7ee7d8ca168e326f57dede4061c6a253b175b3908cd7edfc1d0c5e190c1f2ce1`.

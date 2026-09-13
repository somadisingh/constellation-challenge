# Workstream 3: cross-branch constellation ranking

Scope follows the latest FINDINGS priority: exploit coarse/refined agreement as a
potential additional identification signal. The two branches share imagery, proposals,
and algorithms and are correlated; agreement is not independent confirmation.

`branch_agreement.py` fits each actual candidate branch with the production settings
(80,000 hypotheses, affine, 18px tolerance, gap .03, top-k 8, auxiliary weight 3),
retaining all 48 class hypotheses. It caches fits once, then compares:

1. Baseline maximum raw score (production tie order).
2. Refined branch alone.
3. Coarse branch alone.
4. Reciprocal rank fusion with denominator 3 + rank.
5. Mean within-branch score regret (relative to its top class).
6. Worst within-branch regret.
7. A one-point score bonus when both branches have a verified class fit.
8. A two-point version of that bonus.
9. A one-point bonus only if corresponding mapped reference nodes also agree within
   18px median distance.

Absent class ranks are 49 and missing regrets are 20. These are fixed engineering
choices, not fitted probabilities. No verified fit yields unknown. Fusion selects a
class, then its highest raw-score branch supplies a concrete existing fit. Relocation
is reconstructed from that fit's existing query-group assignments; no invented
coordinates or oracle candidate insertion. Presence threshold remains .72.

Each method is scored with full relocation and as an identification-only diagnostic
that freezes baseline patch outputs (including membership). The latter isolates the
name contribution; it is not a complete membership policy for deployment.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m lab.branch_agreement \
  --output outputs/lab/workstream3/train
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m lab.branch_agreement \
  --dataset outputs/imagebench/v2 --limit 6 \
  --output outputs/lab/workstream3/v2_development
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m lab.branch_agreement \
  --dataset outputs/imagebench/v1 --limit 6 \
  --output outputs/lab/workstream3/v1_development
.venv/bin/python -m lab.workstream3_report
```

The six-scene synthetic screen is a bounded pilot in manifest order, not all-reference
validation. v1/v2 scores are sensitivity checks on different images. No confirmation
access is provided by this experimental runner. Frozen policy order favours baseline
in ties. Three-scene leave-one-scene-out selection is reported with its small-sample
limitation. All algorithms run without labels; labels enter only after fitting.

No production defaults, runnable submission, or leaderboard predictions are changed
by this laboratory experiment. Promotion needs supported gains without material losses
across components, a frozen calibration comparison, and stronger coverage than a pilot.

## Validation protocol and labelled check

All nine rules were fixed before the synthetic screens. Select the highest v2
**development** mean weighted score, breaking ties in listed policy order; inspect
that fixed choice on v2 calibration. Do not select from calibration or confirmation.
Calibration evaluates the same preset rules for transparent diagnostics, but is not
another tuning grid. Require no component regression on labelled data before even
considering a broader promotion experiment.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m lab.branch_agreement \
  --dataset outputs/imagebench/v2 --split calibration --limit 6 \
  --output outputs/lab/workstream3/v2_calibration
```

The experiment's baseline exactly reproduces all coordinates, absence decisions,
memberships and class names of the stored three-scene production predictions.
Its mean weighted score is 0.7286878605463721. Refined-only scores .688018,
coarse-only .607083, reciprocal-rank fusion .581157, and mean/worst regret .711095.
Agreement bonuses and spatial stability tie production on these three scenes.
70 unit tests pass, including fusion ties, single/missing branches, spatial disagreement,
and isolation of the identification-only diagnostic.

## Interpreting failures

Agreement that a class merely has a verified fit can be weak when most references
admit a clutter fit. The spatial variant additionally compares the same reference-node
positions across the two actual fits. Even spatial agreement can be a correlated
failure because both branches originate from the same image and proposals.

If the true class is absent from both retained fitted slates, fusion cannot recover it.
If it appears in the slates but neither stable nor rank fusion separates it, further
score mixing is not demonstrated to help. The next distinct evidence would be support
outside the correspondences that generated a transform, followed by source-quality
measurements at predicted unmatched nodes. Those require separate controlled ablations.

## Completed decision

All 3 labelled + 18 synthetic scene runs completed. The frozen refined-only choice
scores .314888 on v2 calibration versus baseline .282110, with 0/6 identification
for both. Its gain is localization/recovery, while real labelled score falls from
.728688 to .688018. Coarse-only scores .339221 with 1/6 identification on calibration,
but was not the development selection and regresses substantially on labelled data.
Do not select it retrospectively. Leave-one-real-scene-out chooses baseline each time.

Retain production. All shared-fit/spatial bonuses tie baseline across the evaluated
sets. This rules out adoption of these tested configurations, not all consensus methods.
Full tables, selection, protocol, per-scene fits and true-class ranks are in ignored
`outputs/lab/workstream3/`; summary and record copies are exported to deliverables.
The next distinct signals are support beyond seed correspondences and source-quality
measurements. Confirmation remains untouched.

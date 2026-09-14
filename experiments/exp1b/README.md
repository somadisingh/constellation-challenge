# Experiment 1B execution protocol

Controlled HardNet follow-up. Production and Experiment1 are read-only. Results live in `outputs/exp1b/`; the final report is `EXPERIMENT1B_REPORT.md`. No Kaggle submission or geometry change is performed.

## Comparisons

- **Legacy reproduction:** exact original HardNet 2,000-step run for the pisces holdout. Keep the original fixed tensors, negative policy, seed derivation and historical inner bank; redirect output. This is not arm A.
- **A:** 200 original cached queries per allowed sky. Normalize legacy cached positive crops to [0,1], fixing the verified unit mismatch.
- **B:** original source-stratum sampling, freshly regenerated pairs, bounded replay.
- **C:** 200 corrected-source cached queries per allowed sky.
- **D:** corrected source sampling, freshly regenerated pairs, bounded replay.
- **E:** identical anchors/positives/replay as D; genuinely random negative selection/loss.

A–E: native pretrained HardNet, 2,000 optimizer steps, 64 anchors, AdamW 1e-4, weight decay1e-4, 200-step LR warmup/cosine to1e-6, frozen BN statistics, FP32, margin0.5, gradient clip5. Seed31004. Common checkpoint steps500/1000/1500/2000. No separate negative warmup in these new arms. No in-batch mining. The legacy reproduction retains its original negative warmup/mining.

## Source and split controls

Use the original fold manifests and spatial partitions. A held-out sky never contributes to training, mining, augmentation fitting, inner selection or calibration. Real queries from the two allowed skies can support diagnostics, source-distribution fitting and calibration, never gradient updates. Validation competition scenes and imagebench are unused.

The original cached training set balanced source-class/stratum buckets rather than exactly following the advertised50/25/25 bank mixture. B reproduces that effective original sampling law. Corrected sampling uses paired real parent features (mean, standard deviation, core-minus-annulus, source width), spreads each allowed parent's mass over128 nearby FIT sources, and mixes5% uniform mass. Every view uses real parent pixels. Original degradation ranges are retained because paired residuals mix radiometric and alignment errors. No arbitrary brightness offsets or per-query histogram matching.

Fresh arms use75% fresh anchors and25% replay after the first16 steps, before which no replay is available. Replay holds at most256 queries per sky, selected by classical false-match NCC only, with insertion windows of16 steps. Thus D/E remain exactly paired despite different learned scores. Physical-source usage and regenerated query hashes are counted separately; nonzero loss participation is recorded.

## Bounded mining

Every250 steps draw128 fresh FIT source locations per sky, uniformly over the source manifest. Cache36 initial poses per location (angles0:30:330, scales.85/1/1.18), then encode with the current model. This is a bounded classical NCC search pool, not exhaustive production harmonic retrieval; hard-negative conclusions apply to this regime.

For each anchor, compare to the whole permitted-sky pool. A–D take one highest-NCC location, one highest-network-similarity distinct location and one uniform distinct location. All three receive the same nine-trial image-estimated local pose adapter. Loss takes their nearest descriptor negative. E instead takes three distinct uniform locations and uses a uniform one in the loss, independent of scores. Both arms evaluate three final negatives, with no hidden hardest-in-batch selection. Shared NCC alignment of a random location does not make its location selection score-dependent.

Exclusion radius72px exceeds the original36px ambiguity radius and excludes overlapping32px patch supports at the supported scales. Duplicate physical locations cannot fill multiple negative slots. Mining logs record source IDs, new locations and Jaccard overlap after each refresh. Network descriptors are deliberately stale between250-step refreshes.

The faster local pose implementation reuses identical per-scale prefilters. Tests assert the same selected pose, score and invalid-pose count as the reference, including boundary cases.

## Selection and evaluation

**Primary selection:** one newly generated, corrected, fixed inner bank with128 queries per allowed sky (half present/absent), searched blindly using the existing regional proposal/verification protocol. Criterion: equal-sky mean top1 localization reward on present queries. Ties: earliest checkpoint, then arm letter. Freeze before outer scoring.

At each checkpoint also evaluate a separate fresh corrected bank with32 queries per sky and new deterministic draws; and retain the original fixed inner bank as a historical distribution diagnostic. No ground-truth coordinates are injected into any ranking bank. Validation triplet diagnostics use the nearest spatial correct candidate within12px when it was retrieved, versus non-overlapping false candidates. Report their eligible count; they cannot diagnose pool-missing positives.

Use the same fixed bank for all A–E arms in a fold. The old mismatched bank must not select the corrected model. This validation correction was fixed before final matrix execution or any outer model scoring.

Repeat the selected recipe with seed31005 only if its inner criterion beats the matched classical control. Outer scores never choose recipes, checkpoints or the better seed. Only all-three-fold arms receive aggregate means/promotion gates; partial seed repeats remain explicitly partial.

Full-scene calibration uses the same two-feature, scene-balanced logistic procedure as Experiment1. Real-query candidate banks, pose caches, frozen constellation name and membership rule are unchanged; do not relocate coordinates. C1 must reproduce exactly under the same bank/integration. C0 must reproduce at0.7286878605463721.

Promotion: mean total at least.01 aboveC0 and aboveC1, presence/localization/recovery means no lower thanC0, no scene total regression>.02, stable direction in the second seed. Better thanC1 but belowC0 is matcher evidence, not a deployable gain. Identification cannot improve under this frozen integration.

After frozen scoring, audit the nearest correct candidate's validity, NCC and radiometry-fitted residual, and whether a finer local pose search changes its descriptor rank. This diagnostic never edits predictions.

## Reproduction

Requires `.venv-exp1` plus the original Experiment1 artifacts/weights and Kaggle training data. It makes no online downloads. Use the existing Experiment1 preparation commands first if those ignored artifacts are absent; their hashes are recorded in this experiment.

```sh
.venv-exp1/bin/python -m experiments.exp1b legacy
.venv-exp1/bin/python -m experiments.exp1b prepare
.venv-exp1/bin/python -m experiments.exp1b audit
.venv-exp1/bin/python -m unittest tests.test_exp1b -v
caffeinate -i .venv-exp1/bin/python -m experiments.exp1b run
.venv-exp1/bin/python -m experiments.exp1b integrity
.venv-exp1/bin/python -m experiments.exp1b report
```

Only one GPU training process. Read CPU/MPS availability first; the runner rejects unavailable MPS rather than silently falling back. Completed runs are reused only with matching provenance; 500-step checkpoints include optimizer, RNG and sampler/replay/source-use states. Historical/interrupted preflight runs are excluded from final comparisons and retained for audit.

These are exploratory out-of-fold results on three repeatedly studied development scenes, not an untouched test or a promise of leaderboard improvement.

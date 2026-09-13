# Image-level benchmark (Workstream 2)

This package generates actual images and degraded queries, then evaluates the existing
classical pipeline without giving the predictor synthetic coordinates, membership,
transformations, candidate lists, or class labels. It does not change production defaults.

## Build

From the repository root, with its existing dependencies. Version 2 fits source-style summaries on 17 labelled parent crops wholly inside development regions:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m lab.imagebench.realism fit \
  --data . --output outputs/imagebench/source_model_v2.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m lab.imagebench.generate \
  --data . --output outputs/imagebench/v2 \
  --source-model outputs/imagebench/source_model_v2.json
```

Default build: **27 real-region scenes + 144 rendered scenes** (48 references in each
of development/calibration/confirmation). Rendered skies are 3000x3000 uint8 PNGs;
all queries are 32x32 uint8 PNGs. Real-region targets are 808x808 for the supplied
3000x3000 skies. Use `--rendered-per-split 144` in a NEW output directory for three
replicates of each reference per split, covering each degradation profile per class.
The default is a starter benchmark, not a high-powered estimate for every class.

Builds are deterministic for the recorded code, dependencies, source hashes and seed.
A matching rerun resumes after checking existing artifact hashes. Changed settings or
source code require a new output directory. A interrupted scene not committed to the
manifest is deterministically rewritten. Completed corrupt artifacts cause an error.

Files:

- `manifest.json`: frozen configuration, source/reference hashes, split regions, input
  inventory and label hashes. Scene IDs contain track/split/index, not target classes.
- `inputs/<id>/image.png`, `patch_XX.png`: predictor inputs only.
- `labels/<id>.json`: truth, transformations, source identities, nuisance parameters,
  rendering metadata and provenance; read by the evaluator AFTER prediction.
- `patterns/`: the permitted 48 supplied references.
- `cache/`: label-free proposal caches, shared across verification experiments.
- `audit.json`: integrity and descriptive realism evidence.
- `confirmation_access.jsonl`: written only when confirmation evaluation is requested.

Keep the entire dataset under ignored `outputs/`. No validation/leaderboard images are
read. Training coordinates select development-only parent crops for source-style fitting.
No calibration/confirmation crops, cached appearance scores, or neural weights drive fitting. The benchmark generation seed is 610120, independent of earlier experiments.

## Real-region track

Each original sky is divided into a 3x3 grid; each cell is inset by 96 pixels, leaving
192 pixels between adjacent used regions. Each split receives three cells per source
sky. Filters operate only after cropping. Every crop, donor query, and rendered
background uses its assigned split's regions. This avoids pixel/context overlap across
splits; it does **not** make the original skies independent.

Queries use detected interior sources, including subpixel centre phases and a repeated
view per target. Approximately two thirds are present; absent queries come from a
region in another source sky within the same split. Actual morphology, nebulosity,
seams and artifacts are retained. Distinct-source provenance defines absence, not a
claim that no visually convincing accidental match exists.

This track reports presence and localization only. Its arbitrary regions have no target
constellation annotation, so recovery/identification/weighted total are deliberately
omitted. It is not valid to score an arbitrary crop as its parent constellation.

## Rendered track

Each reference is placed with rotation, possible reflection, unequal scaling and mild
shear. Independent schematic jitter has sigma 2, 4.1 or 8 pixels and is recorded
separately from the query-image transform. Only a subset of reference nodes is issued;
small references are retained, including cases with one issued figure node.

Skies combine split-specific smoothed real backgrounds, spatial variation, seams and a
streak, uniform and clustered field stars, elliptical Gaussian/Moffat sources, noise,
unissued target stars, a small fragment of a different reference, and a close pair.
Figure and field stars share the same source-rendering distribution. A repeated
figure query supplies an independent degraded view of the same physical source.
Absent queries use a separately rendered donor field. The donor reuses the split's
background texture (flipped) but has a different source realization; this is documented
and should be varied in future realism tests.

Schematic truth refers to injected nodes. Residual sources in the smoothed background
are unlabelled clutter. The scene is a controlled rendering, not a recreation of the
competition's unknown image-generation process.

## Degradation and centre convention

Query pixel coordinates are mapped into the source by a recorded 2x3 affine matrix.
The continuous midpoint `(15.5,15.5)` maps exactly onto the supervised source centre.
Subpixel phase comes from fractional source centres, never an undocumented translation.
Tests verify identity crops, a rotated image ramp, and centre preservation.

The generator samples a 48x48 neighborhood, blurs it, then takes the centre 32x32,
so blur does not create artificial query borders. Out-of-bounds footprints are rejected.
It then applies gain, offset, directional illumination drift, read/shot-style noise,
and a JPEG round trip before PNG storage. All categories use the same degradation law.

Profiles are **plausible ranges, not fitted estimates**:

| Profile | Sampling scale | Blur sigma | Gain | Read noise sigma | JPEG quality |
|---|---|---|---|---|---|
| mild | .85–1.18 | .25–.65 | .8–1.2 | .5–2 | 92–100 |
| nominal | .75–1.33 | .4–1.1 | .65–1.35 | 1–4 | 80–98 |
| stress | .65–1.5 | .8–1.7 | .45–1.6 | 3–7 | 65–90 |

Scale here maps query pixels to source pixels; it should not be equated to a search
routine's inverse-scale convention without conversion. Stress settings deliberately
extend beyond the default search scale range. Larger anisotropy/reflection belongs to
the global schematic placement, not the competition's local query degradation model.

## Audit and evaluate

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m lab.imagebench.audit \
  --dataset outputs/imagebench/v2 --data .

# Blind local correspondence pilot, without invented constellation labels:
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m lab.imagebench.evaluate \
  --dataset outputs/imagebench/v2 --track real --limit-scenes 2 \
  --output outputs/imagebench/runs/real-fixed

# Paired appearance comparison, reusing label-free proposals:
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m lab.imagebench.evaluate \
  --dataset outputs/imagebench/v2 --track real --limit-scenes 2 \
  --verify-radius adaptive --output outputs/imagebench/runs/real-adaptive

# Full production geometry and all four score components:
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m lab.imagebench.evaluate \
  --dataset outputs/imagebench/v2 --track rendered --full --limit-scenes 1 \
  --output outputs/imagebench/runs/rendered-fixed

# Merge development matching distributions into the descriptive realism audit:
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m lab.imagebench.audit \
  --dataset outputs/imagebench/v2 --data . \
  --evaluation outputs/imagebench/runs/real-fixed
```

Omit `--limit-scenes` to run the selected split completely. `--split calibration`
selects calibration. `--split confirmation` additionally requires `--allow-confirmation`;
freeze settings before using it. Every confirmation invocation is logged, including
cached reruns. The flag is a guardrail, not a claim that local labels are inaccessible.

Full scoring requires rendered 3000x3000 images because the current geometric null uses
a 9e6-pixel area. `--size 512` is available for generator/integrity tests; such images
cannot receive production full-scene scores through this runner.

The runner stores coarse and refined candidate lists generated from the **same image**,
proposal recall, each branch's candidate recall, their union recall, correct ranks,
score/gap distributions, category-specific localization, runtime and source hashes.
Full runs also retain competing branch class/score records and agreement, plus metrics
stratified by reference size, issued-node count and profile. Coarse and refined branches
are correlated; agreement is a potential feature, not independent evidence by fiat.

Per-scene component means are equal-weighted. Category summaries are explicitly
query-weighted and report physical-source counts. No confidence interval treats repeated
queries or related rendered backgrounds as independent samples. A finished invocation
with a scene limit is only a pilot, not a completed evaluation of the entire benchmark.

Outputs resume only when the exact run configuration, dataset build and source match.
Inputs and labels are hash-checked. Existing production output files are never reused as
synthetic predictions. The predictor is called without labels; saved labels only enter
measurement after blind inference. No true location is inserted into a candidate pool.

## Interpretation and next use

Inspect `audit.json` before calibration. It compares development-only brightness,
contrast, saturation and centre/annulus statistics with all actual labelled query images.
Flags identify synthetic medians outside the actual p10–p90 range; absence of a flag is
not proof of distribution equivalence. With `--evaluation`, it also compares matching
scores/gaps against the cached real joint baseline if available.

Build and integrity-check all splits, inspect development, calibrate on calibration,
then evaluate frozen contenders on confirmation. Confirmation pixel-integrity auditing
is allowed, but its model scores are not automatically released. Do not tune away realism
flags on confirmation or infer hidden identification accuracy from the aggregate Kaggle
score. AUC of individual features is not a mathematical ceiling on multi-class accuracy.

This benchmark is infrastructure for Workstreams 3–5, not a new winning pipeline or
proof of a Kaggle improvement. Image morphology, dependence and degradation mismatch
remain material limits. No learned model or external catalog is used.

## Version 2 realism revision

Real-source sampling now matches development parent-crop brightness, contrast, centre/
annulus signal and high-frequency contrast. It selects existing pixels; it does not
histogram-match output queries. Present and absent sources use the same selection law.
The rendered background quantiles use the development source fit, and sources have
broader Gaussian/Moffat profiles with halos. Renderer ranges remain engineering choices.
The query degradation profiles above are unchanged.

The manifest embeds the source model, its source hashes, fitting coordinates and code
hashes. The audit checks that all fitting footprints stay inside development regions.
Version 1 remains a historical comparison. To compare development distributions:

```sh
.venv/bin/python -m lab.imagebench.realism compare --data . \
  --datasets outputs/imagebench/v1 outputs/imagebench/v2 \
  --output outputs/imagebench/realism_comparison.json
```

This is a fitted-development comparison against only 17 known-present queries, not an
independent validation result. Fine texture and saturation tails still differ. Use
this benchmark for controlled experiments alongside the real labelled scenes; do not
interpret its score as an estimate of Kaggle performance. Confirmation remains reserved.

Version 2 contains 14 entirely saturated queries out of 5,278. The integrity audit
lists constant images separately (identical constants do not prove source leakage)
and still rejects nonconstant duplicates across splits. These cases are degenerate
and limit realism; the audit passing does not certify statistical suitability.

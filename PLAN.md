# Constellation detection: updated two-person implementation plan

Updated September 9, 2026. Target completion: September 22, with September 23–24 reserved for submission and reproduction. This plan replaces the earlier implementation sequence.

## 1. Objective and ownership

Build two independently runnable pipelines against a shared data and evaluation contract:

- **You:** classical computer vision, delivered first as a self-contained Colab notebook and an equivalent standalone inference script.
- **Teammate:** learned patch correspondence, delivered as Python train/evaluate/predict commands and configurable Slurm jobs.
- **Shared:** data audit, evaluator verification, reference catalog, fixed validation folds, experiment manifests, and final comparison.

Each pipeline must localize each query or mark it absent, assign figure membership, choose a supplied constellation name, and write a schema-valid submission. Pipeline integration comes after independent evaluation.

## 2. Evidence and new requirements

Local inspection found 19 grayscale 3000 × 3000 skies, 784 grayscale 32 × 32 query patches, and 48 RGBA reference patterns. All 851 PNGs decoded successfully; no exact decoded-pixel duplicates were found. This file-level check does **not** rule out duplicate peaks within an image or repeated views of the same physical star.

| Labeled scene | Figure queries | Present off-figure queries | Absent queries | Total |
|---|---:|---:|---:|---:|
| Pisces | 10 | 17 | 14 | 41 |
| Scorpius | 10 | 16 | 15 | 41 |
| Taurus | 6 | 12 | 16 | 34 |
| Total | 26 | 45 | 45 | 116 |

Exploratory findings, not validated pipeline results:

- A particular DoG detector retaining the strongest 12,000 peaks covered only 21/71 true present centers within 12 pixels. Removing that cap covered 57/71. Star-only proposals are insufficient at these settings.
- At known true centers, a rotation/scale search raised median correlation from approximately 0.36 to 0.74. This is an oracle diagnostic, not blind localization performance.
- Automatic reference extraction found eight templates with fewer than four nodes, including three with two nodes. All 48 node overlays still need visual review.
- The user reports duplicate peaks in many scenes and two or three patches from other constellations. Treat these as explicit test cases. Their frequency and exact appearance have not yet been independently quantified.

Three different situations must remain distinct:

1. Multiple detector responses to one physical star: consolidate for geometry.
2. Multiple queries depicting one physical star: preserve every query and its output, but count the physical star once when ranking geometric hypotheses.
3. Two genuinely close stars: preserve them when image evidence supports two sources.

Present distractor queries must remain present even when they do not belong to the selected target figure.

## 3. Shared contracts, scoring, and validation

Public interfaces:

```python
predict_scene(image, patches, patterns, config) -> ScenePrediction
evaluate(predictions, ground_truth) -> component_metrics
write_submission(predictions, sample_submission, output_path)
```

`ScenePrediction` contains one decision per real query, optional `(x,y,m)`, one reference-catalog class name, and separate diagnostics. Diagnostics include candidate lists, appearance scores, physical-star group IDs, geometric hypotheses, membership confidence, runtime, and memory. Ground truth and scene names must never influence prediction logic.

Coordinates are patch centers, with x as column and y as row. Verify the even-sized patch convention using synthetic transformations and actual labeled pairs. Preserve sample-submission IDs, query order, counts, and `-1` padding; do not assume scene IDs are contiguous.

Local evaluation follows the published weighted scene score: 25% presence, 20% localization, 25% recovery, and 30% identification. Coordinate reward is full through 12 pixels and zero at 36 pixels. Recovery uses the documented nearest-first greedy one-to-one procedure over all predicted-present coordinates. Internal geometric deduplication must **not** silently change the evaluator or remove submitted query predictions. Undefined component conventions remain explicitly unofficial until evaluator source is available.

Use three leave-one-scene-out folds. Fit thresholds and choose configurations on the two development scenes, then evaluate the held-out scene. Keep held-out scene pixels out of learned training and negative mining; inference indexing is allowed. Keep unlabeled scenes out of supervised fitting and threshold selection. Report each held-out score, their equal-scene mean, and worst-scene score.

Synthetic validation uses separate seeds and includes rotation, scale, reflection, unequal axis scaling, missing nodes, photometric degradation, duplicate detections, repeated-query views, close star pairs, random clutter, and small fragments sampled from a second reference. Synthetic success does not demonstrate generalization to all real classes.

## 4. Foundation and Colab deliverable

First repair the current sampling dtype failure and finish the shared contracts before claiming a runnable pipeline. Existing source modules are partial; no clean end-to-end notebook, complete submission, or HPC run has been verified.

The notebook accepts an uploaded ZIP or Drive directory, resolves an optional enclosing `participant/` folder, and supports smoke, labeled-evaluation, and submission modes. It must run top-to-bottom and export a standalone script using the same implementation.

Notebook sections:

1. Setup, dependency versions, input selection, and configuration.
2. File inventory, dimensions, channels, labels, ordering, and padding checks.
3. Sky/patch statistics and labeled overlays separating absent, off-figure, and figure queries.
4. Reference extraction and all 48 node overlays, with supplied-reference provenance for corrections.
5. Candidate coverage and duplicate-peak exploration, including nearest-neighbor spacing and close-pair examples.
6. Classical localization and robust geometric recognition.
7. Held-out experiments, geometry oracles, failure categories, runtime, and memory.
8. Submission, standalone-script export, and reproducibility checks.

Export the executed notebook, audit tables, figures, configuration manifests, and an evidence-based findings report. Separate executed results, oracle diagnostics, and proposed experiments.

## 5. Your classical pipeline

| Stage | Work | Required comparison or decision |
|---|---|---|
| A0 | Tiled translation-only normalized correlation and valid output | Establish coordinate correctness, baseline score, and runtime |
| A1 | Raw, background subtraction, and DoG preprocessing | Measure candidate coverage and match discrimination separately |
| A2 | Union of detected-star proposals and dense appearance proposals | Compare dense strides 8/4 and retrieval budgets 200/1,000/2,000; measure recall before reducing breadth |
| A3 | Physical-star consolidation and close-pair handling | Compare no consolidation, fixed-radius NMS, and adaptive image-supported consolidation |
| A4 | Rotation/scale verification and local refinement | Start at 15° and scales 0.75/0.87/1.0/1.15/1.33; retain top 5/20/50 spatial alternatives |
| A5 | RANSAC recognition across all references | Compare similarity/reflection and affine models, with partial one-to-one verification |
| A6 | Presence/membership calibration and one feedback pass | Compare no geometry feedback against appearance-supported reranking and one refit |
| A7 | Bounded alternatives | SIFT/ORB or Fourier/log-polar registration only after the main path works; half-day cap each |

### Proposal coverage and appearance

Cache scene preprocessing, detections, and descriptors. Use dense proposals because a star detector can miss query centers. Keep a slower tiled fallback for uncertain queries and coverage audits. Union fallback candidates with existing candidates instead of replacing them blindly. Aim for at least 95% proposal coverage on labeled present queries before choosing a fast default; this is an engineering target, not a measured result or generalization guarantee.

Transform validity masks alongside patches. Verify candidates using masked correlation; each query has its own local transformation. Refine translation, angle, and scale around a fixed coarse estimate to avoid traversal-dependent drift. ECC is optional and must fall back on failure. Widen scale bounds only through development-fold experiments, motivated by boundary optima seen in the oracle probe.

### Duplicate handling

Detect peaks at multiple scales and retain source size, intensity, and response diagnostics. Propose duplicate groups using distances relative to estimated source size. Inspect the local intensity profile or compare one-source versus two-source fits for ambiguous close pairs. Do not merge solely because two peaks fall inside the 12-pixel scoring tolerance.

Maintain raw candidates, proposed groups, refined centers, and query-to-star associations separately. Use conservative grouping until verified: detection artifacts can be consolidated early, while association of different queries to the same physical star should follow appearance verification. Avoid transitive distance clustering that joins a chain of distinct stars.

RANSAC support counts each physical star once and uses one-to-one reference correspondences. Submission serialization still emits every query independently, including repeated views assigned to the same location. Record how often consolidation changes centers, recall, and geometric support.

### RANSAC-based classification

RANSAC is a robust fitting component; classification compares the verified fits of all reference patterns.

1. Start from confidently localized, appearance-supported queries and their unique physical-star groups.
2. Generate plausible template-to-scene correspondences using geometric hashing. RANSAC cannot assume correspondences are already known.
3. Sample nondegenerate hypotheses. Use separate similarity/reflection and affine branches; reject unstable transforms and implausible scales or anisotropy using development-calibrated bounds.
4. Verify against points outside the fitting sample, using partial one-to-one assignment and robust residuals.
5. Rank with unique support, template coverage, residual, and appearance consistency. Log these terms separately and calibrate size/model-complexity effects on development folds and synthetic negatives.
6. Retain several competing class hypotheses and their score gap. Select the highest-scoring valid catalog class for scored inference; reserve `unknown` for smoke or explicit failure paths.

An affine transform determined by three points is not evidence of a correct match: require additional independent support for an affine classification hypothesis. For three-node templates, prioritize similarity/reflection with the extra node verifying the fit. For two-node templates, pair geometry alone is ambiguous under free rotation and scale; report ambiguity and test additional supplied-image evidence rather than claiming reliable identification from two points alone.

Initially cap hypothesis evaluations at 50,000 per scene. Allocate trials across references and models without catalog-order bias; log attempted/accepted hypotheses and cap hits. Test processing-order and reference-order invariance.

Use a bounded set of additional scene stars to support missing reference nodes after query-supported hypotheses exist. Keep query-supported and auxiliary support separate. Auxiliary stars must not overwhelm query evidence or create query locations without appearance support.

### Distractor fragments and feedback

The target output remains one constellation per scene. Off-target fragments may form their own small consensus, so compare multiple hypotheses rather than committing to the first RANSAC success. A small template must not win automatically through perfect coverage of two or three distractor stars; raw support alone must not systematically favor large templates either. Evaluate normalized scores against synthetic background/distractor fits.

Presence is primarily an appearance decision. Figure membership depends on the selected target fit. A localized query outside that fit remains `(x,y,0)`. If a star is geometrically shared by multiple plausible patterns, retain the ambiguity in diagnostics and assign membership relative to the selected target.

Allow one geometry-guided reranking pass over plausible appearance candidates, followed by one refit. Never manufacture a candidate at a template-predicted location without checking the query against the image.

## 6. Teammate's learned pipeline and HPC

| Stage | Work |
|---|---|
| B0 | Synthetic independently degraded crop pairs, with verified center supervision and training-scene-only provenance |
| B1 | Small normalized 64-dimensional descriptor trained from scratch on context no larger than 32 × 32 |
| B2 | Tiled dense scene retrieval at stride 4; top-20 distinct candidate locations and independent refinement |
| B3 | Hard negatives mined from training-scene false matches; exclude overlapping crops and known repeated views of the same source |
| B4 | Additional scene scales, then stride 2 only if retrieval recall warrants it |
| B5 | Independently configurable geometric recognizer and presence calibration, including duplicate groups and competing RANSAC hypotheses |
| B6 | Optional synthetic template-conditioned verifier; at most two variants, only after B0–B5 work |

Share contracts, reference extraction, and validated low-level utilities. Keep learned hypothesis generation/ranking separately selectable so the second pipeline provides a meaningful independent comparison. The current reuse of the classical recognizer is only a scaffold.

Initial training budget: two GPU-hours per fold. Slurm defaults: one GPU, eight CPUs, 32 GB RAM, four-hour allocation; job arrays initially limited to one concurrent task. Require user-supplied account, partition, environment, and storage configuration. Provide separate train/evaluate/predict commands, explicit checkpoint resume, independent logs, and aggregation dependent on successful folds. Do not imply that local smoke checks validate cluster execution.

## 7. Required tests and diagnostics

### Correctness fixtures

- Perfect predictions, wrong class, coordinate errors at 12/24/36 pixels, missing present queries, padding, and query-column permutations.
- Nearest-first greedy recovery, duplicate submitted coordinates, and extra correctly localized off-figure queries.
- Two responses from one star, two genuinely close stars, and multiple query views of one star.
- A true target plus two or three distractor-fragment queries; missing target nodes; clustered and uniform clutter.
- Degenerate RANSAC samples, reflections, affine distortions, small templates, hypothesis-cap behavior, and no-evidence paths.
- Scene-folder renaming, scene-processing order, and reference-processing order do not change mapped predictions.

### Reported experiment measurements

Report proposal recall before/after consolidation and retrieval; refined localization recall; all four competition components; per-scene and mean scores; unique query/auxiliary geometric support; runner-up class gap; runtime; peak memory; seed; configuration; dependency versions; and code version or source hash.

Use the two existing geometry oracles: true present coordinates, and true figure membership. Add controlled duplicate and second-pattern-fragment injections to each oracle so geometric robustness is measured independently of localization failures.

Failure categories: missed proposal, incorrect duplicate merge, duplicate support inflation, wrong appearance match, presence error, distractor-class selection, geometric fit failure, small-template ambiguity, and correct location with wrong membership. Any manually reviewed labeled examples are diagnostics, not hardcoded prediction rules.

## 8. Schedule and integration

| Dates | You | Teammate | Joint checkpoint |
|---|---|---|---|
| Sep 9–10 | Repair sampling; complete audit and A0 | Verify scorer; synthetic fixtures; Slurm smoke setup | Correct I/O and executed EDA |
| Sep 11–13 | A1–A4: coverage, duplicates, transformed matching | B1–B3: descriptor, retrieval, hard negatives | Compare proposal recall and duplicate failure cases |
| Sep 14–16 | A5: RANSAC and distractor diagnostics | B5: independent recognition | Two complete inference pipelines |
| Sep 17–19 | A6; highest-value remaining ablations | B4; B6 only if justified | Frozen three-fold comparison |
| Sep 20–22 | Clean Colab/script reproduction | Final training and checkpointed inference | Select pipeline; validate final submission |
| Sep 23–24 | Submission and packaging buffer | Reproduction support | Final checks |

After independent evaluation, compare each localizer with each recognizer. Test one union-of-candidates variant only if errors are complementary; reverify candidates and never average disagreeing coordinates. Stop adding methods after September 19 and prefer the simpler option when measured performance is tied.

## 9. Acceptance and exact next sequence

Acceptance requires a clean-runtime notebook, matching standalone predictions, independently runnable pipelines, complete schema-valid outputs, preserved query ordering/padding, reproducible configurations/checkpoints, and a report separating measured results from proposals. No memorized coordinates, scene-name classification, patch-count class lookup, or manually labeled validation/test answers.

The published public-split descriptions conflict, and scorer zero-denominator behavior remains unresolved. Preserve those uncertainties in documentation. Public leaderboard submissions should check a few frozen milestones within the actual team limit, not tune individual scenes.

Next implementation sequence:

1. Fix float32 sampling maps and complete evaluator/data-contract fixtures.
2. Execute and inspect the audit, reference overlays, and duplicate/close-pair diagnostics.
3. Establish A0, then measure candidate coverage with star and dense proposal unions.
4. Implement conservative physical-star grouping and query-preserving serialization tests.
5. Implement and test correspondence generation, RANSAC verification, and competing-class ranking with structured distractors.
6. Package and clean-run the Colab notebook and matching script.
7. Complete the independent learned/HPC track against the same fixtures and folds.
8. Run the bounded integration comparison and package the selected result.

## Source context

Competition requirements were reviewed earlier in this task: [overview and evaluation](https://www.kaggle.com/competitions/constellation-detection-cs-gy-6643/overview), [data description](https://www.kaggle.com/competitions/constellation-detection-cs-gy-6643/data), and [rules](https://www.kaggle.com/competitions/constellation-detection-cs-gy-6643/rules). Local exploratory measurements concern the supplied `participant/` data. Duplicate peaks and off-target fragments are user-reported requirements pending quantified analysis. RANSAC, grouping choices, budgets, and thresholds above are proposed experiments, not demonstrated competition performance.

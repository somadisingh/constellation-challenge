# Constellation detection: first classical submission milestone

This repository implements the classical track of `PLAN.md`. The original directory contained the dataset and plan, but no source modules. The learned/HPC track and the full plan's experiment matrix are not implemented or claimed complete.

## Collaborator setup

```sh
git clone https://github.com/somadisingh/constellation-challenge.git
cd constellation-challenge
```

Copy your Kaggle download into the repository root with this layout (or pass its existing location with `--data /path/to/participant`):

```text
patterns/
train/
validation/
train_ground_truth.csv
sample_submission.csv
```

The dataset, generated outputs, executed notebook, virtual environment, and model checkpoints are excluded from Git. Keep your local data in place; Git does not upload it. Follow the installation and run commands below. Start with `--mode smoke` for a quick setup check, then use `evaluate` to score the labelled scenes or `submission` to predict the unlabelled scenes.

The editable implementation is in `constellation/` and `run.py`; the notebook and standalone script are exports. After implementation changes, regenerate them with `python build_deliverables.py` followed by `python format_notebook.py`. To execute the notebook locally with `execute_notebook.py`, also install `nbclient` and `ipykernel`.

`FINDINGS.md` preserves the milestone results. Referenced files under `outputs/` are local experiment artifacts and are not included in this repository. Several diagnostic scripts expect those caches and must be used after producing the corresponding experiment outputs. No learned-model training pipeline is implemented yet; the current baseline is classical computer vision.

## Run

Python 3.11+ is intended; this machine was tested with Python 3.14. Install dependencies in an isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python run.py --data . --mode evaluate --workers 2 --threads 1 --output outputs/train
OPENBLAS_NUM_THREADS=1 .venv/bin/python run.py --data . --mode submission --workers 4 --threads 1 --output outputs/submission
```

Use a data directory containing `patterns`, `train`, `validation`, `train_ground_truth.csv`, and `sample_submission.csv`. An enclosing `participant` directory is supported. Submission inference only reads the sample schema, validation images, and supplied reference diagrams; it does not read training labels.

`constellation_inference.py` is a self-contained source export with the same arguments. It temporarily extracts the implementation, then runs it. The dependencies must still be installed. `Constellation_Classical.ipynb` provides input upload/directory selection, audit, diagnostics, inference, and script export. Its executed companion is explicitly a **local smoke run**, not a verified Colab cloud run or a full executed submission notebook.

## Current method (`--pipeline joint`)

1. Dense stride-4 proposals plus local source peaks, described by circular harmonic magnitudes.
2. Top-2,000 appearance retrieval candidates, unioned with exhaustive half-resolution rotation/scale correlation proposals.
3. Full-resolution circular-support correlation over rotation and scale; retain 20 spatial alternatives and refine their local poses.
4. ECC affine refinement with displacement, scale and anisotropy bounds; use background-subtracted correlation for final appearance ranking.
5. Presence threshold 0.72, the maximum of a 0.60-0.84 sweep on the three labelled scenes. Absent decisions retain diagnostics.
6. Extract all reference nodes from opaque white disks in the supplied RGBA diagrams.
7. Classify a query as appearance-ambiguous when its top two alternatives are within 0.03. Verify affine hypotheses against a pool holding every ambiguous query's alternatives, one-to-one per query group, seeded from triangle invariants. Score one-to-one support, residuals, a soft shear penalty and bounded auxiliary image-star evidence, comparing coarse and refined point sets. Choose the strongest candidate class.
8. Adopt the winning fit's chosen alternative as the reported coordinate for each query it matched.
9. Preserve every query independently in the CSV, including repeated locations. Figure membership is assigned relative to the selected geometry; presence is not gated by membership.

Steps 7 and 8 replace the previous stage, which used one point per query and quad-dominant seeding. Steps 1-6 are unchanged, so the two are directly comparable; `--pipeline final` still runs the previous stage. Development mean rose from 0.676 to 0.729 and the leave-one-scene-out threshold holdout from 0.654 to 0.702. On a synthetic geometry benchmark of 192 scenes at a seed unused during tuning, identification rose from 0.125 to 0.474.

`--verify-radius {fixed,adaptive}` selects the verification support radius. `fixed` is the frozen default. `adaptive` follows the radius each pose actually admits, `floor(15.2 * scale)`, instead of a constant 12px. The frozen choice is wrong in both directions: because the patch is read at `15.5 + R(angle) * offset / scale`, a 12px disc makes half the angles inadmissible at scale 0.75 and discards about two thirds of the valid area at scale 1.33. Measured over every cached proposal for all 116 labelled queries, `adaptive` raises recall@12 after selection from 0.901 to 0.972, recall@4 from 0.873 to 0.958, figure retention from 0.885 to 1.000, and present-minus-absent score separation from 0.074 to 0.117, at equal runtime. It is **not** the default because it loses the equal-scene total (best 0.7162 against 0.72869): better scores make queries less ambiguous, so the gate supplies fewer alternatives to the geometric pool, relocation drops and recovery falls from 0.878 to 0.819. That trade-off, and the recalibration attempts that do not close it, are in the fourth-pass section of `FINDINGS.md`.

`--verify-rep {blur,dog}` and `--alternatives N` expose the verification representation and the retained-candidate count. Defaults (`blur`, 20) reproduce the frozen configuration exactly. They exist because verification is where the remaining proposal losses occur: all seven labelled present queries with no candidate within 12px reach the proposal set and are discarded by `verify`, and switching to a DoG representation or retaining 40 candidates raises candidate recall from 0.901 to 0.944 or 0.958. Neither improves the weighted score without recalibrating the downstream ambiguity gate, pool margin and presence threshold, which three labelled scenes cannot support, so both remain off by default. See the second-pass section of `FINDINGS.md`.

Each geometric branch permits up to 80,000 hypothesis verifications, about 160,000 per scene combined. This exceeds the plan's initial 50,000-per-scene target and is an explicit deviation. Two- and three-node references still lack a reliable independent classification path, and the benchmark shows identification is near-hopeless when a reference issues only three or four figure queries. Deduplication is conservative anchor grouping within three pixels, not the planned image-supported one-source/two-source fit.

## Evidence and limitations

The current method's development results are in `outputs/joint_train/metrics.json`, produced by a full rerun from images: mean **0.729**, worst scene 0.487, with correct names for Pisces and Scorpius and an incorrect Serpens Caput prediction for Taurus. The previous stage's results are in `outputs/final_train_cached/metrics.json` (mean 0.676) with an independent rerun in `outputs/reproduction_train/metrics.json`. These three scenes informed method development; neither is an untouched test estimate.

Leave-one-scene-out **presence-threshold** evaluation is in `outputs/lab/joint_threshold_folds.json` for the current method (mean 0.702, worst 0.487) and `outputs/final_threshold_folds.json` for the previous one (mean 0.654, worst 0.418). Both isolate threshold holdout only; all three scenes informed method development. Empty-denominator scorer conventions are unofficial. The published overview/data descriptions disagree about the public split.

Because three labelled scenes cannot support a 30%-weight identification claim, identification is measured on a synthetic geometry benchmark (`lab/synth.py`) whose noise parameters are measured from the real scenes, including the affine template-to-sky residual of about 5px median (`lab/geomerror.py`). It synthesises the candidate lists the recognizer receives, tests geometry in isolation, supplies no auxiliary star map, and is not evidence about the appearance stage. `FINDINGS.md` records the case where an unmeasured noise assumption in that benchmark produced a conclusion that did not transfer.

No scene-name class lookup, memorized coordinates, patch-count class prior, validation labels, external star catalog, or manual validation predictions are used. Reference filenames are used only as the permitted class names and deterministic catalog seeds.

## Tests and diagnostics

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python audit.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python geometry_oracle.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/invariance.py
```

Third-pass experiments, all on cached candidates so no appearance search is repeated:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/oracle_check.py       # localization ceiling by category
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/factorial.py          # seeds x ranking, downstream frozen
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/reloc_experiment.py   # relocation guards
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/reloc_tradeoff.py     # guard strictness sweep
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m lab.proposals          # cache proposals entering verify
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/trace_failures.py     # per-query verification traces
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/selectivity.py        # resolving power of three scenes
```

Fourth-pass experiments:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/verify_full_report.py # full-pool verification variants
OPENBLAS_NUM_THREADS=1 .venv/bin/python run.py --data . --mode evaluate --pipeline joint \
    --verify-radius adaptive --workers 3 --threads 1 --output outputs/lab/adaptive_train
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/adaptive_calibrate.py # threshold x gap recalibration
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/adaptive_pool.py      # pool eligibility widening
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/twopass_experiment.py # geometric pool eligibility
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/twopass_push.py       # recovery push on the finalist
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/recovery_ceiling.py   # oracle recovery per candidate set
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/null_diag.py          # is the chance-fit null calibrated
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/null_sweep.py         # identification under corrected nulls
OPENBLAS_NUM_THREADS=1 .venv/bin/python lab/null_verdict.py       # paired test plus real-scene check
```

`null_mode` selects the chance-fit null used to rank competing classes: `groups` (shipped), `pool` (chance density from the pooled point count rather than the query-group count), `decorrelate` (per-scene score-versus-log-size trend removed), or `pool+decorrelate`. Both corrections address confirmed faults — wrong-class score correlates +0.660 with reference node count, and the pool holds about 124 points where the shipped density counts about 32 groups — but neither yields a supported identification gain (paired n=384: +7 net scenes, p=0.230), so `groups` remains the default. Within-scene AUC of the available fit features is score 0.857, support 0.825, coverage 0.456, residual 0.454; with roughly 40 competing verified classes per scene that puts identification near 0.5, which is where it sits. Identification is limited by the discriminative content of the fit features, not by the null's calibration.

Tests cover scoring rewards, greedy one-to-one recovery, duplicates, extra predictions, CSV ordering/padding, synthetic rotation/scale center preservation, affine/reflection geometry, missing points, clutter, and reference-order invariance. Diagnostic scripts are development tools, not inference dependencies.

`lab/LEDGER.md` is the experiment ledger: what is retained, what was measured and
rejected (with the regime each rejection is scoped to), what is partial, and what is
untested. Read it before proposing an experiment, and read the resumable queue at the
end of `FINDINGS.md` before starting one.

`lab/` holds the measurement and tuning scripts for the joint stage. They read cached candidate lists from `outputs/`, so the expensive appearance search is not repeated: `headroom.py` and `signals.py` bound what reranking can recover, `ambiguity.py` fits the ambiguity gate, `geomerror.py` measures the template-to-sky residual, `synth.py` and `bench.py` provide the synthetic identification benchmark, `truerank.py` and `whylose.py` attribute identification failures, `presence.py` tests presence rules, `valstats.py` compares validation and labelled scene density, and `invariance.py` checks order invariance. Like the other diagnostics they are development tools and are not inference dependencies.

`outputs/audit/reference_nodes.jpg` shows all 48 extracted node overlays. `outputs/audit/inventory.csv` records all 851 decoded PNGs and hashes. No exact decoded-pixel duplicates were found. This does not rule out repeated views or duplicate detector responses within scenes.

## Reproduction and submission

Inference writes per-scene JSON diagnostics, the schema-preserving CSV, and a manifest with configuration, dependency versions, source hash and completion status. The source is checked for changes during each run. Do not edit inference source during a running job.

The submission must have exactly the scene IDs, counts, column order and padding in `sample_submission.csv`. Use `validate_submission.py` before uploading. The CSV is the Kaggle artifact; retain the notebook, standalone script, configuration and report for the course's runnable-code hand-in.

Kaggle access was verified in the user's signed-in Safari session. The competition permits five submissions per day and two final selections. Submission status and any public score are recorded separately after a confirmed upload; local predictions are not evidence of a successful Kaggle submission.


## Confirmed Kaggle result

The previous stage was submitted successfully through Safari as Somadi on September 10, 2026. Kaggle reports **Complete**, with public score **0.58189**. See `outputs/kaggle_submission.json` for source and CSV hashes. No final-selection setting was changed. The live submission UI permits one final selection, while the rules text says two; this discrepancy remains unresolved.

## Current submission candidate

`outputs/joint_submission/submission.csv` holds the joint stage's predictions for all 16 unlabelled scenes, produced by `run.py --mode submission --pipeline joint`. `validate_submission.py` accepts it: 16 scenes, 668 real queries, 375 present, 293 absent, 90 columns, matching the sample schema. Source hash `8580ec0c...` is recorded in its manifest.

Against the previously submitted CSV, presence decisions are identical, 52 coordinates moved, 117 membership flags changed, on-figure flags rose from 95 to 106, and 12 of 16 class names changed.

The user reports a public score of approximately **0.637**, up from the confirmed 0.58189. This is recorded as user-reported only: no local submission entry or CSV hash has been matched to it, so which CSV produced it is unverified, and it is not used for tuning. `0.58189` remains the only score with recorded provenance.

`constellation/twopass.py` decouples geometric pool eligibility from the appearance gap. Eligibility was decided solely by a query's top-two appearance gap, which is why improving the appearance score *reduced* recovery: better scores make queries less ambiguous, so fewer alternatives reach the verification pool and geometry relocates less. A first pass produces competing hypotheses; the node positions they predict then admit any already existing candidate lying within a radius, whatever its appearance gap, and a second pass re-verifies. No location is manufactured from a template prediction. Combined with the adaptive verifier it lifts recovery from 0.756 to 0.822 and figure localization from 0.538 to 0.615, the best figure localization measured and above production's 0.577, for a total of 0.7185 — still below production's 0.72869, so it is available rather than default.

The source hash is now `e61c9e31...`, moved from `8580ec0c...` by configuration additions, by correcting stale `recognize_joint` defaults, and by the `QuerySlate` role separation, and by adding `verify_adaptive`. Predictions are unaffected at every step: under the current source the training metrics reproduce as 0.7286878605463721, the training CSV reproduces as `7cc9d19e...`, and this submission CSV reproduces as `f83b8762...`, all byte-identical to the earlier runs.

## Separated roles in the recognizer

`constellation/slate.py` splits the three jobs a candidate ordering used to conflate. `calib` decides presence, ambiguity and the pool margin; `rank` decides pool ordering and the reported coordinate; `seed` decides the physical-star grouping anchor and the geometric hypothesis seed. `recognize_joint` accepts raw candidate lists, in which case all three collapse onto the single appearance score and the shipped behaviour is reproduced exactly, or `QuerySlate` objects carrying them separately. `build_pool(..., pool_by='calib')` fixes pool membership so that only ordering changes, which is what makes a ranking experiment attributable.

`experiments/exp2_geometry/` integrates the frozen Experiment 1B matcher with recovery.
Its fixed rule uses learned presence/localization by default and adopts or rescues a C0
coordinate only for a query independently supported by C0's relocation map. It scores
0.761959 out of fold (+0.033271 over C0) and 0.751275 with the repeated training seed.
Candidate-level learned rank scoring was also tested under fixed pools and seed anchors;
it is rejected because it is inert on the primary seed and harmful at its largest weight
on the repeat. See `EXPERIMENT2_REPORT.md` and `outputs/exp2_geometry/record.json`.

This exists because a factorial over seeds and ranking showed the earlier re-ranking failures were **seed disruption**: changing seed anchors drops the true class from rank 0 to rank 1, while changing the ranking alone costs only off-figure localization. `constellation/reloc.py` adds relocation guards on the same footing, including one that refits the winning transform with a query's whole physical-star group held out. Neither mechanism improved the weighted objective, so production uses the default policy for both; see the third-pass section of `FINDINGS.md`.

## Image-level benchmark — Workstream 2

Implemented in `lab/imagebench/`; see `lab/imagebench/README.md` for generation,
audit, evaluation and confirmation-access commands. The default dataset contains 27
real-image region scenes and 144 rendered 3000×3000 scenes, covering all 48 references
in each of development/calibration/confirmation. Inputs, separate labels, source-region
provenance, transforms, physical-source IDs and artifact hashes are recorded.

The runner generates blind proposals and both coarse/refined candidates from images.
Real regions receive presence/localization scores only; rendered full-scene runs receive
all four components, branch-agreement diagnostics and size/issued-node breakdowns.
No production defaults were changed. All generated artifacts remain under ignored
`outputs/imagebench/`. Confirmation scoring requires an explicit flag and is logged.

This is a controlled benchmark, not a hidden-score predictor: spatial splits share the
three original skies, rendered backgrounds are reused within splits, and realism audits
identified dimmer/less contrasted queries in v1. Version 2 corrects source sampling and
rendered brightness/contrast using 17 development parent crops; fine texture and
saturation tails remain mismatched. Read the latest Workstream 2 section of
`FINDINGS.md` before interpreting the initial pilot scores or calibrating a new method.

## Workstream 3 — branch agreement experiments

`lab/branch_agreement.py` compares nine classical cross-branch ranking rules using
cached complete class fits; `lab/workstream3_report.py` reports all score components,
name-only controls, agreement reliability and real-scene leave-one-out selection.
See `lab/WORKSTREAM3.md` and the latest `FINDINGS.md` section for measured results.
The reconstructed baseline reproduces existing predictions exactly. No production
change is adopted: the labelled screen does not improve and v2's synthetic gains
come from relocation, with no identification gain in the six-scene screen.

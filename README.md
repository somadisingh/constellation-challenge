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

The source hash is now `0b6b7a0b...`, changed from `8580ec0c...` by configuration additions and by correcting stale `recognize_joint` defaults. Predictions are unaffected: under the current source the training metrics reproduce as 0.7286878605463721, the training CSV reproduces as `7cc9d19e...`, and this submission CSV reproduces as `f83b8762...`, all byte-identical to the earlier runs.

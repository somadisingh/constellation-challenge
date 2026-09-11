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

## Frozen method

1. Dense stride-4 proposals plus local source peaks, described by circular harmonic magnitudes.
2. Top-2,000 appearance retrieval candidates, unioned with exhaustive half-resolution rotation/scale correlation proposals.
3. Full-resolution circular-support correlation over rotation and scale; retain 20 spatial alternatives and refine their local poses.
4. ECC affine refinement with displacement, scale and anisotropy bounds; use background-subtracted correlation for final appearance ranking.
5. Presence threshold 0.72, selected from the three labelled scenes. Absent decisions retain diagnostics.
6. Extract all reference nodes from opaque white disks in the supplied RGBA diagrams.
7. Compare bounded affine quadruple-hash hypotheses from coarse and refined recovered point sets. Score one-to-one support, residuals, a soft shear penalty, and bounded auxiliary image-star evidence. Choose the strongest candidate class; report `unknown` for insufficient evidence.
8. Preserve every query independently in the CSV, including repeated locations. Figure membership is assigned relative to the selected geometry; presence is not gated by membership.

The two geometric branches each permit up to 50,000 full hypothesis verifications, so their combined limit is approximately 100,000 per scene. This exceeds the plan's initial 50,000-per-scene target and is an explicit milestone deviation. Two- and three-node references currently lack a reliable independent classification path. Deduplication is conservative anchor grouping within three pixels, not the planned image-supported one-source/two-source fit.

## Evidence and limitations

The frozen method's development results are in `outputs/final_train_cached/metrics.json`; independent rerun results are in `outputs/reproduction_train/metrics.json`. The entire training CSV reproduced byte for byte. The development mean is approximately **0.676**, with correct names for Pisces and Taurus and an incorrect Cetus prediction for Scorpius. These three scenes informed method development; this is not an untouched test estimate.

`outputs/hybrid_ecc/calibration.json` contains leave-one-scene-out **presence-threshold** evaluation. The final method under these same two-scene threshold fits is recorded separately in `outputs/final_threshold_folds.json`: mean 0.654, worst scene 0.418. Both are threshold holdouts; all three scenes informed method development. Empty-denominator scorer conventions are unofficial. The published overview/data descriptions disagree about the public split.

No scene-name class lookup, memorized coordinates, patch-count class prior, validation labels, external star catalog, or manual validation predictions are used. Reference filenames are used only as the permitted class names and deterministic catalog seeds.

## Tests and diagnostics

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python audit.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python geometry_oracle.py
```

Tests cover scoring rewards, greedy one-to-one recovery, duplicates, extra predictions, CSV ordering/padding, synthetic rotation/scale center preservation, affine/reflection geometry, missing points, clutter, and reference-order invariance. Diagnostic scripts are development tools, not inference dependencies.

`outputs/audit/reference_nodes.jpg` shows all 48 extracted node overlays. `outputs/audit/inventory.csv` records all 851 decoded PNGs and hashes. No exact decoded-pixel duplicates were found. This does not rule out repeated views or duplicate detector responses within scenes.

## Reproduction and submission

Inference writes per-scene JSON diagnostics, the schema-preserving CSV, and a manifest with configuration, dependency versions, source hash and completion status. The source is checked for changes during each run. Do not edit inference source during a running job.

The submission must have exactly the scene IDs, counts, column order and padding in `sample_submission.csv`. Use `validate_submission.py` before uploading. The CSV is the Kaggle artifact; retain the notebook, standalone script, configuration and report for the course's runnable-code hand-in.

Kaggle access was verified in the user's signed-in Safari session. The competition permits five submissions per day and two final selections. Submission status and any public score are recorded separately after a confirmed upload; local predictions are not evidence of a successful Kaggle submission.


## Confirmed Kaggle result

Submitted successfully through Safari as Somadi on September 10, 2026. Kaggle reports **Complete**, with public score **0.58189**. See `outputs/kaggle_submission.json` for source and CSV hashes. No final-selection setting was changed. The live submission UI permits one final selection, while the rules text says two; this discrepancy remains unresolved.

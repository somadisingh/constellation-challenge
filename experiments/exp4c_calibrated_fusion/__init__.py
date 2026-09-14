"""Experiment 4C: calibrated joint-evidence fusion for constellation
identification -- a focused repair of Experiment 4B's diagnosed additive
score-scale defect (geometric-support counts, appearance evidence and
unqueried-star z-scores summed on incompatible numeric scales, which
regressed a previously-correct scene, scorpius, on both seeds).

Scope is intentionally narrow: replace ONLY the combination function over
Experiment 4B's already-frozen hypothesis features with a calibrated,
regularized logistic model, fit leak-free per fold. This experiment does NOT
retrain any patch matcher, does not build a new retrieval system, does not
add new geometric feature families, and does not attempt plate solving --
those are out of scope (Experiment 5 or later).

Frozen, read-only inputs: Experiment 1's real-query alignment caches,
Experiment 1B's HardNet checkpoints, Experiment 2's frozen geometry
integration, Experiment 3's checkpoints/calibrators/deployment policy and
`submission_candidate.csv`, and Experiment 4B's hypothesis-generation
machinery (`independent_scorer.generate_and_score`,
`unqueried_star_evidence.score_unqueried_nodes`) and its own outputs.
Nothing under `constellation/`, `lab/` (code), `experiments/exp1*`,
`experiments/exp2*`, `experiments/exp3_pairwise/`,
`experiments/exp4_joint_identification/`, `experiments/exp4b_joint_solver/`,
or `outputs/{joint_train,joint_submission,exp1,exp1b,exp2_geometry,
exp3_pairwise,exp4_joint_identification,exp4b_joint_solver}` is ever written
by this package.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "exp4c_calibrated_fusion"
SCENES = ("pisces", "scorpius", "taurus")
SEEDS = (31004, 31005)

# Placement-correctness threshold (px): a true-class hypothesis is a POSITIVE
# only if its held-out-matched query-node pairs place the figure within this
# tolerance of the true physical location, not merely the right class name.
# Chosen BEFORE any model fitting, matching the official localization-reward
# full-credit radius used everywhere else in this repo
# (constellation.contracts.reward: full credit <=12px).
PLACEMENT_CORRECT_PX = 12.0

# Predeclared regularization grid (task's exact example).
C_GRID = (0.01, 0.1, 1.0)

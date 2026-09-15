"""Experiment 5: constellation-independent candidate and geometric
hypothesis recovery.

Experiment 4C proved that reweighting Experiment 4B's frozen hypothesis pool
cannot repair identification: Taurus has ZERO correct-placement hypotheses in
that pool, so no combination of scores over it can ever select Taurus
correctly. Experiment 5 asks a different question -- can the pool itself be
made to contain (and independently validate) correct hypotheses, without
learning constellation identity from the three labelled scenes?

Mandatory first step (Stage 1, `oracle_audit.py`): measure whether the
failure is candidate retrieval (not enough correct locations reach the
verification stage) or geometric-hypothesis construction (correct locations
exist but the seeding/scoring never proposes or validates the correct
placement). The predeclared branch decision in `branch_decision.py` is
computed mechanically from that evidence BEFORE any new method is designed,
and is not revisited after seeing final results.

Frozen, read-only inputs (never rewritten by this package): Experiment 1's
real-query alignment caches and candidate banks, Experiment 1B's HardNet
checkpoints, Experiment 2's frozen geometry integration, Experiment 3's
checkpoints/calibrators/deployment policy and `submission_candidate.csv`,
Experiment 4/4B's headroom ladder and hypothesis-generation machinery, and
Experiment 4C's hypothesis dataset and completion/gate records. Nothing under
`constellation/`, `lab/` (code), `experiments/exp1*`, `experiments/exp2*`,
`experiments/exp3_pairwise/`, `experiments/exp4_joint_identification/`,
`experiments/exp4b_joint_solver/`, `experiments/exp4c_calibrated_fusion/`, or
`outputs/{joint_train,joint_submission,exp1,exp1b,exp2_geometry,
exp3_pairwise,exp4_joint_identification,exp4b_joint_solver,
exp4c_calibrated_fusion}` is ever written by this package.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "exp5_hypothesis_recovery"
SCENES = ("pisces", "scorpius", "taurus")
SEEDS = (31004, 31005)

# Official localization-reward full-credit radius (constellation.contracts.reward),
# reused throughout this experiment for "correct location" / "correct placement".
CORRECT_PX = 12.0
RECALL_PX = (12.0, 36.0)
TOPK_GRID = (1, 3, 5, 10, 20)

# Predeclared multi-candidate retention grid (Stage 2). Selected on
# allowed-sky evidence only -- never on the held-out sky.
K_GRID = (5, 10, 20)

# Patterns with fewer than 4 nodes cannot support barycentric fourth-point
# validation; 2-node patterns cannot even fit an affine transform. Recorded
# once here so every module uses the identical, verified list (see
# context-gatherer: constellation.references.extract_patterns, 48 patterns
# total, 8 with <4 nodes, 3 of those with <3 nodes).
PATTERNS_UNDER_4_NODES = ('caelum', 'chamaeleon', 'fornax', 'horologium',
                          'indus', 'mensa', 'sculptor', 'sextans')
PATTERNS_UNDER_3_NODES = ('fornax', 'mensa', 'sextans')

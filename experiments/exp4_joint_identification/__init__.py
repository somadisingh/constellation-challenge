"""Experiment 4: honest deployment evaluation of Experiment 3's fixed arm-F
policy, plus identification-headroom / failure-attribution analysis.

Frozen inputs only, read-only from this package:
  - Experiment 1's real-query alignment caches and blind candidate banks
  - Experiment 1B's fold-specific arm-B HardNet checkpoints
  - Experiment 2's frozen `hybrid_prediction` support-gated integration rule
  - Experiment 3's checkpoints, calibrators, deployment policy and candidate CSV
  - Production's classical (`C0`) geometry/identification (`outputs/joint_train`,
    `outputs/joint_submission`)

Nothing under `constellation/`, `lab/`, `experiments/exp1*`, `experiments/exp2*`,
`experiments/exp3_pairwise/`, or `outputs/{joint_train,joint_submission,exp1,
exp1b,exp2_geometry,exp3_pairwise}` is ever written by this package. See
`integrity.py` for the exact protected-root list and `protected_before.json`
for the pre-experiment hash snapshot.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "exp4_joint_identification"
SCENES = ("pisces", "scorpius", "taurus")

# The single fixed architecture Experiment 3 actually deploys (see
# outputs/exp3_pairwise/deployment_policy.json). Experiment 4's Phase 1
# evaluates THIS policy, not the per-fold oracle-selected arms.
DEPLOYED_ARM = "F"

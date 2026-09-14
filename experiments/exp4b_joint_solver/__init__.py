"""Experiment 4B: candidate-rank fidelity, independent geometric evidence,
unqueried-star null model, and bounded joint multi-candidate identification.

Corrects Experiment 4's inaccurate claim that Experiment 3's verifier
underperforms classical rank-1 ranking on ALL three real labelled scenes (it
does not -- Exp3 beats classical on scorpius; see `prior_claim_corrections.json`).

Frozen inputs only, read-only from this package: Experiment 1's real-query
alignment caches and blind candidate banks, Experiment 1B's HardNet
checkpoints, Experiment 2's frozen integration rule, Experiment 3's
checkpoints/calibrators/deployment policy, Experiment 4's headroom/attribution
records, and production's classical (C0) geometry/identification. Nothing
under `constellation/`, `lab/` (code), `experiments/exp1*`, `experiments/exp2*`,
`experiments/exp3_pairwise/`, `experiments/exp4_joint_identification/`, or
`outputs/{joint_train,joint_submission,exp1,exp1b,exp2_geometry,exp3_pairwise,
exp4_joint_identification}` is ever written by this package.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "exp4b_joint_solver"
SCENES = ("pisces", "scorpius", "taurus")

# Corrected per-scene true-class ranks (verified against
# outputs/exp4_joint_identification/headroom_oracles.json; see
# prior_claim_corrections.json for the full correction record).
EXP3_VS_CLASSICAL_RANKS = {
    'pisces':   {'exp3_rank': 24, 'classical_rank': 5,  'exp3_better': False},
    'scorpius': {'exp3_rank': 5,  'classical_rank': 12, 'exp3_better': True},
    'taurus':   {'exp3_rank': 13, 'classical_rank': 11, 'exp3_better': False},
}

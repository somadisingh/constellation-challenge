"""Experiment 5D: unlabeled stability audit for one validation hypothesis."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'exp5d_stability_audit'
SOURCE = ROOT / 'outputs' / 'exp5c_affine_recovery' / 'reconstructed_validation' / 'constellation_07.json'
BASE_EXP3_CSV = ROOT / 'outputs' / 'exp3_pairwise' / 'submission_candidate.csv'

SCENE = 'constellation_07'
EXISTING_NAME = 'hydra'
CANDIDATE_NAME = 'perseus'
SEEDS = (7301, 7307, 7319)

# Frozen before running any trial. Taurus is not an input to this experiment.
PASS_RULE = {
    'overall_candidate_rate_min': 0.80,
    'candidate_wins_per_family_min': 2,
    'trials_per_family': 3,
    'median_margin_min': 1.00,
    'median_support_min': 7,
    'median_held_out_support_min': 3,
}

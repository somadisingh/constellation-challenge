"""Experiment 5E: calibrate perturbation stability on frozen synthetic truth."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP5C_OUT = ROOT / 'outputs' / 'exp5c_affine_recovery'
OUT = ROOT / 'outputs' / 'exp5e_stability_calibration'

FINAL_SEEDS = (
    ('seed1', EXP5C_OUT / 'synthetic_final_seed1.json'),
    ('seed2', EXP5C_OUT / 'synthetic_final_seed2.json'),
)
PERTURBATION_FAMILIES = ('top1', 'top3', 'jitter_2px', 'query_dropout_10pct')
STABLE_WINS_MIN = 3
INCORRECT_PER_SEED = 5

# Frozen before executing any perturbation solve.
CALIBRATION_GATE = {
    'stable_precision_min': 0.80,
    'stable_correct_count_min': 2,
    'stable_incorrect_rate_max': 0.20,
}

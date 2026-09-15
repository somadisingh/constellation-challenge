"""Experiment 5C: Efficient Affine Hypothesis Recall and Validation-Wide Identification Rescue."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'exp5c_affine_recovery'
PATTERNS_DIR = ROOT / 'patterns'
TRAIN_GROUND_TRUTH = ROOT / 'train_ground_truth.csv'
BASE_EXP3_CSV = ROOT / 'outputs' / 'exp3_pairwise' / 'submission_candidate.csv'

SCENES = ('pisces', 'scorpius', 'taurus')
TOLERANCE = 18.0
DEFAULT_SEED = 31004
SEED_FIT = 42001
SEED_EVAL1 = 880192271
SEED_EVAL2 = 991823145

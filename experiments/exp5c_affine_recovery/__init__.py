"""Experiment 5C: Efficient Affine Hypothesis Recall and Validation-Wide Identification Rescue."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'exp5c_affine_recovery'
DATA_ROOT = ROOT if (ROOT / 'patterns').exists() else ROOT.parent / 'participant'
PATTERNS_DIR = DATA_ROOT / 'patterns'
TRAIN_GROUND_TRUTH = DATA_ROOT / 'train_ground_truth.csv'
SAMPLE_SUBMISSION = DATA_ROOT / 'sample_submission.csv'
BASE_EXP3_CSV = ROOT / 'outputs' / 'exp3_pairwise' / 'submission_candidate.csv'

SCENES = ('pisces', 'scorpius', 'taurus')
TOLERANCE = 18.0
DEFAULT_SEED = 31004
SEED_FIT = 42001
SEED_EVAL1 = 880192271
SEED_EVAL2 = 991823145

# Frozen before either final-evaluation seed is run.
DEFAULT_CONFIG = {
    'max_scene_quads': 30000,
    'descriptor_neighbors': 6,
    'descriptor_max_distance': 0.065,
    'proposal_budget': 18000,
    'progressive_budgets': [2500, 7000, 18000],
    'tolerance_px': TOLERANCE,
    'max_condition_number': 8.0,
    'strict_gate': {'margin_min': 2.0, 'support_min': 9, 'held_out_min': 5},
    'primary_gate': {
        'confidence_min': 0.90,
        'support_min': 7,
        'held_out_min': 3,
        'largest_component_min': 3,
        'stream_agreement_min': 2,
    },
    'safe_peak_memory_gib': 8.0,
}

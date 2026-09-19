"""Experiment 6: duplicate-safe RANSAC constellation identification."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "exp6_ransac_identification"
BASELINE = ROOT / "outputs/exp5c_affine_recovery/submission_recommended_constellation_08_only.csv"
FINAL_RADIUS = 12.0
SEEDS = {"fit": 60401, "selection": 60402, "final1": 60403, "final2": 60404}


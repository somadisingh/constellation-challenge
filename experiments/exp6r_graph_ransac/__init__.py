"""Experiment 6R: evidence repair and invariant graph-pair RANSAC."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/exp6r_graph_ransac"
BASELINE=ROOT/"outputs/exp5c_affine_recovery/submission_recommended_constellation_08_only.csv"
FINAL_RADIUS=12.0
SEEDS={"development":66001,"fit":66002,"selection":66003,"final1":66004,"final2":66005}


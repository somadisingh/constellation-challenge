"""Experiment 5B: genuinely affine four-point proposal recovery."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/'exp5b_affine_proposals'
SCENES=('pisces','scorpius','taurus')
NEIGHBORS=32
PER_CLASS_BUDGET=5000
TOLERANCE=18.0


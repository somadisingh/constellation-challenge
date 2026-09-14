"""Experiment 3: corrected pairwise verifier for presence and localization.

Compares each query jointly with its strongest retrieved candidate crops using a
compact pairwise network, rather than independent per-crop descriptor distance
(Experiment 1/1B). Frozen inputs only: Experiment 1's blind union candidate banks
and real-aligned caches, Experiment 1B's corrected synthesis mechanism and
fold-specific, seed-specific HardNet arm-B fine-tune checkpoints (resolved by
`hardnet_source.frozen_hardnet`, loaded from `outputs/exp1b/runs/{fold}/B_s{seed}/
best.pt`, verified against `outputs/exp1b/selection_frozen.json`), and Experiment
2's support-gated hybrid integration rule. Production, Experiment 1, Experiment
1B and Experiment 2 artifacts are read-only from this package.

This docstring previously claimed fold-specific HardNet arm-B checkpoints were
in use while the code always loaded the generic pretrained backbone instead --
that defect is corrected as of the repair recorded in
`outputs/exp3_pairwise/corrections.json`; see `hardnet_source.py`.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "exp3_pairwise"
SCENES = ("pisces", "scorpius", "taurus")

# Fixed experiment matrix (task §10). Do not add arms after viewing held-out results.
ARMS = ("A", "B", "C", "D", "E", "F")
ARM_SPEC = {
    "A": {"pixel_cnn": True, "hardnet_fusion": False, "objective": "binary_bce",
          "negatives": "random", "offset": False},
    "B": {"pixel_cnn": True, "hardnet_fusion": False, "objective": "binary_bce",
          "negatives": "hard", "offset": False},
    "C": {"pixel_cnn": True, "hardnet_fusion": False, "objective": "listwise",
          "negatives": "hard", "offset": False},
    "D": {"pixel_cnn": True, "hardnet_fusion": True, "objective": "listwise",
          "negatives": "hard", "offset": False},
    "E": {"pixel_cnn": True, "hardnet_fusion": True, "objective": "listwise",
          "negatives": "random", "offset": False},
    "F": {"pixel_cnn": True, "hardnet_fusion": True, "objective": "listwise",
          "negatives": "hard", "offset": True},
}

POSITIVE_RADIUS = 12.0    # px: geometric distance defining a positive candidate
IGNORE_RADIUS = 36.0      # px: 12-36px is ignored for classification, not negative
MAX_OFFSET = 12.0         # px: bound on the residual-correction head (task §8)

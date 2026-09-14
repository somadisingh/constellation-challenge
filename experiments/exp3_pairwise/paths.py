"""Output layout for Experiment 3."""
from __future__ import annotations

from pathlib import Path

from . import OUT


def fold_dir(fold: str) -> Path:
    return OUT / "folds" / fold


def checkpoints_dir(fold: str, arm: str, seed: int) -> Path:
    d = OUT / "checkpoints" / fold / f"{arm}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def panels_dir() -> Path:
    d = OUT / "panels"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT

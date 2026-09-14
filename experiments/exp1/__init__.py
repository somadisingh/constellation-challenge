"""Experiment 1: learned verification of real sky patches.

Implements `cnn/plan_experiment1.md`. Nothing in this package is imported by
`constellation/` or `run.py`; production behaviour is unchanged by construction.
"""
__all__ = ['SCENES', 'FOLDS']

# The three labelled skies. Each becomes an outer held-out fold in turn (plan §5).
SCENES = ('pisces', 'scorpius', 'taurus')
FOLDS = SCENES

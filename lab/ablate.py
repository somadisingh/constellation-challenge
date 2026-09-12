"""Separate the two entangled changes: (a) widening the verification pool and
snapping coordinates, (b) the new scoring terms and similarity branch.

Identification over three scenes moves in 1/3 steps, so it is treated as a
low-power signal here. Localization/recovery/presence average over 71-116
queries and are far more trustworthy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.harness import experiment

FROZEN_SCORING = dict(appearance_weight=0., rank_weight=0., models=('affine',))

print('--- (b) scoring variants, pool held at rank-0 only (gap=0, no snap effect) ---')
experiment('frozen-like scoring',       {**FROZEN_SCORING, 'gap': 0.})
experiment('+similarity branch',        {'appearance_weight': 0., 'rank_weight': 0., 'gap': 0.})
experiment('+appearance term',          {'rank_weight': 0., 'gap': 0.})
experiment('+rank term (all new)',      {'gap': 0.})

print('\n--- (a) pool widening + snap, scoring held frozen-like ---')
for gap in (.01, .02, .03, .05):
    experiment(f'frozen-scoring gap={gap:.2f}', {**FROZEN_SCORING, 'gap': gap})

print('\n--- snap isolated: geometry may relabel but never move coordinates ---')
experiment('gap=0.02 snap=False', {**FROZEN_SCORING, 'gap': .02, 'snap': False})
experiment('gap=0.02 snap=True',  {**FROZEN_SCORING, 'gap': .02, 'snap': True})

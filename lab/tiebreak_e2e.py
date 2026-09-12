"""End-to-end effect of the scale-preserving tie-break at the production config.

No recalibration is applied or needed: the per-query score multiset is preserved, so
threshold 0.72 and gap 0.03 select exactly the same queries as the baseline.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate, ScenePrediction
from constellation.references import extract_patterns
from lab.cache import load_train, TRAIN, ROOT
from lab.samples import build_train
from lab.harness import run_stage, aux_for, DEFAULT
from lab.recache import component_cache, COMPONENTS
from lab.tiebreak import tiebreak_cache


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    cache = load_train()
    data = build_train(cache)
    parts = component_cache(cache, data)
    aux = {n: aux_for(n) for n in TRAIN}
    cfg = dict(DEFAULT)

    rows = {}
    print(f"{'tie-break':14s} {'total':>7s} {'worst':>7s} {'pres':>6s} {'loc':>6s} "
          f"{'rec':>6s} {'id':>5s}  classes")
    for key in ['incumbent', 'multiblur', 'ca_min', 'dog32']:
        rc = cache if key == 'incumbent' else tiebreak_cache(cache, parts, key)
        preds = {}
        for n in TRAIN:
            name, patches, _ = run_stage(rc[n], patterns, aux[n], cfg)
            preds[n] = ScenePrediction(patches, name)
        m = evaluate(preds, truth)
        rows[key] = m
        print(f"{key:14s} {m['mean']['score']:7.4f} {m['worst_score']:7.3f} "
              f"{m['mean']['presence']:6.3f} {m['mean']['localization']:6.3f} "
              f"{m['mean']['recovery']:6.3f} {m['mean']['identification']:5.3f}  "
              + ' '.join(preds[n].constellation for n in TRAIN), flush=True)

    # Presence must be untouched; assert it rather than assume it.
    base = rows['incumbent']['mean']['presence']
    same = all(abs(v['mean']['presence'] - base) < 1e-12 for v in rows.values())
    print(f'\npresence identical across tie-breaks: {same}')
    (ROOT / 'outputs/lab/tiebreak_e2e.json').write_text(
        json.dumps({k: v['mean'] | {'worst': v['worst_score']}
                    for k, v in rows.items()}, indent=2, default=float))


if __name__ == '__main__':
    main()

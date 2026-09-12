"""Leave-one-scene-out presence-threshold evaluation of the joint stage.

Each held-out scene is scored with the threshold that maximises the weighted score
on the other two. This isolates threshold holdout only. All three scenes informed
method development, so it is not an estimate of model-selection generalization.
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate
from constellation.finalize import finalize_joint
from constellation.references import extract_patterns
from lab.cache import load_train, TRAIN, ROOT

GRID = np.round(np.arange(.60, .86, .02), 3)


def main():
    cv2.setNumThreads(4)
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    cache = load_train()
    images = {n: cv2.imread(str(ROOT / 'train' / n / f'{n}_image.png'), 0) for n in TRAIN}

    # Predictions depend on the threshold, so evaluate the whole grid once.
    table = {}
    for t in GRID:
        preds = {n: finalize_joint(images[n], cache[n]['refined'], cache[n]['coarse'],
                                   patterns, threshold=float(t)) for n in TRAIN}
        table[float(t)] = evaluate(preds, truth)
        print(f'  threshold={t:.2f} mean={table[float(t)]["mean"]["score"]:.4f}', flush=True)

    folds, chosen = {}, {}
    for held in TRAIN:
        dev = [n for n in TRAIN if n != held]
        best_t = max(GRID, key=lambda t: np.mean([table[float(t)]['scenes'][n]['score']
                                                 for n in dev]))
        folds[held] = {'development_scenes': dev, 'threshold': float(best_t)}
        chosen[held] = table[float(best_t)]['scenes'][held]
    mean = {k: float(np.mean([chosen[n][k] for n in TRAIN])) for k in chosen[TRAIN[0]]}
    result = {
        'threshold_folds': folds,
        'metrics': {'scenes': chosen, 'mean': mean,
                    'worst_score': min(c['score'] for c in chosen.values())},
        'grid_means': {str(t): table[float(t)]['mean']['score'] for t in GRID},
        'scope': ('Joint stage with the presence threshold fitted on the other two '
                  'scenes. All three scenes informed method development, so this '
                  'isolates threshold holdout, not model-selection holdout.'),
    }
    out = ROOT / 'outputs/lab/joint_threshold_folds.json'
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({'folds': {k: v['threshold'] for k, v in folds.items()},
                      'mean': round(mean['score'], 4),
                      'worst': round(result['metrics']['worst_score'], 4)}, indent=1))


if __name__ == '__main__':
    main()

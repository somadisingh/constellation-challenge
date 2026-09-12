"""Score the production `finalize_joint` on cached train candidates.

This must agree with lab/harness.py; the harness reimplemented the stage for fast
iteration, so a mismatch would mean the tuned configuration was not promoted.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate, write_submission
from constellation.finalize import finalize_joint, finalize
from constellation.references import extract_patterns
from lab.cache import load_train, TRAIN, ROOT


def main(which='joint'):
    cv2.setNumThreads(4)
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    cache = load_train()
    out = ROOT / f'outputs/lab/{which}_train_cached'
    out.mkdir(parents=True, exist_ok=True)
    preds = {}
    for n in TRAIN:
        img = cv2.imread(str(ROOT / 'train' / n / f'{n}_image.png'), 0)
        fn = finalize_joint if which == 'joint' else finalize
        preds[n] = fn(img, cache[n]['refined'], cache[n]['coarse'], patterns)
        (out / f'{n}.json').write_text(json.dumps(asdict(preds[n]), indent=2, default=str))
        print(f'  {n:9s} -> {preds[n].constellation}', flush=True)
    m = evaluate(preds, truth)
    (out / 'metrics.json').write_text(json.dumps(m, indent=2))
    write_submission(preds, ROOT / 'train_ground_truth.csv', out / 'submission.csv')
    print(json.dumps({'mean': {k: round(v, 4) for k, v in m['mean'].items()},
                      'worst': round(m['worst_score'], 4)}, indent=1))
    return m


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'joint')

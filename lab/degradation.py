"""Measure the real scene/patch statistics a synthetic benchmark must reproduce.

Recovered pose comes from candidates that landed within 12px of a labelled
centre, so the angle/scale distribution reflects genuine query degradation
rather than search-grid artefacts.
"""
import json
import sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from lab.cache import load_train, TRAIN, ROOT


def scene_stats(scene, split='train'):
    img = cv2.imread(str(ROOT / split / scene / f'{scene}_image.png'), 0)
    f = img.astype(np.float32)
    dog = cv2.GaussianBlur(f, (0, 0), 1) - cv2.GaussianBlur(f, (0, 0), 2.5)
    peaks = (dog == cv2.dilate(dog, np.ones((5, 5), np.uint8))) & (dog > 1.5)
    ys, xs = np.where(peaks)
    bg = np.percentile(f, 50)
    return dict(scene=scene, shape=img.shape, background=float(bg),
                noise=float(np.std(f[f < np.percentile(f, 80)])),
                stars=int(len(xs)), max=float(f.max()),
                p999=float(np.percentile(f, 99.9)),
                star_peak_median=float(np.median(f[ys, xs])) if len(xs) else 0.)


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    cache = load_train()
    print('--- scene statistics ---')
    for n in TRAIN:
        print('  ', scene_stats(n))

    print('\n--- recovered pose of correctly localized queries ---')
    angles, scales = [], []
    for n in TRAIN:
        for q, tp in zip(cache[n]['refined'], truth[n].patches):
            if tp is None or not len(q):
                continue
            d = np.linalg.norm(np.array([c[:2] for c in q]) - np.array(tp[:2]), axis=1)
            j = int(np.argmin(d))
            if d[j] <= 12 and len(q[j]) >= 5:
                angles.append(q[j][3] % 360); scales.append(q[j][4])
    angles = np.array(angles); scales = np.array(scales)
    print(f'  n={len(angles)}')
    print(f'  angle: histogram over 45deg bins '
          f'{Counter((angles // 45).astype(int).tolist()).most_common()}')
    print(f'  scale: {Counter(np.round(scales, 2).tolist()).most_common()}')

    print('\n--- patch statistics ---')
    vals = []
    for n in TRAIN:
        d = ROOT / 'train' / n / 'patches'
        for p in sorted(d.glob('patch_*.png'))[:40]:
            a = cv2.imread(str(p), 0).astype(np.float32)
            vals.append((float(a.mean()), float(a.std()), float(a.max()), float(a.min())))
    v = np.array(vals)
    print(f'  n={len(v)} mean={v[:,0].mean():.1f} std={v[:,1].mean():.1f} '
          f'max={v[:,2].mean():.1f} min={v[:,3].mean():.1f}')

    # Reference node-count distribution bounds what identification can achieve.
    from constellation.references import extract_patterns
    pat = extract_patterns(ROOT / 'patterns')
    print(f'\n--- reference node counts ---\n  '
          f'{Counter(len(v) for v in pat.values()).most_common()}')


if __name__ == '__main__':
    main()

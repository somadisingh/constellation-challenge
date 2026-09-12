"""Compare the frozen and joint submissions field by field."""
import csv
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import parse_cell
from lab.cache import ROOT


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        n = int(r['n_patches'])
        out[r['Id']] = [parse_cell(r[f'patch_{i:02}']) for i in range(1, n + 1)]
    return out


def main():
    old = load(ROOT / 'outputs/final_submission/submission.csv')
    new = load(ROOT / 'outputs/joint_submission/submission.csv')
    moved, dists, presence_diff, member_diff, total = 0, [], 0, 0, 0
    for k in old:
        for a, b in zip(old[k], new[k]):
            total += 1
            if (a is None) != (b is None):
                presence_diff += 1
                continue
            if a is None:
                continue
            d = float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))
            if d > .5:
                moved += 1; dists.append(d)
            if a[2] != b[2]:
                member_diff += 1
    print(f'{total} real queries')
    print(f'  presence decisions changed : {presence_diff}')
    print(f'  coordinates relocated      : {moved} '
          f'(median move {np.median(dists):.0f}px)' if dists else '')
    print(f'  membership flags changed   : {member_diff}')
    figs_old = sum(1 for k in old for p in old[k] if p and p[2] == 1)
    figs_new = sum(1 for k in new for p in new[k] if p and p[2] == 1)
    print(f'  queries flagged on-figure  : {figs_old} -> {figs_new}')
    reloc = [len(json.load(open(p))['diagnostics'].get('relocated_queries', []))
             for p in sorted((ROOT / 'outputs/joint_submission').glob('constellation_*.json'))]
    print(f'  relocations reported by the stage: total={sum(reloc)} per-scene={reloc}')


if __name__ == '__main__':
    main()

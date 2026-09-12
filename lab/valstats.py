"""How dense are the real validation scenes in the terms that matter?

What drives chance fits is not the patch count but the number of queries that pass
the presence threshold and therefore enter the geometric pool.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.joint import build_pool
from constellation.geometry import consolidate
from lab.cache import load_validation, load_train, TRAIN


def main():
    val = load_validation()
    tr = load_train()
    print(f"{'scene':18s} {'queries':>8s} {'pass .72':>9s} {'pool':>6s} {'ambig':>6s}")
    rows = []
    for name, d in list(tr.items()) + list(val.items()):
        alts = d['refined']
        ids = [i for i, q in enumerate(alts) if len(q) and q[0][2] >= .72]
        sub = [alts[i] for i in ids]
        if len(sub) < 3:
            print(f'{name:18s} {len(alts):8d} {len(ids):9d}   too few')
            continue
        anchors = np.array([q[0][:2] for q in sub]).reshape(-1, 2)
        _, groups = consolidate(anchors)
        pool, tags, _, _ = build_pool(sub, groups, 8, .15, .03)
        amb = sum(1 for q in sub if len(q) > 1 and (q[0][2] - q[1][2]) < .03)
        print(f'{name:18s} {len(alts):8d} {len(ids):9d} {len(pool):6d} {amb:6d}')
        rows.append((name, len(alts), len(ids), len(pool)))
    v = [r for r in rows if r[0].startswith('constellation')]
    t = [r for r in rows if not r[0].startswith('constellation')]
    print(f'\nlabelled  : passing median={np.median([r[2] for r in t]):.0f} '
          f'pool median={np.median([r[3] for r in t]):.0f}')
    print(f'validation: passing median={np.median([r[2] for r in v]):.0f} '
          f'pool median={np.median([r[3] for r in v]):.0f} '
          f'max={max(r[3] for r in v)}')


if __name__ == '__main__':
    main()

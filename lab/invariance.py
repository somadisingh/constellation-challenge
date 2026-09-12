"""Order-invariance checks for the joint recognizer.

PLAN.md requires that neither reference-processing order nor query-processing
order change mapped predictions.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.joint import recognize_joint
from constellation.references import extract_patterns
from lab.cache import load_train, TRAIN, ROOT

CFG = dict(models=('affine',), gap=.03, score_mode='binom', tolerance=18.,
           cap=80000, top_k=8, quad_share=0.)


def main():
    patterns = extract_patterns(ROOT / 'patterns')
    cache = load_train()
    rng = np.random.default_rng(3)
    ok = True
    for n in TRAIN:
        alts = [q for q in cache[n]['refined'] if len(q) and q[0][2] >= .72]
        base_name, base_chosen, _ = recognize_joint(alts, patterns, **CFG)

        # 1. Reference (catalog) order.
        keys = list(patterns); rng.shuffle(keys)
        shuffled = {k: patterns[k] for k in keys}
        n2, c2, _ = recognize_joint(alts, shuffled, **CFG)
        same_ref = (n2 == base_name and
                    all(np.allclose(base_chosen[k], c2[k]) for k in base_chosen)
                    and set(c2) == set(base_chosen))

        # 2. Query order: permute, then map results back to original indices.
        perm = rng.permutation(len(alts))
        n3, c3, _ = recognize_joint([alts[i] for i in perm], patterns, **CFG)
        back = {int(perm[k]): v for k, v in c3.items()}
        same_q = (n3 == base_name and set(back) == set(base_chosen) and
                  all(np.allclose(base_chosen[k], back[k]) for k in base_chosen))

        print(f'{n:9s} class={base_name:16s} relocated={len(base_chosen):3d} '
              f'reference-order={"ok" if same_ref else "CHANGED"} '
              f'query-order={"ok" if same_q else "CHANGED"}')
        if not same_ref:
            print(f'    reference-order gave {n2}')
        if not same_q:
            print(f'    query-order gave {n3}')
        ok &= same_ref and same_q
    print('\nall invariances hold' if ok else '\nINVARIANCE VIOLATED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

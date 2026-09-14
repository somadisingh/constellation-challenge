"""Hard vs matched-random negative selection within an already-retrieved group (task §7).

A "hard negative" here is never a fabricated location: it is one of the negative
candidates ALREADY PRESENT in a `QueryGroup` (produced by real blind retrieval),
selected by one of three difficulty signals:

  classical   highest classical NCC among this group's negative-labelled candidates
  network     highest current pairwise-verifier score after a refresh
  hardnet     highest frozen-HardNet cosine similarity to the query

The matched random control draws from the SAME negative pool, with an IDENTICAL
count of presentations, so a training step under either policy sees the same
group, the same positive, and the same NUMBER of negatives -- only which negative
differs. `hard_vs_random_manifest` records this pairing explicitly so both arms
can be audited for equal presentation counts.
"""
from __future__ import annotations

import numpy as np

from experiments.exp1.mining import NEGATIVE

N_NEGATIVES_PER_GROUP = 3   # matches Exp1B's stream.py 3-way choose_indices convention


def negative_pool(group) -> np.ndarray:
    return group.negative_indices()


def hardnet_similarity(model, device, query: np.ndarray, crops: np.ndarray) -> np.ndarray:
    """Cosine similarity between frozen-HardNet query/candidate descriptors."""
    from experiments.exp1.scoring import encode_batch
    if not len(crops):
        return np.zeros(0)
    z = encode_batch(model, np.concatenate([query[None], crops]), device).numpy()
    zq, zc = z[0], z[1:]
    return zc @ zq   # both already L2-normalised by Descriptor


def select_hard(group, count: int = N_NEGATIVES_PER_GROUP,
                network_scores: np.ndarray | None = None,
                hardnet_sim: np.ndarray | None = None) -> dict:
    """Three distinct hard negatives: classical, network (or hardnet fallback), random.

    Falls back to classical-only when a network/hardnet score is unavailable (task
    §7: "before a network exists, replace its quarter with classical negatives").
    """
    pool = negative_pool(group)
    if len(pool) < 1:
        return {'indices': np.zeros(0, int), 'kinds': [], 'ok': False,
                'reason': 'no negative-labelled candidate in this group'}
    ncc = np.nan_to_num(group.classical_ncc, nan=-np.inf)
    rng = np.random.default_rng(abs(hash(group.group_id)) % (2 ** 32))

    chosen, kinds = [], []
    remaining = pool.copy()
    for slot in range(count):
        if not len(remaining):
            break
        if slot == 0:
            idx = remaining[np.argmax(ncc[remaining])]
            kind = 'classical'
        elif slot == 1 and network_scores is not None:
            idx = remaining[np.argmax(network_scores[remaining])]
            kind = 'network'
        elif slot == 1 and hardnet_sim is not None:
            idx = remaining[np.argmax(hardnet_sim[remaining])]
            kind = 'hardnet'
        elif slot == 1:
            idx = remaining[np.argmax(ncc[remaining])]
            kind = 'classical'
        else:
            idx = int(rng.choice(remaining))
            kind = 'random'
        chosen.append(int(idx))
        kinds.append(kind)
        remaining = remaining[remaining != idx]
    return {'indices': np.array(chosen, int), 'kinds': kinds, 'ok': True}


def select_random(group, count: int = N_NEGATIVES_PER_GROUP) -> dict:
    """Matched random control: same pool, same count, uniform choice."""
    pool = negative_pool(group)
    if len(pool) < 1:
        return {'indices': np.zeros(0, int), 'kinds': [], 'ok': False,
                'reason': 'no negative-labelled candidate in this group'}
    rng = np.random.default_rng(abs(hash(group.group_id)) % (2 ** 32))
    k = min(count, len(pool))
    idx = rng.choice(pool, size=k, replace=False)
    return {'indices': idx, 'kinds': ['random'] * k, 'ok': True}


def hard_vs_random_manifest(groups: list, count: int = N_NEGATIVES_PER_GROUP) -> dict:
    """Presentation-count parity audit between the hard and random policies."""
    hard_n = sum(len(select_hard(g, count)['indices']) for g in groups)
    rand_n = sum(len(select_random(g, count)['indices']) for g in groups)
    return {'n_groups': len(groups), 'hard_presentations': hard_n,
           'random_presentations': rand_n, 'matched': hard_n == rand_n,
           'count_per_group': count}

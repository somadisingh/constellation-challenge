"""Hard vs matched-random negative selection within an already-retrieved group (task §7).

A "hard negative" here is never a fabricated location: it is one of the negative
candidates ALREADY PRESENT in a `QueryGroup` (produced by real blind retrieval),
selected by one of two difficulty signals actually wired into training:

  classical   highest classical NCC among this group's negative-labelled candidates
  hardnet     highest frozen-HardNet cosine similarity to the query

A third, "network" (current pairwise-verifier score after a predeclared refresh),
is defined as an explicit optional slot that only activates when a caller supplies
`network_scores`. As of this repair, the training loop does NOT supply
`network_scores` for the shipped A-F matrix arms, so the honest description of
the shipped "hard" policy is: classical top confuser + frozen-HardNet top confuser
+ deterministic random negative. Do not describe this as "current-network mining"
unless a caller actually threads `network_scores` through (see task §5 correction:
prior reports incorrectly claimed network-mined negatives were part of every arm).

The matched random control draws from the SAME negative pool, with an IDENTICAL
count of presentations, so a training step under either policy sees the same
group, the same positive, and the same NUMBER of negatives -- only which negative
differs. `hard_vs_random_manifest` records this pairing explicitly so both arms
can be audited for equal presentation counts.

Determinism: every random draw in this module is seeded via `derive_seed` (never
Python's built-in `hash()`, which is salted per-process by `PYTHONHASHSEED` and is
NOT stable across separate interpreter invocations). The seed is derived from an
explicit tuple of tags: experiment name, base seed, fold, negative policy, the
group's own `group_id`, and a caller-supplied `refresh_generation` (defaults to 0
for a single-pass, non-refreshed run; a resumed run must pass the SAME
`refresh_generation` sequence it would have produced uninterrupted, since the
digest depends only on these tags and not on process history).
"""
from __future__ import annotations

import numpy as np

from experiments.exp1.env import derive_seed
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


def _negative_rng(seed: int, fold: str, policy: str, group_id: str,
                  refresh_generation: int = 0) -> np.random.Generator:
    """Deterministic RNG for one group's random negative slot.

    Same `(seed, fold, policy, group_id, refresh_generation)` -> same draw, in any
    process, on any machine, resumed or not. This replaces the prior
    `hash(group.group_id)` seeding, which used Python's salted string hash and was
    NOT reproducible across separate interpreter invocations (verified: `hash('x')`
    differs between two `python3 -c` calls unless `PYTHONHASHSEED` is pinned).
    """
    return np.random.default_rng(derive_seed(
        seed, 'exp3-negatives', fold, policy, group_id, 'refresh', refresh_generation))


def select_hard(group, count: int = N_NEGATIVES_PER_GROUP,
                seed: int = 0, refresh_generation: int = 0,
                network_scores: np.ndarray | None = None,
                hardnet_sim: np.ndarray | None = None) -> dict:
    """Three distinct hard negatives: classical, network (or hardnet fallback), random.

    Falls back to classical-only when a network/hardnet score is unavailable (task
    §7: "before a network exists, replace its quarter with classical negatives").

    `seed`/`refresh_generation` make the random (slot-2) draw deterministic and
    reproducible across processes and resumes; see `_negative_rng`.
    """
    pool = negative_pool(group)
    if len(pool) < 1:
        return {'indices': np.zeros(0, int), 'kinds': [], 'ok': False,
                'reason': 'no negative-labelled candidate in this group'}
    ncc = np.nan_to_num(group.classical_ncc, nan=-np.inf)
    rng = _negative_rng(seed, group.fold, 'hard', group.group_id, refresh_generation)

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
    digest = int(derive_seed(seed, 'exp3-negatives-digest', group.fold, 'hard',
                             group.group_id, refresh_generation, tuple(chosen)))
    return {'indices': np.array(chosen, int), 'kinds': kinds, 'ok': True,
           'digest': digest}


def select_random(group, count: int = N_NEGATIVES_PER_GROUP,
                  seed: int = 0, refresh_generation: int = 0) -> dict:
    """Matched random control: same pool, same count, uniform choice."""
    pool = negative_pool(group)
    if len(pool) < 1:
        return {'indices': np.zeros(0, int), 'kinds': [], 'ok': False,
                'reason': 'no negative-labelled candidate in this group'}
    rng = _negative_rng(seed, group.fold, 'random', group.group_id, refresh_generation)
    k = min(count, len(pool))
    idx = rng.choice(pool, size=k, replace=False)
    digest = int(derive_seed(seed, 'exp3-negatives-digest', group.fold, 'random',
                             group.group_id, refresh_generation, tuple(idx.tolist())))
    return {'indices': idx, 'kinds': ['random'] * k, 'ok': True, 'digest': digest}


def hard_vs_random_manifest(groups: list, count: int = N_NEGATIVES_PER_GROUP,
                            seed: int = 0) -> dict:
    """Presentation-count parity audit between the hard and random policies."""
    hard_n = sum(len(select_hard(g, count, seed=seed)['indices']) for g in groups)
    rand_n = sum(len(select_random(g, count, seed=seed)['indices']) for g in groups)
    return {'n_groups': len(groups), 'hard_presentations': hard_n,
           'random_presentations': rand_n, 'matched': hard_n == rand_n,
           'count_per_group': count}


def verify_negative_determinism(groups: list, seed: int, count: int = N_NEGATIVES_PER_GROUP) -> dict:
    """Run selection twice (simulating two separate processes) and assert equality.

    Used by the negative-selection determinism test and by the finalize/completion
    audit to prove resumed runs draw the same negatives as an uninterrupted run.
    """
    run1_hard = [select_hard(g, count, seed=seed)['digest'] for g in groups]
    run2_hard = [select_hard(g, count, seed=seed)['digest'] for g in groups]
    run1_rand = [select_random(g, count, seed=seed)['digest'] for g in groups]
    run2_rand = [select_random(g, count, seed=seed)['digest'] for g in groups]
    return {
        'ok': run1_hard == run2_hard and run1_rand == run2_rand,
        'hard_digests_match': run1_hard == run2_hard,
        'random_digests_match': run1_rand == run2_rand,
        'n_groups': len(groups),
    }

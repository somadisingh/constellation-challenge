"""Phase 1: seven fixed candidate-ranking rules, evaluated leave-one-sky-out.

Rule 1: classical_rank        -- rank purely by classical_ncc (existing C1 signal)
Rule 2: exp3_rank              -- rank purely by exp3_pair_logit
Rule 3: mean_rank_fusion        -- average of (classical rank, exp3 rank), lower better
Rule 4: reciprocal_rank_fusion  -- sum of 1/(60+rank) across the two signals (RRF)
Rule 5: linear_score_fusion     -- standardized (z-scored) classical_ncc + exp3_pair_logit
Rule 6: trained_ranking_head    -- small logistic-regression-style pairwise/listwise
                                   model, fit ONLY on allowed-sky training rows
Rule 7: absent_separated        -- Rule 5's within-present ranking, but the
                                   ABSENT decision is scored independently via
                                   exp3_absent_logit vs best-candidate margin
                                   (never folded into the same linear score as
                                   present-candidate ranking)

Monotonic presence calibration is explicitly NOT one of these seven rules (see
`prior_claim_corrections.json`): it cannot reorder candidates and is therefore
not a ranking method.
"""
from __future__ import annotations

import numpy as np

from .rank_features import relevance


def _valid_candidates(row: dict) -> list:
    return [c for c in row['candidates'] if c['valid']]


def rank_by_classical(row: dict) -> list:
    valid = _valid_candidates(row)
    return sorted(valid, key=lambda c: -c['classical_ncc'])


def rank_by_exp3(row: dict) -> list:
    valid = [c for c in _valid_candidates(row) if c['exp3_pair_logit'] is not None]
    return sorted(valid, key=lambda c: -c['exp3_pair_logit'])


def _ranks_of(valid: list, key: str) -> dict:
    order = sorted(range(len(valid)), key=lambda i: -valid[i][key])
    ranks = {}
    for pos, i in enumerate(order):
        ranks[valid[i]['index']] = pos
    return ranks


def rank_by_mean_rank_fusion(row: dict) -> list:
    valid = [c for c in _valid_candidates(row) if c['exp3_pair_logit'] is not None]
    if not valid:
        return []
    ncc_ranks = _ranks_of(valid, 'classical_ncc')
    exp3_ranks = _ranks_of(valid, 'exp3_pair_logit')
    scored = [(c, (ncc_ranks[c['index']] + exp3_ranks[c['index']]) / 2.0) for c in valid]
    return [c for c, _ in sorted(scored, key=lambda cs: cs[1])]


def rank_by_rrf(row: dict, k: float = 60.0) -> list:
    valid = [c for c in _valid_candidates(row) if c['exp3_pair_logit'] is not None]
    if not valid:
        return []
    ncc_ranks = _ranks_of(valid, 'classical_ncc')
    exp3_ranks = _ranks_of(valid, 'exp3_pair_logit')
    scored = [(c, 1.0 / (k + ncc_ranks[c['index']]) + 1.0 / (k + exp3_ranks[c['index']]))
             for c in valid]
    return [c for c, _ in sorted(scored, key=lambda cs: -cs[1])]


def _zscore(values: np.ndarray) -> np.ndarray:
    mu, sd = values.mean(), values.std()
    return (values - mu) / sd if sd > 1e-9 else values - mu


def rank_by_linear_fusion(row: dict, w_classical: float = 0.5, w_exp3: float = 0.5) -> list:
    valid = [c for c in _valid_candidates(row) if c['exp3_pair_logit'] is not None]
    if not valid:
        return []
    ncc = _zscore(np.array([c['classical_ncc'] for c in valid]))
    exp3 = _zscore(np.array([c['exp3_pair_logit'] for c in valid]))
    score = w_classical * ncc + w_exp3 * exp3
    order = np.argsort(-score)
    return [valid[i] for i in order]


class TrainedRankingHead:
    """Small linear ranking model: score = w . [classical_ncc_z, exp3_logit_z,
    ncc_gap_z, disagreement_z] + b, fit by minimizing a pairwise hinge loss over
    within-query candidate pairs weighted by their relevance difference (a
    RankSVM-style objective), using only rows from the ALLOWED skies of one fold.
    This is a linear model (no new CNN, no descriptor retraining), matching the
    task's "small calibrated ranking model" and "do not replace the full CNN or
    rerun descriptor triplet training" constraints.
    """

    FEATURES = ('classical_ncc', 'exp3_pair_logit', 'ncc_gap_to_second',
               'exp3_ensemble_disagreement')

    def __init__(self):
        self.weights = np.zeros(len(self.FEATURES))
        self.bias = 0.0
        self.feature_mean = np.zeros(len(self.FEATURES))
        self.feature_scale = np.ones(len(self.FEATURES))

    def _feature_vector(self, c: dict) -> np.ndarray:
        return np.array([
            c['classical_ncc'] if c['classical_ncc'] is not None else 0.0,
            c['exp3_pair_logit'] if c['exp3_pair_logit'] is not None else 0.0,
            c['ncc_gap_to_second'] if c['ncc_gap_to_second'] is not None else 0.0,
            c['exp3_ensemble_disagreement'] if c['exp3_ensemble_disagreement'] is not None else 0.0,
        ], float)

    def fit(self, rows: list, seed: int, lr: float = 0.05, epochs: int = 200,
           reg: float = 0.01) -> dict:
        """Pairwise hinge-loss gradient descent over within-query candidate
        pairs; deterministic (fixed init, fixed pair order, no shuffling RNG
        beyond a SHA-256-derived seed used only to break literal weight ties)."""
        from experiments.exp1.env import derive_seed
        pairs = []   # (feat_i, feat_j, relevance_i - relevance_j)
        for row in rows:
            if not row['present']:
                continue
            valid = [c for c in row['candidates'] if c['valid']
                    and c['exp3_pair_logit'] is not None]
            if len(valid) < 2:
                continue
            rel = {c['index']: relevance(c['distance_to_truth']) for c in valid}
            feats = {c['index']: self._feature_vector(c) for c in valid}
            idxs = list(rel)
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    ia, ib = idxs[a], idxs[b]
                    dr = rel[ia] - rel[ib]
                    if abs(dr) < 1e-9:
                        continue
                    if dr > 0:
                        pairs.append((feats[ia], feats[ib], 1.0))
                    else:
                        pairs.append((feats[ib], feats[ia], 1.0))

        if not pairs:
            return {'ok': False, 'reason': 'no informative pairs (all relevances tied)'}

        all_feats = np.array([p[0] for p in pairs] + [p[1] for p in pairs])
        self.feature_mean = all_feats.mean(axis=0)
        self.feature_scale = np.where(all_feats.std(axis=0) < 1e-9, 1.0, all_feats.std(axis=0))

        w = np.zeros(len(self.FEATURES))
        rng_seed = derive_seed(seed, 'exp4b-ranking-head', 'init')
        w += (rng_seed % 1000) * 1e-6   # tiny deterministic perturbation, breaks exact ties
        for epoch in range(epochs):
            grad = np.zeros(len(self.FEATURES))
            n_active = 0
            for fi, fj, margin_target in pairs:
                zi = (fi - self.feature_mean) / self.feature_scale
                zj = (fj - self.feature_mean) / self.feature_scale
                margin = w @ (zi - zj)
                if margin < 1.0:
                    grad += -(zi - zj)
                    n_active += 1
            grad = grad / max(len(pairs), 1) + reg * w
            w -= lr * grad
        self.weights = w
        self.bias = 0.0
        return {'ok': True, 'n_pairs': len(pairs), 'weights': w.tolist(),
               'feature_names': list(self.FEATURES)}

    def score(self, c: dict) -> float:
        z = (self._feature_vector(c) - self.feature_mean) / self.feature_scale
        return float(self.weights @ z)

    def rank(self, row: dict) -> list:
        valid = [c for c in row['candidates'] if c['valid']
                and c['exp3_pair_logit'] is not None]
        if not valid:
            return []
        scored = [(c, self.score(c)) for c in valid]
        return [c for c, _ in sorted(scored, key=lambda cs: -cs[1])]


def rank_by_absent_separated(row: dict) -> tuple:
    """Rule 7: present-candidate ranking uses linear fusion (Rule 5); the
    ABSENT decision is scored SEPARATELY (never mixed into the same linear
    score used for within-present ranking) via best_exp3_logit vs
    exp3_absent_logit margin."""
    ranked_present = rank_by_linear_fusion(row)
    valid = [c for c in row['candidates'] if c['valid'] and c['exp3_pair_logit'] is not None]
    absent_logit = valid[0]['exp3_absent_logit'] if valid else None
    best_logit = max((c['exp3_pair_logit'] for c in valid), default=None)
    absent_margin = (best_logit - absent_logit) if (best_logit is not None
                                                     and absent_logit is not None) else None
    return ranked_present, absent_margin


RULES = {
    '1_classical_rank': rank_by_classical,
    '2_exp3_rank': rank_by_exp3,
    '3_mean_rank_fusion': rank_by_mean_rank_fusion,
    '4_reciprocal_rank_fusion': rank_by_rrf,
    '5_linear_score_fusion': rank_by_linear_fusion,
}

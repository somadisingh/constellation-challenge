"""Leak-free normalization and regularized hypothesis fusion models."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize

from . import C_GRID
from .hypothesis_dataset import FEATURES

NEGATIVE_FEATURES = {'mean_residual', 'stability_mean_shift_px', 'unmatched_nodes',
                     'node_count', 'pool_size', 'attempted_hypotheses',
                     'accepted_hypotheses', 'cap_hit', 'local_star_density',
                     'multiple_testing_term'}
POSITIVE_FEATURES = set(FEATURES) - NEGATIVE_FEATURES

ARM_SPECS = {
    '1_additive_baseline': {'kind': 'additive', 'normalization': 'raw'},
    '2_logistic': {'kind': 'logistic', 'normalization': 'standard'},
    '3_class_balanced_logistic': {'kind': 'logistic', 'normalization': 'standard', 'class_balanced': True},
    '4_pairwise_logistic': {'kind': 'pairwise', 'normalization': 'robust'},
    '5_synthetic_pretrained': {'kind': 'synthetic_anchor', 'normalization': 'robust_null'},
    '6_constrained_sign_logistic': {'kind': 'logistic', 'normalization': 'robust', 'constrained': True},
    '7_without_unqueried': {'kind': 'logistic', 'normalization': 'robust_null', 'drop': 'unqueried'},
    '8_without_appearance': {'kind': 'logistic', 'normalization': 'robust_null', 'drop': 'appearance'},
    '9_without_seed_excluded_geometry': {'kind': 'logistic', 'normalization': 'robust_null', 'drop': 'geometry'},
    '10_complete_calibrated': {'kind': 'logistic', 'normalization': 'robust_null', 'class_balanced': True},
}


def feature_names_for(spec: dict) -> list[str]:
    names = list(FEATURES)
    if spec.get('drop') == 'unqueried':
        names = [x for x in names if x not in {'corrected_unqueried', 'unqueried_raw_count',
                                                'n_unqueried', 'n_unqueried_scored'}]
    if spec.get('drop') == 'appearance':
        names = [x for x in names if x not in {'classical_appearance','exp3_appearance',
                                                'exp3_probability','exp3_disagreement',
                                                'rank_fusion_appearance'}]
    if spec.get('drop') == 'geometry':
        names = [x for x in names if x not in {'total_support', 'seed_support',
                                                'held_out_support', 'support_fraction',
                                                'mean_residual', 'stability_mean_shift_px',
                                                'unique_sources', 'coverage'}]
    return names


@dataclass
class Normalizer:
    kind: str
    names: list[str]
    centre: np.ndarray | None = None
    scale: np.ndarray | None = None
    impute: np.ndarray | None = None

    def fit(self, records: list[dict]):
        x = np.array([[r['features'].get(n, np.nan) if r['features'].get(n) is not None else np.nan
                       for n in self.names] for r in records], float)
        self.impute = np.nanmedian(x, axis=0)
        self.impute = np.where(np.isfinite(self.impute), self.impute, 0.0)
        x = np.where(np.isfinite(x), x, self.impute)
        if self.kind in ('robust', 'robust_null'):
            self.centre = np.median(x, axis=0)
            q25, q75 = np.percentile(x, [25, 75], axis=0)
            self.scale = np.where(q75 - q25 > 1e-9, q75 - q25, 1.0)
        elif self.kind == 'standard':
            self.centre = x.mean(axis=0)
            self.scale = np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
        else:
            self.centre = np.zeros(x.shape[1])
            self.scale = np.ones(x.shape[1])
        return self

    def transform(self, records: list[dict]) -> np.ndarray:
        x = np.array([[r['features'].get(n, np.nan) if r['features'].get(n) is not None else np.nan
                       for n in self.names] for r in records], float)
        missing = ~np.isfinite(x)
        x = np.where(np.isfinite(x), x, self.impute)
        if self.kind == 'rank':
            out = np.zeros_like(x)
            scenes = [r['scene'] for r in records]
            for scene in sorted(set(scenes)):
                ids = np.array([i for i, s in enumerate(scenes) if s == scene])
                for j in range(x.shape[1]):
                    order = np.argsort(np.argsort(x[ids, j], kind='stable'), kind='stable')
                    out[ids, j] = order / max(len(ids) - 1, 1)
            x = out
        else:
            x = (x - self.centre) / self.scale
        return np.c_[x, missing.astype(float)]

    def as_dict(self):
        return {'kind': self.kind, 'names': self.names,
                'centre': self.centre.tolist(), 'scale': self.scale.tolist(),
                'impute': self.impute.tolist(),
                'output_names': self.names + [f'{n}__missing' for n in self.names]}


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove exact/near duplicate hypothesis feature rows deterministically."""
    seen, out = set(), []
    for r in records:
        vals = tuple(None if r['features'].get(n) is None else round(float(r['features'][n]), 6)
                     for n in FEATURES)
        key = (r['scene'], r['class_name'], vals, bool(r['placement_correct']))
        if key not in seen:
            seen.add(key); out.append(r)
    return out


def sample_weights(records: list[dict], class_balanced: bool = False) -> np.ndarray:
    """Equal sky/class mass, with optional label balance capped per class.

    Exact positive/negative balancing can otherwise give the sole positive
    constellation nearly all training mass.  After label reweighting we cap a
    class at four times the original equal-class mass and renormalize each sky
    back to equal total mass.
    """
    w = np.zeros(len(records), float)
    scenes = sorted(set(r['scene'] for r in records))
    for scene in scenes:
        scene_ids = [i for i, r in enumerate(records) if r['scene'] == scene]
        classes = sorted(set(records[i]['class_name'] for i in scene_ids))
        for cls in classes:
            ids = [i for i in scene_ids if records[i]['class_name'] == cls]
            for i in ids:
                w[i] = 1.0 / max(len(scenes) * len(classes) * len(ids), 1)
    if class_balanced:
        y = np.array([r['placement_correct'] for r in records], int)
        for label in (0, 1):
            mass = w[y == label].sum()
            if mass > 0:
                w[y == label] *= 0.5 / mass
        # Bound correlated class groups after label balancing, then bring each
        # sky back to equal total mass WITHOUT letting the renormalization
        # push a capped class back over its bound: redistribute the scene's
        # target mass only across classes not currently at the cap
        # (water-filling), iterating to convergence since freeing capacity in
        # one round can itself need a further cap in the next.
        for scene in scenes:
            ids_scene = [i for i, r in enumerate(records) if r['scene'] == scene]
            classes = sorted(set(records[i]['class_name'] for i in ids_scene))
            n_scenes_total = max(len(scenes), 1)
            target = 1.0 / (n_scenes_total * max(len(classes), 1))
            cap = 4.0 * target
            scene_target_mass = 1.0 / n_scenes_total
            class_ids = {cls: [i for i in ids_scene if records[i]['class_name'] == cls]
                        for cls in classes}
            capped = set()
            for _ in range(len(classes) + 1):
                free_classes = [c for c in classes if c not in capped]
                if not free_classes:
                    break
                free_mass_now = sum(w[class_ids[c]].sum() for c in free_classes)
                remaining_target = scene_target_mass - sum(
                    min(w[class_ids[c]].sum(), cap) for c in capped)
                if free_mass_now <= 1e-12:
                    break
                scale = remaining_target / free_mass_now
                newly_capped = []
                for c in free_classes:
                    ids = class_ids[c]
                    new_mass = w[ids].sum() * scale
                    if new_mass > cap + 1e-12:
                        # clamp this class to the cap now; its excess mass is
                        # redistributed among the still-free classes next round
                        current = w[ids].sum()
                        if current > 1e-12:
                            w[ids] *= cap / current
                        newly_capped.append(c)
                if not newly_capped:
                    for c in free_classes:
                        w[class_ids[c]] *= scale
                    break
                capped.update(newly_capped)
    w /= max(w.sum(), 1e-12)
    return w


class LogisticFusion:
    def __init__(self, names: list[str], normalization='standard', c=0.1,
                 class_balanced=False, constrained=False, pairwise=False,
                 synthetic_anchor=False):
        self.names = names; self.normalizer = Normalizer(normalization, names)
        self.c = float(c); self.class_balanced = class_balanced
        self.constrained = constrained; self.pairwise = pairwise
        self.synthetic_anchor = synthetic_anchor
        self.coef_ = None; self.intercept_ = 0.0; self.fit_info = {}

    def fit(self, records: list[dict]):
        records = deduplicate(records)
        self.normalizer.fit(records)
        x = self.normalizer.transform(records)
        y = np.array([r['placement_correct'] for r in records], float)
        if y.sum() == 0 or y.sum() == len(y):
            self.fit_info = {'ok': False, 'reason': 'one-class training fold',
                             'n': len(y), 'n_positive': int(y.sum())}
            return self
        w = sample_weights(records, self.class_balanced)

        if self.pairwise:
            px, pw = [], []
            for scene in sorted(set(r['scene'] for r in records)):
                pos = [i for i, r in enumerate(records) if r['scene'] == scene and r['placement_correct']]
                neg = [i for i, r in enumerate(records) if r['scene'] == scene and not r['placement_correct']]
                # Strongest negatives under held-out support; deterministic cap.
                neg = sorted(neg, key=lambda i: (-records[i]['features']['held_out_support'],
                                                  records[i]['class_name']))[:64]
                for i in pos:
                    for j in neg:
                        px.append(x[i] - x[j]); pw.append(w[i] + w[j])
            if not px:
                self.fit_info = {'ok': False, 'reason': 'no pairwise examples'}; return self
            xfit = np.asarray(px); yfit = np.ones(len(px)); wfit = np.asarray(pw)
            fit_intercept = False
        else:
            xfit, yfit, wfit, fit_intercept = x, y, w, True

        if self.synthetic_anchor:
            # Monotonic engineering anchors stabilize signs without pretending
            # to be real hypotheses: high support/coverage and low residual/
            # instability form a positive contrast against the reverse.
            d = xfit.shape[1]; pos = np.zeros(d); neg = np.zeros(d)
            for j, n in enumerate(self.names):
                if n in POSITIVE_FEATURES: pos[j] = 1; neg[j] = -1
                elif n in NEGATIVE_FEATURES: pos[j] = -1; neg[j] = 1
            xfit = np.vstack([xfit, pos, neg]); yfit = np.r_[yfit, 1., 0.]
            wfit = np.r_[wfit, .05, .05]

        reg = 1.0 / max(self.c, 1e-12)
        def objective(theta):
            beta = theta[:-1] if fit_intercept else theta
            b = theta[-1] if fit_intercept else 0.0
            z = np.clip(xfit @ beta + b, -40, 40)
            loss = np.logaddexp(0, z) - yfit * z
            val = float(np.sum(wfit * loss) / max(wfit.sum(), 1e-12) + 0.5 * reg * np.dot(beta, beta))
            p = 1 / (1 + np.exp(-z))
            gb = xfit.T @ (wfit * (p - yfit)) / max(wfit.sum(), 1e-12) + reg * beta
            if fit_intercept:
                return val, np.r_[gb, np.sum(wfit * (p - yfit)) / max(wfit.sum(), 1e-12)]
            return val, gb
        npar = xfit.shape[1] + int(fit_intercept)
        bounds = None
        if self.constrained:
            bounds = []
            for n in self.names + [f'{n}__missing' for n in self.names]:
                bounds.append((0, None) if n in POSITIVE_FEATURES else
                              ((None, 0) if n in NEGATIVE_FEATURES else (None, None)))
            if fit_intercept: bounds.append((None, None))
        result = minimize(lambda t: objective(t), np.zeros(npar), jac=True,
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12})
        self.coef_ = result.x[:-1] if fit_intercept else result.x
        self.intercept_ = float(result.x[-1]) if fit_intercept else 0.0
        self.fit_info = {'ok': bool(result.success), 'message': str(result.message),
                         'n': len(records), 'n_positive': int(y.sum()),
                         'objective': float(result.fun)}
        return self

    def decision_function(self, records: list[dict]) -> np.ndarray:
        if self.coef_ is None:
            return np.full(len(records), -np.inf)
        return self.normalizer.transform(records) @ self.coef_ + self.intercept_

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        z = np.clip(self.decision_function(records), -40, 40)
        return 1 / (1 + np.exp(-z))

    def as_dict(self):
        return {'names': self.names, 'c': self.c, 'class_balanced': self.class_balanced,
                'constrained': self.constrained, 'pairwise': self.pairwise,
                'synthetic_anchor': self.synthetic_anchor,
                'normalizer': self.normalizer.as_dict(),
                'coef': None if self.coef_ is None else self.coef_.tolist(),
                'intercept': self.intercept_, 'fit_info': self.fit_info}


def fit_arm(records: list[dict], arm: str, c: float):
    spec = ARM_SPECS[arm]
    if spec['kind'] == 'additive':
        return None
    return LogisticFusion(
        feature_names_for(spec), spec['normalization'], c,
        class_balanced=spec.get('class_balanced', False),
        constrained=spec.get('constrained', False),
        pairwise=spec['kind'] == 'pairwise',
        synthetic_anchor=spec['kind'] == 'synthetic_anchor').fit(records)


def score_records(records: list[dict], arm: str, model=None) -> np.ndarray:
    if arm == '1_additive_baseline':
        return np.array([r['features']['classical_appearance'] +
                         r['features']['held_out_support'] +
                         r['features']['corrected_unqueried'] for r in records])
    return model.decision_function(records)


def rank_classes(records: list[dict], scores: np.ndarray) -> list[dict]:
    best = {}
    for r, score in zip(records, scores):
        key = r['class_name']
        node = {'name': key, 'score': float(score),
                'placement_correct': bool(r['placement_correct']),
                'matrix': r['matrix'], 'mapped_nodes': r['mapped_nodes']}
        if key not in best or score > best[key]['score']:
            best[key] = node
    return sorted(best.values(), key=lambda x: (-x['score'], x['name']))

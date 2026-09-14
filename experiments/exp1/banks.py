"""Blind candidate banks for real queries (plan §7).

One bank is frozen before any model comparison. For each real query it is the
union of three branches from the SAME blind proposal source:

  fixed_coarse   outputs/hybrid/{scene}.json          candidates (verify, fixed radius)
  fixed_ecc      outputs/hybrid_ecc/{scene}.json      candidates (ECC-refined)
  adaptive       outputs/lab/adaptive_train/{scene}.json  coarse_candidates (verify_adaptive)

so at most 60 entries before exact-duplicate removal. Distinct close stars are
kept; only exact coordinate duplicates are merged, and a merged entry records
every branch and pose that produced it.

No true position is ever injected. `check_no_oracle` asserts that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import SCENES
from .env import ROOT, read_json, sha256_file

BRANCH_SOURCES = {
    'fixed_coarse': ('outputs/hybrid/{scene}.json', 'candidates'),
    'fixed_ecc': ('outputs/hybrid_ecc/{scene}.json', 'candidates'),
    'adaptive': ('outputs/lab/adaptive_train/{scene}.json', 'coarse_candidates'),
}
DUP_TOL = 1e-6      # exact duplicate only; close stars stay distinct


def branch_paths(scene: str) -> dict:
    return {b: ROOT / tpl.format(scene=scene) for b, (tpl, _) in BRANCH_SOURCES.items()}


def missing_branches(scenes=SCENES) -> list:
    out = []
    for s in scenes:
        for b, p in branch_paths(s).items():
            if not p.exists():
                out.append({'scene': s, 'branch': b, 'path': str(p)})
    return out


def load_branch_candidates(scene: str) -> dict:
    """{branch: [ [ (x,y,score,angle,scale), ... ] per query ]}."""
    out = {}
    for branch, (tpl, field) in BRANCH_SOURCES.items():
        path = ROOT / tpl.format(scene=scene)
        if not path.exists():
            raise FileNotFoundError(
                f'missing {branch} cache {path}. Regenerate with the command in '
                f'the bank manifest before continuing.')
        doc = read_json(path)
        queries = doc['diagnostics']['queries']
        out[branch] = [[tuple(float(v) for v in c) for c in q[field]] for q in queries]
    return out


def build_bank(scene: str) -> dict:
    """Frozen union bank for one scene, with per-candidate branch/pose provenance."""
    branches = load_branch_candidates(scene)
    n_queries = {b: len(v) for b, v in branches.items()}
    if len(set(n_queries.values())) != 1:
        raise ValueError(f'{scene}: branch query counts disagree {n_queries}')
    n = next(iter(n_queries.values()))
    queries = []
    for i in range(n):
        merged = []
        for branch, per_query in branches.items():
            for rank, (x, y, score, angle, scale) in enumerate(per_query[i]):
                hit = None
                for m in merged:
                    if abs(m['x'] - x) <= DUP_TOL and abs(m['y'] - y) <= DUP_TOL:
                        hit = m
                        break
                if hit is None:
                    merged.append({'x': float(x), 'y': float(y),
                                   'provenance': [{'branch': branch, 'rank': rank,
                                                   'score': float(score),
                                                   'angle': float(angle),
                                                   'scale': float(scale)}]})
                else:
                    hit['provenance'].append({'branch': branch, 'rank': rank,
                                              'score': float(score),
                                              'angle': float(angle),
                                              'scale': float(scale)})
        for m in merged:
            best = max(m['provenance'], key=lambda p: p['score'])
            m['base_angle'] = best['angle']
            m['base_scale'] = best['scale']
            m['classical_score'] = best['score']
            m['branches'] = sorted({p['branch'] for p in m['provenance']})
        queries.append({'index': i, 'n_candidates': len(merged), 'candidates': merged})
    return {
        'scene': scene,
        'n_queries': n,
        'branch_sources': {b: {'path': str(Path(tpl.format(scene=scene))),
                               'field': f,
                               'sha256': sha256_file(ROOT / tpl.format(scene=scene))}
                           for b, (tpl, f) in BRANCH_SOURCES.items()},
        'dup_tolerance': DUP_TOL,
        'queries': queries,
        'sizes': {'min': min(q['n_candidates'] for q in queries),
                  'max': max(q['n_candidates'] for q in queries),
                  'mean': float(np.mean([q['n_candidates'] for q in queries]))},
    }


def bank_xy(bank: dict, index: int) -> np.ndarray:
    return np.array([[c['x'], c['y']] for c in bank['queries'][index]['candidates']],
                    float).reshape(-1, 2)


def bank_poses(bank: dict, index: int) -> np.ndarray:
    return np.array([[c['base_angle'], c['base_scale']]
                     for c in bank['queries'][index]['candidates']], float).reshape(-1, 2)


def bank_scores(bank: dict, index: int) -> np.ndarray:
    return np.array([c['classical_score']
                     for c in bank['queries'][index]['candidates']], float)


# --- recall accounting (plan §7) --------------------------------------------------
def branch_recall(scene: str, records: list, radii=(4, 12, 36)) -> dict:
    """recall@r of each branch and of the union, overall and by stratum.

    A learned reranker cannot fix a location absent from its bank, so this bounds
    every downstream claim.
    """
    branches = load_branch_candidates(scene)
    bank = build_bank(scene)
    present = [r for r in records if r['present']]
    out = {}
    sets = dict(branches)
    sets['union'] = [[(c['x'], c['y'], c['classical_score'], c['base_angle'],
                       c['base_scale']) for c in q['candidates']]
                     for q in bank['queries']]
    for name, per_query in sets.items():
        entry = {}
        for radius in radii:
            hits, strata = [], {}
            for r in present:
                xy = np.array(per_query[r['index']], float).reshape(-1, 5)[:, :2]
                ok = bool(len(xy) and
                          np.linalg.norm(xy - np.array(r['xy']), axis=1).min() <= radius)
                hits.append(ok)
                strata.setdefault(r['stratum'], []).append(ok)
            entry[f'recall@{radius}'] = float(np.mean(hits)) if hits else float('nan')
            for k, v in strata.items():
                entry[f'recall@{radius}:{k}'] = float(np.mean(v))
        entry['n_present'] = len(present)
        out[name] = entry
    return out


def check_no_oracle(bank: dict, records: list, tol: float = 1e-9) -> dict:
    """No true position may enter the bank.

    The meaningful invariant is PROVENANCE COMPLETENESS: every bank entry must
    trace back to an entry in one of the three branch caches, and those caches were
    produced by label-free inference. Set equality with the recomputed union proves
    nothing was added.

    Exact coincidence between a candidate and a label is NOT evidence of injection
    and is only counted, not failed: the dataset labels are integer pixels and the
    classical star detector also fires on integer pixels, so a detector that finds
    the star at its true centre reproduces the label exactly by construction.
    """
    branches = load_branch_candidates(bank['scene'])
    problems = []
    for q in bank['queries']:
        i = q['index']
        recomputed = set()
        for per_query in branches.values():
            for x, y, *_ in per_query[i]:
                recomputed.add((round(float(x), 6), round(float(y), 6)))
        present = set()
        for c in q['candidates']:
            key = (round(c['x'], 6), round(c['y'], 6))
            present.add(key)
            if not c['provenance']:
                problems.append(f'query {i}: candidate {key} has no provenance')
            for p in c['provenance']:
                if p['branch'] not in BRANCH_SOURCES:
                    problems.append(f'query {i}: unknown branch {p["branch"]}')
        added = present - recomputed
        if added:
            problems.append(f'query {i}: {len(added)} candidates are not in any '
                            f'branch cache, e.g. {sorted(added)[:3]}')
    coincidences = []
    for r in records:
        if not r['present']:
            continue
        xy = bank_xy(bank, r['index'])
        if not len(xy):
            continue
        d = np.linalg.norm(xy - np.array(r['xy']), axis=1)
        n_exact = int((d <= tol).sum())
        if n_exact:
            coincidences.append({'query': r['query_id'], 'n_exact': n_exact})
    return {
        'ok': not problems,
        'problems': problems,
        'provenance_complete': not problems,
        'integer_grid_coincidences': coincidences,
        'n_integer_grid_coincidences': len(coincidences),
        'coincidence_note': ('labels are integer pixels and the classical detector '
                             'also fires on integer pixels, so exact agreement is '
                             'expected and is not injection'),
        'tolerance': tol,
    }

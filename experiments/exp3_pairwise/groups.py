"""Leakage-proof training/validation query groups (task §4, §6).

A "query group" is one degraded 32x32 query plus its full masked candidate set,
each candidate labelled by geometric distance to the known source centre:

  positive  : <= POSITIVE_RADIUS (12px)
  ignore    : > POSITIVE_RADIUS and <= IGNORE_RADIUS (36px); excluded from the
              classification loss, never treated as negative
  negative  : > IGNORE_RADIUS

For an absent group the query is generated from the OTHER allowed sky and
candidates are retrieved in the TARGET sky; every returned candidate is a valid
negative because the true source does not exist there.

Retrieval is the ACTUAL blind proposal/verification pipeline
(`experiments.exp1.search.regional_candidates`), restricted to one partition of
one sky exactly as Experiment 1's mining does, so a mined negative is a real
confuser the classical system produces, not a synthetic decoy. A query group
never contains a candidate coordinate that this retrieval did not itself return.

Held-out pixels never enter this module: every source record passed in must
already come from `experiments.exp1.splits.partition_records` restricted to the
two allowed skies of the fold in question (asserted by the caller and re-checked
here defensively).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import IGNORE_RADIUS, POSITIVE_RADIUS
from experiments.exp1.data import load_scene
from experiments.exp1.mining import label_candidate, IGNORE, NEGATIVE, POSITIVE as POS_LABEL
from experiments.exp1.pose import SceneReps, aligned_candidate, prepare_query, select_pose
from experiments.exp1.search import regional_candidates, assert_partition
from experiments.exp1.splits import fold_skies, partition_of, cell_box

CANDIDATE_KEEP = 20     # matches Exp1's bank size convention; bounds compute


@dataclass(eq=False)
class QueryGroup:
    group_id: str
    kind: str                       # 'present' | 'absent'
    fold: str
    target_scene: str                # sky searched for candidates
    source_scene: str                 # sky the query pixels actually came from
    source_id: str
    centre: tuple | None             # true (x, y) in target_scene, None if absent
    query: np.ndarray                # (32, 32) float32 raw query, [0, 1]
    xy: np.ndarray                   # (k, 2) candidate coordinates
    crops: np.ndarray                # (k, 32, 32) float32 aligned candidate crops, [0,1]
    admissible: np.ndarray           # (k,) bool: pose read entirely inside the image
    classical_ncc: np.ndarray        # (k,) masked NCC, the classical control score
    labels: list                     # (k,) 'positive' | 'ignore' | 'negative'
    distance: np.ndarray             # (k,) geometric distance to truth, nan if absent
    pool_missing: bool               # True if kind=='present' and no positive was found
    degenerate: bool

    def __len__(self):
        return len(self.xy)

    def positive_indices(self) -> np.ndarray:
        return np.array([i for i, l in enumerate(self.labels) if l == POS_LABEL], int)

    def negative_indices(self) -> np.ndarray:
        return np.array([i for i, l in enumerate(self.labels) if l == NEGATIVE], int)

    def valid_mask(self) -> np.ndarray:
        return self.admissible


def _align_candidates(reps: SceneReps, query_uint8: np.ndarray, found: list) -> dict:
    """Align every retrieved candidate into the query frame (reuses Exp1's adapter)."""
    qraw, qblur = prepare_query(query_uint8)
    n = len(found)
    xy = np.zeros((n, 2), float)
    crops = np.zeros((n, 32, 32), np.float32)
    admissible = np.zeros(n, bool)
    ncc = np.full(n, np.nan, float)
    for i, (x, y, score, angle, scale) in enumerate(found):
        xy[i] = (x, y)
        sel = select_pose(reps, qblur, (x, y), (angle, scale))
        if sel['pose'] is None:
            continue
        crop, mask = aligned_candidate(reps.raw, (x, y), sel['pose'])
        if not mask.all():
            continue
        admissible[i] = True
        crops[i] = crop / 255.0 if crop.max() > 1.5 else crop
        ncc[i] = sel['ncc']
    return {'qraw': qraw, 'xy': xy, 'crops': crops, 'admissible': admissible,
            'ncc': ncc}


def build_present_group(target_scene: str, source_record: dict, rng,
                        recipe: dict | None, fold: str, partition: str = 'fit',
                        data: str | None = None) -> QueryGroup:
    """One present group: source pixels and candidates both from `target_scene`."""
    from experiments.exp1b.data import generate

    assert source_record['scene'] == target_scene
    assert partition_of(source_record['cell']) == partition, (
        f'source record cell {source_record["cell"]} is not in partition {partition!r}: '
        'a group must never be built from pixels outside its assigned partition')

    scene_obj = load_scene(target_scene, data)
    example = generate(scene_obj.image, source_record, rng, recipe)
    query_uint8 = (example['q'] * 255.0).round().astype(np.uint8)

    found = regional_candidates(target_scene, partition, query_uint8,
                                keep=CANDIDATE_KEEP, data=data)
    check = assert_partition(target_scene, partition, found, data)
    if not check['ok']:
        raise RuntimeError(f'{target_scene}: candidate left partition {partition}: {check}')

    reps = SceneReps(scene_obj.image)
    aligned = _align_candidates(reps, query_uint8, found)
    truth = np.asarray(source_record['xy'], float)
    distance = (np.linalg.norm(aligned['xy'] - truth, axis=1) if len(found)
               else np.zeros(0))
    labels = [label_candidate(float(d), POSITIVE_RADIUS, IGNORE_RADIUS)
             for d in distance]
    pool_missing = not any(l == POS_LABEL for l in labels)

    return QueryGroup(
        group_id=f'{fold}:present:{source_record["source_id"]}',
        kind='present', fold=fold, target_scene=target_scene,
        source_scene=target_scene, source_id=source_record['source_id'],
        centre=(float(truth[0]), float(truth[1])),
        query=aligned['qraw'], xy=aligned['xy'], crops=aligned['crops'],
        admissible=aligned['admissible'], classical_ncc=aligned['ncc'],
        labels=labels, distance=distance, pool_missing=pool_missing,
        degenerate=bool(example.get('q', np.zeros((1,))).std() < 2.0 / 255.0))


def build_absent_group(target_scene: str, donor_record: dict, rng,
                       recipe: dict | None, fold: str, partition: str = 'fit',
                       data: str | None = None) -> QueryGroup:
    """One absent group: query pixels from the DONOR sky, candidates from `target_scene`."""
    from experiments.exp1b.data import generate

    donor_scene = donor_record['scene']
    assert donor_scene != target_scene, 'an absent group must search a DIFFERENT sky'
    assert partition_of(donor_record['cell']) == partition

    donor_image = load_scene(donor_scene, data).image
    example = generate(donor_image, donor_record, rng, recipe)
    query_uint8 = (example['q'] * 255.0).round().astype(np.uint8)

    found = regional_candidates(target_scene, partition, query_uint8,
                                keep=CANDIDATE_KEEP, data=data)
    check = assert_partition(target_scene, partition, found, data)
    if not check['ok']:
        raise RuntimeError(f'{target_scene}: candidate left partition {partition}: {check}')

    reps = SceneReps(load_scene(target_scene, data).image)
    aligned = _align_candidates(reps, query_uint8, found)
    labels = [NEGATIVE] * len(found)   # truth is not in target_scene: all negative

    return QueryGroup(
        group_id=f'{fold}:absent:{donor_record["source_id"]}->{target_scene}',
        kind='absent', fold=fold, target_scene=target_scene,
        source_scene=donor_scene, source_id=donor_record['source_id'], centre=None,
        query=aligned['qraw'], xy=aligned['xy'], crops=aligned['crops'],
        admissible=aligned['admissible'], classical_ncc=aligned['ncc'],
        labels=labels, distance=np.full(len(found), np.nan),
        pool_missing=False,
        degenerate=bool(example.get('q', np.zeros((1,))).std() < 2.0 / 255.0))


def assert_fold_isolated(fold: str, groups: list) -> dict:
    """No group may touch the held-out sky, directly or as a candidate target."""
    held_out, allowed = fold_skies(fold)
    problems = []
    for g in groups:
        if g.target_scene == held_out or g.source_scene == held_out:
            problems.append(f'{g.group_id}: touches held-out sky {held_out!r}')
    return {'ok': not problems, 'problems': problems, 'held_out': held_out,
           'allowed': list(allowed), 'n_groups': len(groups)}

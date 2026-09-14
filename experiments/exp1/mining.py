"""Hard-negative banks and inner-validation banks (plan §7, §11).

Two artifacts, both built from real pixels by the existing classical machinery:

* Mined negative bank, per SCENE, from that scene's FIT cells only. 200 synthetic
  queries per sky, balanced by source stratum, each with up to 8 highest-scoring
  wrong locations and 2 random wrong locations. It is per scene rather than per
  fold because a scene's fit cells do not depend on the fold, and the held-out sky
  is never searched; each fold consumes only the two scenes it is allowed.

* Inner-validation bank, per FOLD, from VAL cells only. 128 queries per allowed
  sky at 1:1 present/absent with frozen source identities. Absent queries are
  generated from the OTHER allowed sky's val partition and searched in this sky, so
  their true source genuinely is not present here. Both classes receive identical
  degradation.

Label policy (plan §7): a candidate within `positive_radius` of the true centre is
the positive; anything from there out to `ignore_radius` is IGNORED rather than
labelled negative, because distinct close stars are hard cases and must not be
called negative from a radius threshold alone. Only beyond `ignore_radius` is a
location a valid negative. The excluded ambiguity rate is reported.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import SCENES
from .augment import (Ranges, degradation_stats, sample_degradation, sample_pose,
                      synthesize_query)
from .data import load_scene
from .env import (SEED_AUGMENT, SEED_INNER_EVAL, SEED_SAMPLING, rng, write_json,
                  read_json)
from .search import assert_partition, cells_of, regional_candidates
from .splits import build_source_bank, fold_skies, partition_records

POSITIVE = 'positive'
IGNORE = 'ignore'
NEGATIVE = 'negative'


def label_candidate(distance: float, positive_radius: float,
                    ignore_radius: float) -> str:
    if distance <= positive_radius:
        return POSITIVE
    if distance <= ignore_radius:
        return IGNORE
    return NEGATIVE


def _stratified_sources(records: list, count: int, generator) -> list:
    """Round-robin over source strata so the query set is stratum balanced."""
    buckets: dict = {}
    for r in records:
        buckets.setdefault((r['source_class'], r['stratum']), []).append(r)
    keys = sorted(buckets)
    for k in keys:
        arr = buckets[k]
        order = generator.permutation(len(arr))
        buckets[k] = [arr[i] for i in order]
    out, cursor = [], {k: 0 for k in keys}
    while len(out) < count:
        progressed = False
        for k in keys:
            if len(out) >= count:
                break
            if cursor[k] < len(buckets[k]):
                out.append(buckets[k][cursor[k]])
                cursor[k] += 1
                progressed = True
        if not progressed:
            break
    return out


def _random_negatives(scene: str, partition: str, truth_xy, count: int,
                      generator, ignore_radius: float,
                      data: str | None = None) -> list:
    """Random wrong locations inside the partition, beyond the ignore radius."""
    bank = build_source_bank(scene, data)
    pool = partition_records(bank, partition)
    out, guard = [], 0
    while len(out) < count and guard < count * 200:
        guard += 1
        r = pool[int(generator.integers(0, len(pool)))]
        if truth_xy is not None:
            d = float(np.hypot(r['xy'][0] - truth_xy[0], r['xy'][1] - truth_xy[1]))
            if d <= ignore_radius:
                continue
        out.append({'x': r['xy'][0], 'y': r['xy'][1], 'source_id': r['source_id'],
                    'kind': 'random'})
    return out


def build_mined_bank(scene: str, config: dict, data: str | None = None,
                     progress=None) -> dict:
    """Mined negatives for one scene from its FIT cells (plan §7)."""
    cfg = config['mining']
    n_queries = int(cfg['queries_per_sky'])
    ranges = Ranges(**{k: (tuple(v) if isinstance(v, list) else v)
                       for k, v in config['augment'].items()
                       if k in Ranges().as_dict()})
    scene_obj = load_scene(scene, data)
    bank = build_source_bank(scene, data)
    fit = partition_records(bank, 'fit')
    gen = rng(SEED_SAMPLING, 'mine-sources', scene)
    chosen = _stratified_sources(fit, n_queries, gen)

    aug = rng(SEED_AUGMENT, 'mine', scene)
    queries, images, synth_records = [], [], []
    started = time.perf_counter()
    for k, source in enumerate(chosen):
        pose = sample_pose(aug, ranges)
        degradation = sample_degradation(aug, ranges)
        synth = synthesize_query(scene_obj.image, source['xy'], pose, degradation)
        synth_records.append(synth)
        if not synth['ok']:
            continue
        found = regional_candidates(scene, 'fit', synth['query'],
                                    keep=int(cfg['hard_per_query']) + 12, data=data)
        check = assert_partition(scene, 'fit', found, data)
        if not check['ok']:
            raise RuntimeError(f'{scene}: mined location left the fit partition: {check}')
        truth = np.array(source['xy'], float)
        labelled = []
        for x, y, score, angle, scale in found:
            d = float(np.hypot(x - truth[0], y - truth[1]))
            labelled.append({'x': x, 'y': y, 'classical_score': score,
                             'angle': angle, 'scale': scale,
                             'distance_to_truth': d,
                             'label': label_candidate(d, cfg['positive_radius'],
                                                      cfg['ignore_radius'])})
        hard = [c for c in labelled if c['label'] == NEGATIVE][:int(cfg['hard_per_query'])]
        ignored = [c for c in labelled if c['label'] == IGNORE]
        positives = [c for c in labelled if c['label'] == POSITIVE]
        randoms = _random_negatives(scene, 'fit', truth,
                                   int(cfg['random_per_query']), aug,
                                   cfg['ignore_radius'], data)
        images.append(synth['query'])
        queries.append({
            'query_index': k,
            'image_slot': len(images) - 1,
            'source_id': source['source_id'],
            'cell': source['cell'],
            'centre': [float(truth[0]), float(truth[1])],
            'source_class': source['source_class'],
            'stratum': source['stratum'],
            'pose': {'angle': pose['angle'], 'scale': pose['scale']},
            'degradation': degradation,
            'candidates': labelled,
            'hard_negatives': hard,
            'random_negatives': randoms,
            'n_positive_found': len(positives),
            'n_ignored': len(ignored),
            'degenerate': bool(synth['degenerate']),
        })
        if progress and (k + 1) % 25 == 0:
            rate = (time.perf_counter() - started) / (k + 1)
            progress(f'    {scene} mine {k + 1}/{n_queries} '
                     f'({rate:.2f}s/query, eta {rate * (n_queries - k - 1) / 60:.1f}m)')

    total_c = sum(len(q['candidates']) for q in queries)
    n_ignored = sum(q['n_ignored'] for q in queries)
    return {
        'scene': scene,
        'partition': 'fit',
        'cells': list(cells_of('fit')),
        'n_requested': n_queries,
        'n_queries': len(queries),
        'queries': queries,
        'images': images,
        'label_policy': {'positive_radius': cfg['positive_radius'],
                         'ignore_radius': cfg['ignore_radius']},
        'ambiguity': {
            'n_candidates': total_c,
            'n_ignored': n_ignored,
            'excluded_ambiguity_rate': (n_ignored / total_c) if total_c else 0.0,
        },
        'recall': _mined_recall(queries),
        'degradation_stats': degradation_stats(synth_records),
        'augment_ranges': ranges.as_dict(),
        'seconds': time.perf_counter() - started,
    }


def _mined_recall(queries: list) -> dict:
    out = {}
    for radius in (4, 12, 36):
        hits = []
        for q in queries:
            d = [c['distance_to_truth'] for c in q['candidates']]
            hits.append(bool(d) and min(d) <= radius)
        out[f'recall@{radius}'] = float(np.mean(hits)) if hits else float('nan')
    hard = [len(q['hard_negatives']) for q in queries if 'hard_negatives' in q]
    out['mean_hard_per_query'] = float(np.mean(hard)) if hard else 0.0
    out['n_queries'] = len(queries)
    return out


def build_inner_bank(fold: str, config: dict, data: str | None = None,
                     progress=None) -> dict:
    """Inner-validation bank for one fold from VAL cells (plan §11)."""
    held_out, allowed = fold_skies(fold)
    cfg = config['inner_eval']
    mcfg = config['mining']
    total = int(cfg['queries_per_sky'])
    n_present = int(round(total * float(cfg['present_fraction'])))
    n_absent = total - n_present
    ranges = Ranges(**{k: (tuple(v) if isinstance(v, list) else v)
                       for k, v in config['augment'].items()
                       if k in Ranges().as_dict()})

    skies = {}
    started = time.perf_counter()
    for evaluated in allowed:
        donor = [s for s in allowed if s != evaluated][0]
        gen = rng(SEED_INNER_EVAL, 'inner-sources', fold, evaluated)
        aug = rng(SEED_AUGMENT, 'inner', fold, evaluated)
        present_src = _stratified_sources(
            partition_records(build_source_bank(evaluated, data), 'val'),
            n_present, gen)
        absent_src = _stratified_sources(
            partition_records(build_source_bank(donor, data), 'val'),
            n_absent, gen)

        queries, images, synth_records = [], [], []
        plan = ([('present', evaluated, s) for s in present_src] +
                [('absent', donor, s) for s in absent_src])
        for k, (kind, source_scene, source) in enumerate(plan):
            pose = sample_pose(aug, ranges)
            degradation = sample_degradation(aug, ranges)   # identical law both classes
            synth = synthesize_query(load_scene(source_scene, data).image,
                                     source['xy'], pose, degradation)
            synth_records.append(synth)
            if not synth['ok']:
                continue
            found = regional_candidates(evaluated, 'val', synth['query'],
                                        keep=20, data=data)
            check = assert_partition(evaluated, 'val', found, data)
            if not check['ok']:
                raise RuntimeError(f'{evaluated}: inner location left val: {check}')
            truth = np.array(source['xy'], float) if kind == 'present' else None
            labelled = []
            for x, y, score, angle, scale in found:
                d = (float(np.hypot(x - truth[0], y - truth[1]))
                     if truth is not None else None)
                labelled.append({
                    'x': x, 'y': y, 'classical_score': score,
                    'angle': angle, 'scale': scale, 'distance_to_truth': d,
                    'label': (label_candidate(d, mcfg['positive_radius'],
                                              mcfg['ignore_radius'])
                              if d is not None else 'absent_scene')})
            images.append(synth['query'])
            queries.append({
                'query_index': k,
                'image_slot': len(images) - 1,
                'kind': kind,
                'evaluated_scene': evaluated,
                'source_scene': source_scene,
                'source_id': source['source_id'],
                'centre': ([float(truth[0]), float(truth[1])]
                           if truth is not None else None),
                'stratum': source['stratum'],
                'source_class': source['source_class'],
                'pose': {'angle': pose['angle'], 'scale': pose['scale']},
                'degradation': degradation,
                'candidates': labelled,
                'degenerate': bool(synth['degenerate']),
            })
            if progress and (k + 1) % 32 == 0:
                rate = (time.perf_counter() - started) / (k + 1)
                progress(f'    {fold}/{evaluated} inner {k + 1}/{len(plan)} '
                         f'({rate:.2f}s/query)')
        present = [q for q in queries if q['kind'] == 'present']
        skies[evaluated] = {
            'evaluated_scene': evaluated,
            'donor_scene': donor,
            'cells': list(cells_of('val')),
            'n_queries': len(queries),
            'n_present': len(present),
            'n_absent': len(queries) - len(present),
            'queries': queries,
            'images': images,
            'recall': _mined_recall(present),
            'degradation_stats': degradation_stats(synth_records),
        }
    return {
        'fold': fold, 'held_out': held_out, 'allowed': list(allowed),
        'partition': 'val',
        'skies': skies,
        'seeds': {'sources': SEED_INNER_EVAL, 'augment': SEED_AUGMENT},
        'augment_ranges': ranges.as_dict(),
        'seconds': time.perf_counter() - started,
    }


# --- persistence ------------------------------------------------------------------
def save_mined(paths, scene: str, doc: dict) -> Path:
    target = Path(paths.root) / 'mined' / f'{scene}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    images = doc.pop('images', [])
    arr = np.stack(images).astype(np.uint8) if images else np.zeros((0, 32, 32), np.uint8)
    np.savez_compressed(target.with_suffix('.npz'), queries=arr)
    doc['images_npz'] = target.with_suffix('.npz').name
    doc['n_images'] = int(len(arr))
    write_json(target, doc)
    return target


def load_mined(paths, scene: str) -> dict:
    target = Path(paths.root) / 'mined' / f'{scene}.json'
    doc = read_json(target)
    with np.load(target.with_suffix('.npz')) as blob:
        doc['images'] = blob['queries']
    return doc


def save_inner(paths, fold: str, doc: dict) -> Path:
    target = Path(paths.root) / 'inner' / f'{fold}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for scene, node in doc['skies'].items():
        images = node.pop('images', [])
        arrays[scene] = (np.stack(images).astype(np.uint8) if images
                         else np.zeros((0, 32, 32), np.uint8))
        node['n_images'] = int(len(arrays[scene]))
    np.savez_compressed(target.with_suffix('.npz'), **arrays)
    doc['images_npz'] = target.with_suffix('.npz').name
    write_json(target, doc)
    return target


def load_inner(paths, fold: str) -> dict:
    target = Path(paths.root) / 'inner' / f'{fold}.json'
    doc = read_json(target)
    with np.load(target.with_suffix('.npz')) as blob:
        for scene, node in doc['skies'].items():
            node['images'] = blob[scene]
    return doc


# --- CLI --------------------------------------------------------------------------
def run_cli(args, paths, config) -> int:
    from .stages import folds_for
    data = args.data
    quiet = getattr(args, 'quiet', False)
    refresh = getattr(args, 'refresh', False)

    def say(msg):
        if not quiet:
            print(msg, flush=True)

    folds = folds_for(args)
    scenes_needed = sorted({s for f in folds for s in fold_skies(f)[1]})
    summary = {'mined': {}, 'inner': {}}

    for scene in scenes_needed:
        target = Path(paths.root) / 'mined' / f'{scene}.json'
        if target.exists() and not refresh:
            doc = read_json(target)
            say(f'mined {scene:9s} cached: {doc["n_queries"]} queries, '
                f'recall@4={doc["recall"]["recall@4"]:.3f}')
        else:
            say(f'mining {scene} on fit cells {list(cells_of("fit"))} ...')
            doc = build_mined_bank(scene, config, data, progress=say)
            save_mined(paths, scene, doc)
            say(f'mined {scene:9s} {doc["n_queries"]} queries in '
                f'{doc["seconds"] / 60:.1f}m recall@4={doc["recall"]["recall@4"]:.3f} '
                f'ambiguity_excluded={doc["ambiguity"]["excluded_ambiguity_rate"]:.3f}')
        summary['mined'][scene] = {
            'n_queries': doc['n_queries'], 'recall': doc['recall'],
            'ambiguity': doc['ambiguity'],
            'degradation_stats': doc['degradation_stats']}

    for fold in folds:
        target = Path(paths.root) / 'inner' / f'{fold}.json'
        if target.exists() and not refresh:
            doc = read_json(target)
            say(f'inner {fold:9s} cached')
        else:
            say(f'building inner bank for fold {fold} on val cells '
                f'{list(cells_of("val"))} ...')
            doc = build_inner_bank(fold, config, data, progress=say)
            save_inner(paths, fold, doc)
            say(f'inner {fold:9s} built in {doc["seconds"] / 60:.1f}m')
        summary['inner'][fold] = {
            scene: {'n_present': node['n_present'], 'n_absent': node['n_absent'],
                    'recall': node['recall']}
            for scene, node in doc['skies'].items()}
        for scene, node in summary['inner'][fold].items():
            say(f'   {scene:9s} present={node["n_present"]:3d} '
                f'absent={node["n_absent"]:3d} '
                f'recall@4={node["recall"]["recall@4"]:.3f} '
                f'recall@12={node["recall"]["recall@12"]:.3f}')

    write_json(Path(paths.root) / 'mining.json', summary)
    return 0

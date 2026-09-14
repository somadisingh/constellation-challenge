"""Controls, calibration and end-to-end evaluation (plan §8, §11).

Integration deliberately isolates correspondence:

* the frozen classical recognizer supplies the class and the mapped reference nodes,
* the learned matcher picks a coordinate from the blind bank, or absent, using its
  own calibrated presence decision, and is NEVER snapped back to the classical
  geometry's chosen coordinate,
* membership is distance < 18px to the frozen selected figure nodes,
* the classical constellation name is unchanged, so identification is expected to
  stay fixed and any gain or loss must come from correspondence.

C1 and the pretrained controls are run through the SAME integration, so pool
expansion cannot be mistaken for a learning benefit.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import SCENES
from .banks import bank_poses, bank_scores, bank_xy
from .calibration import (apply_calibrator, extract_features, fallback_threshold,
                          fit_calibrator, select_threshold)
from .data import MEMBER_RADIUS, load_scene, query_records
from .env import ROOT, append_jsonl, read_json, write_json
from .pose import SceneReps
from .scoring import (NEG_INF, AlignedSet, align_query, score_classical,
                      score_descriptor, score_pair_head, summarise_query)
from .splits import fold_skies

ARM_LABELS = {
    'C1': 'classical masked NCC on the expanded aligned bank',
    'P-H': 'pretrained HardNet, no fine-tuning',
    'P-Y': 'pretrained HyNet, no fine-tuning',
    'P-S': 'pretrained SOSNet, no fine-tuning',
}
PRETRAINED_ARMS = {'P-H': 'hardnet', 'P-Y': 'hynet', 'P-S': 'sosnet'}


# --- real-query alignment cache ---------------------------------------------------
def align_real_scene(scene: str, bank: dict, config: dict, data=None,
                     progress=None) -> dict:
    """Align every bank candidate of every real query of one scene."""
    aa = float(config['pose']['aa_factor'])
    scene_obj = load_scene(scene, data)
    reps = SceneReps(scene_obj.image)
    n = bank['n_queries']
    kmax = max(q['n_candidates'] for q in bank['queries'])
    crops = np.zeros((n, kmax, 32, 32), np.float32)
    ncc = np.full((n, kmax), np.nan, np.float32)
    poses = np.zeros((n, kmax, 2), np.float32)
    adm = np.zeros((n, kmax), bool)
    low = np.zeros((n, kmax), bool)
    xy = np.zeros((n, kmax, 2), np.float32)
    counts = np.zeros(n, int)
    qimg = np.zeros((n, 32, 32), np.float32)
    rejected = np.zeros(n, int)
    for i in range(n):
        aligned = align_query(reps, scene_obj.patches[i], bank_xy(bank, i),
                              bank_poses(bank, i), query_id=f'{scene}:{i}',
                              aa_factor=aa)
        k = len(aligned)
        counts[i] = k
        qimg[i] = aligned.query_raw
        rejected[i] = aligned.rejected_poses
        if k:
            crops[i, :k] = aligned.crops
            ncc[i, :k] = aligned.ncc
            poses[i, :k] = aligned.poses
            adm[i, :k] = aligned.admissible
            low[i, :k] = aligned.low_info
            xy[i, :k] = aligned.xy
        if progress and (i + 1) % 20 == 0:
            progress(f'    {scene} align real {i + 1}/{n}')
    return {'scene': scene, 'crops': crops, 'ncc': ncc, 'poses': poses,
            'admissible': adm, 'low_info': low, 'xy': xy, 'counts': counts,
            'query': qimg, 'rejected': rejected, 'pose_trials': aligned.pose_trials}


def save_real_aligned(paths, scene: str, doc: dict) -> Path:
    target = Path(paths.root) / 'real' / f'{scene}_aligned.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{k: v for k, v in doc.items()
                                   if isinstance(v, np.ndarray)})
    write_json(target.with_suffix('.meta.json'),
               {'scene': scene, 'pose_trials': int(doc['pose_trials'])})
    return target


def load_real_aligned(paths, scene: str) -> dict:
    target = Path(paths.root) / 'real' / f'{scene}_aligned.npz'
    meta = read_json(target.with_suffix('.meta.json'))
    with np.load(target) as blob:
        out = {k: blob[k] for k in blob.files}
    out.update(meta)
    return out


def ensure_real_aligned(paths, config, data, say) -> dict:
    out = {}
    for scene in SCENES:
        target = Path(paths.root) / 'real' / f'{scene}_aligned.npz'
        if target.exists():
            out[scene] = load_real_aligned(paths, scene)
            say(f'  real {scene:9s} alignment cached')
        else:
            say(f'  aligning real queries for {scene} ...')
            bank = read_json(paths.bank(scene))
            doc = align_real_scene(scene, bank, config, data, progress=say)
            save_real_aligned(paths, scene, doc)
            out[scene] = doc
    return out


def _aligned_set(node: dict, i: int) -> AlignedSet:
    k = int(node['counts'][i])
    return AlignedSet(query_id=f'{node.get("scene", "?")}:{i}',
                      xy=node['xy'][i, :k].astype(float),
                      crops=node['crops'][i, :k],
                      ncc=node['ncc'][i, :k].astype(float),
                      poses=node['poses'][i, :k].astype(float),
                      admissible=node['admissible'][i, :k],
                      low_info=node['low_info'][i, :k],
                      pose_trials=int(node.get('pose_trials', 9)),
                      rejected_poses=int(node['rejected'][i]),
                      query_raw=node['query'][i])


def score_real_scene(node: dict, scene: str, arm: str, model=None, head=None,
                     device='cpu', data=None) -> list:
    records = query_records(scene, data)
    rows = []
    for r in records:
        aligned = _aligned_set(node, r['index'])
        if arm == 'classical':
            scores = score_classical(aligned)
        elif arm == 'descriptor':
            scores = score_descriptor(model, aligned, device)
        elif arm == 'pair_head':
            scores = score_pair_head(model, head, aligned, device)
        else:
            raise ValueError(arm)
        summary = summarise_query(scores, aligned,
                                  truth_xy=r['xy'] if r['present'] else None)
        summary.update({'scene': scene, 'index': r['index'],
                        'query_id': r['query_id'], 'present': r['present'],
                        'stratum': r['stratum'], 'truth_xy': r['xy'],
                        'figure': r['figure'],
                        'pose_trials': aligned.pose_trials,
                        'rejected_poses': aligned.rejected_poses})
        rows.append(summary)
    return rows


# --- frozen classical geometry ----------------------------------------------------
def frozen_geometry(scene: str) -> dict:
    """Class and mapped reference nodes from the frozen production recognizer."""
    path = ROOT / 'outputs' / 'joint_train' / f'{scene}.json'
    doc = read_json(path)
    geometry = doc['diagnostics'].get('geometry', {})
    hyps = geometry.get('hypotheses') or []
    nodes = np.array(hyps[0].get('nodes', []), float).reshape(-1, 2) if hyps else \
        np.empty((0, 2))
    return {'scene': scene, 'constellation': doc['constellation'],
            'nodes': nodes, 'n_nodes': int(len(nodes)),
            'source': str(path.relative_to(ROOT))}


def integrate(scene: str, rows: list, probs: np.ndarray, threshold: float,
              geometry: dict, member_radius: float = MEMBER_RADIUS) -> dict:
    """Build a complete ScenePrediction from a learned arm (plan §11)."""
    from constellation.contracts import ScenePrediction
    nodes = geometry['nodes']
    patches, detail = [], []
    for r, p in zip(rows, probs):
        chosen = r.get('best_xy')
        usable = chosen is not None and np.isfinite(p)
        present = bool(usable and p >= threshold)
        if present:
            x, y = float(chosen[0]), float(chosen[1])
            member = int(len(nodes) > 0 and
                         np.linalg.norm(nodes - [x, y], axis=1).min() < member_radius)
            patches.append((x, y, member))
        else:
            patches.append(None)
        detail.append({'query_id': r['query_id'], 'present_pred': present,
                       'probability': (float(p) if np.isfinite(p) else None),
                       'selected_xy': chosen, 'membership': patches[-1][2]
                       if patches[-1] else None,
                       'empty_bank': r['empty_bank'],
                       'alignment_failed_all': r['alignment_failed_all'],
                       'flagged_single_candidate': r['flagged']})
    prediction = ScenePrediction(patches, geometry['constellation'], {
        'integration': 'learned_correspondence_frozen_identification',
        'threshold': float(threshold),
        'frozen_nodes': int(len(nodes)),
        'frozen_class_source': geometry['source']})
    return {'prediction': prediction, 'detail': detail}


def evaluate_predictions(predictions: dict, data=None) -> dict:
    from constellation.contracts import evaluate
    from .data import load_truth
    truth = load_truth(data)
    return evaluate(predictions, {k: truth[k] for k in predictions})


# --- arm assembly -----------------------------------------------------------------
def load_arm(name: str, kind: str, device: str, checkpoint: Path | None = None):
    """Return (model, head) for an arm."""
    import torch
    from .models import Descriptor, build_for_training, load_backbone
    if kind == 'pretrained':
        backbone, _ = load_backbone(name, pretrained=True, device=device)
        return Descriptor(backbone, name).to(device).eval(), None
    if kind == 'trained':
        model, _ = build_for_training(name, pretrained=True, device=device)
        blob = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(blob['model'])
        return model.eval(), None
    raise ValueError(kind)


# --- CLI --------------------------------------------------------------------------
def run_cli(args, paths, config) -> int:
    from .inner import inner_metrics, load_inner_aligned, score_inner
    from .stages import folds_for, _device, _seed
    device = _device(args, config)
    seed = _seed(args, config)
    data = args.data
    say = (lambda m: None) if getattr(args, 'quiet', False) else \
        (lambda m: print(m, flush=True))

    real = ensure_real_aligned(paths, config, data, say)
    geometries = {s: frozen_geometry(s) for s in SCENES}
    say(f'frozen classical classes: '
        f'{ {s: g["constellation"] for s, g in geometries.items()} }')

    # C0 is the historical production run, unchanged.
    c0 = read_json(ROOT / 'outputs' / 'joint_train' / 'metrics.json')
    write_json(Path(paths.root) / 'c0.json',
               {'source': 'outputs/joint_train/metrics.json', 'metrics': c0,
                'note': 'historical production output, retained with its own pool'})

    results = {'c0': c0, 'arms': {}, 'folds': {}}

    # --- controls on the inner banks (before training influences anything) --------
    inner_controls = {}
    for fold in folds_for(args):
        aligned = load_inner_aligned(paths, fold)
        entry = {}
        scored = score_inner(aligned, 'classical')
        entry['C1'] = inner_metrics(scored)
        for arm, backbone in PRETRAINED_ARMS.items():
            model, _ = load_arm(backbone, 'pretrained', device)
            scored = score_inner(aligned, 'descriptor', model=model, device=device)
            entry[arm] = inner_metrics(scored)
        inner_controls[fold] = entry
        say(f'fold {fold} inner controls: ' + '  '.join(
            f'{a}={entry[a]["equal_sky_top1_localization_reward"]:.4f}'
            for a in ('C1', 'P-H', 'P-Y', 'P-S')))
    write_json(Path(paths.root) / 'inner_controls.json', inner_controls)
    results['inner_controls'] = inner_controls

    # --- per-fold: calibrate on allowed skies, apply to the held-out sky ----------
    for fold in folds_for(args):
        held_out, allowed = fold_skies(fold)
        fold_out = {'held_out': held_out, 'allowed': list(allowed), 'arms': {}}
        arms = _arms_for_fold(paths, fold, seed, device, say)
        for arm_name, spec in arms.items():
            rows_by_scene = {}
            for scene in SCENES:
                rows_by_scene[scene] = score_real_scene(
                    real[scene], scene, spec['arm_kind'], model=spec.get('model'),
                    head=spec.get('head'), device=device, data=data)
            calib_rows = [r for s in allowed for r in rows_by_scene[s]]
            cal = fit_calibrator(calib_rows, config)
            if cal.get('ok'):
                thr = select_threshold(cal, calib_rows, config)
                threshold = thr['selected']
                fallback = None
            else:
                fallback = fallback_threshold(calib_rows, config)
                thr = {'selected': None, 'fallback': fallback}
                threshold = None
            predictions, details = {}, {}
            for scene in SCENES:
                rows = rows_by_scene[scene]
                if cal.get('ok'):
                    probs = apply_calibrator(cal, rows)
                    t = threshold
                else:
                    X, usable = extract_features(rows)
                    probs = np.where(usable, X[:, 0], -np.inf)
                    t = fallback['selected_score_threshold'] if fallback.get('ok') else 0.0
                built = integrate(scene, rows, probs, t, geometries[scene],
                                  float(config['integration']['member_radius']))
                predictions[scene] = built['prediction']
                details[scene] = built['detail']
            metrics = evaluate_predictions(predictions, data)
            fold_out['arms'][arm_name] = {
                'label': spec.get('label', arm_name),
                'calibration': cal,
                'threshold': thr,
                'metrics': metrics,
                'held_out_metrics': metrics['scenes'][held_out],
                'per_query': {s: _slim(details[s], rows_by_scene[s]) for s in SCENES},
                'strata': {s: _strata_summary(rows_by_scene[s]) for s in SCENES},
            }
            say(f'  {fold}/{arm_name:16s} held-out({held_out}) '
                f'score={metrics["scenes"][held_out]["score"]:.4f} '
                f'mean={metrics["mean"]["score"]:.4f} '
                f'thr={threshold if threshold is not None else "fallback"}')
        results['folds'][fold] = fold_out
        write_json(Path(paths.fold(fold)) / f'evaluate_s{seed}.json', fold_out)

    results['oof'] = _out_of_fold(results['folds'])
    write_json(Path(paths.root) / f'evaluate_s{seed}.json', results)
    append_jsonl(paths.runs, {'stage': 'evaluate', 'seed': seed,
                              'folds': list(results['folds'])})
    _print_oof(results['oof'], c0, say)
    return 0


def _selected_arm(paths, fold: str, seed: int, device: str, say):
    """The recipe this fold actually selected on its own inner validation.

    Every fold contributes one, so SELECTED is the only learned arm defined on all
    three held-out skies and therefore the only learned arm whose out-of-fold mean
    is comparable with C0 and C1. Arm names like `T:hardnet_continue` exist in some
    folds only, because the selected backbone and whether the continuation ran both
    depend on the fold.
    """
    schedule = Path(paths.fold(fold)) / f'schedule_s{seed}.json'
    if not schedule.exists():
        return None
    doc = read_json(schedule)
    backbone = doc['selected_backbone']
    tag = (f'{backbone}_continue_s{seed}' if doc.get('continued')
           else f'{backbone}_screen_s{seed}')
    ckpt = Path(paths.checkpoints(fold)) / tag / 'best.pt'
    if not ckpt.exists():
        return None
    adopt_head = bool((doc.get('pair_head') or {}).get('pair_head_beats_distance'))
    model, _ = load_arm(backbone, 'trained', device, checkpoint=ckpt)
    entry = {'arm_kind': 'descriptor', 'model': model,
             'label': (f'SELECTED: {backbone} ({tag}), scoring='
                       f'{"pair head" if adopt_head else "descriptor distance"}'),
             'selected_backbone': backbone, 'selected_tag': tag,
             'pair_head_adopted': adopt_head}
    if adopt_head:
        head_path = ckpt.parent / 'pair_head.pt'
        if head_path.exists():
            entry['arm_kind'] = 'pair_head'
            entry['head'] = _load_head(head_path, device)
    return entry


def _arms_for_fold(paths, fold: str, seed: int, device: str, say) -> dict:
    """C1 and the pretrained controls always; trained arms when a checkpoint exists."""
    arms = {'C1': {'arm_kind': 'classical', 'label': ARM_LABELS['C1']}}
    selected = _selected_arm(paths, fold, seed, device, say)
    if selected is not None:
        arms['SELECTED'] = selected
        say(f'    selected recipe for {fold}: {selected["label"]}')
    for arm, backbone in PRETRAINED_ARMS.items():
        model, _ = load_arm(backbone, 'pretrained', device)
        arms[arm] = {'arm_kind': 'descriptor', 'model': model,
                     'label': ARM_LABELS[arm]}
    ckpt_root = Path(paths.checkpoints(fold))
    if ckpt_root.exists():
        for sub in sorted(ckpt_root.glob('*/best.pt')):
            tag = sub.parent.name
            if f'_s{seed}' not in tag:
                continue
            backbone = tag.split('_')[0]
            try:
                model, _ = load_arm(backbone, 'trained', device, checkpoint=sub)
            except Exception as exc:
                say(f'    skip {tag}: {exc}')
                continue
            arms[f'T:{tag}'] = {'arm_kind': 'descriptor', 'model': model,
                                'label': f'fine-tuned {backbone} ({tag})'}
            head_path = sub.parent / 'pair_head.pt'
            if head_path.exists():
                try:
                    arms[f'H:{tag}'] = {
                        'arm_kind': 'pair_head', 'model': model,
                        'head': _load_head(head_path, device),
                        'label': f'pair head on frozen {backbone} ({tag})'}
                except Exception as exc:
                    say(f'    skip pair head {tag}: {exc}')
    return arms


def _load_head(path: Path, device: str):
    import torch
    from .models import PairHead
    blob = torch.load(path, map_location=device, weights_only=False)
    head = PairHead().to(device)
    head.load_state_dict(blob['head'])
    return head.eval()


def _slim(detail: list, rows: list) -> list:
    out = []
    for d, r in zip(detail, rows):
        out.append({**d,
                    'present_truth': r['present'], 'stratum': r['stratum'],
                    'error_distance': r.get('error_distance'),
                    'localization_reward': r.get('localization_reward'),
                    'top1_correct': r.get('top1_correct'),
                    'top5_correct': r.get('top5_correct'),
                    'rank_of_correct': r.get('rank_of_correct'),
                    'bank_has_correct': r.get('bank_has_correct'),
                    'nearest_bank_distance': r.get('nearest_bank_distance'),
                    'best_match_score': r.get('best_match_score'),
                    'best_minus_second_score': r.get('best_minus_second_score')})
    return out


def _strata_summary(rows: list) -> dict:
    out = {}
    for stratum in ('figure', 'offfigure', 'absent'):
        sel = [r for r in rows if r['stratum'] == stratum]
        if not sel:
            continue
        present = [r for r in sel if r['present']]
        out[stratum] = {
            'n': len(sel),
            'top1_correct': float(np.mean([r['top1_correct'] for r in present]))
            if present else None,
            'mean_reward': float(np.mean([r['localization_reward'] for r in present]))
            if present else None,
            'pool_missing': int(sum(1 for r in present
                                    if not r.get('bank_has_correct'))),
            'alignment_failed': int(sum(1 for r in sel if r['alignment_failed_all'])),
            'conditional_top1': float(np.mean(
                [r['top1_correct'] for r in present if r.get('bank_has_correct')]))
            if any(r.get('bank_has_correct') for r in present) else None,
        }
    return out


def _out_of_fold(folds: dict) -> dict:
    """Each scene's metrics come from the fold that held it out (plan §11).

    `complete` marks arms defined on every fold. Only complete arms are comparable
    with C0 and C1: a partial arm's mean is over a scene subset, and the three skies
    differ enormously (C0 per-scene totals run 0.487 to 0.876), so a partial mean
    says more about which scenes it covers than about the arm.
    """
    n_folds = len(folds)
    arms = sorted({a for f in folds.values() for a in f['arms']})
    out = {}
    for arm in arms:
        per_scene = {}
        for fold, node in folds.items():
            if arm not in node['arms']:
                continue
            per_scene[node['held_out']] = node['arms'][arm]['held_out_metrics']
        if not per_scene:
            continue
        keys = ('presence', 'localization', 'recovery', 'identification', 'score')
        out[arm] = {
            'per_scene': per_scene,
            'mean': {k: float(np.mean([v[k] for v in per_scene.values()]))
                     for k in keys},
            'worst_score': min(v['score'] for v in per_scene.values()),
            'n_scenes': len(per_scene),
            'complete': len(per_scene) == n_folds,
            'folds_present': sorted(per_scene),
        }
    return out


def _print_oof(oof: dict, c0: dict, say) -> None:
    base = c0['mean']['score']
    header = (f'  {"arm":28s} {"mean":>7s} {"gain/C0":>8s} {"worst":>7s} '
              f'{"pres":>6s} {"loc":>6s} {"rec":>6s} {"ident":>6s} {"n":>2s}')

    def row(name, m, worst, n):
        say(f'  {name:28s} {m["score"]:7.4f} {m["score"] - base:+8.4f} '
            f'{worst:7.4f} {m["presence"]:6.3f} {m["localization"]:6.3f} '
            f'{m["recovery"]:6.3f} {m["identification"]:6.3f} {n:>2d}')

    complete = {a: n for a, n in oof.items() if n['complete']}
    partial = {a: n for a, n in oof.items() if not n['complete']}

    say('\nout-of-fold, ALL THREE held-out skies (comparable with C0)')
    say(header)
    row('C0 (production)', c0['mean'], c0['worst_score'], 3)
    for arm, node in sorted(complete.items(), key=lambda kv: -kv[1]['mean']['score']):
        row(arm, node['mean'], node['worst_score'], node['n_scenes'])
    if partial:
        say('\npartial arms: a scene SUBSET only, NOT comparable with the above')
        say(header)
        for arm, node in sorted(partial.items(),
                                key=lambda kv: -kv[1]['mean']['score']):
            row(f'{arm} [{",".join(node["folds_present"])}]', node['mean'],
                node['worst_score'], node['n_scenes'])

"""Run the frozen deployment policy on the 16 unlabelled validation scenes and
build a submission CANDIDATE (repair task §9). Never overwrites the production
CSV; never uploads to Kaggle.

Candidate bank source: the SAME blind classical proposals production's C0
pipeline already computed and cached at inference time, read from
`outputs/joint_submission/{scene_id}.json`'s
`diagnostics.queries[i].candidates` (each `(x, y, score, angle, scale)`, produced
by `constellation.pipeline.predict_scene`'s label-free retrieval, exactly the
candidate source `experiments.exp1.banks` also traces its bank provenance
against). This module never invents a candidate location that source did not
already return.

C0 geometry (constellation identification, node positions) is reused verbatim
from the SAME `outputs/joint_submission/{scene_id}.json` file: this module does
not re-run or second-guess the classical geometry recognizer, per §10 ("finish
and validate presence/localization deployment first" -- identification is out
of scope for this repair).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import OUT
from experiments.exp1.env import ROOT, sha256_file, write_json
from experiments.exp1.pose import SceneReps
from experiments.exp1.scoring import align_query


def validation_scene_ids(sample_submission: Path | None = None) -> list:
    path = sample_submission or (ROOT / 'sample_submission.csv')
    with open(path, newline='') as f:
        return [row['Id'] for row in csv.DictReader(f)]


def load_validation_candidates(scene_id: str) -> dict:
    """Read C0's already-cached blind candidate bank + geometry for one scene."""
    path = ROOT / 'outputs' / 'joint_submission' / f'{scene_id}.json'
    if not path.exists():
        raise FileNotFoundError(
            f'{scene_id}: no cached C0 prediction at {path}. Run '
            f'`run.py --mode submission --pipeline joint` first; Experiment 3 '
            f'reuses that candidate bank rather than re-running classical search.')
    doc = json.loads(path.read_text())
    return {'scene_id': scene_id, 'constellation': doc['constellation'],
           'patches': doc['patches'], 'diagnostics': doc['diagnostics'],
           'source_path': str(path.relative_to(ROOT)),
           'source_sha256': sha256_file(path)}


def load_validation_image_and_patches(scene_id: str, data: str | None = None):
    import cv2
    from experiments.exp1.data import data_root
    root = data_root(data)
    scene_dir = root / 'validation' / scene_id
    image_path = scene_dir / f'{scene_id}_image.png'
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    patch_paths = sorted((scene_dir / 'patches').glob('patch_*.png'))
    patches = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in patch_paths]
    return image, patches, image_path, patch_paths


def score_validation_scene(models: list, calibrators: list, threshold: float,
                          scene_id: str, device: str, data: str | None = None) -> list:
    """Score every query of one unseen validation scene with the ensemble.

    `models`/`calibrators` are parallel lists (one per fold/seed ensemble member).
    Every model scores every query unconditionally -- no branching on scene
    identity, filename, or patch count.
    """
    from .calibration import apply_calibrator, listwise_absent_probability
    cand = load_validation_candidates(scene_id)
    image, patches, image_path, patch_paths = load_validation_image_and_patches(
        scene_id, data)
    reps = SceneReps(image)
    rows = []
    for i, patch in enumerate(patches):
        query_diag = cand['diagnostics']['queries'][i]
        candidates = query_diag.get('candidates') or []
        xy = np.array([(c[0], c[1]) for c in candidates], float).reshape(-1, 2)
        base_poses = np.array([(c[3], c[4]) for c in candidates], float).reshape(-1, 2)
        aligned = align_query(reps, patch, xy, base_poses, query_id=f'{scene_id}:{i:02d}')

        member_logits, member_absent = [], []
        for model in models:
            from .real_scoring import score_aligned_set
            scored = score_aligned_set(model, aligned, device)
            member_logits.append(scored['pair_logits'])
            member_absent.append(scored['absent_logit'])
        valid = aligned.admissible & ~aligned.low_info
        stacked_logits = np.stack(member_logits) if member_logits else np.zeros((0, 0))
        mean_logits = stacked_logits.mean(axis=0) if stacked_logits.size else np.zeros(0)
        mean_absent = float(np.mean(member_absent)) if member_absent else 0.0
        masked = np.where(valid, mean_logits, -np.inf) if len(mean_logits) else mean_logits
        has_valid = bool(valid.any())
        best_idx = int(np.argmax(masked)) if has_valid else None
        best_logit = float(masked[best_idx]) if has_valid else None
        second_logit = None
        if has_valid and int(valid.sum()) >= 2:
            order = np.argsort(-masked)
            second_logit = float(masked[order[1]])
        rows.append({
            'scene_id': scene_id, 'index': i, 'query_id': f'{scene_id}:{i:02d}',
            'best_index': best_idx, 'best_logit': best_logit,
            'second_logit': second_logit,
            'best_minus_second': (best_logit - second_logit)
            if second_logit is not None else 0.0,
            'absent_logit': mean_absent, 'n_valid': int(valid.sum()),
            'best_xy': (aligned.xy[best_idx].tolist() if best_idx is not None else None),
            'empty_bank': len(aligned) == 0,
            'alignment_failed_all': len(aligned) > 0 and not has_valid,
            'present': None,   # unknown for validation; never available to inference
        })

    calib_rows = [{'present': False, 'best_logit': r['best_logit'],
                   'best_minus_second': r['best_minus_second'],
                   'absent_logit': r['absent_logit'], 'n_valid': r['n_valid']}
                 for r in rows]
    cal = calibrators[0]
    probs = (apply_calibrator(cal, calib_rows) if cal.get('ok')
            else listwise_absent_probability(calib_rows))
    for r, p in zip(rows, probs):
        r['probability'] = float(p)
    return rows, cand


def build_scene_prediction(scene_id: str, rows: list, cand: dict, threshold: float):
    """Calibrated presence + Exp2 geometry snap/rescue, C0 identification reused."""
    from constellation.contracts import ScenePrediction
    from .integration import apply_geometry_stage
    nodes_doc = cand['diagnostics'].get('geometry', {})
    hyps = nodes_doc.get('hypotheses') or []
    nodes = np.array(hyps[0].get('nodes', []), float).reshape(-1, 2) if hyps else \
        np.empty((0, 2))
    patches = []
    for r in rows:
        present = bool(r['best_xy'] is not None and r['probability'] >= threshold)
        if present:
            x, y = r['best_xy']
            member = int(len(nodes) > 0 and
                         np.linalg.norm(nodes - [x, y], axis=1).min() < 18.0)
            patches.append((float(x), float(y), member))
        else:
            patches.append(None)
    learned = ScenePrediction(patches, cand['constellation'], {
        'integration': 'exp3_pairwise_ensemble', 'threshold': float(threshold)})
    classical = ScenePrediction(cand['patches'], cand['constellation'],
                                cand['diagnostics'])
    final = apply_geometry_stage(learned, classical, 'verifier_snap_rescue')
    return final


def run_validation_inference(policy_path: str | Path | None = None,
                             device: str = 'cpu', data: str | None = None,
                             say=print) -> dict:
    """The full frozen policy applied to all 16 unlabelled validation scenes."""
    from .hardnet_source import frozen_hardnet
    from .training import build_model, _load_trainable_state_dict
    import torch

    policy = json.loads((Path(policy_path) if policy_path
                         else OUT / 'deployment_policy.json').read_text())
    arm = policy['selected_arm']
    from . import ARM_SPEC
    spec = ARM_SPEC[arm]

    models, calibrators = [], []
    manifest_members = []
    for member in policy['ensemble_members']:
        fold, seed = member['fold'], member['seed']
        ckpt_path = ROOT / member['checkpoint']
        if not ckpt_path.exists():
            say(f'  SKIP {fold}/s{seed}: checkpoint missing at {ckpt_path}')
            continue
        hardnet_model = None
        if spec['hardnet_fusion']:
            hardnet_model, _ = frozen_hardnet(device, fold, seed)
        model = build_model(arm, hardnet_backbone=hardnet_model).to(device)
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        _load_trainable_state_dict(model, blob['model'])
        model.eval()
        models.append(model)
        # Each member calibrates on its OWN allowed skies (already frozen in
        # held_out_s{seed}.json); reuse that calibrator/threshold rather than
        # refitting on validation data.
        held_out_path = OUT / 'folds' / fold / f'held_out_s{seed}.json'
        if held_out_path.exists():
            fold_record = json.loads(held_out_path.read_text())
            cal = fold_record['calibration']
            cal = dict(cal, threshold=fold_record.get('threshold', {}))
        else:
            cal = {'ok': False, 'threshold': {}}
        calibrators.append(cal)
        manifest_members.append({'fold': fold, 'seed': seed,
                                 'checkpoint_sha256': sha256_file(ckpt_path)})
    if not models:
        raise RuntimeError('no ensemble member checkpoints found; cannot run inference')

    threshold = 0.5   # deterministic default; see docstring in deployment.py for
                      # why this is not re-tuned on validation/held-out data
    thresholds = [c.get('threshold', {}).get('selected') for c in calibrators
                 if c.get('ok')]
    if thresholds:
        threshold = float(np.mean(thresholds))

    scene_ids = validation_scene_ids()
    predictions_dir = OUT / 'validation_predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions = {}
    manifest = {'policy': policy, 'threshold_used': threshold,
               'ensemble_members_loaded': manifest_members, 'scenes': {}}
    for scene_id in scene_ids:
        say(f'  scoring {scene_id} ...')
        rows, cand = score_validation_scene(models, calibrators, threshold,
                                            scene_id, device, data)
        pred = build_scene_prediction(scene_id, rows, cand, threshold)
        predictions[scene_id] = pred
        write_json(predictions_dir / f'{scene_id}.json',
                  {'patches': pred.patches, 'constellation': pred.constellation,
                   'diagnostics': pred.diagnostics})
        n_present = sum(1 for p in pred.patches if p is not None)
        manifest['scenes'][scene_id] = {
            'n_patches': len(pred.patches), 'n_present': n_present,
            'n_absent': len(pred.patches) - n_present,
            'constellation': pred.constellation,
            'n_snap': pred.diagnostics.get('n_snap'),
            'n_rescue': pred.diagnostics.get('n_rescue'),
        }
    write_json(OUT / 'validation_manifest.json', manifest)
    return {'predictions': predictions, 'manifest': manifest}


def write_submission_candidate(predictions: dict, out_path: str | Path | None = None) -> Path:
    from constellation.contracts import write_submission
    out = Path(out_path) if out_path else OUT / 'submission_candidate.csv'
    write_submission(predictions, ROOT / 'sample_submission.csv', out)
    return out


def validate_submission_candidate(csv_path: Path) -> dict:
    """Run the SAME checks `validate_submission.py` performs, in-process."""
    import subprocess
    proc = subprocess.run(
        [str(ROOT / '.venv' / 'bin' / 'python'), str(ROOT / 'validate_submission.py'),
         str(csv_path)], cwd=ROOT, capture_output=True, text=True)
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {'valid': False, 'stdout': proc.stdout, 'stderr': proc.stderr}
    result['exit_code'] = proc.returncode
    write_json(OUT / 'submission_validation.json', result)
    return result


def run(device: str = 'cpu', data: str | None = None, say=print) -> dict:
    say('=== Experiment 3 validation inference (unseen scenes; no Kaggle upload) ===')
    result = run_validation_inference(device=device, data=data, say=say)
    csv_path = write_submission_candidate(result['predictions'])
    say(f'wrote candidate submission: {csv_path}')
    validation = validate_submission_candidate(csv_path)
    say(f'submission validator: {validation}')
    say('CONFIRMED: no Kaggle upload was performed by this module.')
    return {'manifest': result['manifest'], 'csv_path': str(csv_path),
           'validation': validation}


if __name__ == '__main__':
    run()

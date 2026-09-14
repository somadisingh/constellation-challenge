"""Precomputed training tensors from the mined banks (plan §6, §7, §9).

Alignment is model independent, so every aligned crop a training run needs is built
once here and the training loop becomes tensor indexing. Two positive variants are
stored:

  positive_warmup  aligned at the true centre using the KNOWN synthetic pose plus a
                   residual perturbation (angle +/-10 deg, relative scale +/-10%,
                   centre +/-2px). Used for the 500 warmup steps only.
  positive         aligned at the true centre using the SAME image-estimated pose
                   adapter as inference. Used for all hard-negative training, so
                   training and inference share one alignment path.

The negative pool keeps every mined location beyond the ignore radius plus the
random source-bank negatives, each with its classical score, so the 50/25/25
classical/network/random mix can be selected later without re-searching.
"""
from __future__ import annotations

import numpy as np

from .augment import perturbed_pose
from .data import load_scene, to_float
from .env import SEED_AUGMENT, rng
from .mining import NEGATIVE, POSITIVE
from .pose import AA_FACTOR, SceneReps, aligned_candidate, prepare_query, select_pose
from .scoring import align_query


def _aligned_or_nan(reps, qblur, xy, base_pose, aa_factor):
    sel = select_pose(reps, qblur, xy, base_pose, aa_factor)
    if sel['pose'] is None:
        return None, sel
    crop, _ = aligned_candidate(reps.raw, xy, sel['pose'], aa_factor)
    return crop, sel


def build_training_tensors(scene: str, mined: dict, config: dict,
                           data: str | None = None, progress=None) -> dict:
    """Aligned crops for every mined query of one scene."""
    aa = float(config['pose']['aa_factor'])
    scene_obj = load_scene(scene, data)
    reps = SceneReps(scene_obj.image)
    aug = rng(SEED_AUGMENT, 'pairs', scene)

    queries = mined['queries']
    images = mined['images']
    max_neg = max((sum(1 for c in q['candidates'] if c['label'] == NEGATIVE)
                   + len(q['random_negatives'])) for q in queries)

    n = len(queries)
    q_arr = np.zeros((n, 32, 32), np.float32)
    pos = np.zeros((n, 32, 32), np.float32)
    pos_warm = np.zeros((n, 32, 32), np.float32)
    pos_ok = np.zeros(n, bool)
    neg = np.zeros((n, max_neg, 32, 32), np.float32)
    neg_ok = np.zeros((n, max_neg), bool)
    neg_kind = np.zeros((n, max_neg), np.int8)      # 0 classical, 1 random
    neg_cls = np.full((n, max_neg), -np.inf, np.float32)
    neg_xy = np.zeros((n, max_neg, 2), np.float32)
    centres = np.zeros((n, 2), np.float64)
    stats = {'positive_alignment_failed': 0, 'negative_alignment_failed': 0,
             'positive_pose_fallback': 0}

    for i, q in enumerate(queries):
        patch = images[q['image_slot']]
        qraw, qblur = prepare_query(patch)
        q_arr[i] = qraw
        centre = np.array(q['centre'], float)
        centres[i] = centre

        # Warmup positive: known synthetic pose plus residual perturbation.
        pert = perturbed_pose(aug, q['pose'])
        crop, _ = aligned_candidate(reps.raw,
                                    (centre[0] + pert['dx'], centre[1] + pert['dy']),
                                    (pert['angle'], pert['scale']), aa)
        pos_warm[i] = crop

        # Inference-path positive: image-estimated pose at the true centre. The base
        # pose comes from the mined positive candidate when the classical search
        # found one, otherwise from the known synthetic pose (counted separately).
        found_pos = [c for c in q['candidates'] if c['label'] == POSITIVE]
        if found_pos:
            base = (found_pos[0]['angle'], found_pos[0]['scale'])
        else:
            base = (q['pose']['angle'], q['pose']['scale'])
            stats['positive_pose_fallback'] += 1
        crop, sel = _aligned_or_nan(reps, qblur, centre, base, aa)
        if crop is None:
            stats['positive_alignment_failed'] += 1
        else:
            pos[i] = crop
            pos_ok[i] = True

        slot = 0
        for c in q['candidates']:
            if c['label'] != NEGATIVE or slot >= max_neg:
                continue
            crop, sel = _aligned_or_nan(reps, qblur, (c['x'], c['y']),
                                        (c['angle'], c['scale']), aa)
            if crop is None:
                stats['negative_alignment_failed'] += 1
                continue
            neg[i, slot] = crop
            neg_ok[i, slot] = True
            neg_kind[i, slot] = 0
            neg_cls[i, slot] = c['classical_score']
            neg_xy[i, slot] = (c['x'], c['y'])
            slot += 1
        for c in q['random_negatives']:
            if slot >= max_neg:
                break
            # A random negative has no classical pose, so its base pose is drawn
            # from the same augmentation law the queries use.
            base = (float(aug.uniform(0, 360)), float(np.exp(aug.uniform(
                np.log(config['augment']['scale_log'][0]),
                np.log(config['augment']['scale_log'][1])))))
            crop, sel = _aligned_or_nan(reps, qblur, (c['x'], c['y']), base, aa)
            if crop is None:
                stats['negative_alignment_failed'] += 1
                continue
            neg[i, slot] = crop
            neg_ok[i, slot] = True
            neg_kind[i, slot] = 1
            neg_cls[i, slot] = -np.inf
            neg_xy[i, slot] = (c['x'], c['y'])
            slot += 1
        if progress and (i + 1) % 50 == 0:
            progress(f'    {scene} tensors {i + 1}/{n}')

    return {
        'scene': scene,
        'queries': q_arr,
        'positive': pos,
        'positive_warmup': pos_warm,
        'positive_ok': pos_ok,
        'negatives': neg,
        'negative_ok': neg_ok,
        'negative_kind': neg_kind,
        'negative_classical': neg_cls,
        'negative_xy': neg_xy,
        'centres': centres,
        'source_ids': [q['source_id'] for q in queries],
        'strata': np.array([q['stratum'] for q in queries], int),
        'degenerate': np.array([q['degenerate'] for q in queries], bool),
        'max_negatives': int(max_neg),
        'stats': stats,
    }


def identity_mask(centres_a: np.ndarray, scene_a: list, centres_b: np.ndarray,
                  scene_b: list, ignore_radius: float) -> np.ndarray:
    """True where b must NOT be used as a negative for a.

    Same physical source, an overlapping source, or a position inside the ignore
    radius in the SAME sky. Positions in a different sky are never the same source.
    """
    same_scene = np.array(scene_a)[:, None] == np.array(scene_b)[None, :]
    d = np.linalg.norm(centres_a[:, None, :] - centres_b[None, :, :], axis=2)
    return same_scene & (d <= ignore_radius)

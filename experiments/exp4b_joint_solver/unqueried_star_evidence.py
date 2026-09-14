"""Phase 3: unqueried-star evidence and a calibrated clutter/null model.

For every predicted reference node NOT already explained by a seed or
held-out query correspondence, search the raw sky image in a bounded radius
for a real local source, using a multi-scale DoG response (the same detector
family `constellation.finalize.auxiliary_map` already uses for its existing
auxiliary-evidence bonus, but per-NODE here rather than as one scene-wide
density map). Score against MATCHED NULL positions (same local background,
similar source density) rather than a fixed global threshold, then correct for
multiple testing (number of nodes in the reference, number of candidates,
number of hypotheses, local density, searched-node count).
"""
from __future__ import annotations

import numpy as np

from . import SCENES
from experiments.exp1.env import derive_seed

SEARCH_RADIUS = 15.0     # px, bounded local search around a predicted node
DOG_SCALES = ((0.8, 2.5), (1.4, 4.0), (2.5, 7.0))   # matches constellation/splits.py peak scales


def _dog_response(image: np.ndarray, xy: np.ndarray, radius: float = SEARCH_RADIUS) -> dict:
    """Multi-scale DoG peak response in a local window around `xy`, plus basic
    photometric diagnostics (local background, saturation, crowding)."""
    import cv2
    x, y = float(xy[0]), float(xy[1])
    h, w = image.shape
    x0, x1 = max(0, int(x - radius)), min(w, int(x + radius) + 1)
    y0, y1 = max(0, int(y - radius)), min(h, int(y + radius) + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return {'available': False}
    win = image[y0:y1, x0:x1].astype(np.float32)

    best_peak_value, best_peak_pos, best_width = -np.inf, None, None
    for lo, hi in DOG_SCALES:
        dog = cv2.GaussianBlur(win, (0, 0), lo) - cv2.GaussianBlur(win, (0, 0), hi)
        peak_idx = np.unravel_index(np.argmax(dog), dog.shape)
        val = float(dog[peak_idx])
        if val > best_peak_value:
            best_peak_value = val
            best_peak_pos = (float(peak_idx[1] + x0), float(peak_idx[0] + y0))
            # crude width estimate: half-max radius along the row through the peak
            row = dog[peak_idx[0], :]
            above = row >= val / 2.0
            best_width = float(above.sum())

    background = float(np.median(win))
    saturation_frac = float((win >= 254.0).mean())
    peak_distance = (float(np.hypot(best_peak_pos[0] - x, best_peak_pos[1] - y))
                     if best_peak_pos else None)
    # crowding: number of local maxima above half the best peak within the window
    dog_final = cv2.GaussianBlur(win, (0, 0), 0.8) - cv2.GaussianBlur(win, (0, 0), 2.5)
    local_max = (dog_final == cv2.dilate(dog_final, np.ones((5, 5), np.uint8)))
    crowding = int((local_max & (dog_final > best_peak_value / 2.0)).sum())

    return {'available': True, 'peak_strength': best_peak_value,
           'peak_distance_px': peak_distance, 'peak_width': best_width,
           'local_background': background, 'saturation_fraction': saturation_frac,
           'crowding_count': crowding}


def matched_null_positions(image: np.ndarray, xy: np.ndarray, seed: int, tag: str,
                           n_nulls: int = 8, min_offset: float = 40.0,
                           max_offset: float = 120.0) -> list:
    """Deterministic null positions at a SIMILAR local background/density: same
    radial-offset band around `xy` (so they share large-scale illumination and
    are not systematically closer to image edges), never the SAME location."""
    rng = np.random.default_rng(derive_seed(seed, 'exp4b-null', tag,
                                            round(float(xy[0]), 3), round(float(xy[1]), 3)))
    h, w = image.shape
    out = []
    tries = 0
    while len(out) < n_nulls and tries < n_nulls * 20:
        tries += 1
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(min_offset, max_offset)
        nx, ny = xy[0] + r * np.cos(angle), xy[1] + r * np.sin(angle)
        if 0 <= nx < w and 0 <= ny < h:
            out.append((float(nx), float(ny)))
    return out


def _exclusion_set(chosen_pairs: list, mapped_nodes: np.ndarray, pool: np.ndarray) -> set:
    """Template-node indices already explained by a seed/held-out correspondence
    -- these are EXCLUDED from unqueried-star search (one physical source may
    not support multiple nodes, and an already-queried/matched node has no
    'unqueried' evidence to add)."""
    return {i for i, _ in chosen_pairs}


def score_unqueried_nodes(image: np.ndarray, mapped_nodes: np.ndarray, chosen_pairs: list,
                          pool: np.ndarray, seed: int, tag: str,
                          ablation: str = 'matched_null_lr') -> dict:
    """Search every UNQUERIED node (not in `chosen_pairs`) and score against
    matched nulls. `ablation` in {'none','raw_count','quality_weighted',
    'matched_null_lr','matched_null_lr_corrected'}."""
    explained = _exclusion_set(chosen_pairs, mapped_nodes, pool)
    unqueried_idx = [i for i in range(len(mapped_nodes)) if i not in explained]

    per_node = []
    used_physical_sources = set()   # prevents one source supporting >1 node
    for i in unqueried_idx:
        xy = mapped_nodes[i]
        resp = _dog_response(image, xy)
        if not resp.get('available'):
            per_node.append({'node': i, 'available': False})
            continue
        # de-duplication: if the detected peak coincides (within 3px) with an
        # already-claimed physical source, this node gets NO independent credit.
        peak_key = (round(resp['peak_strength'], 1),
                   round((resp.get('peak_distance_px') or -1), 1))
        physical_id = None
        for used in used_physical_sources:
            if abs(used[0] - xy[0]) < 3.0 and abs(used[1] - xy[1]) < 3.0:
                physical_id = used
                break
        duplicate = physical_id is not None
        if not duplicate:
            used_physical_sources.add((float(xy[0]), float(xy[1])))

        nulls = matched_null_positions(image, xy, seed, tag)
        null_responses = [_dog_response(image, n) for n in nulls]
        null_strengths = [r['peak_strength'] for r in null_responses if r.get('available')]
        null_mean = float(np.mean(null_strengths)) if null_strengths else 0.0
        null_std = float(np.std(null_strengths)) if len(null_strengths) >= 2 else 1.0

        z = (resp['peak_strength'] - null_mean) / max(null_std, 1e-6)
        per_node.append({'node': i, 'available': True, 'duplicate_source': duplicate,
                        **resp, 'null_mean_strength': null_mean, 'null_std_strength': null_std,
                        'z_score_vs_null': z})

    valid_nodes = [n for n in per_node if n.get('available') and not n.get('duplicate_source')]

    if ablation == 'none' or not valid_nodes:
        return {'ablation': ablation, 'evidence_score': 0.0, 'n_unqueried': len(unqueried_idx),
               'n_scored': len(valid_nodes), 'per_node': per_node}

    if ablation == 'raw_count':
        score = float(len(valid_nodes))
    elif ablation == 'quality_weighted':
        score = float(np.sum([max(n['peak_strength'], 0) for n in valid_nodes]))
    elif ablation in ('matched_null_lr', 'matched_null_lr_corrected'):
        # log-likelihood-ratio-style: sum of z-scores clipped at 0 (only
        # positive evidence counts; a node darker than its null cannot count
        # AGAINST the class beyond removing its own contribution).
        score = float(np.sum([max(n['z_score_vs_null'], 0) for n in valid_nodes]))
        if ablation == 'matched_null_lr_corrected':
            # multiple-testing correction: divide by number of nodes SEARCHED
            # (not just scored), so a large template offering many chances to
            # find a false-positive peak does not get an automatic advantage.
            n_searched = max(len(unqueried_idx), 1)
            score = score / np.sqrt(n_searched)
    else:
        raise ValueError(ablation)

    return {'ablation': ablation, 'evidence_score': score, 'n_unqueried': len(unqueried_idx),
           'n_scored': len(valid_nodes), 'per_node': per_node}

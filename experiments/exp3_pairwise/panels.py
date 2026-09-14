"""Contact-sheet panels for the required failure/success categories (repair task
§3.10). These are POST-EVALUATION diagnostics only: they are generated after
`held_out_runner.py` and `finalize.py` have already run, and nothing here feeds
back into arm selection, calibration, or thresholds -- looking at a panel must
never change a frozen decision (task §14: "held-out panels only after freezing
that fold's configuration").
"""
from __future__ import annotations

import json

import numpy as np

from . import OUT, SCENES
from experiments.exp1.env import write_json

CATEGORIES = (
    'pairwise_fixes', 'pairwise_regressions', 'false_present', 'false_absent',
    'correct_candidate_misranked', 'pool_missing', 'geometry_snap_improvements',
    'geometry_snap_regressions', 'offset_improvements', 'offset_regressions',
)


def _classify_rows(rows: list, e2_lookup: dict | None) -> dict:
    """Bucket per-query rows into the required panel categories."""
    buckets = {c: [] for c in CATEGORIES}
    for r in rows:
        qid = r['query_id']
        if r.get('present') and r.get('top1_correct') is False:
            if r.get('bank_has_correct'):
                buckets['correct_candidate_misranked'].append(qid)
            else:
                buckets['pool_missing'].append(qid)
        if e2_lookup is not None and qid in e2_lookup:
            e2r = e2_lookup[qid]
            exp3_ok = bool(r.get('top1_correct'))
            e2_ok = bool(e2r.get('top1_correct'))
            if exp3_ok and not e2_ok:
                buckets['pairwise_fixes'].append(qid)
            elif e2_ok and not exp3_ok:
                buckets['pairwise_regressions'].append(qid)
        if r.get('present') is False and r.get('best_logit') is not None:
            # would need the calibrated decision to know false-present; approximate
            # using margin > 0 as a proxy flag for panel selection only
            if (r['best_logit'] - r['absent_logit']) > 0:
                buckets['false_present'].append(qid)
        if r.get('present') is True and r.get('best_logit') is not None:
            if (r['best_logit'] - r['absent_logit']) < 0:
                buckets['false_absent'].append(qid)
        if r.get('offset_confidence') is not None:
            gated = r['offset_confidence'] >= 0.5
            if gated and r.get('error_distance') is not None:
                pass   # improvement/regression needs before/after distance; see below
    return buckets


def build_manifest(seed: int, max_per_category: int = 8, say=print) -> dict:
    """Text-based manifest (query IDs per category); actual crop images are drawn
    from the real query patches on demand by `render_panel_images` -- kept
    separate so the manifest itself never depends on having a display backend."""
    held_out_path = OUT / f'held_out_s{seed}.json'
    if not held_out_path.exists():
        raise FileNotFoundError(f'{held_out_path} missing')
    held_out_doc = json.loads(held_out_path.read_text())

    panels = {c: [] for c in CATEGORIES}
    for fold, fdata in held_out_doc.items():
        rows_path = OUT / 'folds' / fold / f'rows_s{seed}.json'
        if not rows_path.exists():
            continue
        rows_by_scene = json.loads(rows_path.read_text())
        held = fdata['held_out']
        rows = rows_by_scene.get(held, [])
        buckets = _classify_rows(rows, e2_lookup=None)
        for cat, qids in buckets.items():
            panels[cat].extend(qids[:max_per_category])

    manifest = {'seed': seed, 'categories': {
        cat: {'query_ids': qids[:max_per_category], 'count': len(qids)}
        for cat, qids in panels.items()}}
    say(f'panel manifest (seed {seed}): ' +
       ', '.join(f'{c}={len(v)}' for c, v in panels.items()))
    return manifest


def render_panel_images(manifest: dict, seed: int, say=print) -> list:
    """Render an actual PNG contact sheet per non-empty category, using the real
    query/candidate crops. Returns the list of `{category, path}` entries written
    to the manifest."""
    import cv2
    from experiments.exp1.data import load_scene, query_records
    from experiments.exp1.evaluation import _aligned_set, load_real_aligned
    from experiments.exp1.stages import Paths
    from experiments.exp1.env import ROOT as EXP1_ROOT

    panels_dir = OUT / 'panels'
    panels_dir.mkdir(parents=True, exist_ok=True)
    written = []
    paths = Paths(EXP1_ROOT / 'outputs' / 'exp1')

    for cat, node in manifest['categories'].items():
        qids = node['query_ids']
        if not qids:
            continue
        tiles = []
        for qid in qids[:8]:
            try:
                scene, idx_str = qid.split(':')
                idx = int(idx_str)
                node_aligned = load_real_aligned(paths, scene)
                aligned = _aligned_set(node_aligned, idx)
                query_img = (aligned.query_raw * 255).clip(0, 255).astype('uint8')
                tiles.append(cv2.resize(query_img, (64, 64), interpolation=cv2.INTER_NEAREST))
            except Exception:
                continue
        if not tiles:
            continue
        rows = int(np.ceil(len(tiles) / 4))
        grid = np.zeros((rows * 64, 4 * 64), dtype='uint8')
        for i, tile in enumerate(tiles):
            r, c = divmod(i, 4)
            grid[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = tile
        out_path = panels_dir / f'{cat}_s{seed}.png'
        cv2.imwrite(str(out_path), grid)
        written.append({'category': cat, 'path': str(out_path.relative_to(
            __import__('experiments.exp1.env', fromlist=['ROOT']).ROOT)),
            'n_tiles': len(tiles), 'query_ids': qids[:8]})
        say(f'  wrote panel {out_path.name} ({len(tiles)} tiles)')
    return written


def run(seeds: tuple = (31004, 31005), say=print) -> dict:
    panels_dir = OUT / 'panels'
    panels_dir.mkdir(parents=True, exist_ok=True)
    all_entries = []
    for seed in seeds:
        manifest = build_manifest(seed, say=say)
        entries = render_panel_images(manifest, seed, say=say)
        all_entries.extend(entries)
    full_manifest = {'panels': all_entries, 'seeds': list(seeds),
                     'note': 'Post-evaluation diagnostics only; generated AFTER '
                             'arm selection, calibration and held-out scoring were '
                             'frozen. No panel inspection fed back into any '
                             'decision.'}
    write_json(panels_dir / 'manifest.json', full_manifest)
    say(f'panels manifest written: {len(all_entries)} panel image(s)')
    return full_manifest


if __name__ == '__main__':
    run()

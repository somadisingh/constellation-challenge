"""Post-evaluation diagnostic panels for Experiment 4B (task's `panels/`
requirement). Generated strictly AFTER `gates.json` was frozen -- nothing here
feeds back into any gate, ablation choice, or selected rule.

One contact sheet per real labelled scene, showing that scene's real query
patches (the same crops `rank_features.py` scored), for visual audit of what
the pipeline is actually looking at.
"""
from __future__ import annotations

import numpy as np

from . import OUT, SCENES
from experiments.exp1.env import ROOT, write_json


def render_scene_panel(scene: str, say=print) -> dict:
    import cv2
    from experiments.exp1.data import query_records
    from experiments.exp1.evaluation import _aligned_set, load_real_aligned
    from experiments.exp1.stages import Paths

    paths = Paths(ROOT / 'outputs' / 'exp1')
    node = load_real_aligned(paths, scene)
    records = query_records(scene)

    tiles = []
    labels = []
    for r in records[:16]:
        aligned = _aligned_set(node, r['index'])
        if len(aligned) == 0:
            continue
        query_img = (aligned.query_raw * 255).clip(0, 255).astype('uint8')
        tiles.append(cv2.resize(query_img, (64, 64), interpolation=cv2.INTER_NEAREST))
        labels.append(f"{r['query_id']}:{'present' if r['present'] else 'absent'}")

    if not tiles:
        return {'scene': scene, 'path': None, 'n_tiles': 0}

    rows = int(np.ceil(len(tiles) / 4))
    grid = np.zeros((rows * 64, 4 * 64), dtype='uint8')
    for i, tile in enumerate(tiles):
        r, c = divmod(i, 4)
        grid[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = tile

    panels_dir = OUT / 'panels'
    panels_dir.mkdir(parents=True, exist_ok=True)
    out_path = panels_dir / f'{scene}_queries.png'
    cv2.imwrite(str(out_path), grid)
    say(f'  wrote panel {out_path.name} ({len(tiles)} tiles)')
    return {'scene': scene, 'path': str(out_path.relative_to(ROOT)),
           'n_tiles': len(tiles), 'query_ids': labels}


def run(say=print) -> dict:
    entries = []
    for scene in SCENES:
        entries.append(render_scene_panel(scene, say=say))
    manifest = {
        'panels': entries,
        'note': ('Post-evaluation diagnostics only; generated AFTER gates.json was '
                'frozen. No panel inspection fed back into any gate, ablation '
                'choice, or selected rule.'),
    }
    write_json((OUT / 'panels') / 'manifest.json', manifest)
    say(f'panels manifest written: {len(entries)} panel image(s)')
    return manifest


if __name__ == '__main__':
    run()

"""Diagnostic panels and contact sheets (plan §6, §14).

Panels are POST-EVALUATION diagnostics. Nothing here feeds back into a prediction,
a threshold or a checkpoint choice.

Two artifacts:
  contact sheet   generated queries beside the ALLOWED real queries, for a realism
                  check. Held-out real queries are never shown.
  query panels    real-query cases: fixes, regressions, absence errors and
                  alignment failures, each with the source crop and candidate
                  overlays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Circle, Rectangle   # noqa: E402

from . import SCENES
from .data import load_scene, query_records
from .env import read_json, write_json
from .splits import fold_skies


def contact_sheet(paths, fold: str, config: dict, data=None, n: int = 12) -> Path:
    """Generated queries beside real queries from the ALLOWED skies only."""
    from .mining import load_mined
    held_out, allowed = fold_skies(fold)
    rows = []
    for scene in allowed:
        mined = load_mined(paths, scene)
        for q in mined['queries'][:n // len(allowed)]:
            rows.append(('generated', scene, mined['images'][q['image_slot']],
                         f"s={q['pose']['scale']:.2f}"))
        scene_obj = load_scene(scene, data)
        for r in query_records(scene, data)[:n // len(allowed)]:
            rows.append(('real (allowed)', scene, scene_obj.patches[r['index']],
                         r['stratum'][:6]))
    cols = 6
    rowcount = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(rowcount, cols, figsize=(cols * 1.5, rowcount * 1.7))
    for ax in np.ravel(axes):
        ax.axis('off')
    for ax, (kind, scene, img, note) in zip(np.ravel(axes), rows):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'{kind}\n{scene[:4]} {note}', fontsize=5)
    fig.suptitle(f'fold {fold}: generated vs allowed real queries '
                 f'(held-out {held_out} never shown)', fontsize=8)
    fig.tight_layout()
    out = Path(paths.root) / 'panels' / f'contact_sheet_{fold}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _crop(image: np.ndarray, xy, half: int = 60) -> tuple:
    h, w = image.shape
    x0 = int(np.clip(round(xy[0]) - half, 0, w - 1))
    y0 = int(np.clip(round(xy[1]) - half, 0, h - 1))
    x1 = int(np.clip(x0 + 2 * half, 0, w))
    y1 = int(np.clip(y0 + 2 * half, 0, h))
    return image[y0:y1, x0:x1], (x0, y0)


def query_panels(paths, fold: str, arm: str, seed: int, config: dict,
                 data=None, limit: int = 16) -> dict:
    """Panels for the most informative real-query cases of one arm."""
    fold_doc = read_json(Path(paths.fold(fold)) / f'evaluate_s{seed}.json')
    if arm not in fold_doc['arms']:
        return {'ok': False, 'reason': f'arm {arm} not in fold {fold}'}
    c1 = fold_doc['arms'].get('C1', {}).get('per_query', {})
    per_query = fold_doc['arms'][arm]['per_query']
    banks = {s: read_json(paths.bank(s)) for s in SCENES}

    cases = []
    for scene, rows in per_query.items():
        base = {r['query_id']: r for r in c1.get(scene, [])}
        for r in rows:
            ref = base.get(r['query_id'])
            kind = None
            if r['alignment_failed_all']:
                kind = 'alignment failure'
            elif r['present_truth'] and not r['present_pred']:
                kind = 'false absent'
            elif not r['present_truth'] and r['present_pred']:
                kind = 'false present'
            elif ref is not None and r['present_truth']:
                a = r.get('localization_reward') or 0.0
                b = ref.get('localization_reward') or 0.0
                if a - b > 0.15:
                    kind = 'fix vs C1'
                elif b - a > 0.15:
                    kind = 'regression vs C1'
            if r['present_truth'] and not r.get('bank_has_correct'):
                kind = 'pool missing'
            if kind:
                cases.append({'scene': scene, 'kind': kind, **r})

    order = {'fix vs C1': 0, 'regression vs C1': 1, 'false absent': 2,
             'false present': 3, 'alignment failure': 4, 'pool missing': 5}
    cases.sort(key=lambda c: (order.get(c['kind'], 9),
                              -(c.get('error_distance') or 0)))
    chosen = cases[:limit]
    if not chosen:
        return {'ok': False, 'reason': 'no informative case found'}

    fig, axes = plt.subplots(len(chosen), 3,
                             figsize=(7.5, 2.4 * len(chosen)), squeeze=False)
    records = {s: {r['query_id']: r for r in query_records(s, data)} for s in SCENES}
    for row, case in enumerate(chosen):
        scene = case['scene']
        scene_obj = load_scene(scene, data)
        truth_rec = records[scene][case['query_id']]
        idx = truth_rec['index']

        axes[row][0].imshow(scene_obj.patches[idx], cmap='gray', vmin=0, vmax=255)
        axes[row][0].set_title(f"{case['query_id']} query\n{case['kind']}", fontsize=6)

        centre = truth_rec['xy'] or (case.get('selected_xy') or (1500, 1500))
        crop, (ox, oy) = _crop(scene_obj.image, centre)
        for col, (title, mark) in enumerate((('truth neighbourhood', 'truth'),
                                             ('selection', 'selected')), start=1):
            ax = axes[row][col]
            ax.imshow(crop, cmap='gray')
            if truth_rec['xy']:
                ax.add_patch(Circle((truth_rec['xy'][0] - ox,
                                     truth_rec['xy'][1] - oy), 12,
                                    fill=False, color='lime', lw=1.2))
            bank = banks[scene]['queries'][idx]['candidates']
            for c in bank:
                ax.plot(c['x'] - ox, c['y'] - oy, '.', color='deepskyblue', ms=2)
            if case.get('selected_xy'):
                ax.add_patch(Circle((case['selected_xy'][0] - ox,
                                     case['selected_xy'][1] - oy), 8,
                                    fill=False, color='red', lw=1.2))
            err = case.get('error_distance')
            ax.set_title(f"{title}\nerr={err if err is None else round(err, 1)}px "
                         f"score={_fmt(case.get('best_match_score'))}", fontsize=6)
            ax.axis('off')
        axes[row][0].axis('off')
    fig.suptitle(f'fold {fold} arm {arm}: real-query diagnostics '
                 f'(post-evaluation only, never used to override)', fontsize=8)
    fig.tight_layout()
    out = Path(paths.root) / 'panels' / f'panels_{fold}_{arm.replace(":", "_")}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    summary = {'ok': True, 'path': str(out.relative_to(Path(paths.root))),
               'n_panels': len(chosen),
               'kinds': {k: sum(1 for c in chosen if c['kind'] == k)
                         for k in order},
               'cases': [{k: c.get(k) for k in
                          ('query_id', 'kind', 'error_distance', 'present_truth',
                           'present_pred', 'bank_has_correct', 'rank_of_correct')}
                         for c in chosen]}
    return summary


def _fmt(v):
    return '-' if v is None else f'{v:.3f}'


def learning_curves(paths, fold: str, seed: int) -> Path | None:
    """Loss and inner-metric curves for every arm trained on this fold."""
    ckpt = Path(paths.checkpoints(fold))
    files = sorted(ckpt.parent.glob(f'checkpoints/*_s{seed}.json')) + \
        sorted(ckpt.parent.glob(f'*_s{seed}.json'))
    runs = []
    for path in sorted(set(ckpt.glob('*.json'))):
        doc = read_json(path)
        if 'history' in doc and doc.get('history'):
            runs.append((path.stem, doc))
    if not runs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, doc in runs:
        h = doc['history']
        axes[0].plot([r['step'] for r in h], [r['loss'] for r in h], label=name,
                     lw=1)
        ev = doc.get('evaluations') or []
        if ev:
            axes[1].plot([r['step'] for r in ev], [r['metric'] for r in ev],
                         marker='o', ms=3, label=name, lw=1)
    axes[0].set_xlabel('step'); axes[0].set_ylabel('triplet loss')
    axes[0].set_title(f'fold {fold}: training loss')
    axes[1].set_xlabel('step')
    axes[1].set_ylabel('equal-sky top1 localization reward')
    axes[1].set_title('inner validation')
    for ax in axes:
        ax.legend(fontsize=6)
        ax.grid(alpha=.3)
    fig.tight_layout()
    out = Path(paths.root) / 'panels' / f'curves_{fold}_s{seed}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_all(paths, config, seed: int, data=None, say=print) -> dict:
    out = {'contact_sheets': {}, 'panels': {}, 'curves': {}}
    for fold in SCENES:
        try:
            out['contact_sheets'][fold] = str(
                contact_sheet(paths, fold, config, data).relative_to(Path(paths.root)))
        except Exception as exc:
            out['contact_sheets'][fold] = f'failed: {exc!r}'
        curve = None
        try:
            curve = learning_curves(paths, fold, seed)
        except Exception as exc:
            out['curves'][fold] = f'failed: {exc!r}'
        if curve:
            out['curves'][fold] = str(curve.relative_to(Path(paths.root)))
        fold_file = Path(paths.fold(fold)) / f'evaluate_s{seed}.json'
        if not fold_file.exists():
            continue
        doc = read_json(fold_file)
        for arm in doc['arms']:
            if arm == 'C1':
                continue
            try:
                res = query_panels(paths, fold, arm, seed, config, data)
            except Exception as exc:
                res = {'ok': False, 'reason': repr(exc)}
            out['panels'][f'{fold}/{arm}'] = res
            if res.get('ok'):
                say(f'  panels {fold}/{arm}: {res["n_panels"]} cases -> {res["path"]}')
    write_json(Path(paths.root) / 'panels.json', out)
    return out

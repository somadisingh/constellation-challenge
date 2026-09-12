"""Score a configurable final stage on the three labelled scenes using cached
candidates. Candidate generation is fixed, so every variant here is comparable.
"""
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth, evaluate, ScenePrediction
from constellation.references import extract_patterns
from constellation.finalize import auxiliary_map
from constellation.joint import recognize_joint
from lab.cache import load_train, TRAIN, ROOT

AUX_CACHE = ROOT / 'outputs/lab/aux'


def aux_for(scene, split='train'):
    """Auxiliary star-evidence map, cached to disk (recomputation is ~2s/scene)."""
    AUX_CACHE.mkdir(parents=True, exist_ok=True)
    path = AUX_CACHE / f'{scene}.npy'
    if path.exists():
        return np.load(path)
    img = cv2.imread(str(ROOT / split / scene / f'{scene}_image.png'), 0)
    m = auxiliary_map(img)
    np.save(path, m)
    return m


def run_stage(scene_alts, patterns, aux, cfg):
    """Apply the joint stage to one scene. Returns (name, patches, diagnostics)."""
    stages = []
    for label, key, cutoff in [('refined', 'refined', cfg['threshold']),
                               ('coarse', 'coarse', cfg['coarse_cutoff'])]:
        alts = scene_alts[key]
        ids = [i for i, q in enumerate(alts) if len(q) and q[0][2] >= cutoff]
        sub = [alts[i] for i in ids]
        if len(sub) < 3:
            continue
        name, chosen, diag = recognize_joint(
            sub, patterns, tolerance=cfg['tolerance'], top_k=cfg['top_k'],
            margin=cfg['margin'], shear_penalty=cfg['shear_penalty'],
            auxiliary_map=aux if cfg['use_aux'] else None, models=cfg['models'],
            appearance_weight=cfg['appearance_weight'], rank_weight=cfg['rank_weight'],
            gap=cfg['gap'], score_mode=cfg['score_mode'], sigma=cfg['sigma'],
            size_penalty=cfg['size_penalty'], cap=cfg['cap'], min_support=cfg['min_support'],
            quad_share=cfg['quad_share'], aux_weight=cfg['aux_weight'])
        top = diag['hypotheses'][0] if diag.get('hypotheses') else {'score': -1e9}
        # Relocation is only trustworthy when the class itself is: on 192
        # synthetic scenes a winner-to-runner-up gap of 2.0 carries 0.97
        # identification precision, versus 0.47 unconditionally.
        if (diag.get('score_gap') or 0.) < cfg['snap_min_gap']:
            chosen = {}
        stages.append((top.get('score', -1e9), label, name,
                       {ids[k]: v for k, v in chosen.items()}, diag))
    if not stages:
        return 'unknown', [None] * len(scene_alts['refined']), {}
    stages.sort(key=lambda s: (-s[0], s[2], s[1]))
    _, label, name, chosen, diag = stages[0]

    nodes = np.array(diag['hypotheses'][0].get('nodes', [])).reshape(-1, 2)
    patches = []
    for i, q in enumerate(scene_alts['refined']):
        if not len(q):
            patches.append(None); continue
        x, y, score = q[0][0], q[0][1], q[0][2]
        if cfg['snap'] and i in chosen:
            x, y = chosen[i]
        present = score >= cfg['threshold']
        if cfg['geometry_rescue'] and i in chosen:
            present = present or score >= cfg['rescue_threshold']
        member = int(len(nodes) > 0
                     and np.linalg.norm(nodes - [x, y], axis=1).min() < cfg['member_radius'])
        patches.append((float(x), float(y), member) if present else None)
    return name, patches, {'stage': label, 'geometry': diag}


# Tuned on the synthetic geometry benchmark with measured ~5px template-to-sky
# model error, then confirmed on 192 fresh scenes at an unseen seed:
# identification 0.125 (frozen) -> 0.474, snapping 230 fixes / 0 regressions.
DEFAULT = dict(threshold=.72, coarse_cutoff=.65, tolerance=18., top_k=8, margin=.15,
               shear_penalty=2., use_aux=True, models=('affine',),
               appearance_weight=0., rank_weight=0., snap=True, member_radius=18.,
               geometry_rescue=False, rescue_threshold=.60, gap=.03,
               score_mode='binom', sigma=5., size_penalty=0., cap=80000,
               min_support=4, quad_share=0., snap_min_gap=0., aux_weight=3.)


def experiment(label, overrides=None, verbose=True, scenes=None):
    cfg = dict(DEFAULT); cfg.update(overrides or {})
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    patterns = extract_patterns(ROOT / 'patterns')
    cache = load_train()
    names = scenes or TRAIN
    preds, t0 = {}, time.perf_counter()
    detail = {}
    for n in names:
        name, patches, diag = run_stage(cache[n], patterns, aux_for(n), cfg)
        preds[n] = ScenePrediction(patches, name)
        detail[n] = diag
    sub_truth = {k: v for k, v in truth.items() if k in names}
    m = evaluate(preds, sub_truth)
    if verbose:
        print(f"{label:34s} score={m['mean']['score']:.4f} worst={m['worst_score']:.3f} "
              f"pres={m['mean']['presence']:.3f} loc={m['mean']['localization']:.3f} "
              f"rec={m['mean']['recovery']:.3f} id={m['mean']['identification']:.3f} "
              f"[{time.perf_counter()-t0:.0f}s]  "
              + ' '.join(f"{n}:{preds[n].constellation}" for n in names))
    return m, preds, detail


if __name__ == '__main__':
    experiment('joint (defaults)')

import cv2
import numpy as np
from .geometry import recognize
from .joint import recognize_joint
from .contracts import ScenePrediction

# Coarse-stage appearance scores are on a different scale from ECC-refined ones.
COARSE_CUTOFF = .65


def auxiliary_map(image):
    raw = image.astype(np.float32)
    dog = cv2.GaussianBlur(raw, (0, 0), 1) - cv2.GaussianBlur(raw, (0, 0), 8)
    response = cv2.dilate(dog, np.ones((25, 25), np.uint8))
    reference = np.sort(response.ravel()[::10])
    return (np.searchsorted(reference, response) / len(reference)).astype(np.float32)


def finalize(image, queries, raw_queries, patterns, threshold=.72, seed=6643):
    """Frozen milestone stage: one point per query, quad+triangle affine hashing."""
    aux = auxiliary_map(image); competing = []
    for stage, qs, cutoff in [('refined', queries, threshold),
                              ('coarse', raw_queries, COARSE_CUTOFF)]:
        ids = [i for i, q in enumerate(qs) if q[0][2] >= cutoff]
        name, m, d = recognize([qs[i][0][:2] for i in ids], patterns, seed=seed,
                               use_quads=True, shear_penalty=2., tolerance=18.,
                               auxiliary_map=aux)
        best = d['hypotheses'][0] if d.get('hypotheses') else {'score': -1e9, 'support': 0}
        competing.append((best['score'], name, d, stage))
    competing.sort(key=lambda x: (-x[0], x[1], x[3]))
    _, name, geometry, stage = competing[0]
    nodes = (np.array(geometry['hypotheses'][0]['nodes'])
             if geometry.get('hypotheses') and 'nodes' in geometry['hypotheses'][0]
             else np.empty((0, 2)))
    patches = []
    for q in queries:
        x, y, score, *_ = q[0]
        member = int(len(nodes) > 0 and np.linalg.norm(nodes - [x, y], axis=1).min() < 18)
        patches.append((x, y, member) if score >= threshold else None)
    return ScenePrediction(patches, name, {
        'geometry': geometry, 'selected_geometry_stage': stage,
        'competing_geometry': [{'stage': s, 'name': n, 'score': float(v)}
                               for v, n, d, s in competing]})


def finalize_joint(image, queries, raw_queries, patterns, threshold=.72, seed=6643,
                   tolerance=18., gap=.03, top_k=8, cap=80000, quad_share=0.,
                   member_radius=18., aux_weight=3., snap=True, snap_min_gap=0.):
    """Joint localization and recognition.

    Differs from `finalize` in three measured ways. Verification runs against the
    alternatives of appearance-ambiguous queries rather than one point each, and
    the winning fit's chosen alternative replaces the reported coordinate. Seeding
    uses triangle invariants only, because four-point quad invariants compound the
    ~5px template-to-sky model error. The hypothesis budget is raised, which only
    matters once that model error is represented.

    On 192 synthetic scenes at an unseen seed, identification is 0.474 against
    0.125 for `finalize`, and relocation produced 230 fixes with 0 regressions.
    On the three labelled scenes the weighted mean is 0.729 against 0.676.
    """
    aux = auxiliary_map(image)
    competing = []
    for stage, qs, cutoff in [('refined', queries, threshold),
                              ('coarse', raw_queries, COARSE_CUTOFF)]:
        ids = [i for i, q in enumerate(qs) if len(q) and q[0][2] >= cutoff]
        if len(ids) < 3:
            continue
        name, chosen, d = recognize_joint(
            [qs[i] for i in ids], patterns, seed=seed, tolerance=tolerance, gap=gap,
            top_k=top_k, cap=cap, quad_share=quad_share, aux_weight=aux_weight,
            models=('affine',), shear_penalty=2., auxiliary_map=aux)
        best = d['hypotheses'][0] if d.get('hypotheses') else {'score': -1e9}
        if not snap or (d.get('score_gap') or 0.) < snap_min_gap:
            chosen = {}
        competing.append((best.get('score', -1e9), name,
                          {ids[k]: v for k, v in chosen.items()}, d, stage))
    if not competing:
        return ScenePrediction([None] * len(queries), 'unknown',
                               {'reason': 'too few candidate points'})
    competing.sort(key=lambda x: (-x[0], x[1], x[4]))
    _, name, chosen, geometry, stage = competing[0]
    nodes = (np.array(geometry['hypotheses'][0].get('nodes', [])).reshape(-1, 2)
             if geometry.get('hypotheses') else np.empty((0, 2)))
    patches = []
    for i, q in enumerate(queries):
        if not len(q):
            patches.append(None); continue
        x, y, score = float(q[0][0]), float(q[0][1]), float(q[0][2])
        if i in chosen:
            x, y = chosen[i]
        member = int(len(nodes) > 0
                     and np.linalg.norm(nodes - [x, y], axis=1).min() < member_radius)
        patches.append((x, y, member) if score >= threshold else None)
    return ScenePrediction(patches, name, {
        'geometry': geometry, 'selected_geometry_stage': stage,
        'relocated_queries': sorted(chosen),
        'competing_geometry': [{'stage': s, 'name': n, 'score': float(v)}
                               for v, n, _, _, s in competing]})

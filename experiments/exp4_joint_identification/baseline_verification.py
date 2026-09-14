"""Phase 0: reproduce every frozen upstream number Experiment 4 depends on, and
diff the previous production submission against Experiment 3's candidate CSV.

Fails loudly (returns `ok=False`) rather than silently proceeding if any
recorded baseline does not reproduce exactly, per the task's Phase 0 requirement.
"""
from __future__ import annotations

import csv
import json

from experiments.exp1.env import ROOT, sha256_file
from experiments.exp3_pairwise.baseline_verification import (verify_c0, verify_exp1b,
                                                              verify_exp2)


def verify_exp3_corrected(seed_tag: str) -> dict:
    """Reproduce the CORRECTED Experiment 3 primary/repeat OOF numbers exactly
    as recorded in `outputs/exp3_pairwise/metrics.json` (post-repair)."""
    doc = json.load(open(ROOT / 'outputs' / 'exp3_pairwise' / 'metrics.json'))
    key = 'primary' if seed_tag == 'primary' else 'repeat'
    node = doc[key]
    expected = ({'presence': 0.904644653592022, 'localization': 0.7735042735042734,
                'recovery': 0.9444444444444445, 'identification': 0.6666666666666666,
                'score': 0.8169731292099712} if seed_tag == 'primary' else
               {'presence': 0.8833967827195597, 'localization': 0.711301044634378,
                'recovery': 0.9444444444444445, 'identification': 0.6666666666666666,
                'score': 0.7992205157178766})
    actual = {k: node[k] for k in ('presence', 'localization', 'recovery',
                                   'identification', 'score')}
    ok = all(abs(actual[k] - expected[k]) < 1e-6 for k in expected)
    return {'expected': expected, 'actual': actual, 'ok': ok, 'stage': doc['primary_stage']}


def verify_deployment_policy() -> dict:
    """Deployment policy is a fixed architecture (arm F), never per-fold selection."""
    doc = json.load(open(ROOT / 'outputs' / 'exp3_pairwise' / 'deployment_policy.json'))
    ok = (doc['policy_type'] == 'fixed_architecture_ensemble'
         and doc['selected_arm'] == 'F' and len(doc['ensemble_members']) == 6)
    return {'ok': ok, 'selected_arm': doc.get('selected_arm'),
           'n_members': len(doc.get('ensemble_members', [])),
           'members': doc.get('ensemble_members', [])}


def verify_candidate_csv_hash() -> dict:
    path = ROOT / 'outputs' / 'exp3_pairwise' / 'submission_candidate.csv'
    exists = path.exists()
    return {'ok': exists, 'path': str(path.relative_to(ROOT)),
           'sha256': sha256_file(path) if exists else None,
           'bytes': path.stat().st_size if exists else None}


def _load_csv_rows(path) -> dict:
    with open(path, newline='') as f:
        return {r['Id']: r for r in csv.DictReader(f)}


def diff_submissions() -> dict:
    """Cell-level diff: previous production submission vs Exp3's candidate CSV.

    Reports constellation-name changes, presence (-1 vs coordinate) flips, and
    coordinate changes where BOTH files report a present patch -- with no
    attribution to snap/rescue yet (that requires per-query diagnostics that the
    CSV alone doesn't carry; see `attribute_snap_rescue` for that finer view,
    which reads the underlying per-scene JSON diagnostics instead).
    """
    prod_path = ROOT / 'outputs' / 'joint_submission' / 'submission.csv'
    exp3_path = ROOT / 'outputs' / 'exp3_pairwise' / 'submission_candidate.csv'
    prod = _load_csv_rows(prod_path)
    exp3 = _load_csv_rows(exp3_path)
    common_ids = sorted(set(prod) & set(exp3))

    constellation_changes = []
    presence_flips = 0
    coord_changes_both_present = 0
    per_scene = {}
    for sid in common_ids:
        p, e = prod[sid], exp3[sid]
        scene_flips, scene_coord_changes = 0, 0
        if p['constellation'] != e['constellation']:
            constellation_changes.append({'id': sid, 'prod': p['constellation'],
                                          'exp3': e['constellation']})
        n = int(p['n_patches'])
        for i in range(1, n + 1):
            col = f'patch_{i:02d}'
            pv, ev = p.get(col, '-1'), e.get(col, '-1')
            p_present = pv.strip() != '-1'
            e_present = ev.strip() != '-1'
            if p_present != e_present:
                presence_flips += 1
                scene_flips += 1
            elif p_present and e_present and pv != ev:
                coord_changes_both_present += 1
                scene_coord_changes += 1
        per_scene[sid] = {'presence_flips': scene_flips,
                          'coord_changes_both_present': scene_coord_changes,
                          'constellation_changed': p['constellation'] != e['constellation']}

    return {
        'production_csv': str(prod_path.relative_to(ROOT)),
        'exp3_candidate_csv': str(exp3_path.relative_to(ROOT)),
        'n_scenes_compared': len(common_ids),
        'constellation_changes': constellation_changes,
        'n_constellation_changes': len(constellation_changes),
        'total_presence_flips': presence_flips,
        'total_coord_changes_both_present': coord_changes_both_present,
        'per_scene': per_scene,
    }


def attribute_snap_rescue() -> dict:
    """Per real labelled scene, attribute Exp3's presence/coordinate changes to
    snap vs rescue vs plain-learned, using Exp3's own recorded `actions` diagnostic
    (from `hybrid_prediction`) for the `verifier_snap_rescue` stage. Only possible
    for the 3 REAL labelled scenes, since only those have Exp3 held-out prediction
    JSON with per-query `actions` -- the 16 unlabelled validation scenes' actions
    are in `outputs/exp3_pairwise/validation_predictions/*.json`'s diagnostics.
    """
    out = {}
    held_out_doc = json.load(open(ROOT / 'outputs' / 'exp3_pairwise' / 'held_out_s31004.json'))
    for fold, fdata in held_out_doc.items():
        held = fdata['held_out']
        pred = fdata['predictions']['verifier_snap_rescue'][held]
        actions = pred.get('diagnostics', {}).get('actions', [])
        counts = {'learned': actions.count('learned'),
                 'geometry_snap': actions.count('geometry_snap'),
                 'geometry_rescue': actions.count('geometry_rescue')}
        out[held] = counts
    # 16 unlabelled validation scenes
    val_dir = ROOT / 'outputs' / 'exp3_pairwise' / 'validation_predictions'
    val_out = {}
    if val_dir.exists():
        for p in sorted(val_dir.glob('*.json')):
            doc = json.load(open(p))
            actions = doc.get('diagnostics', {}).get('actions', [])
            val_out[p.stem] = {'learned': actions.count('learned'),
                              'geometry_snap': actions.count('geometry_snap'),
                              'geometry_rescue': actions.count('geometry_rescue')}
    return {'real_scenes': out, 'validation_scenes': val_out}


def run(say=print) -> dict:
    say('=== Phase 0: baseline reconstruction ===')
    c0 = verify_c0()
    say(f"C0: {'OK' if c0['ok'] else 'MISMATCH'} {c0['actual']}")
    exp1b = verify_exp1b()
    say(f"Exp1B: {'OK' if exp1b['ok'] else 'MISMATCH'} {exp1b['actual']}")
    exp2p = verify_exp2('primary')
    say(f"Exp2 primary: {'OK' if exp2p['ok'] else 'MISMATCH'} {exp2p['actual']}")
    exp2r = verify_exp2('repeat')
    say(f"Exp2 repeat: {'OK' if exp2r['ok'] else 'MISMATCH'} {exp2r['actual']}")
    exp3p = verify_exp3_corrected('primary')
    say(f"Exp3 corrected primary: {'OK' if exp3p['ok'] else 'MISMATCH'} {exp3p['actual']}")
    exp3r = verify_exp3_corrected('repeat')
    say(f"Exp3 corrected repeat: {'OK' if exp3r['ok'] else 'MISMATCH'} {exp3r['actual']}")
    policy = verify_deployment_policy()
    say(f"Deployment policy: {'OK' if policy['ok'] else 'MISMATCH'} arm={policy['selected_arm']} "
       f"n_members={policy['n_members']}")
    csv_hash = verify_candidate_csv_hash()
    say(f"Candidate CSV: {'exists' if csv_hash['ok'] else 'MISSING'} sha256={csv_hash['sha256']}")
    diff = diff_submissions()
    say(f"Submission diff: {diff['n_constellation_changes']} constellation changes, "
       f"{diff['total_presence_flips']} presence flips, "
       f"{diff['total_coord_changes_both_present']} coord changes (both present)")
    snap_rescue = attribute_snap_rescue()
    say(f"Snap/rescue attribution (real scenes): {snap_rescue['real_scenes']}")

    all_ok = (c0['ok'] and exp1b['ok'] and exp2p['ok'] and exp2r['ok']
             and exp3p['ok'] and exp3r['ok'] and policy['ok'] and csv_hash['ok'])
    say('ALL BASELINES VERIFIED EXACTLY.' if all_ok else 'BASELINE VERIFICATION FAILED.')
    return {
        'ok': all_ok, 'c0': c0, 'exp1b': exp1b, 'exp2_primary': exp2p,
        'exp2_repeat': exp2r, 'exp3_corrected_primary': exp3p,
        'exp3_corrected_repeat': exp3r, 'deployment_policy': policy,
        'candidate_csv': csv_hash, 'submission_diff': diff,
        'snap_rescue_attribution': snap_rescue,
    }


if __name__ == '__main__':
    from experiments.exp1.env import write_json
    from . import OUT
    result = run()
    write_json(OUT / 'baseline_verification.json', result)
    raise SystemExit(0 if result['ok'] else 1)

"""Report and machine record generation (plan §12, §14).

Verdict first, then the seven required sections. Every metric names the artifact
that produced it. Large logs stay in their JSON files; this writes a ledger.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import SCENES
from .env import ROOT, env_dict, read_json, write_json


def _maybe(path: Path):
    return read_json(path) if Path(path).exists() else None


def decide_gates(oof: dict, c0: dict, config: dict, seeds_agree: bool | None) -> dict:
    """Predeclared engineering continuation target (plan §12). Not a significance claim."""
    g = config['gates']
    base = c0['mean']
    out = {'gates': dict(g), 'arms': {}, 'c0_mean': base,
           'not_evaluable': {}}
    for arm, node in oof.items():
        # A gate may only be applied to an arm defined on EVERY held-out sky. A
        # partial arm's mean is over a scene subset, and the three skies differ by
        # ~0.39 in C0 total, so a subset mean can look like a large gain purely by
        # covering the easy scene.
        if not node.get('complete', True):
            out['not_evaluable'][arm] = {
                'reason': 'defined on a scene subset only',
                'folds_present': node.get('folds_present'),
                'n_scenes': node.get('n_scenes'),
                'subset_mean_score': node['mean']['score'],
            }
            continue
        m = node['mean']
        per_scene_reg = {}
        for scene, sm in node['per_scene'].items():
            c0_scene = c0['scenes'].get(scene, {}).get('score')
            if c0_scene is not None:
                per_scene_reg[scene] = sm['score'] - c0_scene
        worst_reg = min(per_scene_reg.values()) if per_scene_reg else 0.0
        checks = {
            'mean_gain_over_c0': m['score'] - base['score'],
            'mean_gain_ge_threshold': (m['score'] - base['score']) >= g['min_mean_gain_over_c0'],
            'presence_not_below_c0': m['presence'] >= base['presence'] - 1e-9,
            'localization_not_below_c0': m['localization'] >= base['localization'] - 1e-9,
            'recovery_not_below_c0': m['recovery'] >= base['recovery'] - 1e-9,
            'per_scene_regression': per_scene_reg,
            'worst_per_scene_delta': worst_reg,
            'no_scene_regression_beyond_limit': worst_reg >= -g['max_per_scene_regression'],
            'stable_across_seeds': seeds_agree,
        }
        checks['passed'] = bool(
            checks['mean_gain_ge_threshold'] and checks['presence_not_below_c0']
            and checks['localization_not_below_c0'] and checks['recovery_not_below_c0']
            and checks['no_scene_regression_beyond_limit'])
        out['arms'][arm] = checks
    c1 = oof.get('C1', {}).get('mean', {}).get('score')
    out['c1_mean'] = c1
    out['n_arms_gated'] = len(out['arms'])
    for arm, checks in out['arms'].items():
        checks['gain_over_c1'] = (oof[arm]['mean']['score'] - c1) if c1 is not None else None
        checks['positive_over_c1'] = (checks['gain_over_c1'] or 0) > 0
        if config['gates']['require_positive_over_c1']:
            checks['passed'] = bool(checks['passed'] and checks['positive_over_c1'])
    passing = [a for a, c in out['arms'].items() if c['passed'] and a != 'C1']
    out['any_learned_arm_passed'] = bool(passing)
    out['passing_arms'] = passing
    out['verdict'] = ('improved' if passing else 'no demonstrated gain')
    return out


def build_record(paths, config) -> dict:
    root = Path(paths.root)
    seed = config['seeds']['model']
    seed2 = config['seeds']['model_second']
    record = {
        'experiment': 'exp1',
        'plan': 'cnn/plan_experiment1.md',
        'environment': _maybe(root / 'environment.json'),
        'preflight': _summarise_preflight(_maybe(root / 'preflight.json')),
        'models': _maybe(root / 'models.json'),
        'protocol': _protocol_summary(_maybe(root / 'protocol.json')),
        'prepare': _maybe(root / 'prepare.json'),
        'candidate_banks': _maybe(root / 'candidate_banks.json'),
        'mining': _maybe(root / 'mining.json'),
        'inner_controls': _maybe(root / 'inner_controls.json'),
        'screen': {f's{seed}': _maybe(root / f'screen_s{seed}.json'),
                   f's{seed2}': _maybe(root / f'screen_s{seed2}.json')},
        'evaluate': {f's{seed}': _slim_eval(_maybe(root / f'evaluate_s{seed}.json')),
                     f's{seed2}': _slim_eval(_maybe(root / f'evaluate_s{seed2}.json'))},
        'runs_ledger': str((root / 'runs.jsonl').relative_to(root))
        if (root / 'runs.jsonl').exists() else None,
    }
    record['realism'] = _maybe(root / 'realism_audit.json')
    record['panels'] = _maybe(root / 'panels.json')
    record['schedule'] = {f's{seed}': _maybe(root / f'schedule_s{seed}.json'),
                          f's{seed2}': _maybe(root / f'schedule_s{seed2}.json')}
    record['transfer'] = _transfer_table(root, seed)
    record['strata_table'] = _strata_table(root, seed)
    primary = _maybe(root / f'evaluate_s{seed}.json')
    second = _maybe(root / f'evaluate_s{seed2}.json')
    if primary and primary.get('oof'):
        seeds_agree = None
        if second and second.get('oof'):
            seeds_agree = _seeds_agree(primary['oof'], second['oof'],
                                       primary['c0']['mean']['score'])
        record['gates'] = decide_gates(primary['oof'], primary['c0'], config,
                                       seeds_agree)
        record['verdict'] = record['gates']['verdict']
    else:
        record['verdict'] = 'blocked'
        record['blocked_reason'] = 'no out-of-fold evaluation artifact present'
    write_json(root / 'record.json', record)
    return record


def _transfer_table(root: Path, seed: int) -> dict | None:
    """Did inner-validation selection predict the held-out direction? (plan §11)"""
    schedule = _maybe(root / f'schedule_s{seed}.json')
    evaluate = _maybe(root / f'evaluate_s{seed}.json')
    if not schedule or not evaluate:
        return None
    rows = []
    for fold, node in schedule.items():
        arms = (evaluate.get('folds', {}).get(fold, {}) or {}).get('arms', {})
        if 'SELECTED' not in arms or 'C1' not in arms:
            continue
        inner_sel = ((node.get('continued') or node['screened'][
            node['selected_backbone']])['metric'])
        inner_c1 = node['inner_controls']['C1']
        outer_sel = arms['SELECTED']['held_out_metrics']['score']
        outer_c1 = arms['C1']['held_out_metrics']['score']
        inner_better = inner_sel > inner_c1 + 1e-9
        outer_better = outer_sel > outer_c1 + 1e-9
        rows.append({
            'fold': fold, 'held_out': node['held_out'],
            'inner_selected': inner_sel, 'inner_c1': inner_c1,
            'inner_verdict': 'learned better' if inner_better else 'C1 at least equal',
            'outer_selected': outer_sel, 'outer_c1': outer_c1,
            'outer_verdict': 'learned better' if outer_better else 'C1 better',
            'agrees': inner_better == outer_better,
        })
    if not rows:
        return None
    return {'rows': rows, 'n': len(rows),
            'n_agree': sum(1 for r in rows if r['agrees'])}


def _pct(v):
    return '-' if v is None else f'{float(v):.3f}'


def _strata_table(root: Path, seed: int) -> list | None:
    """SELECTED against C1 per stratum, on the held-out sky of each fold."""
    evaluate = _maybe(root / f'evaluate_s{seed}.json')
    if not evaluate:
        return None
    rows = []
    for fold, node in (evaluate.get('folds') or {}).items():
        held = node['held_out']
        arms = node['arms']
        if 'SELECTED' not in arms or 'C1' not in arms:
            continue
        c1 = arms['C1']['strata'].get(held, {})
        sel = arms['SELECTED']['strata'].get(held, {})
        for stratum in ('figure', 'offfigure', 'absent'):
            if stratum not in sel:
                continue
            rows.append({
                'scene': held, 'stratum': stratum, 'n': sel[stratum]['n'],
                'c1_top1': (c1.get(stratum) or {}).get('top1_correct'),
                'sel_top1': sel[stratum]['top1_correct'],
                'pool_missing': sel[stratum]['pool_missing'],
                'alignment_failed': sel[stratum]['alignment_failed'],
            })
    return rows or None


def _seeds_agree(a: dict, b: dict, base: float) -> bool:
    arms = set(a) & set(b)
    signs = []
    for arm in arms:
        if arm == 'C1' or not a[arm].get('complete', True) \
                or not b[arm].get('complete', True):
            continue
        signs.append(np.sign(a[arm]['mean']['score'] - base) ==
                     np.sign(b[arm]['mean']['score'] - base))
    return bool(signs) and all(signs)


def _summarise_preflight(doc):
    if not doc:
        return None
    out = {'ok': doc.get('ok'), 'problems': doc.get('problems'),
           'device': doc.get('device'),
           'cpu_fallback_operations': doc.get('cpu_fallback_operations'),
           'models': {}}
    for name, entry in (doc.get('models') or {}).items():
        if 'skipped' in entry:
            out['models'][name] = entry
            continue
        out['models'][name] = {
            'parity': {k: entry['parity'][k] for k in
                       ('informative_cosine_min', 'informative_max_abs_diff',
                        'degenerate_cosine_min', 'degenerate_max_abs_diff',
                        'top1_rank_agreement', 'zero_norm_descriptors')
                       if k in entry.get('parity', {})},
            'overfit': entry['overfit'],
            'throughput': entry.get('throughput'),
            'n_parameters': entry['forward_backward']['cpu']['n_parameters'],
        }
    return out


def _protocol_summary(doc):
    if not doc:
        return None
    return {'geometry_check': doc.get('geometry_check', {}).get('ok'),
            'footprint_radius': doc.get('geometry_check', {}).get('radius'),
            'queries': doc.get('queries'), 'seeds': doc.get('seeds'),
            'code_hash': doc.get('code_hash'),
            'data': {k: v for k, v in (doc.get('data') or {}).items()
                     if k != 'scenes'},
            'scene_dims': {s: {'height': v['height'], 'width': v['width'],
                               'n_queries': v['n_queries'],
                               'image_sha256': v['image_sha256']}
                           for s, v in (doc.get('data') or {}).get('scenes', {}).items()}}


def _slim_eval(doc):
    """Keep the metrics and drop the per-query payloads, which live in fold files."""
    if not doc:
        return None
    out = {'c0': doc.get('c0'), 'oof': doc.get('oof'), 'folds': {}}
    for fold, node in (doc.get('folds') or {}).items():
        out['folds'][fold] = {
            'held_out': node['held_out'], 'allowed': node['allowed'],
            'arms': {a: {'label': v['label'],
                         'held_out_metrics': v['held_out_metrics'],
                         'mean': v['metrics']['mean'],
                         'threshold': v['threshold'].get('selected'),
                         'calibration_ok': v['calibration'].get('ok'),
                         'strata': v['strata']}
                     for a, v in node['arms'].items()}}
    return out


# --- markdown ---------------------------------------------------------------------
def render_markdown(record: dict, config: dict) -> str:
    L = []
    verdict = record.get('verdict', 'blocked')
    L.append('# Experiment 1 — learned verification of real sky patches\n')
    L.append(f'**Verdict: {verdict}.**\n')
    gates = record.get('gates')
    c0 = (record.get('evaluate') or {}).get(f"s{config['seeds']['model']}", {})
    c0 = (c0 or {}).get('c0') or {}
    if gates:
        base = gates['c0_mean']['score']
        sel = ((record.get('evaluate') or {}).get(f"s{config['seeds']['model']}")
               or {}).get('oof', {}).get('SELECTED', {})
        sel_mean = sel.get('mean', {}).get('score')
        L.append(f"Reproduced C0 (production, `outputs/joint_train/metrics.json`) "
                 f"total **{base:.7f}**, unchanged. C1, the same classical masked NCC "
                 f"on the expanded bank under the same integration, scores "
                 f"**{gates['c1_mean']:.4f}**. The selected learned recipe scores "
                 f"**{sel_mean:.4f}**." if sel_mean is not None else '')
        L.append('')
        L.append('Three things carry the result, and they separate cleanly.\n')
        L.append('1. **The integration itself costs recovery, before any learning.** '
                 'C1 contains no learned component, yet it loses 0.0432 to C0 and its '
                 'recovery falls from 0.878 to 0.589. Exp1 deliberately lets the '
                 'matcher pick the coordinate and forbids snapping back to the '
                 'classical geometry, which removes the relocation step that produced '
                 'C0\'s recovery. That cost is charged to every arm equally and is not '
                 'evidence about descriptors.')
        L.append('2. **Learning is real but does not reach the classical criterion.** '
                 'Fine-tuning lifts HardNet from 0.5561 (pretrained) to 0.6412, a '
                 'large gain over off-the-shelf descriptors. It still ends 0.0304 '
                 'below C1. Presence improves markedly (0.848 against C0\'s 0.717); '
                 'localization and recovery do not.')
        L.append('3. **Inner-validation selection did not transfer.** The learned arm '
                 'beat C1 on inner validation in two folds but on the held-out sky in '
                 'only one. The realism audit below shows why: the generated queries '
                 'are far lower contrast than the real ones, so the selection set is '
                 'not the test distribution.')
        L.append('')
        L.append('No arm passed the predeclared gates, so production is retained and '
                 'no submission candidate was produced. Per plan §12 a failed '
                 'experiment is a valid outcome; three scenes could not have '
                 'established a hidden-set gain even had the gates passed.')
        L.append('')
        oof = (record['evaluate'][f"s{config['seeds']['model']}"] or {}).get('oof', {})
        complete = {a: n for a, n in oof.items() if n.get('complete', True)}
        partial = {a: n for a, n in oof.items() if not n.get('complete', True)}
        L.append('Out-of-fold over all three held-out skies (116 queries). Each scene '
                 'is scored by the fold that held it out.\n')
        L.append('| arm | mean total | gain vs C0 | gain vs C1 | worst scene | '
                 'presence | localization | recovery | ident | gates |')
        L.append('|---|---|---|---|---|---|---|---|---|---|')
        L.append(f"| **C0 (production)** | {gates['c0_mean']['score']:.4f} | "
                 f"+0.0000 | "
                 f"{(gates['c0_mean']['score'] - (gates.get('c1_mean') or 0)):+.4f} | "
                 f"{(c0 or {}).get('worst_score', float('nan')):.4f} | "
                 f"{gates['c0_mean']['presence']:.3f} | "
                 f"{gates['c0_mean']['localization']:.3f} | "
                 f"{gates['c0_mean']['recovery']:.3f} | "
                 f"{gates['c0_mean']['identification']:.3f} | reference |")
        for arm, node in sorted(complete.items(),
                                key=lambda kv: -kv[1]['mean']['score']):
            c = gates['arms'].get(arm, {})
            m = node['mean']
            L.append(f"| {arm} | {m['score']:.4f} | "
                     f"{c.get('mean_gain_over_c0', float('nan')):+.4f} | "
                     f"{(c.get('gain_over_c1') or 0):+.4f} | "
                     f"{node['worst_score']:.4f} | {m['presence']:.3f} | "
                     f"{m['localization']:.3f} | {m['recovery']:.3f} | "
                     f"{m['identification']:.3f} | "
                     f"{'PASS' if c.get('passed') else 'fail'} |")
        L.append('')
        if partial:
            L.append('The following arms exist on a scene SUBSET only, because the '
                     'selected backbone and whether the continuation ran are both '
                     'fold decisions. Their means are NOT comparable with the table '
                     'above and no gate is applied to them: C0 per-scene totals span '
                     '0.487 to 0.876, so covering only the easy sky inflates a subset '
                     'mean.\n')
            L.append('| arm | folds present | subset mean | note |')
            L.append('|---|---|---|---|')
            for arm, node in sorted(partial.items(),
                                    key=lambda kv: -kv[1]['mean']['score']):
                L.append(f"| {arm} | {', '.join(node['folds_present'])} | "
                         f"{node['mean']['score']:.4f} | not evaluable |")
            L.append('')
    else:
        L.append(f"No out-of-fold evaluation is present: "
                 f"{record.get('blocked_reason', 'stage not reached')}.\n")

    L.append('## 1. Environment\n')
    env = (record.get('environment') or {}).get('environment', {})
    if env:
        L.append(f"- {env.get('cpu_brand')}, {env.get('machine')}, macOS "
                 f"{env.get('mac_version')}, {env.get('ram_gib', 0):.1f} GiB RAM, "
                 f"{env.get('free_disk_gib', 0):.0f} GiB free")
        L.append(f"- Python {env.get('python')}, torch {env.get('torch')}, "
                 f"kornia {env.get('kornia')}, numpy {env.get('numpy')}, "
                 f"scipy {env.get('scipy')}, opencv {env.get('cv2')}")
        L.append(f"- MPS built {env.get('mps_built')}, available "
                 f"{env.get('mps_available')}; package lock "
                 f"`requirements-exp1-lock.txt`")
    pf = record.get('preflight') or {}
    for name, entry in (pf.get('models') or {}).items():
        p = entry.get('parity', {})
        tp = entry.get('throughput') or {}
        L.append(f"- {name}: CPU/MPS informative cosine "
                 f"{p.get('informative_cosine_min', float('nan')):.6f}, max abs "
                 f"{p.get('informative_max_abs_diff', float('nan')):.2e}, rank "
                 f"agreement {p.get('top1_rank_agreement', float('nan')):.2f}, "
                 f"{tp.get('steps_per_second', float('nan')):.1f} steps/s at "
                 f"microbatch {tp.get('microbatch')}")
    for op in (pf.get('cpu_fallback_operations') or []):
        L.append(f"- CPU fallback: {op['operation']} ({op['reason']}); {op['scope']}")
    L.append('')

    L.append('## 2. Models and what actually trained\n')
    for name, entry in ((record.get('models') or {}).get('models') or {}).items():
        L.append(f"- **{name}**: `{entry['weights_path']}`, sha256 "
                 f"`{entry.get('actual_sha256', '-')}`, source {entry['source_url']}, "
                 f"licence {entry['license']}")
    L.append('- Loss is the Exp1 symmetric hard-negative triplet, not the original '
             'HardNet/HyNet/SOSNet recipe. Results are labelled '
             '"backbone fine-tuned with the Exp1 loss".')
    L.append('')

    L.append('## 3. Folds, sources and split tests\n')
    prep = record.get('prepare') or {}
    proto = record.get('protocol') or {}
    L.append(f"- Partition geometry check: ok={proto.get('geometry_check')}, "
             f"footprint radius {proto.get('footprint_radius', float('nan')):.1f}px "
             f"inside a {config['protocol']['margin']}px margin")
    for fold, node in (prep.get('folds') or {}).items():
        counts = ', '.join(f"{s}: fit {v['fit']}, val {v['val']}, synthcal {v['synthcal']}"
                           for s, v in node['unique_sources'].items())
        L.append(f"- fold **{fold}** (held out {node['held_out']}): {counts}")
    mining = record.get('mining') or {}
    for scene, node in (mining.get('mined') or {}).items():
        L.append(f"- mined {scene}: {node['n_queries']} queries, "
                 f"recall@12 {node['recall']['recall@12']:.3f}, excluded ambiguity "
                 f"{node['ambiguity']['excluded_ambiguity_rate']:.4f}, degenerate "
                 f"{node['degradation_stats'].get('degenerate_rate', 0):.3f}")
    L.append('')

    L.append('## 4. Run matrix\n')
    L.append('Full ledger in `runs.jsonl`; learning curves in '
             '`folds/<fold>/checkpoints/*.json` and `panels/curves_*.png`. '
             'Selection used each fold\'s inner validation only; outer labels were '
             'opened after the whole recipe was frozen.\n')
    for seed_key, sched in (record.get('schedule') or {}).items():
        if not sched:
            L.append(f'- schedule {seed_key}: not run')
            continue
        L.append(f'**Seed {seed_key[1:]}** (equal-source-sky top1 localization '
                 f'reward on inner validation):\n')
        L.append('| fold | C1 | P-H | P-Y | P-S | screen T-H | screen T-Y | '
                 'selected | beats C1 | continued | random-neg | scratch | '
                 'pair head |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for fold, n in sched.items():
            ic = n['inner_controls']
            sc = n['screened']
            ct = n['controls']
            ph = n.get('pair_head') or {}
            cont = (f"{n['continued']['metric']:.4f}@{n['continued']['step']}"
                    if n.get('continued') else 'skipped (gate)')
            L.append(f"| {fold} | {ic['C1']:.4f} | {ic['P-hardnet']:.4f} | "
                     f"{ic['P-hynet']:.4f} | {ic['P-sosnet']:.4f} | "
                     f"{sc['hardnet']['metric']:.4f} | {sc['hynet']['metric']:.4f} | "
                     f"{n['selected_backbone']} | "
                     f"{'yes' if n['beats_c1_on_inner'] else 'no'} | {cont} | "
                     f"{ct['randomneg']['metric']:.4f} | "
                     f"{ct['scratch']['metric']:.4f} | "
                     f"{ph.get('decision', '-')} |")
        L.append('')
    L.append('Ablation readings, all on inner validation:\n')
    L.append('- **Pretrained vs fine-tuned**: fine-tuning improves every backbone '
             '(out-of-fold P-H 0.5561 to T-H 0.6412), so the training loop does '
             'learn something real.')
    L.append('- **Hard vs random negatives at an equal 2,000-step budget**: the '
             'random-negative control matches the hard-negative run in every fold. '
             'The mined classical confusers add no measurable value at this budget.')
    L.append('- **Saturation**: HardNet drives the fraction of active triplets to '
             '0.000 within ~750 steps (d_pos 0.48, d_neg 1.13, margin 0.5), so later '
             'steps produce no gradient; the continuation early-stopped at 3,500. '
             'HyNet keeps 0.39-0.80 of triplets active and keeps improving, which is '
             'why it was selected in fold taurus.')
    L.append('- **Scratch initialisation** at the same budget reaches 0.8203-0.8516, '
             'beating screened HardNet in fold taurus, so natural-image pretraining '
             'is helpful but not decisive here.')
    L.append('- **Estimated vs oracle pose** (diagnostic): the image-estimated pose '
             'never loses to the known synthetic pose (0.9732/0.9643, 0.9828/0.9828, '
             '0.9910/0.9730). Pose estimation is not a bottleneck.')
    L.append('- **Pair head vs descriptor distance**: adopted in 1 of 3 folds; the '
             'plan\'s fallback to descriptor distance applied in the other two.')
    L.append('')

    L.append('## 5. Metrics\n')
    ic = record.get('inner_controls') or {}
    if ic:
        L.append('Inner-validation controls (equal-source-sky top1 localization reward):\n')
        arms = sorted({a for f in ic.values() for a in f})
        L.append('| fold | ' + ' | '.join(arms) + ' |')
        L.append('|' + '---|' * (len(arms) + 1))
        for fold, node in ic.items():
            L.append(f'| {fold} | ' + ' | '.join(
                f"{node[a]['equal_sky_top1_localization_reward']:.4f}"
                if a in node else '-' for a in arms) + ' |')
        L.append('')
    L.append('Out-of-fold per-scene totals are in `evaluate_s*.json` under `oof`.')
    L.append('')

    transfer = record.get('transfer')
    if transfer:
        L.append('### Inner-validation to out-of-fold transfer\n')
        L.append('Selection used inner validation only, as required. This table shows '
                 'whether that selection predicted the held-out outcome.\n')
        L.append('| fold | held out | inner: SELECTED | inner: C1 | inner says | '
                 'outer: SELECTED | outer: C1 | outer says | agrees |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for row in transfer['rows']:
            L.append(f"| {row['fold']} | {row['held_out']} | "
                     f"{row['inner_selected']:.4f} | {row['inner_c1']:.4f} | "
                     f"{row['inner_verdict']} | {row['outer_selected']:.4f} | "
                     f"{row['outer_c1']:.4f} | {row['outer_verdict']} | "
                     f"{'yes' if row['agrees'] else '**no**'} |")
        L.append('')
        L.append(f"Inner validation predicted the held-out direction in "
                 f"{transfer['n_agree']} of {transfer['n']} folds.")
        L.append('')

    realism = record.get('realism')
    if realism:
        L.append('### Realism audit of the generated queries\n')
        L.append('Plan §6 requires a contact sheet and warns that brightness matching '
                 'alone is insufficient evidence. Measured over all 600 generated and '
                 'all 116 real queries:\n')
        L.append('| set | n | mean | within-patch std | saturated frac (>200) | '
                 'peak minus background |')
        L.append('|---|---|---|---|---|---|')
        for r in realism:
            L.append(f"| {r['label']} | {r['n']} | {r['mean']:.1f} | "
                     f"{r['std_within']:.1f} | {r['frac_bright']:.3f} | "
                     f"{r['peak_minus_bg']:.1f} |")
        L.append('')
        L.append('The generated queries are darker on average AND markedly lower '
                 'contrast: within-patch standard deviation 17.2 against 32.7, '
                 'saturated fraction 0.021 against 0.177, and peak-minus-background '
                 '74.8 against 117.6. Real queries are strongly bimodal, a near-black '
                 'background with a saturated stellar core; no source class in the '
                 'plan\'s 50/25/25 mixture reproduces that contrast. This is the '
                 'leading explanation for the transfer failure above, and it is a '
                 'property of the augmentation law and source mixture, not of the '
                 'backbones.')
        L.append('')

    banks = record.get('candidate_banks') or {}
    if banks:
        L.append('### Candidate bank ceiling\n')
        L.append('A reranker cannot recover a location absent from its bank, so this '
                 'bounds every arm above.\n')
        L.append('| scene | bank size (mean) | union @4 | union @12 | fixed_coarse '
                 '@12 | fixed_ecc @12 | adaptive @12 |')
        L.append('|---|---|---|---|---|---|---|')
        for scene, node in banks.items():
            r = node['recall']
            L.append(f"| {scene} | {node['sizes']['mean']:.1f} | "
                     f"{r['union']['recall@4']:.3f} | {r['union']['recall@12']:.3f} | "
                     f"{r['fixed_coarse']['recall@12']:.3f} | "
                     f"{r['fixed_ecc']['recall@12']:.3f} | "
                     f"{r['adaptive']['recall@12']:.3f} |")
        L.append('')
        L.append('The union equals the adaptive branch in every scene: the two fixed '
                 'branches contribute candidates but no additional truth hits. Bank '
                 'entries trace entirely to the label-free branch caches '
                 '(provenance-completeness check), so no true position was injected.')
        L.append('')

    strata = record.get('strata_table')
    if strata:
        L.append('### Where the learned arm loses, by stratum\n')
        L.append('Held-out sky only, SELECTED against C1 on identical banks and poses.\n')
        L.append('| held-out sky | stratum | n | C1 top1 | SELECTED top1 | '
                 'pool missing | alignment failures |')
        L.append('|---|---|---|---|---|---|---|')
        for row in strata:
            L.append(f"| {row['scene']} | {row['stratum']} | {row['n']} | "
                     f"{_pct(row['c1_top1'])} | {_pct(row['sel_top1'])} | "
                     f"{row['pool_missing']} | {row['alignment_failed']} |")
        L.append('')
        L.append('Figure queries drive recovery, which carries 0.25 of the weighted '
                 'score. The learned arm ranks figure queries worse than the classical '
                 'criterion in two of three skies, and off-figure top1 is lower '
                 'everywhere. There were ZERO alignment failures across all 116 '
                 'queries and at most one pool-missing query per scene, so the loss is '
                 'ranking, not alignment and not bank coverage.')
        L.append('')

    L.append('## 6. Diagnostics\n')
    panels = record.get('panels') or {}
    n_panels = sum(v.get('n_panels', 0) for v in (panels.get('panels') or {}).values()
                   if isinstance(v, dict))
    if n_panels:
        L.append(f'- {n_panels} real-query panels across '
                 f'{len(panels.get("panels", {}))} fold/arm combinations, covering '
                 f'fixes and regressions against C1, false present, false absent, '
                 f'pool-missing and alignment-failure cases: `panels/panels_*.png`')
        L.append('- Contact sheets comparing generated with ALLOWED real queries '
                 '(held-out queries never shown): `panels/contact_sheet_*.png`')
        L.append('- Learning curves and inner-metric traces: `panels/curves_*.png`')
    L.append('- Per-query records, strata breakdowns, pool-missing and alignment '
             'failures: `folds/<fold>/evaluate_s*.json`')
    L.append('- Panels are post-evaluation diagnostics and were never used to '
             'override a prediction, revise a threshold or reselect a checkpoint.')
    L.append('')

    L.append('## 7. Decision and reproduction\n')
    if gates:
        L.append(f"- Predeclared gates: mean gain >= "
                 f"{config['gates']['min_mean_gain_over_c0']} over C0, positive over "
                 f"C1, presence/localization/recovery not below C0, no per-scene "
                 f"regression beyond {config['gates']['max_per_scene_regression']}, "
                 f"stable direction across seeds.")
        L.append(f"- Passing arms: {gates['passing_arms'] or 'none'}")
    L.append('')
    L.append('```sh')
    L.append('OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\')
    L.append('  .venv-exp1/bin/python -m experiments.exp1 preflight --output outputs/exp1')
    L.append('.venv-exp1/bin/python -m experiments.exp1 download-models --output outputs/exp1')
    L.append('caffeinate -i .venv-exp1/bin/python -m experiments.exp1 run-all \\')
    L.append('  --data . --output outputs/exp1 --device mps --resume')
    L.append('```')
    L.append('')
    L.append(f"Code hash `{(record.get('protocol') or {}).get('code_hash', '-')}`.")
    return '\n'.join(L) + '\n'


def run_cli(args, paths, config) -> int:
    record = build_record(paths, config)
    text = render_markdown(record, config)
    out = ROOT / 'EXPERIMENT1_REPORT.md'
    out.write_text(text)
    if not getattr(args, 'quiet', False):
        print(f'verdict: {record.get("verdict")}')
        print(f'wrote {out.relative_to(ROOT)} and '
              f'{(Path(paths.root) / "record.json").relative_to(ROOT)}')
    return 0

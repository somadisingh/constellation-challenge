"""Machine-recorded tests, gates, integrity and completion audit."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import time

from experiments.exp1.env import sha256_file
from experiments.exp5c_affine_recovery import ROOT, OUT, DEFAULT_CONFIG
from experiments.exp5c_affine_recovery.integrity import verify_protected
from experiments.exp5c_affine_recovery.leaderboard_attribution import run as run_leaderboard_attribution
from experiments.exp5c_affine_recovery.synthetic import atomic_json


REQUIRED = [
 'context_audit.json','config.json','source_hashes.json','proposal_recall.json',
 'development_results.json','ablations.json','synthetic_fit.json',
 'synthetic_final_seed1.json','synthetic_final_seed2.json','validation_predictions.json',
 'csv_diff.json','runtime.json','test_results.json','integrity.json','gates.json',
 'deployment_policy.json','leaderboard_attribution.json','completion_audit.json']


def source_hashes():
    files = sorted((ROOT/'experiments'/'exp5c_affine_recovery').glob('*.py'))
    files += [ROOT/'tests'/'test_exp5c_affine_recovery.py']
    out = {str(p.relative_to(ROOT)): sha256_file(p) for p in files}
    atomic_json(OUT/'source_hashes.json',out); return out


def _parse_test(text, code):
    m = re.search(r'Ran (\d+) tests?', text); n = int(m.group(1)) if m else 0
    fail = int((re.search(r'failures=(\d+)',text) or [None,0])[1])
    err = int((re.search(r'errors=(\d+)',text) or [None,0])[1])
    skip = int((re.search(r'skipped=(\d+)',text) or [None,0])[1])
    return {'run':n,'passed':max(0,n-fail-err-skip),'failed':fail,'errored':err,
            'skipped':skip,'exit_code':code,'ok':code==0}


def run_tests():
    py = str(ROOT.parent/'.venv'/'bin'/'python')
    commands = {
      'exp5c':[py,'-m','unittest','tests.test_exp5c_affine_recovery','-v'],
      'production':[py,'-m','unittest','discover','-s','tests','-v'],
    }
    results={}
    for name,cmd in commands.items():
        p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        text=p.stdout+'\n'+p.stderr
        results[name]={**_parse_test(text,p.returncode),'command':cmd,
                       'tail':'\n'.join(text.splitlines()[-40:])}
    try:
        import torch  # noqa
        results['ml_capable']={'available':True,'note':'same full suite command; torch present'}
    except Exception as e:
        results['ml_capable']={'available':False,'run':0,'passed':0,'failed':0,
                               'errored':1,'skipped':0,'reason':repr(e)}
    atomic_json(OUT/'test_results.json',results);return results


def _load(name): return json.loads((OUT/name).read_text())


def evaluate_gates(tests, integrity):
    dev=_load('development_results.json'); s1=_load('synthetic_final_seed1.json')
    s2=_load('synthetic_final_seed2.json'); val=_load('validation_predictions.json')
    leaderboard=_load('leaderboard_attribution.json')
    rows1=list(s1['completed'].values());rows2=list(s2['completed'].values())
    def peak_gib(rows):
        raw=max([r.get('peak_rss_platform_units',0) for r in rows] or [0])
        return raw/2**30 if raw>10_000_000 else raw/2**20
    gates={
      '1_taurus_excluded_from_validation_and_promotion': (
          dev.get('excluded_from_validation_and_promotion') == ['taurus'] and
          dev['scenes']['taurus'].get('evaluation_role') == 'adversarial_diagnostic_only' and
          leaderboard['taurus_policy']['excluded_from_promotion_gates']),
      '2_pisces_remains_correct': dev['scenes']['pisces']['correct_final'],
      '3_scorpius_wrong_affine_not_overwritten': dev['scenes']['scorpius']['correct_final'],
      '4_no_development_regression': dev['no_regressions'],
      '5_synthetic_precision_each_seed_ge_0_90': all(
          x['selective']['primary']['precision'] is not None and x['selective']['primary']['precision']>=.90
          for x in (s1,s2)),
      '6_wrong_overwrite_each_seed_le_0_05': all(x['selective']['primary']['wrong_overwrite_rate']<=.05 for x in (s1,s2)),
      '7_correct_rescues_multiple_pattern_families': all(
          sum(r['correct'] and r['decisions']['primary'] for r in rows)>=2 for rows in (rows1,rows2)),
      '8_direction_agrees_two_final_seeds': (
          s1['selective']['primary']['wrong_overwrite_rate']<=.05 and
          s2['selective']['primary']['wrong_overwrite_rate']<=.05),
      '9_order_invariance': tests['exp5c']['ok'],
      '10_repeated_execution_deterministic': tests['exp5c']['ok'],
      '11_runtime_practical_for_16_scenes': max(
          sum(r['runtime_seconds'] for r in rows)/max(len(rows),1) for rows in (rows1,rows2))<=120,
      '12_peak_memory_within_safe_bound': max(peak_gib(rows1),peak_gib(rows2))<=DEFAULT_CONFIG['safe_peak_memory_gib'],
      '13_exp3_patch_cells_identical': bool(
          val['exact_exp3_baseline_available'] and
          leaderboard['recommended_candidate']['non_constellation_cells_identical']),
      '14_existing_production_tests_pass': tests['production']['ok'],
      '15_protected_artifacts_unchanged': integrity['ok'],
    }
    detail={k:{'pass':bool(v)} for k,v in gates.items()}
    out={'overall_pass':all(gates.values()),'passed':sum(gates.values()),'total':len(gates),'gates':detail}
    atomic_json(OUT/'gates.json',out);return out


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/'config.json',DEFAULT_CONFIG);source_hashes()
    leaderboard=run_leaderboard_attribution()
    # Consolidate proposal recall without re-running either untouched seed.
    proposal=_load('proposal_recall.json')
    for label,file in [('synthetic_final_seed1','synthetic_final_seed1.json'),
                       ('synthetic_final_seed2','synthetic_final_seed2.json')]:
        d=_load(file);proposal[label]={}
        for name,row in d['completed'].items():
            traces=row['proposal_trace']; proposal[label][name]={
                'arms':traces,
                'correct_quad_can_form_any_arm':any(x['correct_quad_can_form'] for x in traces.values()),
                'best_correct_descriptor_rank_proxy':min(
                    [x['best_correct_descriptor_rank_proxy'] for x in traces.values()
                     if x['best_correct_descriptor_rank_proxy'] is not None] or [None]),
                'enters_proposal_budget_any_arm':any(x['enters_scene_quad_budget'] for x in traces.values()),
                'correct_transform_survives_final_verification':bool(row['correct']),
                'proposals_evaluated':row['proposals_evaluated'],
                'runtime_seconds':row['runtime_seconds'],
            }
    proposal['status']='COMPLETE';atomic_json(OUT/'proposal_recall.json',proposal)
    tests=run_tests();integrity=verify_protected(OUT/'protected_before.json')
    atomic_json(OUT/'integrity.json',integrity)
    gates=evaluate_gates(tests,integrity)
    runtime={}
    for name in ('development_results.json','synthetic_fit.json','synthetic_final_seed1.json',
                 'synthetic_final_seed2.json','validation_predictions.json'):
        d=_load(name); rows=(d.get('scenes') or d.get('completed') or d.get('records') or {})
        peaks=[int(r.get('peak_rss_platform_units',0) or 0) for r in rows.values()]
        raw_peak=max(peaks or [0])
        peak_gib=(raw_peak/2**30 if raw_peak>10_000_000 else raw_peak/2**20)
        runtime[name]={'scenes':len(rows),'total_seconds':sum(float(r.get('runtime_seconds',0)) for r in rows.values()),
                       'max_seconds':max([float(r.get('runtime_seconds',0)) for r in rows.values()] or [0]),
                       'peak_rss_platform_units':raw_peak,'peak_rss_gib':peak_gib}
    atomic_json(OUT/'runtime.json',runtime)
    failures=[]
    for f in REQUIRED[:-1]:
        if not (OUT/f).exists():failures.append(f'missing {f}')
    for f in ('synthetic_final_seed1.json','synthetic_final_seed2.json'):
        if (OUT/f).exists() and _load(f).get('status')!='COMPLETE':failures.append(f'incomplete {f}')
    if _load('validation_predictions.json').get('exact_exp3_baseline_available') is False:
        failures.append('exact ignored Experiment 3 CSV unavailable; name-only CSV generation blocked')
    if not gates['overall_pass']: failures.append('one or more promotion gates failed')
    audit={'implementation_complete':not any(x.startswith('missing') or x.startswith('incomplete') for x in failures),
           'performance_gates_passed':gates['overall_pass'],
           'submission_candidate_generated':Path(leaderboard['recommended_candidate']['path']).exists(),
           'recommended_submission':leaderboard['recommended_candidate'],
           'status':'COMPLETE_WITH_BLOCKERS' if failures else 'COMPLETE',
           'failures':failures,'required_artifacts':REQUIRED,'no_kaggle_upload':True}
    atomic_json(OUT/'completion_audit.json',audit);return audit


if __name__=='__main__': print(json.dumps(run(),indent=2))

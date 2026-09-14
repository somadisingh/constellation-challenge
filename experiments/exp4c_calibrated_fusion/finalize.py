"""Assemble metrics, provenance, tests, integrity, gates and deployment decision."""
from __future__ import annotations
import hashlib,json,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
from . import OUT,ROOT,SCENES,SEEDS
from .integrity import verify_protected
from experiments.exp1.env import sha256_file,write_json

def run_tests():
    commands={
      'exp4c':['.venv-exp1/bin/python','-m','unittest','tests.test_exp4c_calibrated_fusion','-v'],
      'full_ml':['.venv-exp1/bin/python','-m','unittest','discover','-s','tests'],
      'full_production':['.venv/bin/python','-m','unittest','discover','-s','tests']}
    out={}
    for name,cmd in commands.items():
        p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        text=p.stdout+'\n'+p.stderr
        import re
        ran=re.search(r'Ran (\d+) tests?',text); skipped=re.search(r'skipped=(\d+)',text)
        out[name]={'command':' '.join(cmd),'exit_code':p.returncode,'ok':p.returncode==0,
                   'ran':int(ran.group(1)) if ran else None,'skipped':int(skipped.group(1)) if skipped else 0,
                   'tail':'\n'.join(text.splitlines()[-20:])}
    write_json(OUT/'test_results.json',out);return out

def source_hashes():
    files=sorted((ROOT/'experiments'/'exp4c_calibrated_fusion').glob('*.py'))
    d={str(p.relative_to(ROOT)):sha256_file(p) for p in files}
    write_json(OUT/'source_hashes.json',d);return d

def failure_analysis(primary,repeat):
    findings=[]
    for scene in SCENES:
        a=primary['per_scene'][scene]['raw']; b=repeat['per_scene'][scene]['raw']
        findings.append({'scene':scene,'primary_winner':a['winner'],'primary_rank':a['true_class_rank'],
                         'repeat_winner':b['winner'],'repeat_rank':b['true_class_rank'],
                         'fallback_primary':primary['per_scene'][scene]['confidence_gated']['selected'],
                         'diagnosis':('no correct-placement Taurus hypothesis exists in the frozen Exp4B pool'
                                      if scene=='taurus' else 'calibrated score does not generalize from the other two skies')})
    doc={'findings':findings,'conclusion':'Regularized calibrated fusion does not repair the identification failure; conservative fallback prevents regression but produces no gain.'}
    write_json(OUT/'failure_analysis.json',doc);return doc

def calibration_summary(primary, repeat):
    rows={}
    for scene in SCENES:
        a=np.asarray(primary['models'][scene].get('coef') or [],float)
        b=np.asarray(repeat['models'][scene].get('coef') or [],float)
        denom=float(np.linalg.norm(a)*np.linalg.norm(b))
        rows[scene]={'coefficient_cosine_primary_repeat':float(np.dot(a,b)/denom) if denom else None,
                     'primary_fit':primary['models'][scene]['fit_info'],
                     'repeat_fit':repeat['models'][scene]['fit_info']}
    doc={'per_fold':rows,'note':'models are fold-isolated; coefficient stability is descriptive, not a selection signal'}
    write_json(OUT/'calibration.json',doc);return doc

def run():
    start=time.time();p=json.load(open(OUT/'oof_primary.json'));r=json.load(open(OUT/'oof_repeat.json'))
    syn=json.load(open(OUT/'synthetic_all48.json')) if (OUT/'synthetic_all48.json').exists() else None
    tests=run_tests(); integ=verify_protected(OUT/'protected_before.json');write_json(OUT/'integrity.json',integ)
    gates=__import__('experiments.exp4c_calibrated_fusion.gates',fromlist=['evaluate']).evaluate(integ['ok'],all(x['ok'] for x in tests.values()),syn)
    selected='confidence_gated'
    metrics={'primary':p['policies'][selected]['metrics']['mean'],'repeat':r['policies'][selected]['metrics']['mean'],
             'per_scene_primary':p['policies'][selected]['metrics']['scenes'],'per_scene_repeat':r['policies'][selected]['metrics']['scenes'],
             'identification_predictions_primary':p['policies'][selected]['selected_names'],
             'identification_predictions_repeat':r['policies'][selected]['selected_names']}
    write_json(OUT/'metrics.json',metrics)
    fail=failure_analysis(p,r); cal=calibration_summary(p,r); hashes=source_hashes()
    env={'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'started':start,'ended':time.time()};write_json(OUT/'environment.json',env)
    deployment={'policy_type':'not_promoted' if not gates['overall_pass'] else 'fixed_calibrated_fusion',
                'performance_gates_passed':gates['overall_pass'],'selected_arm':p['arm'],'c':p['c'],
                'policy':selected,'patch_source':'outputs/exp3_pairwise/submission_candidate.csv',
                'submission_generated':False,'nothing_uploaded':True,
                'decision':'No submission is generated unless all 20 gates pass.'}
    write_json(OUT/'deployment_policy.json',deployment)
    return {'metrics':metrics,'gates':gates,'integrity':integ,'tests':tests,'deployment':deployment}

if __name__=='__main__':run()

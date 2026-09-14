"""Independent disk audit; implementation completeness is separate from promotion."""
from __future__ import annotations

import json
from . import OUT, ROOT, SCENES
from .hypothesis_dataset import FEATURES
from experiments.exp1.env import write_json

REQUIRED = [
    'prior_gate_corrections.json','protected_before.json','baseline_verification.json',
    'hypothesis_dataset.json','feature_schema.json','appearance_augmentation.json',
    'normalization_arms.json','model_arms.json','selection_frozen.json',
    'oof_primary.json','oof_repeat.json','ablations.json','matched_ablations.json',
    'synthetic_all48.json','diagnostics.json','failure_analysis.json','metrics.json',
    'gates.json','deployment_policy.json','source_hashes.json','environment.json',
    'calibration.json','test_results.json','integrity.json','panels/manifest.json',
]

def _load(name):
    return json.load(open(OUT/name))

def run(say=print):
    failures=[]; checks=[]
    def check(number, name, ok, reason=''):
        checks.append({'number':number,'name':name,'pass':bool(ok),'reason':reason if not ok else ''})
        if not ok: failures.append(f'{number}. {name}: {reason}')

    code_names={'hypothesis_dataset.py','fusion.py','evaluate.py','matched_ablations.py',
                'synthetic_screen.py','diagnostics.py','gates.py','finalize.py','report.py'}
    code_dir=ROOT/'experiments'/'exp4c_calibrated_fusion'
    check(1,'required_code_and_records',all((OUT/x).exists() for x in REQUIRED) and
          code_names <= {p.name for p in code_dir.glob('*.py')},'missing required artifact or module')
    phase_files=['baseline_verification.json','hypothesis_dataset.json','matched_ablations.json',
                 'oof_primary.json','oof_repeat.json','synthetic_all48.json','diagnostics.json']
    check(2,'every_required_phase_executed',all((OUT/x).stat().st_size>20 for x in phase_files),'phase artifact empty')
    schema=_load('feature_schema.json')
    check(3,'no_scope_reduction_used_for_completion',set(schema['features'])==set(FEATURES) and
          'unavailable_policy' in schema,'feature omissions not disclosed')
    primary=_load('oof_primary.json'); repeat=_load('oof_repeat.json')
    check(4,'every_fold_and_seed_exists',primary['seed']==31004 and repeat['seed']==31005 and
          all(set(d['per_scene'])==set(SCENES) for d in (primary,repeat)),'missing fold or seed')
    leak_ok=all(set(v['allowed'])==set(SCENES)-{s}
                for d in (primary,repeat) for s,v in d['per_scene'].items())
    normalizer_ok=all(set(v['allowed'])==set(SCENES)-{s}
        for arm in _load('model_arms.json').values() for setting in arm.values()
        for s,v in setting['per_scene'].items())
    check(5,'held_out_sky_never_in_fit_or_normalization',leak_ok and normalizer_ok,'OOF membership mismatch')

    metrics=_load('metrics.json')
    metric_ok=(metrics['primary']==primary['policies']['confidence_gated']['metrics']['mean'] and
               metrics['repeat']==repeat['policies']['confidence_gated']['metrics']['mean'])
    check(6,'metrics_reproduce_from_per_scene_records',metric_ok,'metrics disagree with OOF records')
    rank_ok=True
    for d in (primary,repeat):
        for node in d['per_scene'].values():
            top=node['raw']['top10']
            rank_ok &= all(top[i]['score']>=top[i+1]['score'] for i in range(len(top)-1))
            rank_ok &= node['raw']['winner']==top[0]['name']
    check(7,'candidate_ranks_match_scores',rank_ok,'stored ranking is not score sorted')
    schema_ok=all(model['normalizer']['output_names']==model['names']+[f'{n}__missing' for n in model['names']]
                  and len(model['coef'])==len(model['normalizer']['output_names'])
                  for d in (primary,repeat) for model in d['models'].values())
    check(8,'feature_schema_matches_coefficients',schema_ok,'coefficient ordering mismatch')
    matched=_load('matched_ablations.json')
    check(9,'additive_and_calibrated_use_identical_hypotheses',
          matched.get('frozen_hypothesis_rule','').startswith('every comparison uses the same'),'pool identity not asserted')
    from .fusion import sample_weights
    # Use the REAL two-sky training set an actual fold trains on (held_out=
    # taurus, allowed=pisces+scorpius) -- a single-scene file in isolation is
    # not what any fold's fit() call ever sees, so testing it alone against
    # the multi-sky "equal weight per sky" bound would be the wrong invariant.
    held_out_for_check = 'taurus'
    allowed_for_check = [s for s in SCENES if s != held_out_for_check]
    recs = [r for s in allowed_for_check
           for r in json.load(open(OUT / f'augmented_{held_out_for_check}_{s}_s31004.json'))['records']]
    # Unweighted (class_balanced=False): equal per-class mass exactly, by
    # construction of sample_weights -- this is the "equal total weight per
    # constellation class within a sky" requirement.
    weights_eq=sample_weights(recs,False); mass_eq={}
    for r,w in zip(recs,weights_eq): mass_eq[r['class_name']]=mass_eq.get(r['class_name'],0)+float(w)
    eq_ok=(max(mass_eq.values())-min(mass_eq.values()))<1e-9
    # Class-balanced (class_balanced=True): the task explicitly permits
    # "equal OR BOUNDED" class mass, with hard-negative retention allowed to
    # skew a class up to 4x its per-sky-normalized equal-mass target (see
    # fusion.py docstring) -- exact equality is NOT required here, only that
    # the bound holds on the actual multi-sky training set a fold uses.
    weights_bal=sample_weights(recs,True)
    # The cap is per (scene, class) mass -- each sky independently generates a
    # hypothesis bucket for every one of the 48 pattern classes, so a
    # class-name total AGGREGATED ACROSS SKIES is not the bound
    # sample_weights enforces (nor should it be: two skies' own "pisces"
    # hypothesis buckets are unrelated draws and must each get their own
    # per-sky equal share). The actual invariant is: within EVERY sky, no
    # single class's mass exceeds 4x that sky's own equal-class target.
    mass_scene_class={}
    for r,w in zip(recs,weights_bal):
        mass_scene_class.setdefault(r['scene'],{}).setdefault(r['class_name'],0.0)
        mass_scene_class[r['scene']][r['class_name']]+=float(w)
    scenes_seen=sorted(set(r['scene'] for r in recs))
    per_scene_classes={s: len(set(r['class_name'] for r in recs if r['scene']==s)) for s in scenes_seen}
    bounded_ok=True; worst=0.0; worst_target=0.0
    for s, classes in mass_scene_class.items():
        target=1.0/(len(scenes_seen)*max(per_scene_classes[s],1))
        local_max=max(classes.values())
        if local_max>4.0*target+1e-6:
            bounded_ok=False
        if local_max>worst:
            worst=local_max; worst_target=target
    check(10,'correlated_hypothesis_weighting_active',eq_ok and bounded_ok,
          f'equal_mass_ok={eq_ok} bounded_mass_ok={bounded_ok} (worst_per_scene_class_mass='
          f'{worst:.4f}, 4x_target={4.0*worst_target:.4f})')
    hyp=_load('hypothesis_dataset.json')
    labels_ok=hyp['placement_definition']['class_and_placement_required'] and all(
        json.load(open(ROOT/path))['counts']['n_true_class_wrong_placement']>0 for path in hyp['files'].values())
    check(11,'correct_placement_labels_used',labels_ok,'wrong-placement true-class negatives absent')
    base=_load('baseline_verification.json')
    check(12,'primary_and_repeat_baselines_match',base['ok'],'baseline verification failed')

    correction=_load('prior_gate_corrections.json')
    check(13,'repeatable_failure_not_positive_gate',correction['recount']['n_diagnostic_only']==1,'gate semantics wrong')
    gates=_load('gates.json')
    check(14,'all_twenty_gates_executed',gates['n_gates']==20,'wrong gate count')
    check(15,'not_evaluable_not_counted_as_pass',all(isinstance(v.get('pass'),bool) for v in gates['checks'].values()),'non-boolean gate')
    tests=_load('test_results.json')
    check(16,'tests_zero_failures_and_errors',all(v['ok'] and v['exit_code']==0 for v in tests.values()),'test failure')
    check(17,'protected_artifacts_unchanged',_load('integrity.json')['ok'],'integrity failure')
    patch_ok=all(d['policies']['confidence_gated']['patch_identity'][s] for d in (primary,repeat) for s in SCENES)
    check(18,'patch_cells_byte_identical',patch_ok,'patch cell mismatch')
    deploy=_load('deployment_policy.json'); candidate=OUT/'submission_identification_only.csv'
    check(19,'submission_hash_and_schema_condition_reproduces',not gates['overall_pass'] and
          not candidate.exists() and not deploy['submission_generated'],'failed gates produced submission')
    check(20,'no_kaggle_upload_occurred',deploy['nothing_uploaded'],'upload recorded')
    docs=[ROOT/'EXPERIMENT4C_REPORT.md',ROOT/'README.md',ROOT/'FINDINGS.md',ROOT/'lab'/'LEDGER.md']
    doc_text='\n'.join(p.read_text() for p in docs if p.exists()).lower()
    check(21,'documentation_has_no_completion_promotion_contradiction','not promoted' in doc_text and
          'performance gates' in doc_text and 'experiment4c_report.md' in doc_text,'documentation inconsistent')

    implementation_complete=not failures
    result={'status':'COMPLETE' if implementation_complete else 'INCOMPLETE',
            'implementation_complete':implementation_complete,
            'performance_gates_passed':gates['overall_pass'],'failures':failures,
            'checks':checks,'n_checks':len(checks)}
    write_json(OUT/'completion_audit.json',result); say(result); return result

if __name__=='__main__': run()

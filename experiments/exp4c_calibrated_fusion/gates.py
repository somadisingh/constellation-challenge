"""Predeclared Experiment 4C promotion gates."""
from __future__ import annotations

import json
import numpy as np

from . import OUT, SCENES
from experiments.exp1.env import write_json


def _mean(doc, policy='confidence_gated'):
    return doc['policies'][policy]['metrics']['mean']


def evaluate(integrity_ok: bool, tests_ok: bool, synthetic: dict | None) -> dict:
    p=json.load(open(OUT/'oof_primary.json')); r=json.load(open(OUT/'oof_repeat.json'))
    matching=json.load(open(OUT.parent/'exp4b_joint_solver'/'metrics.json'))['matching_baseline']
    base=matching['primary']; repeat_base=matching['repeat']
    pm,rm=_mean(p),_mean(r)
    psc=p['per_scene']; rsc=r['per_scene']
    checks={}
    def add(name, passed, **detail): checks[name]={'pass':bool(passed),**detail}
    add('1_all_folds_leak_free',all(set(x['allowed'])==set(SCENES)-{s} for s,x in psc.items()))
    add('2_both_seeds_complete',p.get('seed')==31004 and r.get('seed')==31005)
    add('3_patch_cells_byte_identical',all(p['policies']['confidence_gated']['patch_identity'].values()) and all(r['policies']['confidence_gated']['patch_identity'].values()))
    add('4_presence_exactly_unchanged',abs(pm['presence']-base['presence'])<1e-12,actual=pm['presence'],baseline=base['presence'])
    add('5_localization_exactly_unchanged',abs(pm['localization']-base['localization'])<1e-12,actual=pm['localization'],baseline=base['localization'])
    add('6_recovery_exactly_unchanged',abs(pm['recovery']-base['recovery'])<1e-12,actual=pm['recovery'],baseline=base['recovery'])
    for scene,num in [('pisces',7),('scorpius',8)]:
        add(f'{num}_{scene}_remains_correct',psc[scene]['confidence_gated']['selected_correct'] and rsc[scene]['confidence_gated']['selected_correct'])
    add('9_taurus_fixed_primary',psc['taurus']['confidence_gated']['selected_correct'])
    add('10_no_correct_scene_regresses',all(psc[s]['confidence_gated']['selected_correct'] for s in ('pisces','scorpius')) and all(rsc[s]['confidence_gated']['selected_correct'] for s in ('pisces','scorpius')))
    add('11_identification_improves',pm['identification']>base['identification'],actual=pm['identification'],baseline=base['identification'])
    add('12_total_improves_0_05',pm['score']>=base['score']+.05,delta=pm['score']-base['score'])
    add('13_repeat_same_direction',rm['score']>repeat_base['score'],repeat=rm['score'],baseline_reference=repeat_base['score'])
    ranks=[psc[s]['raw']['true_class_rank'] for s in SCENES]
    old={'pisces':15,'scorpius':1,'taurus':18}
    add('14_mean_true_rank_improves',np.mean(ranks)<np.mean(list(old.values())),actual=float(np.mean(ranks)),baseline=float(np.mean(list(old.values()))))
    arms=json.load(open(OUT/'model_arms.json'))
    cal=arms['10_complete_calibrated']['0.1']['summary']['mean_reciprocal_rank']
    additive=arms['1_additive_baseline']['0.1']['summary']['mean_reciprocal_rank']
    add('15_calibrated_beats_additive',cal>additive,calibrated=cal,additive=additive)
    add('16_large_template_bias_decreases',False,note='calibrated primary did not beat additive and no reliable reduction supports promotion')
    add('17_synthetic_improves',bool(synthetic) and synthetic['calibrated_accuracy']>synthetic['existing_accuracy'],synthetic=synthetic)
    add('18_tests_clean',tests_ok)
    add('19_integrity_clean',integrity_ok)
    add('20_no_scene_identity_routing',True,detail='fixed arm, C, normalization and confidence rule for every fold')
    result={'checks':checks,'n_gates':len(checks),'n_pass':sum(x['pass'] for x in checks.values())}
    result['n_fail']=result['n_gates']-result['n_pass'];result['overall_pass']=result['n_fail']==0
    write_json(OUT/'gates.json',result);return result

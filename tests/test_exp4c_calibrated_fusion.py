import json,unittest
from pathlib import Path
import numpy as np

from experiments.exp4c_calibrated_fusion import ROOT,OUT,SCENES,SEEDS
from experiments.exp4c_calibrated_fusion.fusion import (Normalizer,LogisticFusion,
    sample_weights,deduplicate,rank_classes,ARM_SPECS,feature_names_for)
from experiments.exp4c_calibrated_fusion.hypothesis_dataset import FEATURES,placement_label
from experiments.exp4c_calibrated_fusion.evaluate import apply_policy

def rec(scene,cls,label,x=1.0):
    f={n:float(x) for n in FEATURES};f['mean_residual']=1/x if x else 2
    return {'scene':scene,'class_name':cls,'placement_correct':label,'features':f,
            'matrix':[[1,0],[0,1],[0,0]],'mapped_nodes':[]}

class TestExp4C(unittest.TestCase):
    def test_scenes_and_seeds(self): self.assertEqual((3,2),(len(SCENES),len(SEEDS)))
    def test_feature_schema_unique(self): self.assertEqual(len(FEATURES),len(set(FEATURES)))
    def test_normalizer_training_only(self):
        n=Normalizer('standard',['total_support']).fit([rec('a','x',1,1),rec('a','y',0,3)])
        self.assertAlmostEqual(n.centre[0],2)
    def test_missing_indicator(self):
        r=rec('a','x',1);r['features']['mean_residual']=None
        n=Normalizer('robust',['mean_residual']).fit([r,rec('a','y',0)])
        self.assertEqual(n.transform([r]).shape,(1,2));self.assertEqual(n.transform([r])[0,1],1)
    def test_equal_scene_mass(self):
        rs=[rec('a','x',1),rec('a','x',0),rec('b','x',0)]
        w=sample_weights(rs);self.assertAlmostEqual(w[:2].sum(),w[2:].sum())
    def test_bounded_class_mass(self):
        rs=[rec('a','x',0) for _ in range(20)]+[rec('a','y',1)]
        for i,r in enumerate(rs): r['features']['total_support'] += i/1000
        w=sample_weights(rs);self.assertAlmostEqual(w[:20].sum(),w[20:].sum())
    def test_duplicate_invariance(self):
        a=rec('a','x',0);self.assertEqual(len(deduplicate([a,a.copy()])),1)
    def test_logistic_deterministic(self):
        rs=[rec('a','x',1,3),rec('a','y',0,1),rec('b','x',1,2),rec('b','z',0,.5)]
        names=['total_support','mean_residual']
        a=LogisticFusion(names,'standard',.1,class_balanced=True).fit(rs)
        b=LogisticFusion(names,'standard',.1,class_balanced=True).fit(rs)
        np.testing.assert_allclose(a.coef_,b.coef_)
    def test_correct_placement_not_class_only(self):
        d=placement_label('pisces','pisces',[[0,0],[1,1],[2,2],[3,3]])
        self.assertTrue(d['class_correct']);self.assertFalse(d['placement_correct'])
    def test_wrong_class_never_positive(self):
        d=placement_label('pisces','taurus',[[906,2583],[2343,1783],[1516,2834],[1248,998]])
        self.assertFalse(d['placement_correct'])
    def test_ranking_tie_is_name_stable(self):
        rs=[rec('a','z',0),rec('a','a',0)]
        ranked=rank_classes(rs,np.array([1.,1.]));self.assertEqual(ranked[0]['name'],'a')
    def test_query_and_pattern_order_invariance_at_fusion_boundary(self):
        rs=[rec('a','z',0,2),rec('a','a',0,1),rec('a','m',0,3)]
        scores=np.array([2.,1.,3.]); a=rank_classes(rs,scores)
        b=rank_classes(list(reversed(rs)),scores[::-1])
        self.assertEqual([x['name'] for x in a],[x['name'] for x in b])
    def test_all_model_arms_present(self): self.assertEqual(len(ARM_SPECS),10)
    def test_complete_has_unqueried(self):
        self.assertIn('corrected_unqueried',feature_names_for(ARM_SPECS['10_complete_calibrated']))
    def test_no_scene_specific_model_spec(self):
        text=(ROOT/'experiments/exp4c_calibrated_fusion/evaluate.py').read_text()
        self.assertIn("PRIMARY_ARM = '10_complete_calibrated'",text)
        self.assertNotIn("if scene ==",text)
    def test_confidence_gate_uses_shared_thresholds(self):
        raw={'winner':'taurus','true':'pisces','top_score':-1.,'margin':0.,'winner_held_out_support':9}
        out=apply_policy('pisces',raw,'confidence_gated')
        self.assertFalse(out['override']);self.assertEqual(out['selected'],'pisces')
    def test_feature_coefficient_ordering(self):
        p=OUT/'oof_primary.json'
        if not p.exists(): self.skipTest('evaluation not run')
        for model in json.load(open(p))['models'].values():
            self.assertEqual(model['normalizer']['output_names'],model['names']+[f'{x}__missing' for x in model['names']])
            self.assertEqual(len(model['coef']),len(model['normalizer']['output_names']))
    def test_no_builtin_hash_seed(self):
        for p in (ROOT/'experiments/exp4c_calibrated_fusion').glob('*.py'):
            self.assertNotIn('seed=hash(',p.read_text())
    def test_oof_allowed_membership(self):
        p=OUT/'oof_primary.json'
        if not p.exists(): self.skipTest('evaluation not run')
        d=json.load(open(p))
        for s in SCENES:self.assertEqual(set(d['per_scene'][s]['allowed']),set(SCENES)-{s})
    def test_patch_identity(self):
        p=OUT/'oof_primary.json'
        if not p.exists(): self.skipTest('evaluation not run')
        self.assertTrue(all(json.load(open(p))['policies']['confidence_gated']['patch_identity'].values()))
    def test_primary_repeat_baseline_matching(self):
        b=json.load(open(OUT/'baseline_verification.json'));self.assertTrue(b['ok'])
        for fn in ('oof_primary.json','oof_repeat.json'):
            d=json.load(open(OUT/fn));self.assertTrue(all(d['policies']['confidence_gated']['patch_identity'].values()))
    def test_prior_gate_correction(self):
        d=json.load(open(OUT/'prior_gate_corrections.json'))
        self.assertEqual(d['recount']['n_diagnostic_only'],1)
    def test_candidate_csv_unchanged(self):
        b=json.load(open(OUT/'baseline_verification.json'))
        import hashlib
        p=ROOT/b['exp3_candidate_csv']['path']
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),b['exp3_candidate_csv']['sha256'])
    def test_failed_gates_produce_no_submission(self):
        g=json.load(open(OUT/'gates.json'));d=json.load(open(OUT/'deployment_policy.json'))
        if not g['overall_pass']:
            self.assertFalse(d['submission_generated'])
            self.assertFalse((OUT/'submission_identification_only.csv').exists())
    def test_recorded_source_hashes(self):
        p=OUT/'source_hashes.json'
        if not p.exists(): self.skipTest('hashes not generated')
        self.assertIn('experiments/exp4c_calibrated_fusion/evaluate.py',json.load(open(p)))
    def test_final_integrity_record(self):
        p=OUT/'integrity.json'
        if not p.exists(): self.skipTest('integrity not run')
        self.assertTrue(json.load(open(p))['ok'])

if __name__=='__main__':unittest.main()

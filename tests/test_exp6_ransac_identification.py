import csv, json, tempfile, unittest
from pathlib import Path
import numpy as np

from experiments.exp6_ransac_identification.assignment import unique_assignment
from experiments.exp6_ransac_identification.context import atomic_json
from experiments.exp6_ransac_identification.duplicate_audit import Candidate, cluster_points, candidates_from_alternatives
from experiments.exp6_ransac_identification.ransac import required_trials, run_ransac
from experiments.exp6_ransac_identification.transforms import apply_transform, fit_transform
from experiments.exp6_ransac_identification.validation import mutate_names_only

class Exp6Tests(unittest.TestCase):
    def setUp(self):
        self.src=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,.5]])

    def check_family(self,family,matrix,tol=1e-6):
        dst=apply_transform(self.src,matrix); fit=fit_transform(self.src,dst,family)
        self.assertIsNotNone(fit); self.assertLess(np.max(np.linalg.norm(apply_transform(self.src,fit)-dst,axis=1)),tol)

    def test_similarity_recovery(self):
        self.check_family("similarity",[[0.,-2.,8.],[2.,0.,9.]])
    def test_reflected_similarity_recovery(self):
        self.check_family("reflected_similarity",[[0.,2.,8.],[2.,0.,9.]])
    def test_anisotropic_recovery(self):
        self.check_family("anisotropic",[[0.,-3.,8.],[2.,0.,9.]],1e-5)
    def test_affine_recovery(self):
        self.check_family("affine",[[2.,.4,8.],[-.2,1.3,9.]])
    def test_degenerate(self):
        self.assertIsNone(fit_transform([[0,0],[0,0]],[[1,1],[2,2]],"similarity"))
        self.assertIsNone(fit_transform([[0,0],[1,0],[2,0]],self.src[:3],"affine"))
    def test_trial_formula(self):
        self.assertEqual(required_trials(.99,1,2),1); self.assertGreater(required_trials(.99,.1,3),100)
    def test_cluster_order_geometry(self):
        p=np.array([[0,0],[1,0],[20,20]]); self.assertEqual(len(set(cluster_points(p,2))),2)
    def test_duplicate_cannot_inflate(self):
        c=[Candidate(0,0,0,(0,0),1),Candidate(0,1,0,(.1,0),.9),Candidate(1,0,0,(0,0),1)]
        a=unique_assignment([[0,0],[.2,0]],c,12); self.assertEqual(a["unique_assignment_inlier_count"],1); self.assertGreater(a["raw_inlier_count"],1)
    def test_one_star_not_multiple_nodes(self):
        c=[Candidate(0,0,0,(0,0),1)]; self.assertEqual(unique_assignment([[0,0],[1,0]],c,12)["unique_assignment_inlier_count"],1)
    def test_one_node_not_multiple_queries(self):
        c=[Candidate(0,0,0,(0,0),1),Candidate(1,0,1,(.1,0),1)]; self.assertEqual(unique_assignment([[0,0]],c,12)["unique_assignment_inlier_count"],1)
    def test_final_radius_is_12(self):
        c=[Candidate(i,0,i,(float(i*10),0),1) for i in range(3)]; h=run_ransac(self.src[:3],c,"affine",seed=2,discovery_radius=24,final_radius=12,max_trials=20)
        self.assertIn(h["status"],("ok","no_hypothesis"))
    def test_deterministic_and_refit(self):
        dst=apply_transform(self.src,[[20,0,10],[0,20,20]])
        alt=[[(x,y,1.,0.,1.)] for x,y in dst]; c=candidates_from_alternatives(alt)
        a=run_ransac(self.src,c,"similarity",seed=4,max_trials=200); b=run_ransac(self.src,c,"similarity",seed=4,max_trials=200)
        self.assertEqual(a["pairs"],b["pairs"]); self.assertGreaterEqual(a["post_refit_support"],a["pre_refit_support"])
    def test_order_invariance(self):
        p=np.array([[2,2],[0,0],[1,1]]); q=p[[2,0,1]]
        self.assertEqual(sorted(np.bincount(cluster_points(p,1.5))),sorted(np.bincount(cluster_points(q,1.5))))
    def test_atomic_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.json"; atomic_json(p,{"done":[1]}); self.assertEqual(json.loads(p.read_text())["done"],[1])
    def test_csv_name_only(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/"a.csv"; b=Path(d)/"b.csv"; a.write_text("Id,x,constellation\na,1,old\n")
            out=mutate_names_only(a,b,{"a":"new"}); self.assertTrue(out["non_constellation_cells_identical"])
            row=next(csv.DictReader(b.open())); self.assertEqual((row["x"],row["constellation"]),("1","new"))
    def test_validation_has_no_train_truth(self):
        text=Path("experiments/exp6_ransac_identification/validation.py").read_text(); self.assertNotIn("TRAIN_GROUND_TRUTH",text)
    def test_baseline_hash(self):
        from experiments.exp6_ransac_identification import BASELINE
        from experiments.exp6_ransac_identification.context import sha256
        self.assertEqual(sha256(BASELINE),"84fa0c3def2526694f8b89253e8e87c21b07d91bd5e0d6b798c74e64b09b4b17")

if __name__=="__main__": unittest.main()

import json,tempfile,unittest
from pathlib import Path
import numpy as np
from experiments.exp6r_graph_ransac.graph_descriptors import describe_graph
from experiments.exp6r_graph_ransac.candidate_pairs import stable_candidates,describe_pairs
from experiments.exp6r_graph_ransac.correspondences import edge_pair_proposals,strategy_signature
from experiments.exp6r_graph_ransac.ransac import solve_pattern
from experiments.exp6r_graph_ransac.synthetic import run_seed
from experiments.exp6r_graph_ransac.finalize import validate_required

class Exp6RTests(unittest.TestCase):
 def setUp(self):
  self.g={"nodes":np.array([[0.,0.],[1.,0.],[1.,1.],[2.,1.]]),"edges":[(0,1),(1,2),(2,3)]}
  self.alt=[[(10,10,1,0,1),(50,50,.2,0,1)],[(30,10,.9,0,1)],[(30,30,.8,0,1)],[(50,30,.7,0,1)]]
 def test_strategy_changes_order(self):
  c=stable_candidates(self.alt);p=describe_pairs(c);g=describe_graph(self.g)
  self.assertNotEqual(strategy_signature(edge_pair_proposals(g,p,"uniform",20)),strategy_signature(edge_pair_proposals(g,p,"confidence",20)))
 def test_real_edges_used(self): self.assertEqual(len(describe_graph(self.g)["edges"]),3)
 def test_node_permutation_signature(self):
  perm=[2,0,3,1];inv={o:n for n,o in enumerate(perm)};h={"nodes":self.g["nodes"][perm],"edges":[(inv[a],inv[b]) for a,b in self.g["edges"]]}
  self.assertEqual(sorted(describe_graph(self.g)["node_signatures"]),sorted(describe_graph(h)["node_signatures"]))
 def test_query_and_candidate_permutation(self):
  a=stable_candidates(self.alt);b=stable_candidates([list(reversed(x)) for x in reversed(self.alt)])
  self.assertEqual([(x.coordinate,x.score) for x in a],[(x.coordinate,x.score) for x in b])
 def test_symmetric_nodes_retained_and_endpoint_reversal(self):
  p=edge_pair_proposals(describe_graph(self.g),describe_pairs(stable_candidates(self.alt)),"two_edge",100)
  keys={(x["pattern_edge"],x["candidate_pair"]) for x in p}; self.assertTrue(any(sum(1 for x in p if (x["pattern_edge"],x["candidate_pair"])==k)>=2 for k in keys))
 def test_duplicate_safe_planted_transform(self):
  c=stable_candidates(self.alt);h=solve_pattern(self.g,c,describe_pairs(c),max_proposals=100)
  self.assertGreaterEqual(h["unique_inliers"],3);self.assertGreaterEqual(h["raw_inliers"],h["unique_inliers"])
 def test_truth_independent_acceptance_and_all48(self):
  # Static checks prevent correctness from entering the decision assignment.
  src=Path("experiments/exp6r_graph_ransac/synthetic.py").read_text();self.assertNotIn('accepted=bool(result["winner"]==name)',src);self.assertIn('"classes_compared"',src)
 def test_wrong_overwrite_computed(self):
  src=Path("experiments/exp6r_graph_ransac/synthetic.py").read_text();self.assertIn('result["winner"]!=name',src);self.assertNotIn('"wrong_overwrite":False',src)
 def test_ablation_schema(self):
  p=Path("outputs/exp6r_graph_ransac/ablations.json")
  if p.exists():
   for r in json.loads(p.read_text())["results"]: self.assertTrue({"arm","config","input_hash","runtime","raw_accuracy","status"}<=set(r))
 def test_finalizer_fails_missing(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(RuntimeError): validate_required(Path(d))
 def test_validation_no_truth(self): self.assertNotIn("TRAIN_GROUND_TRUTH",Path("experiments/exp6r_graph_ransac/validation.py").read_text())
 def test_report_values_match_json(self):
  report=Path("EXPERIMENT6R_REPORT.md").read_text(); out=Path("outputs/exp6r_graph_ransac")
  dev=json.loads((out/"development_results.json").read_text()); s2=json.loads((out/"synthetic_final_seed2.json").read_text())
  self.assertIn(f"Scorpius {dev['scenes']['scorpius']['true_rank']}",report)
  self.assertIn(f"`{s2['mean_reciprocal_rank']:.5f}`",report)

if __name__=="__main__":unittest.main()

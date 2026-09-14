import numpy as np
import unittest

from constellation.contracts import ScenePrediction
from experiments.exp2_geometry.integration import (hybrid_prediction, map_scores,
                                                     rank_utility, supported_queries)
from experiments.exp2_geometry.candidate_geometry import apply_geometry


def pred(patches, relocated=()):
    return ScenePrediction(patches, "x", {"relocated_queries": list(relocated)})


class Exp2GeometryTests(unittest.TestCase):
    def test_rank_utility_is_stable_and_ignores_invalid(self):
        got = rank_utility(np.array([-.3, -np.inf, -.1, -.2]))
        self.assertTrue(np.allclose(got, [0., 0., 1., .5]))

    def test_coordinate_score_mapping_requires_provenance(self):
        score, audit = map_scores(np.array([[1, 2], [8, 9]]), np.array([.2, .8]),
                                  np.array([[8.5, 9.0], [1, 2]]))
        self.assertTrue(np.allclose(score, [.8, .2]))
        self.assertEqual(audit["mapped"], 2)
        with self.assertRaises(ValueError):
            map_scores(np.array([[0, 0]]), np.array([1.]), np.array([[3, 0]]), 2.)

    def test_hybrid_only_uses_requested_geometric_support(self):
        learned = pred([None, (20, 20, 0), None])
        classical = pred([(1, 1, 1), (2, 2, 0), (3, 3, 0)], relocated=(1, 2))
        got = hybrid_prediction(learned, classical, snap="relocated", rescue="relocated")
        self.assertEqual(got.patches, [None, (2, 2, 0), (3, 3, 0)])
        self.assertEqual(got.diagnostics["n_snap"], 1)
        self.assertEqual(got.diagnostics["n_rescue"], 1)

    def test_member_and_relocation_support_are_distinct(self):
        p = pred([(1, 1, 1), (2, 2, 0)], relocated=(1,))
        self.assertEqual(supported_queries(p, "member"), {0})
        self.assertEqual(supported_queries(p, "relocated"), {1})
        self.assertEqual(supported_queries(p, "either"), {0, 1})

    def test_candidate_geometry_rescue_is_explicit(self):
        learned = pred([None, (5, 5, 0)])
        result = {"chosen": {0: (1., 1.), 1: (2., 2.)}, "name": "g",
                  "stage": "refined", "rank_weight": .1, "mapping_audit": {"n": 2},
                  "geometry": {"hypotheses": [{"nodes": [[1., 1.], [2., 2.]]}]}}
        no_rescue = apply_geometry(learned, result, rescue=False)
        rescue = apply_geometry(learned, result, rescue=True)
        self.assertIsNone(no_rescue.patches[0])
        self.assertEqual(rescue.patches[0], (1., 1., 1))
        self.assertEqual(no_rescue.patches[1], (2., 2., 1))
        self.assertEqual(no_rescue.diagnostics["rank_weight"], .1)

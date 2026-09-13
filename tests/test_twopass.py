"""Tests for geometric pool eligibility and the two-pass recognizer."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.slate import slate_from_candidates, mark_eligible, with_scores
from constellation.joint import build_pool
from constellation.twopass import predicted_nodes, recognize_two_pass


def candidates(*rows):
    return [(float(x), float(y), float(s), 0., 1.) for x, y, s in rows]


class Eligibility(unittest.TestCase):
    def setUp(self):
        # Confident query: gap 0.50, so the appearance gate admits one candidate only.
        self.confident = slate_from_candidates(
            candidates((10, 10, .90), (500, 500, .40), (900, 900, .30)))

    def test_mark_eligible_flags_only_nearby_candidates(self):
        s = mark_eligible(self.confident, np.array([[502., 501.]]), 18.)
        self.assertEqual(s.eligible_indices(), [1])

    def test_no_nodes_marks_nothing(self):
        s = mark_eligible(self.confident, np.empty((0, 2)), 18.)
        self.assertEqual(s.eligible_indices(), [])

    def test_radius_is_respected(self):
        s = mark_eligible(self.confident, np.array([[540., 500.]]), 18.)
        self.assertEqual(s.eligible_indices(), [])

    def test_eligibility_admits_a_candidate_the_gate_would_exclude(self):
        """The appearance gate gives one point; eligibility adds the flagged one."""
        base, _, _, _ = build_pool([self.confident], [0], top_k=8, margin=.15, gap=.03)
        self.assertEqual(len(base), 1)
        s = mark_eligible(self.confident, np.array([[500., 500.]]), 18.)
        pool, _, _, _ = build_pool([s], [0], top_k=8, margin=.15, gap=.03)
        self.assertEqual(len(pool), 2)
        self.assertTrue(any(np.allclose(p, [500, 500]) for p in pool))

    def test_eligibility_never_invents_a_location(self):
        """Only existing candidate coordinates can enter the pool."""
        s = mark_eligible(self.confident, np.array([[123., 456.]]), 1e6)
        pool, _, _, _ = build_pool([s], [0], top_k=8, margin=.15, gap=.03)
        for p in pool:
            self.assertTrue(any(np.allclose(p, c) for c in self.confident.xy))

    def test_eligibility_does_not_change_presence_or_ambiguity(self):
        s = mark_eligible(self.confident, np.array([[500., 500.]]), 18.)
        self.assertEqual(s.presence_score, self.confident.presence_score)
        self.assertEqual(s.ambiguity_gap, self.confident.ambiguity_gap)

    def test_default_slate_has_no_eligibility(self):
        pool_a, _, _, _ = build_pool([self.confident], [0], top_k=8, margin=.15, gap=.03)
        explicit = with_scores(self.confident, eligible=np.zeros(3, bool))
        pool_b, _, _, _ = build_pool([explicit], [0], top_k=8, margin=.15, gap=.03)
        self.assertTrue(np.allclose(pool_a, pool_b))


class PredictedNodes(unittest.TestCase):
    def test_only_verified_hypotheses_contribute(self):
        diag = {'hypotheses': [{'nodes': [[1., 2.]], 'support': 5},
                               {'nodes': [[3., 4.]], 'support': 2}]}
        n = predicted_nodes(diag, 3)
        self.assertEqual(len(n), 1)

    def test_missing_hypotheses_is_safe(self):
        self.assertEqual(len(predicted_nodes({}, 3)), 0)

    def test_limit_is_respected(self):
        diag = {'hypotheses': [{'nodes': [[float(i), 0.]], 'support': 5}
                               for i in range(6)]}
        self.assertEqual(len(predicted_nodes(diag, 2)), 2)


class TwoPass(unittest.TestCase):
    def test_falls_back_when_no_hypothesis_verifies(self):
        lists = [candidates((10. * i, 10., .9)) for i in range(4)]
        name, chosen, diag = recognize_two_pass(lists, {'a': np.zeros((5, 2))})
        self.assertEqual(name, 'unknown')
        self.assertFalse(diag.get('two_pass', {}).get('applied', False))

    def test_reports_whether_the_second_pass_ran(self):
        rng = np.random.default_rng(1)
        pts = rng.uniform(300, 2700, size=(9, 2))
        lists = [candidates((x, y, .9 - .002 * i)) for i, (x, y) in enumerate(pts)]
        patterns = {'alpha': pts[:7] + rng.normal(0, 1.5, (7, 2))}
        _, _, diag = recognize_two_pass(lists, patterns, node_radius=18.)
        self.assertIn('two_pass', diag)
        self.assertIn('applied', diag['two_pass'])


if __name__ == '__main__':
    unittest.main()

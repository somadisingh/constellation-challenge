import unittest

from experiments.exp5d_stability_audit import CANDIDATE_NAME
from experiments.exp5d_stability_audit.run import evaluate, perturb


def alternatives(n=10, depth=5):
    return [[(q * 10.0 + r, q * 7.0 + r, 1.0 - r / 10, 0.0, 1.0)
             for r in range(depth)] for q in range(n)]


class PerturbationTests(unittest.TestCase):
    def test_depth_variants(self):
        source = alternatives()
        self.assertTrue(all(len(row) == 1 for row in perturb(source, 'top1', 1)))
        self.assertTrue(all(len(row) == 3 for row in perturb(source, 'top3', 1)))
        self.assertTrue(all(len(row) == 5 for row in perturb(source, 'full_top5', 1)))

    def test_jitter_is_deterministic(self):
        a = perturb(alternatives(), 'jitter_2px', 17)
        b = perturb(alternatives(), 'jitter_2px', 17)
        self.assertEqual(a, b)
        self.assertNotEqual(a, alternatives())

    def test_query_dropout_keeps_alignment_and_minimum(self):
        dropped = perturb(alternatives(20), 'query_dropout_10pct', 17)
        self.assertEqual(len(dropped), 18)
        self.assertTrue(all(len(row) == 5 for row in dropped))


class DecisionTests(unittest.TestCase):
    @staticmethod
    def _record(family, seed, winner=CANDIDATE_NAME, margin=1.2,
                support=7, held_out_support=4):
        return {'family': family, 'seed': seed, 'winner': winner,
                'margin': margin, 'support': support,
                'held_out_support': held_out_support}

    def test_frozen_rule_passes_stable_candidate(self):
        families = ('full_top5', 'top3', 'top1', 'jitter_2px', 'query_dropout_10pct')
        records = [self._record(family, seed) for family in families for seed in range(3)]
        self.assertTrue(evaluate(records)['passed'])

    def test_frozen_rule_rejects_family_instability(self):
        families = ('full_top5', 'top3', 'top1', 'jitter_2px', 'query_dropout_10pct')
        records = [self._record(family, seed) for family in families for seed in range(3)]
        for row in records:
            if row['family'] == 'top1':
                row['winner'] = 'other'
        result = evaluate(records)
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['each_family_candidate_wins'])


if __name__ == '__main__':
    unittest.main()

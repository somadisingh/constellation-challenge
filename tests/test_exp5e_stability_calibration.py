import unittest

from experiments.exp5e_stability_calibration.run import evaluate


class CalibrationDecisionTests(unittest.TestCase):
    @staticmethod
    def _cases():
        return [
            {'seed_label': 's1', 'pattern': 'a', 'baseline_correct': True},
            {'seed_label': 's1', 'pattern': 'b', 'baseline_correct': True},
            {'seed_label': 's1', 'pattern': 'c', 'baseline_correct': False},
            {'seed_label': 's1', 'pattern': 'd', 'baseline_correct': False},
            {'seed_label': 's1', 'pattern': 'e', 'baseline_correct': False},
            {'seed_label': 's1', 'pattern': 'f', 'baseline_correct': False},
            {'seed_label': 's1', 'pattern': 'g', 'baseline_correct': False},
        ]

    @staticmethod
    def _trials(cases, stable_patterns):
        rows = []
        for case in cases:
            for index in range(4):
                rows.append({
                    'seed_label': case['seed_label'], 'pattern': case['pattern'],
                    'winner_matches_baseline': (
                        case['pattern'] in stable_patterns and index < 3),
                })
        return rows

    def test_gate_passes_precise_stability(self):
        cases = self._cases()
        result = evaluate(cases, self._trials(cases, {'a', 'b'}))
        self.assertTrue(result['passed'])
        self.assertEqual(result['metrics']['stable_precision'], 1.0)

    def test_gate_rejects_stable_incorrect_cases(self):
        cases = self._cases()
        result = evaluate(cases, self._trials(cases, {'a', 'b', 'c', 'd'}))
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['stable_precision'])
        self.assertFalse(result['checks']['stable_incorrect_rate'])

    def test_taurus_fails_exclusion_check(self):
        cases = self._cases() + [
            {'seed_label': 's2', 'pattern': 'taurus', 'baseline_correct': True}]
        result = evaluate(cases, self._trials(cases, {'a', 'b', 'taurus'}))
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['taurus_excluded'])


if __name__ == '__main__':
    unittest.main()

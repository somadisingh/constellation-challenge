import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp5c_affine_recovery import PATTERNS_DIR
from experiments.exp5c_affine_recovery.hashing import compute_quad_descriptors
from experiments.exp5c_affine_recovery.pattern_index import extract_pattern_graph, PatternIndex
from experiments.exp5c_affine_recovery.solver import (
    ARMS, _assign, build_scene_quads, verify_transform, SceneQuads,
)
from experiments.exp5c_affine_recovery.synthetic import atomic_json
from experiments.exp5c_affine_recovery.validation import mutate_names_only
from experiments.exp5c_affine_recovery.leaderboard_attribution import _differences


def alternatives(n=7, depth=5):
    out = []
    for q in range(n):
        out.append([(q * 100. + r * 2, (q * q % 11) * 53. + r * (q + 1),
                     .95 - .02 * r, 0., 1.)
                    for r in range(depth)])
    return out


class HashingTests(unittest.TestCase):
    def test_general_affine_invariance_including_reflection(self):
        p = np.array([[0., 0.], [2., .3], [.4, 1.7], [2.4, 2.1]])
        a = np.array([[-1.4, .55], [.2, .73]])
        q = p @ a.T + [19., -7.]
        _, x = compute_quad_descriptors(p, [[0, 1, 2, 3]])
        _, y = compute_quad_descriptors(q, [[0, 1, 2, 3]])
        np.testing.assert_allclose(x, y, atol=1e-10)

    def test_point_order_stability(self):
        p = np.array([[0., 0.], [2., .3], [.4, 1.7], [2.4, 2.1]])
        _, x = compute_quad_descriptors(p, [[0, 1, 2, 3]])
        _, y = compute_quad_descriptors(p, [[2, 0, 3, 1]])
        np.testing.assert_allclose(x, y, atol=1e-12)


class CandidateTests(unittest.TestCase):
    def test_all_arms_use_four_distinct_queries(self):
        for arm in ARMS:
            q = build_scene_quads(alternatives(), arm, max_quads=200, seed=5)
            for ids in q.ids:
                self.assertEqual(len(set(map(int, q.point_queries[ids]))), 4, arm)

    def test_alternates_are_not_collapsed(self):
        q = build_scene_quads(alternatives(), 'one_alt_top5', 500, 4)
        self.assertTrue(np.any(q.point_ranks[q.ids] > 0))

    def test_proposal_order_deterministic(self):
        a = build_scene_quads(alternatives(10), 'prosac', 400, 91)
        b = build_scene_quads(alternatives(10), 'prosac', 400, 91)
        np.testing.assert_array_equal(a.ids, b.ids)
        np.testing.assert_allclose(a.descriptors, b.descriptors)

    def test_query_order_invariance_for_complete_small_arm(self):
        source = alternatives(7)
        a = build_scene_quads(source, 'rank1', 1000, 3)
        b = build_scene_quads(list(reversed(source)), 'rank1', 1000, 3)
        ax = sorted(map(tuple, np.round(a.descriptors, 12)))
        bx = sorted(map(tuple, np.round(b.descriptors, 12)))
        self.assertEqual(ax, bx)

    def test_stream_budgets_are_separate(self):
        rank1 = build_scene_quads(alternatives(10), 'rank1', 80, 9)
        alternate = build_scene_quads(alternatives(10), 'one_alt_top5', 80, 9)
        self.assertGreater(len(rank1.ids), 0)
        self.assertGreater(len(alternate.ids), 0)
        self.assertTrue(np.any(alternate.point_ranks[alternate.ids] > 0))

    def test_no_query_or_location_reuse_in_assignment(self):
        mapped = np.array([[0., 0.], [0.5, 0.], [10., 0.]])
        pool = np.array([[0., 0.], [.4, 0.], [10., 0.]])
        pairs, _ = _assign(mapped, pool, np.array([1, 1, 2]), np.array([0, 0, 1]), 2.)
        self.assertEqual(len(pairs), 2)


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.index = PatternIndex()

    def test_all_patterns_and_real_edges_extracted(self):
        self.assertEqual(len(self.index.graphs), 48)
        self.assertTrue(any(g['edges'] for g in self.index.graphs.values()))

    def test_graph_is_not_complete(self):
        for g in self.index.graphs.values():
            n = len(g['nodes'])
            if n >= 4 and g['edges']:
                self.assertLess(len(g['edges']), n * (n - 1) // 2)
                return
        self.fail('no eligible graph')

    def test_pattern_order_invariance(self):
        reversed_graphs = dict(reversed(list(self.index.graphs.items())))
        other = PatternIndex(reversed_graphs)
        self.assertEqual(sorted(self.index.usable_classes), sorted(other.usable_classes))
        for name in self.index.usable_classes:
            np.testing.assert_allclose(self.index.template_quads[name][1],
                                       other.template_quads[name][1])


class VerificationTests(unittest.TestCase):
    def test_seed_nodes_excluded_but_observed_fourth_point_counts(self):
        nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 1.]])
        points = nodes * 500 + [800, 700]
        sq = SceneQuads('rank1', points, np.arange(5), np.zeros(5, int),
                        np.ones(5), np.array([[0, 1, 2, 3]], int), np.zeros((1, 4)))
        graph = {'nodes': nodes, 'edges': [(0, 1), (0, 2), (1, 3), (3, 4)]}
        h = verify_transform('x', graph, [0, 1, 2, 3], [0, 1, 2, 3], sq,
                             points, np.arange(5), np.ones(5), np.arange(5), (3000, 3000))
        self.assertIsNotNone(h)
        self.assertEqual(h['held_out_support'], 1)
        self.assertIn((4, 4), h['pairs'])
        self.assertEqual(h['unique_matched_queries'], 5)


class CheckpointAndCsvTests(unittest.TestCase):
    def test_atomic_checkpoint_and_resume_shape(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'state.json'
            atomic_json(p, {'status': 'RUNNING', 'completed': {'a': 1}})
            self.assertEqual(json.loads(p.read_text())['completed'], {'a': 1})
            self.assertEqual(list(Path(d).glob('*.tmp')), [])

    def test_resume_code_skips_completed_patterns(self):
        source = (Path(__file__).parents[1] / 'experiments' / 'exp5c_affine_recovery' /
                  'synthetic.py').read_text()
        self.assertIn("if name in existing['completed']", source)

    def test_csv_name_only_mutation_and_padding(self):
        with tempfile.TemporaryDirectory() as d:
            base, target = Path(d) / 'base.csv', Path(d) / 'out.csv'
            fields = ['Id', 'n_patches', 'patch_01', 'patch_02', 'constellation']
            with base.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
                w.writerow({'Id': 's', 'n_patches': 1, 'patch_01': '(1, 2, 0)',
                            'patch_02': '-1', 'constellation': 'old'})
            result = mutate_names_only(base, target, {'s': 'new'})
            self.assertTrue(result['non_constellation_cells_identical'])
            with target.open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row['patch_02'], '-1'); self.assertEqual(row['constellation'], 'new')

    def test_validation_source_does_not_read_train_truth(self):
        source = (Path(__file__).parents[1] / 'experiments' / 'exp5c_affine_recovery' /
                  'validation.py').read_text()
        self.assertNotIn('train_ground_truth.csv', source)
        self.assertNotIn('TRAIN_GROUND_TRUTH', source)

    def test_taurus_is_excluded_from_validation_and_promotion(self):
        root = Path(__file__).parents[1] / 'experiments' / 'exp5c_affine_recovery'
        development = (root / 'development.py').read_text()
        finalize = (root / 'finalize.py').read_text()
        self.assertIn("'taurus': 'adversarial_diagnostic_only'", development)
        self.assertIn("'excluded_from_validation_and_promotion': ['taurus']", development)
        self.assertNotIn('1_taurus_correct_transform_reaches_budget', finalize)

    def test_leaderboard_recommendation_changes_only_constellation_08(self):
        with tempfile.TemporaryDirectory() as d:
            base, target = Path(d) / 'base.csv', Path(d) / 'target.csv'
            fields = ['Id', 'patch_01', 'constellation']
            with base.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
                w.writerow({'Id': 'constellation_08', 'patch_01': '-1',
                            'constellation': 'eridanus'})
            mutate_names_only(base, target, {'constellation_08': 'orion'})
            self.assertEqual(_differences(base, target), [{
                'scene': 'constellation_08', 'column': 'constellation',
                'before': 'eridanus', 'after': 'orion'}])


if __name__ == '__main__': unittest.main()

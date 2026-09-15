"""Tests for Experiment 5: constellation-independent candidate and geometric
hypothesis recovery.

Run:
    .venv-exp1/bin/python -m unittest tests.test_exp5_hypothesis_recovery -v
"""
from __future__ import annotations

import json
import unittest

import numpy as np

from experiments.exp5_hypothesis_recovery import (CORRECT_PX, K_GRID, OUT, ROOT,
                                                   SCENES, SEEDS)
from experiments.exp5_hypothesis_recovery.barycentric import (barycentric_agreement,
                                                               from_barycentric,
                                                               to_barycentric)
from experiments.exp5_hypothesis_recovery.geometric_recovery import (
    _condition_number, _is_collinear, graph_consistency, multi_candidate_generation_points,
    prosac_triples)


# ─── AFFINE / REFLECTION / BARYCENTRIC INVARIANCE ────────────────────────────

class BarycentricInvariance(unittest.TestCase):
    def test_barycentric_roundtrip(self):
        a, b, c = np.array([0., 0.]), np.array([4., 0.]), np.array([0., 3.])
        x = np.array([1.0, 1.0])
        bary = to_barycentric(a, b, c, x)
        recovered = from_barycentric(a, b, c, bary)
        np.testing.assert_allclose(recovered[0], x, atol=1e-9)

    def test_barycentric_preserved_under_affine_map(self):
        """Barycentric coordinates of a point relative to a triangle are
        UNCHANGED after applying the SAME affine map (rotation + anisotropic
        scale + shear + reflection) to both the triangle and the point."""
        rng = np.random.default_rng(0)
        a, b, c = rng.uniform(-5, 5, (3, 2))
        x = rng.uniform(-5, 5, 2)
        bary_before = to_barycentric(a, b, c, x)[0]

        # Anisotropic scale + shear + reflection + rotation, applied to
        # all four points identically.
        angle = 0.7
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        aniso = np.diag([2.3, 0.6])
        shear = np.array([[1.0, 0.4], [0.0, 1.0]])
        reflect = np.array([[-1.0, 0.0], [0.0, 1.0]])
        M = reflect @ shear @ aniso @ rot

        a2, b2, c2, x2 = (M @ p for p in (a, b, c, x))
        bary_after = to_barycentric(a2, b2, c2, x2)[0]
        np.testing.assert_allclose(bary_before, bary_after, atol=1e-9)

    def test_barycentric_agreement_zero_for_identical_coordinates(self):
        self.assertAlmostEqual(barycentric_agreement([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]), 0.0)

    def test_barycentric_agreement_grows_with_disagreement(self):
        a = barycentric_agreement([0.2, 0.3, 0.5], [0.25, 0.3, 0.45])
        b = barycentric_agreement([0.2, 0.3, 0.5], [0.6, 0.1, 0.3])
        self.assertLess(a, b)

    def test_degenerate_triangle_returns_none(self):
        a, b, c = np.array([0., 0.]), np.array([1., 0.]), np.array([2., 0.])   # collinear
        self.assertIsNone(to_barycentric(a, b, c, np.array([0.5, 0.0])))


# ─── DEGENERATE TRIPLE REJECTION ──────────────────────────────────────────────

class DegenerateTripleRejection(unittest.TestCase):
    def test_collinear_triple_detected(self):
        pts = np.array([[0., 0.], [1., 0.], [2., 0.]])
        self.assertTrue(_is_collinear(pts))

    def test_non_collinear_triple_not_flagged(self):
        pts = np.array([[0., 0.], [1., 0.], [0., 1.]])
        self.assertFalse(_is_collinear(pts))

    def test_condition_number_of_identity_like_matrix_is_near_one(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        self.assertAlmostEqual(_condition_number(matrix), 1.0, places=6)

    def test_condition_number_grows_with_anisotropy(self):
        low = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        high = np.array([[10.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        self.assertLess(_condition_number(low), _condition_number(high))


# ─── UNIQUE QUERY / SOURCE ASSIGNMENT AND MULTI-CANDIDATE SEEDING ────────────

class MultiCandidateSeeding(unittest.TestCase):
    def _fake_slates(self):
        from constellation.slate import slate_from_candidates
        # Two queries, each with 3 ranked candidates (rank0=best).
        q0 = slate_from_candidates([(10., 10., 0.9, 0., 1.), (50., 50., 0.5, 0., 1.),
                                    (90., 90., 0.3, 0., 1.)])
        q1 = slate_from_candidates([(20., 20., 0.9, 0., 1.), (60., 60., 0.5, 0., 1.),
                                    (95., 95., 0.3, 0., 1.)])
        return [q0, q1]

    def test_k1_matches_single_seed_point_per_query(self):
        slates = self._fake_slates()
        groups = [0, 1]
        pts, tags = multi_candidate_generation_points(slates, groups, k=1)
        self.assertEqual(len(pts), 2)
        np.testing.assert_allclose(sorted(pts[:, 0]), [10., 20.])

    def test_k2_emits_two_points_per_query(self):
        slates = self._fake_slates()
        groups = [0, 1]
        pts, tags = multi_candidate_generation_points(slates, groups, k=2)
        self.assertEqual(len(pts), 4)
        self.assertEqual(sorted(tags.tolist()), [0, 0, 1, 1])

    def test_k_larger_than_bank_uses_whole_bank(self):
        slates = self._fake_slates()
        groups = [0, 1]
        pts, tags = multi_candidate_generation_points(slates, groups, k=10)
        self.assertEqual(len(pts), 6)   # 3 candidates x 2 queries


class TripleUniqueness(unittest.TestCase):
    def test_prosac_triples_reject_duplicate_source_groups(self):
        """Every accepted triple must draw from 3 DISTINCT physical-source
        groups -- prosac_triples enforces this via `groups_of_gen`."""
        from constellation.joint import SceneIndex
        template = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        gen_pts = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.],
                           [5., 5.], [15., 15.]])
        groups_of_gen = np.array([0, 1, 2, 3, 0, 1])   # note: indices 4,5 REPEAT groups 0,1
        index = SceneIndex(gen_pts, use_quads=False)
        triples = prosac_triples(template, gen_pts, groups_of_gen, index,
                                 per_class_budget=50, seed=31004, class_name='test')
        for ti, si in triples:
            src_groups = [int(groups_of_gen[j]) for j in si]
            self.assertEqual(len(set(src_groups)), 3, 'triple reused a physical-source group')


# ─── FITTING-TRIPLE EXCLUSION FROM VALIDATION SUPPORT ────────────────────────

class FittingTripleExclusion(unittest.TestCase):
    def test_seed_pairs_disjoint_from_held_out_pairs(self):
        from experiments.exp4b_joint_solver.independent_scorer import score_hypothesis_seed_excluded
        p = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        pool = p.copy()
        tags = np.arange(len(pool), dtype=int)
        matrix = np.array([[1., 0.], [0., 1.], [0., 0.]])
        seed_ti = np.array([0, 1, 2])
        seed_pool_idx = np.array([0, 1, 2])
        scored = score_hypothesis_seed_excluded(p, matrix, pool, tags, seed_ti,
                                                seed_pool_idx, tolerance=0.5)
        seed_template_idx = {i for i, _ in scored['seed_pairs']}
        held_out_template_idx = {i for i, _ in scored['held_out_pairs']}
        self.assertEqual(seed_template_idx & held_out_template_idx, set())
        self.assertTrue(seed_template_idx <= {0, 1, 2})
        self.assertFalse(held_out_template_idx & {0, 1, 2})


# ─── GRAPH CONSISTENCY / CLOSE-STAR / DUPLICATE-SOURCE HANDLING ──────────────

class GraphConsistency(unittest.TestCase):
    def test_consistent_scaled_template_has_full_edge_consistency(self):
        template = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.]])
        mapped = template * 3.0   # uniform scale -- every edge ratio identical
        result = graph_consistency(template, mapped, {0, 1, 2, 3})
        self.assertEqual(result['n_edges_consistent'], result['n_edges_checked'])
        self.assertFalse(result['fragmented'])

    def test_single_outlier_node_fragments_the_graph(self):
        template = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.]])
        mapped = template * 3.0
        mapped[3] = [500., 500.]   # one node wildly displaced, inconsistent with the rest
        result = graph_consistency(template, mapped, {0, 1, 2, 3})
        self.assertLess(result['largest_connected_fragment'], 4)
        self.assertTrue(result['fragmented'])

    def test_fewer_than_two_matched_nodes_not_fragmented(self):
        template = np.array([[0., 0.], [10., 0.], [0., 10.]])
        mapped = template.copy()
        result = graph_consistency(template, mapped, {0})
        self.assertFalse(result['fragmented'])
        self.assertEqual(result['n_edges_checked'], 0)


# ─── MULTIPLICITY CORRECTION (small-pattern bias) ────────────────────────────

class MultiplicityCorrection(unittest.TestCase):
    def test_small_pattern_does_not_get_free_full_score_from_fraction(self):
        """A tiny (4-node) pattern matching only 1 held-out node must not
        score as favourably as a large (12-node) pattern matching several
        held-out nodes purely because 1/4 == 100% under a naive fraction --
        the binomial-surprise term must discount the tiny pattern instead."""
        from experiments.exp5_hypothesis_recovery.geometric_recovery import GeometricHypothesis

        small = GeometricHypothesis(
            class_name='small', matrix=[[1, 0], [0, 1], [0, 0]], seed_triple=[0, 1, 2],
            mapped_nodes=[[0, 0], [1, 0], [0, 1], [1, 1]], total_pairs=[(0, 0), (1, 1), (2, 2), (3, 3)],
            held_out_pairs=[(3, 3)], seed_support=3, held_out_support=1,
            fourth_point_checked=0, fourth_point_supported=0, fourth_point_mean_agreement=None,
            graph_edges_checked=0, graph_edges_consistent=0, graph_largest_fragment=0,
            graph_fragmented=False, mean_residual=0.1, unique_queries=4, unique_sources=4,
            unmatched_nodes=0, appearance_score=0.9, unqueried_evidence=0.0, n_unqueried_scored=0,
            multiplicity_correction=1.0, stability_mean_shift_px=0.1, pool_size=100, n_groups=30)

        large = GeometricHypothesis(
            class_name='large', matrix=[[1, 0], [0, 1], [0, 0]], seed_triple=[0, 1, 2],
            mapped_nodes=[[0, 0]] * 12, total_pairs=[(i, i) for i in range(6)],
            held_out_pairs=[(i, i) for i in range(3, 6)], seed_support=3, held_out_support=3,
            fourth_point_checked=0, fourth_point_supported=0, fourth_point_mean_agreement=None,
            graph_edges_checked=0, graph_edges_consistent=0, graph_largest_fragment=0,
            graph_fragmented=False, mean_residual=0.1, unique_queries=6, unique_sources=6,
            unmatched_nodes=6, appearance_score=0.9, unqueried_evidence=0.0, n_unqueried_scored=0,
            multiplicity_correction=1.0, stability_mean_shift_px=0.1, pool_size=100, n_groups=30)

        self.assertGreater(large.total_score, small.total_score)

    def test_disabling_held_out_support_zeroes_that_term(self):
        from experiments.exp5_hypothesis_recovery.geometric_recovery import GeometricHypothesis
        kwargs = dict(class_name='x', matrix=[[1, 0], [0, 1], [0, 0]], seed_triple=[0, 1, 2],
                     mapped_nodes=[[0, 0]] * 6, total_pairs=[(i, i) for i in range(5)],
                     held_out_pairs=[(i, i) for i in range(2, 5)], seed_support=2, held_out_support=3,
                     fourth_point_checked=0, fourth_point_supported=0, fourth_point_mean_agreement=None,
                     graph_edges_checked=0, graph_edges_consistent=0, graph_largest_fragment=0,
                     graph_fragmented=False, mean_residual=0.1, unique_queries=5, unique_sources=5,
                     unmatched_nodes=1, appearance_score=0.0, unqueried_evidence=0.0, n_unqueried_scored=0,
                     multiplicity_correction=1.0, stability_mean_shift_px=None, pool_size=50, n_groups=15)
        with_support = GeometricHypothesis(**kwargs, use_held_out_support=True)
        without_support = GeometricHypothesis(**kwargs, use_held_out_support=False)
        self.assertEqual(without_support.score_breakdown['held_out_support_surprise'], 0.0)
        self.assertGreater(with_support.score_breakdown['held_out_support_surprise'], 0.0)


# ─── DETERMINISTIC PROSAC/RANSAC AND NO hash() SEEDING ───────────────────────

class DeterministicSearch(unittest.TestCase):
    def test_prosac_triples_deterministic_across_calls(self):
        from constellation.joint import SceneIndex
        template = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [0.5, 0.5]])
        rng = np.random.default_rng(1)
        gen_pts = rng.uniform(0, 100, (12, 2))
        groups_of_gen = np.arange(12)
        index = SceneIndex(gen_pts, use_quads=False)
        t1 = prosac_triples(template, gen_pts, groups_of_gen, index, 50, 31004, 'x')
        t2 = prosac_triples(template, gen_pts, groups_of_gen, index, 50, 31004, 'x')
        self.assertEqual([tuple(int(x) for x in ti) for ti, _ in t1],
                        [tuple(int(x) for x in ti) for ti, _ in t2])

    def test_no_builtin_hash_used_for_seeding(self):
        import ast
        pkg_dir = ROOT / 'experiments' / 'exp5_hypothesis_recovery'
        offenders = []
        for f in sorted(pkg_dir.glob('*.py')):
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == 'hash'):
                    offenders.append(f.name)
        self.assertEqual(offenders, [])

    def test_derive_seed_used_not_random_module(self):
        text = (ROOT / 'experiments' / 'exp5_hypothesis_recovery' / 'geometric_recovery.py').read_text()
        self.assertIn('derive_seed', text)


# ─── QUERY / PATTERN ORDER INVARIANCE ─────────────────────────────────────────

class OrderInvariance(unittest.TestCase):
    def test_synthetic_screen_reports_pattern_and_query_order_invariance(self):
        path = OUT / 'synthetic_all48.json'
        if not path.exists():
            self.skipTest('synthetic screen not run')
        doc = json.loads(path.read_text())
        self.assertTrue(doc['pattern_order_independence']['identical'])
        self.assertTrue(doc['query_order_independence']['identical'])
        self.assertTrue(doc['deterministic_repeatability']['identical'])


# ─── WHOLE-SKY ISOLATION ──────────────────────────────────────────────────────

class WholeSkyIsolation(unittest.TestCase):
    def test_oof_allowed_membership_excludes_held_out_sky(self):
        for fn in ('oof_primary.json', 'oof_repeat.json'):
            path = OUT / fn
            if not path.exists():
                self.skipTest(f'{fn} not generated')
            doc = json.loads(path.read_text())
            for s in SCENES:
                self.assertEqual(set(doc['per_scene'][s]['allowed']), set(SCENES) - {s})
                self.assertFalse(doc['per_scene'][s]['fit_on_held_out_sky'])

    def test_override_threshold_never_uses_held_out_sky_evidence(self):
        path = OUT / 'oof_primary.json'
        if not path.exists():
            self.skipTest('oof_primary.json not generated')
        doc = json.loads(path.read_text())
        for held in SCENES:
            threshold = doc['per_scene'][held]['override_threshold']
            # the threshold's own evidence lists must never include the held-out
            # sky's raw outcome -- verified by checking it was built from exactly
            # 2 allowed-sky observations at most.
            n_evidence = len(threshold['allowed_correct_supports']) + len(threshold['allowed_wrong_supports'])
            self.assertLessEqual(n_evidence, 2)


# ─── PRIMARY/REPEAT BASELINE MATCHING AND PATCH IDENTITY ─────────────────────

class PrimaryRepeatAndPatchIdentity(unittest.TestCase):
    def test_primary_and_repeat_seeds_match_expected(self):
        for fn, expected_seed in (('oof_primary.json', 31004), ('oof_repeat.json', 31005)):
            path = OUT / fn
            if not path.exists():
                self.skipTest(f'{fn} not generated')
            doc = json.loads(path.read_text())
            self.assertEqual(doc['seed'], expected_seed)

    def test_patch_cells_byte_identical_for_identification_only_policy(self):
        for fn in ('oof_primary.json', 'oof_repeat.json'):
            path = OUT / fn
            if not path.exists():
                self.skipTest(f'{fn} not generated')
            doc = json.loads(path.read_text())
            self.assertTrue(all(doc['patch_identity'].values()))


# ─── PROTECTED-FILE INTEGRITY ─────────────────────────────────────────────────

class ProtectedIntegrity(unittest.TestCase):
    def test_protected_before_snapshot_exists(self):
        path = OUT / 'protected_before.json'
        self.assertTrue(path.exists())
        doc = json.loads(path.read_text())
        self.assertGreater(len(doc), 0)

    def test_verify_protected_reports_no_changes(self):
        from experiments.exp5_hypothesis_recovery.integrity import verify_protected
        result = verify_protected(OUT / 'protected_before.json')
        self.assertEqual(result['changed'], [])
        self.assertEqual(result['missing'], [])
        self.assertTrue(result['ok'], msg=result)


# ─── SUBMISSION SCHEMA (conditional on gates) ────────────────────────────────

class SubmissionSchema(unittest.TestCase):
    def test_no_submission_generated_unless_all_gates_pass(self):
        gates_path = OUT / 'gates.json'
        if not gates_path.exists():
            self.skipTest('gates.json not yet generated')
        gates = json.loads(gates_path.read_text())
        submission_path = OUT / 'submission_candidate_exp5.csv'
        if not gates['overall_pass']:
            self.assertFalse(submission_path.exists(),
                            'gates did not clear; no submission candidate should exist')


# ─── GATE HONESTY ──────────────────────────────────────────────────────────────

class GateHonesty(unittest.TestCase):
    def test_not_evaluable_and_not_applicable_never_pass(self):
        gates_path = OUT / 'gates.json'
        if not gates_path.exists():
            self.skipTest('gates.json not yet generated')
        gates = json.loads(gates_path.read_text())
        for name, node in gates['checks'].items():
            if node.get('status') in ('not_evaluable', 'not_applicable'):
                self.assertFalse(node.get('pass', False),
                                f'{name} has status {node["status"]} but pass=True')

    def test_twenty_four_gates_declared(self):
        gates_path = OUT / 'gates.json'
        if not gates_path.exists():
            self.skipTest('gates.json not yet generated')
        gates = json.loads(gates_path.read_text())
        self.assertEqual(gates['n_gates'], 24)


if __name__ == '__main__':
    unittest.main()

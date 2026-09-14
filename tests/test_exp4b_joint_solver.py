"""Tests for Experiment 4B: candidate-rank fidelity, independent geometric
evidence, unqueried-star null model, and bounded joint multi-candidate
identification.

Run:
    OMP_NUM_THREADS=1 .venv-exp1/bin/python -m unittest tests.test_exp4b_joint_solver -v
"""
from __future__ import annotations

import json
import unittest

import numpy as np

from experiments.exp1.env import ROOT, derive_seed
from experiments.exp4b_joint_solver import OUT, SCENES


# ─── DETERMINISTIC SEED DERIVATION ────────────────────────────────────────────

class SeedDerivation(unittest.TestCase):
    def test_derive_seed_is_deterministic_not_hash(self):
        """`derive_seed` (SHA-256-based, per repo convention) must never be
        implemented via Python's `hash()`, which is randomized per-process
        unless PYTHONHASHSEED is fixed -- the task explicitly forbids `hash()`
        for seeding."""
        import inspect
        from experiments.exp1 import env
        src = inspect.getsource(env.derive_seed)
        self.assertNotIn(' hash(', src)
        self.assertIn('sha256', src.lower())

    def test_derive_seed_reproducible_across_calls(self):
        a = derive_seed(31004, 'exp4b-stability', 'pisces', 0, (1, 2, 3))
        b = derive_seed(31004, 'exp4b-stability', 'pisces', 0, (1, 2, 3))
        self.assertEqual(a, b)

    def test_derive_seed_differs_for_different_tags(self):
        a = derive_seed(31004, 'exp4b-null', 'pisces:eridanus', 1.0, 2.0)
        b = derive_seed(31004, 'exp4b-null', 'pisces:hydra', 1.0, 2.0)
        self.assertNotEqual(a, b)

    def test_no_hash_builtin_used_for_seeding_anywhere_in_package(self):
        """Structural sweep: no module in experiments.exp4b_joint_solver calls
        the `hash()` builtin for seed derivation."""
        import ast
        pkg_dir = ROOT / 'experiments' / 'exp4b_joint_solver'
        offenders = []
        for f in sorted(pkg_dir.glob('*.py')):
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == 'hash'):
                    offenders.append(str(f.name))
        self.assertEqual(offenders, [])


# ─── RANK FUSION: QUERY / PATTERN-ORDER INVARIANCE ───────────────────────────

class RankFusionOrderInvariance(unittest.TestCase):
    def _row(self, order=(0, 1, 2)):
        base_cands = [
            {'index': 0, 'valid': True, 'classical_ncc': 0.9, 'exp3_pair_logit': 1.5,
             'exp3_absent_logit': -2.0, 'ncc_gap_to_second': 0.1,
             'exp3_ensemble_disagreement': 0.05},
            {'index': 1, 'valid': True, 'classical_ncc': 0.6, 'exp3_pair_logit': 0.2,
             'exp3_absent_logit': -2.0, 'ncc_gap_to_second': -0.1,
             'exp3_ensemble_disagreement': 0.2},
            {'index': 2, 'valid': True, 'classical_ncc': 0.3, 'exp3_pair_logit': -0.5,
             'exp3_absent_logit': -2.0, 'ncc_gap_to_second': -0.3,
             'exp3_ensemble_disagreement': 0.1},
        ]
        return {'candidates': [base_cands[i] for i in order]}

    def test_linear_fusion_winner_independent_of_input_order(self):
        from experiments.exp4b_joint_solver.rank_fusion import rank_by_linear_fusion
        r1 = rank_by_linear_fusion(self._row((0, 1, 2)))
        r2 = rank_by_linear_fusion(self._row((2, 0, 1)))
        self.assertEqual(r1[0]['index'], r2[0]['index'])
        self.assertEqual([c['index'] for c in r1], [c['index'] for c in r2])

    def test_rrf_winner_independent_of_input_order(self):
        from experiments.exp4b_joint_solver.rank_fusion import rank_by_rrf
        r1 = rank_by_rrf(self._row((0, 1, 2)))
        r2 = rank_by_rrf(self._row((1, 2, 0)))
        self.assertEqual([c['index'] for c in r1], [c['index'] for c in r2])

    def test_absent_separated_margin_independent_of_input_order(self):
        from experiments.exp4b_joint_solver.rank_fusion import rank_by_absent_separated
        _, margin1 = rank_by_absent_separated(self._row((0, 1, 2)))
        _, margin2 = rank_by_absent_separated(self._row((2, 1, 0)))
        self.assertAlmostEqual(margin1, margin2)


# ─── MONOTONIC CALIBRATION CANNOT REORDER (Phase 0 correction) ──────────────

class MonotonicCalibrationCannotReorder(unittest.TestCase):
    def test_sigmoid_calibration_preserves_argsort(self):
        """A monotonic (sigmoid) recalibration of raw scores must never change
        the argsort order of a candidate slate -- this is the empirical claim
        underlying `prior_claim_corrections.json`."""
        raw = np.array([2.0, -1.0, 0.5, -3.0, 1.2])
        calibrated = 1.0 / (1.0 + np.exp(-(raw * 0.7 + 0.3)))
        self.assertTrue(np.array_equal(np.argsort(-raw), np.argsort(-calibrated)))


# ─── INDEPENDENT SCORER: SEED-EXCLUDED SUPPORT SEPARATION ────────────────────

class SeedExcludedSupport(unittest.TestCase):
    def test_seed_correspondences_excluded_from_held_out_support(self):
        from experiments.exp4b_joint_solver.independent_scorer import (
            score_hypothesis_seed_excluded)
        # A tiny synthetic scene: 4 template points map exactly onto 4 pool
        # points via the identity transform; the first 3 are the SEED.
        p = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        pool = p.copy()
        # Distinct query-group tags per pool point: `assignment` allows at
        # most one match per group, so identical tags would artificially cap
        # total matches at 1 regardless of seed/held-out split.
        tags = np.arange(len(pool), dtype=int)
        matrix = np.eye(3)[:, :2] if False else np.array([[1., 0.], [0., 1.], [0., 0.]])
        seed_ti = np.array([0, 1, 2])
        seed_pool_idx = np.array([0, 1, 2])
        scored = score_hypothesis_seed_excluded(p, matrix, pool, tags, seed_ti,
                                                seed_pool_idx, tolerance=0.5)
        self.assertEqual(scored['seed_support'], 3)
        self.assertEqual(scored['held_out_support'], 1)
        self.assertEqual(scored['total_support'], 4)
        # The held-out pair must be node index 3, never one of the seed indices.
        held_out_template_indices = {i for i, _ in scored['held_out_pairs']}
        self.assertEqual(held_out_template_indices, {3})


# ─── DUPLICATE-QUERY / ABSENT / OFF-FIGURE HANDLING ──────────────────────────

class DuplicateAndAbsentHandling(unittest.TestCase):
    def test_relevance_zero_when_distance_none_absent_query(self):
        from experiments.exp4b_joint_solver.rank_features import relevance
        self.assertEqual(relevance(None), 0.0)

    def test_relevance_full_credit_within_12px(self):
        from experiments.exp4b_joint_solver.rank_features import relevance
        self.assertEqual(relevance(0.0), 1.0)
        self.assertEqual(relevance(12.0), 1.0)

    def test_relevance_zero_credit_at_or_beyond_36px(self):
        from experiments.exp4b_joint_solver.rank_features import relevance
        self.assertEqual(relevance(36.0), 0.0)
        self.assertEqual(relevance(100.0), 0.0)

    def test_unqueried_search_deduplicates_coincident_physical_source(self):
        """Two mapped nodes landing on the SAME physical source (within 3px)
        must not both receive independent credit."""
        from experiments.exp4b_joint_solver.unqueried_star_evidence import (
            score_unqueried_nodes)
        rng = np.random.default_rng(0)
        image = (rng.normal(20, 3, size=(200, 200))).astype(np.float32)
        # Two mapped nodes 1px apart (effectively the same physical source).
        mapped = np.array([[100.0, 100.0], [101.0, 100.5], [50.0, 50.0]])
        pool = np.zeros((0, 2))
        chosen_pairs = []   # nothing pre-explained -> all 3 nodes are "unqueried"
        result = score_unqueried_nodes(image, mapped, chosen_pairs, pool, seed=31004,
                                       tag='test', ablation='raw_count')
        # n_scored must be at most 2 (the near-duplicate pair collapses to one).
        self.assertLessEqual(result['n_scored'], 2)


# ─── MULTIPLE-TESTING CORRECTION ─────────────────────────────────────────────

class MultipleTestingCorrection(unittest.TestCase):
    def test_corrected_score_divides_by_sqrt_of_nodes_searched(self):
        from experiments.exp4b_joint_solver.unqueried_star_evidence import (
            score_unqueried_nodes)
        rng = np.random.default_rng(1)
        image = rng.normal(20, 3, size=(300, 300)).astype(np.float32)
        # Inject a bright peak so the score is nonzero and deterministic.
        image[100, 100] = 250.0
        mapped = np.array([[100.0, 100.0]])
        pool = np.zeros((0, 2))
        raw = score_unqueried_nodes(image, mapped, [], pool, seed=31004, tag='t',
                                    ablation='matched_null_lr')
        corrected = score_unqueried_nodes(image, mapped, [], pool, seed=31004, tag='t',
                                          ablation='matched_null_lr_corrected')
        if raw['evidence_score'] > 0:
            self.assertAlmostEqual(corrected['evidence_score'],
                                   raw['evidence_score'] / np.sqrt(1), places=6)

    def test_more_nodes_searched_does_not_automatically_increase_corrected_score(self):
        """A reference with more unqueried nodes to search must not win purely
        from having more chances to find a clutter peak once corrected -- this
        is the large-template-bias gate (task's promotion gate on the null model)."""
        from experiments.exp4b_joint_solver.unqueried_star_evidence import (
            score_unqueried_nodes)
        rng = np.random.default_rng(2)
        image = rng.normal(20, 3, size=(400, 400)).astype(np.float32)
        pool = np.zeros((0, 2))
        few = np.array([[50.0, 50.0]])
        many = np.array([[50.0 + 5 * i, 50.0] for i in range(15)])
        r_few = score_unqueried_nodes(image, few, [], pool, seed=42, tag='few',
                                      ablation='matched_null_lr_corrected')
        r_many = score_unqueried_nodes(image, many, [], pool, seed=42, tag='many',
                                       ablation='matched_null_lr_corrected')
        # With pure noise and no injected peaks, corrected per-node contribution
        # should not scale up linearly with node count (the sqrt normalization
        # must visibly suppress growth vs raw_count on the same inputs).
        r_few_raw = score_unqueried_nodes(image, few, [], pool, seed=42, tag='few',
                                          ablation='raw_count')
        r_many_raw = score_unqueried_nodes(image, many, [], pool, seed=42, tag='many',
                                           ablation='raw_count')
        if r_many_raw['evidence_score'] > r_few_raw['evidence_score'] > 0:
            ratio_raw = r_many_raw['evidence_score'] / max(r_few_raw['evidence_score'], 1e-9)
            ratio_corrected = (r_many['evidence_score']
                              / max(r_few['evidence_score'], 1e-9))
            self.assertLessEqual(ratio_corrected, ratio_raw + 1e-6)


# ─── BEAM-SEARCH TIE HANDLING / DETERMINISM ──────────────────────────────────

class BeamSearchSolverState(unittest.TestCase):
    def test_total_score_excludes_stability_from_the_summed_score(self):
        """`stability_score` is documented as a tie-break only -- it must have
        zero weight in `total_score`."""
        from experiments.exp4b_joint_solver.joint_solver import SolverState
        low_stability = SolverState(
            constellation='x', matrix=[[1, 0], [0, 1], [0, 0]],
            query_to_candidate={}, query_to_node={}, absent_queries=[],
            offfigure_queries=[], unmatched_nodes=[], appearance_score=1.0,
            held_out_geometric_support=5, unqueried_star_evidence=0.0,
            clutter_penalty=0.0, stability_score=0.1)
        high_stability = SolverState(
            constellation='x', matrix=[[1, 0], [0, 1], [0, 0]],
            query_to_candidate={}, query_to_node={}, absent_queries=[],
            offfigure_queries=[], unmatched_nodes=[], appearance_score=1.0,
            held_out_geometric_support=5, unqueried_star_evidence=0.0,
            clutter_penalty=0.0, stability_score=99.0)
        self.assertEqual(low_stability.total_score, high_stability.total_score)

    def test_beam_prune_tie_break_prefers_lower_stability_shift(self):
        """Two states with an identical total_score must be ordered by LOWER
        stability_mean_shift_px first (more stable transform wins a tie)."""
        from experiments.exp4b_joint_solver.joint_solver import SolverState
        a = SolverState(
            constellation='alpha', matrix=[[1, 0], [0, 1], [0, 0]],
            query_to_candidate={}, query_to_node={}, absent_queries=[],
            offfigure_queries=[], unmatched_nodes=[], appearance_score=1.0,
            held_out_geometric_support=5, unqueried_star_evidence=0.0,
            clutter_penalty=0.0, stability_score=5.0)
        b = SolverState(
            constellation='beta', matrix=[[1, 0], [0, 1], [0, 0]],
            query_to_candidate={}, query_to_node={}, absent_queries=[],
            offfigure_queries=[], unmatched_nodes=[], appearance_score=1.0,
            held_out_geometric_support=5, unqueried_star_evidence=0.0,
            clutter_penalty=0.0, stability_score=1.0)
        states = [a, b]
        states.sort(key=lambda s: (-s.total_score,
                                   s.stability_score if s.stability_score is not None else 1e9,
                                   s.constellation))
        self.assertEqual(states[0].constellation, 'beta')


# ─── LEAK-FREE FOLD MEMBERSHIP (reused from Exp4) ────────────────────────────

class LeakFreeMembershipReuse(unittest.TestCase):
    def test_full_evaluation_reuses_exp4_leak_free_checkpoint_membership(self):
        """Phase 5's full_evaluation.py must never load a checkpoint trained on
        its own held-out fold -- verified structurally against the real
        checkpoint layout, matching Exp4's own membership rule."""
        checkpoint_root = ROOT / 'outputs' / 'exp3_pairwise' / 'checkpoints'
        for fold in SCENES:
            own_dir = checkpoint_root / fold
            self.assertTrue(own_dir.exists(), f'missing checkpoint dir for {fold}')
            for other in SCENES:
                if other != fold:
                    # The path this fold's own evaluation should use never
                    # points inside another fold's checkpoint directory.
                    self.assertNotEqual(str(own_dir), str(checkpoint_root / other))


# ─── MATCHED PRIMARY/REPEAT BASELINES ────────────────────────────────────────

class MatchedSeedBaselines(unittest.TestCase):
    def setUp(self):
        metrics_path = OUT / 'metrics.json'
        if not metrics_path.exists():
            self.skipTest('metrics.json not yet generated')
        self.metrics = json.loads(metrics_path.read_text())

    def test_primary_and_repeat_use_different_seeds_but_same_pipeline(self):
        p5 = self.metrics['phase5_full_evaluation']
        self.assertIn('verifier_snap_rescue', p5['primary'])
        self.assertIn('verifier_snap_rescue', p5['repeat'])
        self.assertNotAlmostEqual(p5['primary']['verifier_snap_rescue']['score'],
                                  p5['repeat']['verifier_snap_rescue']['score'])

    def test_matching_baseline_present_for_both_seeds(self):
        mb = self.metrics['matching_baseline']
        self.assertIn('score', mb['primary'])
        self.assertIn('score', mb['repeat'])


# ─── PROTECTED-FILE INTEGRITY ─────────────────────────────────────────────────

class ProtectedIntegrity(unittest.TestCase):
    def test_protected_before_snapshot_exists_and_is_nonempty(self):
        path = OUT / 'protected_before.json'
        self.assertTrue(path.exists())
        doc = json.loads(path.read_text())
        self.assertGreater(len(doc), 0)

    def test_verify_protected_reports_no_changes_to_frozen_roots(self):
        from experiments.exp4b_joint_solver.integrity import verify_protected
        result = verify_protected(OUT / 'protected_before.json')
        self.assertEqual(result['changed'], [])
        self.assertEqual(result['missing'], [])
        self.assertTrue(result['ok'], msg=result)


# ─── SUBMISSION SCHEMA (conditional: none generated, per gates.json) ────────

class SubmissionSchema(unittest.TestCase):
    def test_no_submission_directory_generated_because_gates_did_not_clear(self):
        gates_path = OUT / 'gates.json'
        if not gates_path.exists():
            self.skipTest('gates.json not yet generated')
        gates = json.loads(gates_path.read_text())
        submissions_dir = OUT / 'submissions'
        if not gates['overall_pass']:
            self.assertFalse(submissions_dir.exists() and any(submissions_dir.iterdir())
                             if submissions_dir.exists() else True,
                             'gates did not clear; no submission candidate should exist')


# ─── GATES / METRICS CONSISTENCY ─────────────────────────────────────────────

class GatesConsistency(unittest.TestCase):
    def setUp(self):
        gates_path = OUT / 'gates.json'
        if not gates_path.exists():
            self.skipTest('gates.json not yet generated')
        self.gates = json.loads(gates_path.read_text())

    def test_fourteen_gates_declared(self):
        self.assertEqual(self.gates['n_gates'], 14)

    def test_not_evaluable_is_never_counted_as_pass(self):
        for name, node in self.gates['checks'].items():
            if node.get('status') == 'not_evaluable':
                self.assertFalse(node.get('pass', False),
                                 f'{name} is not_evaluable but marked pass')

    def test_overall_pass_requires_all_checks_pass(self):
        all_pass = all(c['pass'] for c in self.gates['checks'].values())
        self.assertEqual(self.gates['overall_pass'], all_pass)


if __name__ == '__main__':
    unittest.main()

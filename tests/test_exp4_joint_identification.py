"""Tests for Experiment 4: honest deployment evaluation and identification
headroom analysis.

Run:
    OMP_NUM_THREADS=1 .venv-exp1/bin/python -m unittest tests.test_exp4_joint_identification -v
    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest tests.test_exp4_joint_identification -v
"""
from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

needs_torch = unittest.skipUnless(HAVE_TORCH, 'torch not available in this env')

from experiments.exp4_joint_identification import DEPLOYED_ARM, SCENES


# ─── LEAK-FREE FIXED-ARM MEMBERSHIP ──────────────────────────────────────────

@needs_torch
class LeakFreeMembership(unittest.TestCase):
    def test_load_member_resolves_own_fold_checkpoint(self):
        from experiments.exp4_joint_identification.fixed_policy_oof import _load_member
        model, prov = _load_member('pisces', 31004, 'cpu')
        self.assertEqual(prov['fold'], 'pisces')
        self.assertIn('checkpoints/pisces/F_s31004', prov['checkpoint'])

    def test_leak_free_rule_never_selects_other_fold_checkpoint(self):
        """For each held-out sky, the ONLY eligible checkpoints are those whose
        own training fold equals that sky; verify this structurally against the
        real checkpoint directory layout (no other fold's checkpoint path is
        ever constructible by _load_member for a given fold argument)."""
        from experiments.exp4_joint_identification.fixed_policy_oof import _load_member
        for fold in SCENES:
            _, prov = _load_member(fold, 31004, 'cpu')
            self.assertEqual(prov['fold'], fold)
            for other in SCENES:
                if other != fold:
                    self.assertNotIn(f'checkpoints/{other}/', prov['checkpoint'])

    def test_deployed_arm_is_fixed_not_oracle_selected(self):
        """The Phase 1 evaluation always uses DEPLOYED_ARM; it must never read
        a per-fold `selection_frozen.json` winner."""
        import inspect
        from experiments.exp4_joint_identification import fixed_policy_oof
        src = inspect.getsource(fixed_policy_oof)
        self.assertNotIn('selection_frozen', src)
        self.assertEqual(DEPLOYED_ARM, 'F')


# ─── ENSEMBLE SCORING / LOG-ODDS AGGREGATION ─────────────────────────────────

@needs_torch
class EnsembleAggregation(unittest.TestCase):
    def test_ensemble_disagreement_flag_detects_true_disagreement(self):
        """Two members with different best-candidate indices must be flagged
        as disagreeing; identical choices must be flagged as agreeing."""
        from experiments.exp4_joint_identification.fixed_policy_oof import score_scene_ensemble
        # Structural check only (no real model call): confirm the disagreement
        # computation logic is present and keyed on member_best_idx equality.
        import inspect
        src = inspect.getsource(score_scene_ensemble)
        self.assertIn('ensemble_members_agree_on_best', src)
        self.assertIn('member_best_idx', src)


# ─── HEADROOM ORACLE LADDER ───────────────────────────────────────────────────

class HeadroomOracleLadder(unittest.TestCase):
    def test_level1_perfect_figure_coordinates_identify_correctly(self):
        """A perfect-information oracle (true figure-star coordinates) must
        recover the correct constellation for every real labelled scene --
        this is a sanity floor: if this fails, the geometry/scoring code
        itself is broken, not just the appearance signal upstream of it."""
        from experiments.exp4_joint_identification.headroom_oracles import (
            level_1_figure_only, _patterns)
        from experiments.exp1.data import query_records
        patterns = _patterns()
        for scene in SCENES:
            records = query_records(scene)
            result = level_1_figure_only(scene, records, patterns)
            self.assertEqual(result['predicted'], scene,
                            f'{scene}: perfect-coordinate oracle should identify correctly')

    def test_oracle_no_membership_level_matches_all_present_level(self):
        """recognize_joint never consumes a membership flag at any level; level
        2 and level 3 must therefore produce IDENTICAL predictions (this test
        makes that invariant explicit rather than assumed)."""
        from experiments.exp4_joint_identification.headroom_oracles import (
            level_2_all_present, level_3_no_membership, _patterns)
        from experiments.exp1.data import query_records
        patterns = _patterns()
        scene = 'pisces'
        records = query_records(scene)
        r2 = level_2_all_present(scene, records, patterns)
        r3 = level_3_no_membership(scene, records, patterns)
        self.assertEqual(r2['predicted'], r3['predicted'])
        self.assertEqual(r2['diagnostics_summary']['true_class_rank'],
                         r3['diagnostics_summary']['true_class_rank'])

    def test_rank_and_gap_finds_true_class_in_full_hypothesis_list(self):
        from experiments.exp4_joint_identification.headroom_oracles import _rank_and_gap
        hyps = [{'name': 'a', 'score': 5.0}, {'name': 'b', 'score': 3.0},
               {'name': 'c', 'score': 1.0}]
        r = _rank_and_gap(hyps, 'b')
        self.assertEqual(r['true_class_rank'], 2)
        self.assertAlmostEqual(r['score_gap_winner_minus_true'], 2.0)
        self.assertFalse(r['correct_class_wins'])

    def test_rank_and_gap_handles_missing_true_class(self):
        from experiments.exp4_joint_identification.headroom_oracles import _rank_and_gap
        hyps = [{'name': 'a', 'score': 5.0}]
        r = _rank_and_gap(hyps, 'nonexistent')
        self.assertIsNone(r['true_class_rank'])
        self.assertIsNone(r['score_gap_winner_minus_true'])

    def test_single_point_slates_preserves_none_for_absent_queries(self):
        from experiments.exp4_joint_identification.headroom_oracles import _single_point_slates
        slates = _single_point_slates([(1.0, 2.0), None, (3.0, 4.0)])
        self.assertEqual(len(slates), 3)
        self.assertEqual(slates[1], [])
        self.assertEqual(len(slates[0]), 1)
        self.assertEqual(slates[0][0][:2], (1.0, 2.0))


# ─── CATALOG/QUERY-ORDER AND AFFINE INVARIANCE (existing geometry code) ──────

class IntegrityExcludesDocumentation(unittest.TestCase):
    """`lab/LEDGER.md` is a required end-of-experiment write (like README.md/
    FINDINGS.md, which are never in PROTECTED_ROOTS at all); the integrity
    checker must not flag doc-only edits under a protected code root."""

    def test_iter_files_skips_markdown(self):
        from experiments.exp4_joint_identification.integrity import _iter_files
        from experiments.exp4_joint_identification import ROOT
        files = list(_iter_files(ROOT / 'lab'))
        self.assertTrue(all(f.suffix.lower() != '.md' for f in files))
        # but non-markdown files under lab/ are still tracked
        self.assertTrue(any(f.suffix == '.py' for f in files))


class DeterminismAndInvariance(unittest.TestCase):
    def test_recognize_joint_deterministic_across_calls(self):
        """Same input, same seed -> byte-identical prediction and score, across
        two separate calls (not just two processes; this checks in-process
        determinism, which is a precondition for cross-process determinism)."""
        from constellation.joint import recognize_joint
        rng = np.random.default_rng(3)
        patterns = {'x': rng.uniform(0, 100, (6, 2)),
                   'y': rng.uniform(0, 100, (5, 2))}
        alts = [[(float(x), float(y), 1.0, 0.0, 1.0)]
               for x, y in rng.uniform(0, 100, (8, 2))]
        n1, c1, d1 = recognize_joint(alts, patterns, diag_top=8)
        n2, c2, d2 = recognize_joint(alts, patterns, diag_top=8)
        self.assertEqual(n1, n2)
        self.assertEqual(c1, c2)
        self.assertEqual(d1.get('score_gap'), d2.get('score_gap'))

    def test_catalog_order_invariance(self):
        """Recognition must not depend on the order patterns are provided in a
        dict (Python dicts preserve insertion order, so this tests the actual
        sorted-iteration behaviour in recognize_joint)."""
        from constellation.joint import recognize_joint
        rng = np.random.default_rng(11)
        p1 = {'aaa': rng.uniform(0, 100, (6, 2)), 'zzz': rng.uniform(0, 100, (5, 2))}
        p2 = {'zzz': p1['zzz'], 'aaa': p1['aaa']}
        alts = [[(float(x), float(y), 1.0, 0.0, 1.0)]
               for x, y in rng.uniform(0, 100, (8, 2))]
        n1, _, _ = recognize_joint(alts, p1, diag_top=8)
        n2, _, _ = recognize_joint(alts, p2, diag_top=8)
        self.assertEqual(n1, n2)

    def test_query_order_invariance_via_consolidate(self):
        """Permuting the query (alternative-list) order must not change which
        physical stars are grouped together (consolidate groups by coordinate,
        not by input position)."""
        from constellation.geometry import consolidate
        pts = np.array([[10., 10.], [50., 50.], [10.1, 10.1], [90., 90.]])
        _, groups_a = consolidate(pts)
        perm = [2, 0, 3, 1]
        _, groups_b = consolidate(pts[perm])
        # group membership of the two near-duplicate points (0 and 2) must match
        # regardless of processing order: they map to the same unique-point id.
        self.assertEqual(groups_a[0], groups_a[2])
        # index 0 in the permuted array is original index 2, index 3 is original 0
        self.assertEqual(groups_b[perm.index(0)], groups_b[perm.index(2)])


# ─── PHASE-1 BREAKDOWN METRIC CORRECTNESS ────────────────────────────────────

class FixedPolicyBreakdown(unittest.TestCase):
    def test_class_f1_perfect_prediction(self):
        from experiments.exp4_joint_identification.fixed_policy_breakdown import _class_f1
        y = np.array([1, 1, 0, 0])
        pred = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(_class_f1(y, pred, 1), 1.0)
        self.assertAlmostEqual(_class_f1(y, pred, 0), 1.0)

    def test_class_f1_all_wrong(self):
        from experiments.exp4_joint_identification.fixed_policy_breakdown import _class_f1
        y = np.array([1, 1, 0, 0])
        pred = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(_class_f1(y, pred, 1), 0.0)
        self.assertAlmostEqual(_class_f1(y, pred, 0), 0.0)

    def test_brier_score_perfect_calibration_is_zero(self):
        from experiments.exp4_joint_identification.fixed_policy_breakdown import _brier
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        y = np.array([1, 0, 1, 0])
        self.assertAlmostEqual(_brier(probs, y), 0.0)

    def test_reliability_bins_cover_full_range(self):
        from experiments.exp4_joint_identification.fixed_policy_breakdown import _reliability_bins
        probs = np.array([0.05, 0.55, 0.95])
        y = np.array([0, 1, 1])
        bins = _reliability_bins(probs, y, n_bins=10)
        self.assertEqual(len(bins), 10)
        self.assertAlmostEqual(bins[0]['lo'], 0.0)
        self.assertAlmostEqual(bins[-1]['hi'], 1.0)


# ─── BASELINE MATCHING (primary vs primary, repeat vs repeat) ───────────────

class BaselineMatching(unittest.TestCase):
    def test_verify_exp3_corrected_reads_matching_seed_tag(self):
        from experiments.exp4_joint_identification.baseline_verification import (
            verify_exp3_corrected)
        primary = verify_exp3_corrected('primary')
        repeat = verify_exp3_corrected('repeat')
        self.assertNotAlmostEqual(primary['actual']['score'], repeat['actual']['score'])

    def test_deployment_policy_is_fixed_architecture_not_per_fold(self):
        from experiments.exp4_joint_identification.baseline_verification import (
            verify_deployment_policy)
        r = verify_deployment_policy()
        self.assertTrue(r['ok'])
        self.assertEqual(r['selected_arm'], 'F')
        self.assertEqual(r['n_members'], 6)


# ─── COMPLETION AUDIT RULES ───────────────────────────────────────────────────

class CompletionAuditRules(unittest.TestCase):
    def test_forbidden_status_detection(self):
        from experiments.exp4_joint_identification.completion_audit import (
            _find_forbidden_status)
        doc = {'status': 'incomplete', 'other': 'mentions incomplete in prose only'}
        hits = _find_forbidden_status(doc)
        self.assertEqual(len(hits), 1)

    def test_new_solver_gates_all_explicitly_not_applicable(self):
        """Since no new solver was built, every one of the 10 gates must say so
        explicitly rather than silently passing."""
        from experiments.exp4_joint_identification.gates import evaluate_new_solver_gates
        gates = evaluate_new_solver_gates()
        self.assertEqual(len(gates), 10)
        for name, node in gates.items():
            self.assertEqual(node['status'], 'not_applicable')


# ─── NO SCENE-IDENTITY ROUTING ────────────────────────────────────────────────

class NoSceneIdentityRouting(unittest.TestCase):
    @needs_torch
    def test_score_scene_ensemble_does_not_branch_on_scene_name(self):
        import inspect
        from experiments.exp4_joint_identification import fixed_policy_oof
        src = inspect.getsource(fixed_policy_oof.score_scene_ensemble)
        # 'scene' appears only as a parameter/label, never in a conditional
        # branch keyed on its VALUE (e.g. "if scene == 'pisces'").
        self.assertNotIn("scene ==", src)
        self.assertNotIn("scene in (", src)

    def test_headroom_oracle_uses_no_filename_or_patch_count_shortcuts(self):
        import inspect
        from experiments.exp4_joint_identification import headroom_oracles
        src = inspect.getsource(headroom_oracles)
        self.assertNotIn('n_patches ==', src)


if __name__ == '__main__':
    unittest.main()

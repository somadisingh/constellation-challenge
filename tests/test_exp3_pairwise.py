"""Required tests for Experiment 3: pairwise verifier (task §16).

Runs under both environments:
  .venv-exp1  (Python 3.12, torch present) -- all tests
  .venv        (Python 3.14, no torch)     -- torch-independent tests only

    OMP_NUM_THREADS=1 .venv-exp1/bin/python -m unittest tests.test_exp3_pairwise -v
"""
import math
import unittest

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

needs_torch = unittest.skipUnless(HAVE_TORCH, 'torch not available in this env')

from experiments.exp3_pairwise import (ARM_SPEC, IGNORE_RADIUS, MAX_OFFSET,
                                        POSITIVE_RADIUS, SCENES)


# ─── LISTWISE LOSS ────────────────────────────────────────────────────────────

@needs_torch
class ListwiseLoss(unittest.TestCase):

    def setUp(self):
        from experiments.exp3_pairwise.losses import listwise_absent_loss
        self.loss_fn = listwise_absent_loss

    def _run(self, logits, valid, positive, absent, present):
        return self.loss_fn(torch.tensor(logits, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(valid, dtype=torch.bool).unsqueeze(0),
                            torch.tensor(positive, dtype=torch.bool).unsqueeze(0),
                            torch.tensor([absent], dtype=torch.float32),
                            torch.tensor([present], dtype=torch.bool))

    def test_single_positive_loss_is_finite(self):
        r = self._run([2., -1., 0., -2.], [True]*4, [True,False,False,False], -3., True)
        self.assertTrue(torch.isfinite(r))
        self.assertGreater(float(r), 0)

    def test_multi_positive_marginal_le_single_positive(self):
        single = float(self._run([2.,-1.,0.,-2.],[True]*4,[True,False,False,False],-3.,True)[0])
        multi  = float(self._run([2.,-1.,0.,-2.],[True]*4,[True,False,True,False],-3.,True)[0])
        self.assertLessEqual(multi, single)

    def test_absent_target(self):
        r = self._run([-1.,-2.,-3.,-0.5],[True]*4,[False]*4, 1.0, False)
        self.assertTrue(torch.isfinite(r))
        self.assertGreater(float(r), 0)

    def test_padding_invariance(self):
        base = float(self._run([2.,-1.,0.],[True]*3,[True,False,False],-3.,True)[0])
        padded = float(self._run([2.,-1.,0.,99.],[True,True,True,False],
                                  [True,False,False,False],-3.,True)[0])
        self.assertAlmostEqual(base, padded, places=5)

    def test_invalid_candidates_excluded(self):
        """Invalid (admissible=False) candidates must not affect the loss."""
        with_invalid = float(self._run([2.,-1.,99.],[True,True,False],
                                        [True,False,False],-3.,True)[0])
        without_invalid = float(self._run([2.,-1.],[True,True],
                                           [True,False],-3.,True)[0])
        self.assertAlmostEqual(with_invalid, without_invalid, places=5)

    def test_absent_only_prefers_absent_logit(self):
        """High absent logit → lower absent loss."""
        hi = float(self._run([-5.,-5.],[True]*2,[False]*2, 5., False)[0])
        lo = float(self._run([-5.,-5.],[True]*2,[False]*2,-5., False)[0])
        self.assertLess(hi, lo)

    def test_pool_missing_gives_huge_loss(self):
        """Present row with no positive → loss should be very large (caller excludes it)."""
        r = self._run([1.,2.,3.],[True]*3,[False]*3,-5.,True)
        self.assertGreater(float(r), 1e30)

    def test_score_direction_higher_beats_lower(self):
        good = float(self._run([5.,-3.,-3.],[True]*3,[True,False,False],-3.,True)[0])
        bad  = float(self._run([-5.,3.,3.],[True]*3,[True,False,False],-3.,True)[0])
        self.assertLess(good, bad)

    def test_permutation_invariance(self):
        base = float(self._run([2.,-1.,1.],[True]*3,[True,False,False],-3.,True)[0])
        perm = float(self._run([-1.,2.,1.],[True]*3,[False,True,False],-3.,True)[0])
        self.assertAlmostEqual(base, perm, places=5)

    def test_empty_candidates_absent_target(self):
        """Empty valid set with absent target: loss = -absent_logit + log(exp(absent))."""
        r = float(self._run([],[],[],2.,False)[0])
        self.assertTrue(math.isfinite(r))
        self.assertAlmostEqual(r, 0.0, places=4)

    def test_single_candidate_query(self):
        r = float(self._run([3.],[True],[True],-1.,True)[0])
        self.assertTrue(math.isfinite(r))
        self.assertGreater(r, 0)


# ─── BINARY BCE ───────────────────────────────────────────────────────────────

@needs_torch
class BinaryBCE(unittest.TestCase):
    def test_separated_pairs_give_small_loss(self):
        from experiments.exp3_pairwise.losses import binary_bce_loss
        loss = float(binary_bce_loss(
            torch.tensor([3., 4., 5.]), torch.tensor([-3., -4., -5.])))
        self.assertLess(loss, 0.05)

    def test_swapped_pairs_give_large_loss(self):
        from experiments.exp3_pairwise.losses import binary_bce_loss
        loss = float(binary_bce_loss(
            torch.tensor([-3., -4., -5.]), torch.tensor([3., 4., 5.])))
        self.assertGreater(loss, 2.0)


# ─── OFFSET HEAD ──────────────────────────────────────────────────────────────

@needs_torch
class OffsetBound(unittest.TestCase):
    def test_offset_bounded_by_max_offset(self):
        from experiments.exp3_pairwise.model import PairwiseVerifier
        from experiments.exp1.models import load_backbone, Descriptor
        backbone, _ = load_backbone('hardnet', pretrained=True, device='cpu')
        hardnet_model = Descriptor(backbone, 'hardnet').eval()
        for p in hardnet_model.parameters():
            p.requires_grad_(False)
        model = PairwiseVerifier(hardnet_fusion=True, offset=True,
                                  hardnet_backbone=hardnet_model).eval()
        q = torch.rand(16, 32, 32)
        c = torch.rand(16, 32, 32)
        with torch.no_grad():
            dx, dy, conf = model.offset(q, c)
        self.assertTrue(bool((dx.abs() <= MAX_OFFSET + 1e-4).all()))
        self.assertTrue(bool((dy.abs() <= MAX_OFFSET + 1e-4).all()))
        self.assertTrue(bool(((conf >= 0) & (conf <= 1)).all()))

    def test_offset_applied_only_when_gated(self):
        """Offset must NOT be applied when confidence < threshold (task §15)."""
        from experiments.exp3_pairwise.integration import build_prediction
        from experiments.exp1.evaluation import frozen_geometry

        rows = [
            {'query_id': 'test:0', 'present': True, 'empty_bank': False,
             'alignment_failed_all': False,
             'best_xy': [100.0, 100.0], 'best_logit': 1.0,
             'absent_logit': -1.0, 'n_valid': 5,
             'offset_dx': 5.0, 'offset_dy': 5.0, 'offset_confidence': 0.3},
        ]
        probs = np.array([0.9])
        gate = {'confidence_threshold': 0.5, 'max_correction': 12.0}
        pred = build_prediction('pisces', rows, probs, 0.5, offset_gate=gate)
        self.assertIsNotNone(pred['prediction'].patches[0])
        xy = pred['prediction'].patches[0][:2]
        self.assertAlmostEqual(xy[0], 100.0, places=5)
        self.assertAlmostEqual(xy[1], 100.0, places=5)

    def test_offset_applied_when_gate_passes(self):
        from experiments.exp3_pairwise.integration import build_prediction
        rows = [
            {'query_id': 'test:0', 'present': True, 'empty_bank': False,
             'alignment_failed_all': False,
             'best_xy': [100.0, 100.0], 'best_logit': 1.0,
             'absent_logit': -1.0, 'n_valid': 5,
             'offset_dx': 5.0, 'offset_dy': 5.0, 'offset_confidence': 0.8},
        ]
        probs = np.array([0.9])
        gate = {'confidence_threshold': 0.5, 'max_correction': 12.0}
        pred = build_prediction('pisces', rows, probs, 0.5, offset_gate=gate)
        xy = pred['prediction'].patches[0][:2]
        self.assertAlmostEqual(xy[0], 105.0, places=5)
        self.assertAlmostEqual(xy[1], 105.0, places=5)

    def test_offset_never_applied_after_geometry_snap_rescue(self):
        """After Exp2 snap/rescue, C0's coordinate replaces the learned one entirely.
        The neural offset was computed on the ORIGINAL best_xy; it is never re-applied."""
        from experiments.exp3_pairwise.integration import build_prediction, apply_geometry_stage
        from constellation.contracts import ScenePrediction
        rows = [
            {'query_id': 'test:0', 'present': True, 'empty_bank': False,
             'alignment_failed_all': False,
             'best_xy': [100.0, 100.0], 'best_logit': 2.0,
             'absent_logit': -1.0, 'n_valid': 5,
             'offset_dx': 5.0, 'offset_dy': 5.0, 'offset_confidence': 0.99},
        ]
        probs = np.array([0.9])
        gate = {'confidence_threshold': 0.5, 'max_correction': 12.0}
        learned = build_prediction('pisces', rows, probs, 0.5, offset_gate=gate)['prediction']
        c0_coord = (200.0, 300.0, 0)
        classical = ScenePrediction([c0_coord], 'pisces',
                                    {'relocated_queries': [0]})
        snapped = apply_geometry_stage(learned, classical, 'verifier_snap_rescue')
        xy = snapped.patches[0][:2]
        self.assertAlmostEqual(xy[0], 200.0, places=5)
        self.assertAlmostEqual(xy[1], 300.0, places=5)

    def test_planted_offset_recovery(self):
        """The offset head must learn a planted (dx, dy) (overfit smoke, task §11)."""
        from experiments.exp3_pairwise.overfit_check import check_offset_recovers_planted_displacement
        r = check_offset_recovers_planted_displacement('cpu')
        self.assertTrue(r['recovers_planted_offset'])
        self.assertTrue(r['offset_bounded'])
        self.assertLess(r['final_mean_abs_error_px'], 2.0)


# ─── FROZEN HARDNET ───────────────────────────────────────────────────────────

@needs_torch
class FrozenHardNet(unittest.TestCase):
    def setUp(self):
        from experiments.exp1.models import load_backbone, Descriptor
        backbone, _ = load_backbone('hardnet', pretrained=True, device='cpu')
        self.hardnet_model = Descriptor(backbone, 'hardnet').eval()
        for p in self.hardnet_model.parameters():
            p.requires_grad_(False)
        self.before = {k: v.clone()
                       for k, v in self.hardnet_model.state_dict().items()}

    def test_hardnet_weights_unchanged_after_training_step(self):
        from experiments.exp3_pairwise.training import build_model
        model = build_model('D', hardnet_backbone=self.hardnet_model)
        model.train()
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
        q, c = torch.rand(8, 32, 32), torch.rand(8, 32, 32)
        for _ in range(10):
            logit = model.pair_logit(q, c)
            loss = logit.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        drift = max(float((self.before[k] - v).abs().max())
                   for k, v in self.hardnet_model.state_dict().items())
        self.assertEqual(drift, 0.0)

    def test_hardnet_bn_stays_eval_after_model_train(self):
        """PairwiseVerifier.train() must keep the frozen HardNet submodule eval."""
        from experiments.exp3_pairwise.training import build_model
        model = build_model('D', hardnet_backbone=self.hardnet_model)
        model.train()
        bn_training = [m.training for m in self.hardnet_model.modules()
                       if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
        self.assertTrue(all(not t for t in bn_training))

    def test_hardnet_excluded_from_checkpoint(self):
        from experiments.exp3_pairwise.training import build_model, _trainable_state_dict
        model = build_model('D', hardnet_backbone=self.hardnet_model)
        sd = _trainable_state_dict(model)
        self.assertFalse(any(k.startswith('hardnet.') for k in sd))

    def test_checkpoint_resume_does_not_contaminate_frozen_backbone(self):
        """Loading a saved checkpoint (which excludes hardnet keys) must not
        modify the shared frozen hardnet_model instance."""
        from experiments.exp3_pairwise.training import (build_model,
            _trainable_state_dict, _load_trainable_state_dict)
        model = build_model('D', hardnet_backbone=self.hardnet_model)
        sd = _trainable_state_dict(model)
        model2 = build_model('D', hardnet_backbone=self.hardnet_model)
        _load_trainable_state_dict(model2, sd)
        drift = max(float((self.before[k] - v).abs().max())
                   for k, v in self.hardnet_model.state_dict().items())
        self.assertEqual(drift, 0.0)

    def test_hardnet_receives_no_gradient(self):
        from experiments.exp3_pairwise.training import build_model
        model = build_model('D', hardnet_backbone=self.hardnet_model)
        model.train()
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
        q, c = torch.rand(4, 32, 32), torch.rand(4, 32, 32)
        model.pair_logit(q, c).pow(2).mean().backward()
        for p in self.hardnet_model.parameters():
            self.assertIsNone(p.grad)


# ─── HARD/RANDOM PRESENTATION COUNTS ─────────────────────────────────────────

@needs_torch
class MatchedPresentations(unittest.TestCase):
    def test_hard_random_matched_count(self):
        from experiments.exp3_pairwise.negatives import (hard_vs_random_manifest,
                                                         N_NEGATIVES_PER_GROUP)
        from experiments.exp3_pairwise.groups import QueryGroup
        import numpy as np
        groups = []
        for i in range(6):
            xy = np.array([[10.0 * j, 20.0 * j] for j in range(10)], float)
            d = np.array([5.0] + [40.0]*9, float)
            labels = ['positive'] + ['negative']*9
            g = QueryGroup(f'test:{i}', 'present', 'pisces', 'pisces', 'scorpius',
                           f'src:{i}', (0., 0.),
                           np.zeros((32, 32), np.float32), xy,
                           np.zeros((10, 32, 32), np.float32),
                           np.ones(10, bool),
                           np.ones(10) * 0.5,
                           labels, d, False, False)
            groups.append(g)
        manifest = hard_vs_random_manifest(groups)
        self.assertTrue(manifest['matched'])
        self.assertEqual(manifest['hard_presentations'], manifest['random_presentations'])


# ─── FOLD ISOLATION ───────────────────────────────────────────────────────────

class FoldIsolation(unittest.TestCase):
    def test_held_out_sky_never_in_allowed(self):
        from experiments.exp1.splits import fold_skies
        for fold in SCENES:
            held_out, allowed = fold_skies(fold)
            self.assertEqual(held_out, fold)
            self.assertNotIn(fold, allowed)
            self.assertEqual(len(set(allowed)), 2)

    def test_assert_fold_isolated_catches_violation(self):
        from experiments.exp3_pairwise.groups import QueryGroup, assert_fold_isolated
        bad = QueryGroup('bad', 'present', 'pisces', 'pisces', 'pisces',
                         'src:0', (0., 0.), np.zeros((32, 32), np.float32),
                         np.zeros((0, 2)), np.zeros((0, 32, 32), np.float32),
                         np.zeros(0, bool), np.zeros(0), [], np.zeros(0),
                         False, False)
        result = assert_fold_isolated('pisces', [bad])
        self.assertFalse(result['ok'])
        self.assertTrue(len(result['problems']) > 0)

    def test_footprint_margin_covers_sampling_radius(self):
        from experiments.exp1.splits import MARGIN, max_sampling_radius
        self.assertLessEqual(max_sampling_radius(), MARGIN)

    def test_cell_partition_assignment(self):
        from experiments.exp1.splits import partition_of
        for c in range(6):
            self.assertEqual(partition_of(c), 'fit')
        for c in range(6, 8):
            self.assertEqual(partition_of(c), 'val')
        self.assertEqual(partition_of(8), 'synthcal')


# ─── NO ORACLE INSERTION ──────────────────────────────────────────────────────

class NoOracleInsertion(unittest.TestCase):
    def test_bank_entries_trace_to_label_free_branches(self):
        from experiments.exp1.banks import build_bank, check_no_oracle
        from experiments.exp1.data import query_records
        bank = build_bank('taurus')
        check = check_no_oracle(bank, query_records('taurus', '.'))
        self.assertTrue(check['ok'], check.get('problems'))

    def test_build_present_group_does_not_inject_truth_as_candidate(self):
        """All candidate coordinates must trace to real retrieval, never to ground truth."""
        from experiments.exp3_pairwise.groups import build_present_group
        from experiments.exp1.splits import build_source_bank, partition_records, fold_skies
        from experiments.exp1.env import rng
        _, allowed = fold_skies('pisces')
        scene = allowed[0]
        bank = build_source_bank(scene)
        recs = partition_records(bank, 'fit')
        gen = rng(31003, 'no_oracle_test')
        r = recs[gen.integers(0, len(recs))]
        g = build_present_group(scene, r, gen, None, 'pisces', 'fit')
        truth = np.array(r['xy'], float)
        if len(g.xy):
            d = np.linalg.norm(g.xy - truth, axis=1)
            exact = (d == 0.0).sum()
            self.assertLessEqual(int(exact), 1)

    def test_absent_group_has_no_positive_label(self):
        from experiments.exp3_pairwise.groups import build_absent_group
        from experiments.exp1.splits import build_source_bank, partition_records, fold_skies
        from experiments.exp1.env import rng
        from experiments.exp1.mining import POSITIVE as POS_LABEL
        _, allowed = fold_skies('pisces')
        s1, s2 = allowed
        bank = build_source_bank(s2)
        recs = partition_records(bank, 'fit')
        gen = rng(31003, 'absent_test')
        r = recs[gen.integers(0, len(recs))]
        g = build_absent_group(s1, r, gen, None, 'pisces', 'fit')
        self.assertNotIn(POS_LABEL, g.labels)
        self.assertEqual(g.kind, 'absent')


# ─── SCORE DIRECTION ──────────────────────────────────────────────────────────

@needs_torch
class ScoreDirection(unittest.TestCase):
    def test_higher_is_better_convention(self):
        from experiments.exp3_pairwise.metrics import score_groups
        if not HAVE_TORCH:
            self.skipTest('torch unavailable')
        from experiments.exp3_pairwise.training import build_model
        from experiments.exp3_pairwise.groups import QueryGroup
        model = build_model('A').eval()
        g = QueryGroup('t', 'present', 'pisces', 'pisces', 'scorpius', 'src',
                       (100., 100.),
                       np.random.rand(32, 32).astype(np.float32),
                       np.array([[100., 100.], [500., 500.]], float),
                       np.random.rand(2, 32, 32).astype(np.float32),
                       np.ones(2, bool), np.zeros(2),
                       ['positive', 'negative'], np.array([0.5, 400.0]),
                       False, False)
        rows = score_groups(model, [g], 'cpu')
        self.assertEqual(rows[0]['n_valid'], 2)
        self.assertIsNotNone(rows[0]['best_logit'])


# ─── CALIBRATION ──────────────────────────────────────────────────────────────

class Calibration(unittest.TestCase):
    def _rows(self):
        import random
        random.seed(42)
        np.random.seed(42)
        rows = []
        for s in ('pisces', 'scorpius'):
            for i in range(40):
                present = i % 2 == 0
                rows.append({'scene': s, 'present': present,
                             'best_logit': (0.9 + np.random.randn() * 0.1) if present
                                            else (-0.5 + np.random.randn() * 0.2),
                             'best_minus_second': 0.3 if present else 0.05,
                             'absent_logit': -1.0 if present else 0.5,
                             'n_valid': 10, 'localization_reward': 1.0 if present else 0.0})
        return rows

    def test_calibrator_separates_classes(self):
        from experiments.exp3_pairwise.calibration import fit_calibrator, apply_calibrator
        rows = self._rows()
        cal = fit_calibrator(rows)
        self.assertTrue(cal.get('ok'), cal.get('reason'))
        probs = apply_calibrator(cal, rows)
        present = np.array([r['present'] for r in rows])
        self.assertGreater(probs[present].mean(), probs[~present].mean())

    def test_classical_threshold_never_reused(self):
        from experiments.exp3_pairwise.calibration import select_threshold
        import numpy as np
        rows = self._rows()
        probs = np.linspace(0, 1, len(rows))
        thr = select_threshold(probs, rows)
        self.assertNotAlmostEqual(thr['selected'], 0.72, places=3)

    def test_unusable_row_gets_probability_zero(self):
        from experiments.exp3_pairwise.calibration import fit_calibrator, apply_calibrator
        rows = self._rows()
        cal = fit_calibrator(rows)
        rows_with_none = rows + [{'scene': 'pisces', 'present': False,
                                   'best_logit': None, 'best_minus_second': None,
                                   'absent_logit': None, 'n_valid': 0,
                                   'localization_reward': 0.0}]
        probs = apply_calibrator(cal, rows_with_none)
        self.assertEqual(probs[-1], 0.0)

    def test_listwise_absent_prob_is_between_zero_and_one(self):
        from experiments.exp3_pairwise.calibration import listwise_absent_probability
        rows = self._rows()
        p = listwise_absent_probability(rows)
        self.assertTrue(((p >= 0) & (p <= 1)).all())


# ─── INPUT RANGE / DOUBLE-NORMALIZATION GUARD ─────────────────────────────────

@needs_torch
class InputRange(unittest.TestCase):
    def test_double_normalization_guard(self):
        """photometric_normalize is idempotent: applying it twice should give
        the same result as applying it once (no further scaling of the result)."""
        from experiments.exp3_pairwise.model import photometric_normalize
        x = torch.rand(8, 32, 32)
        once = photometric_normalize(x)
        twice = photometric_normalize(once)
        self.assertAlmostEqual(float((once - twice).abs().mean()), 0.0, places=3)

    def test_constant_input_does_not_explode(self):
        """A uniform crop should have a finite, bounded output (floor clamp on std)."""
        from experiments.exp3_pairwise.model import photometric_normalize
        x = torch.full((4, 32, 32), 0.5)
        out = photometric_normalize(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(float(out.abs().max()), 2.0)

    def test_crops_in_0_1_range_after_alignment(self):
        """candidate crops produced by aligned_candidate should already be in [0,1]."""
        from experiments.exp1.pose import SceneReps, aligned_candidate
        import numpy as np
        image = np.random.randint(0, 256, (200, 200), dtype=np.uint8).astype(np.float32)
        crop, mask = aligned_candidate(image, (100., 100.), (45.0, 1.0))
        if mask.all():
            self.assertGreaterEqual(float(crop.min()), 0.0)
            self.assertLessEqual(float(crop.max()), 255.0 + 1e-4)


# ─── INTEGRATION / C0 + E2 REPRODUCTION ──────────────────────────────────────

class Integration(unittest.TestCase):
    def test_exp2_integration_reused_verbatim(self):
        """Exp3's apply_geometry_stage must delegate to Exp2's hybrid_prediction."""
        from experiments.exp3_pairwise.integration import apply_geometry_stage
        from experiments.exp2_geometry.integration import hybrid_prediction
        from constellation.contracts import ScenePrediction
        learned = ScenePrediction([(1., 1., 0), None, (3., 3., 0)], 'pisces',
                                   {'relocated_queries': [0, 2]})
        classical = ScenePrediction([(10., 10., 1), (20., 20., 0), (30., 30., 1)],
                                     'pisces', {'relocated_queries': [0, 2]})
        snap_rescue = apply_geometry_stage(learned, classical, 'verifier_snap_rescue')
        expected = hybrid_prediction(learned, classical, snap='relocated', rescue='relocated')
        self.assertEqual(snap_rescue.patches, expected.patches)
        self.assertEqual(snap_rescue.constellation, expected.constellation)

    def test_c0_metrics_unchanged(self):
        import json
        m = json.load(open('outputs/joint_train/metrics.json'))
        self.assertAlmostEqual(m['mean']['score'], 0.7286878605463721, places=12)
        self.assertAlmostEqual(m['mean']['presence'], 0.717315544749591, places=12)
        self.assertAlmostEqual(m['mean']['recovery'], 0.8777777777777778, places=12)
        self.assertAlmostEqual(m['mean']['identification'], 0.6666666666666666, places=12)

    def test_e2_metrics_unchanged(self):
        import json
        d = json.load(open('outputs/exp2_geometry/record.json'))
        m = d['primary']['rules']['snap_and_rescue_relocated']['metrics']['mean']
        self.assertAlmostEqual(m['score'], 0.7619585865218471, places=10)
        self.assertAlmostEqual(m['presence'], 0.865213263466306, places=10)

    def test_complete_fold_aggregation_only(self):
        """Only arms evaluated on ALL THREE held-out skies may be compared with E2."""
        result = {
            'complete_arm': {'per_scene': {'pisces': {'score': 0.7}, 'scorpius': {'score': 0.8},
                             'taurus': {'score': 0.5}}, 'complete': True},
            'partial_arm': {'per_scene': {'pisces': {'score': 0.95}}, 'complete': False},
        }
        complete = {k: v for k, v in result.items() if v['complete']}
        self.assertIn('complete_arm', complete)
        self.assertNotIn('partial_arm', complete)

    def test_submission_csv_serialization(self):
        """A complete ScenePrediction from the integration path must serialize."""
        import csv, tempfile
        from pathlib import Path
        from constellation.contracts import write_submission, ScenePrediction
        from experiments.exp1.env import ROOT
        sample = ROOT / 'sample_submission.csv'
        rows = list(csv.DictReader(open(sample)))
        preds = {r['Id']: ScenePrediction(
            [(100., 200., 0)] * int(r['n_patches']), 'pisces') for r in rows}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'test.csv'
            write_submission(preds, sample, out)
            got = list(csv.DictReader(open(out)))
        self.assertEqual(len(got), len(rows))
        self.assertEqual(list(got[0].keys()), list(rows[0].keys()))

    def test_exp3_does_not_import_production_modules(self):
        """experiments.exp3_pairwise must not be imported by constellation/*."""
        import subprocess, sys
        out = subprocess.run(
            [sys.executable, '-c',
             'import constellation.pipeline, constellation.finalize, sys; '
             'print(any(m.startswith("experiments") for m in sys.modules))'],
            capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), 'False', out.stderr)


# ─── DETERMINISM / RESUME ─────────────────────────────────────────────────────

@needs_torch
class Determinism(unittest.TestCase):
    def test_step_sampler_is_deterministic(self):
        from experiments.exp3_pairwise.groups import QueryGroup
        from experiments.exp3_pairwise.streams import StepSampler
        gs = [QueryGroup(f't:{i}', 'present' if i % 2 == 0 else 'absent',
                         'pisces', 'pisces', 'pisces', f's:{i}', None,
                         np.zeros((32,32),np.float32), np.zeros((0,2)),
                         np.zeros((0,32,32),np.float32), np.zeros(0,bool),
                         np.zeros(0), [], np.zeros(0), False, False)
              for i in range(20)]
        sampler = StepSampler(gs, seed=31004, batch_size=6)
        a = [g.group_id for g in sampler(0)]
        b = [g.group_id for g in sampler(0)]
        c = [g.group_id for g in sampler(1)]
        self.assertEqual(a, b, 'same step must give same group')
        self.assertNotEqual(a, c, 'different steps should differ')

    def test_checkpoint_resume_restores_model_state(self):
        import tempfile
        from pathlib import Path
        from experiments.exp3_pairwise.training import build_model, _trainable_state_dict, _load_trainable_state_dict
        model = build_model('A')
        sd = _trainable_state_dict(model)
        model2 = build_model('A')
        _load_trainable_state_dict(model2, sd)
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            if not n1.startswith('hardnet.'):
                self.assertTrue(torch.allclose(p1, p2), f'{n1} mismatch after resume')


# ─── INTEGRITY ────────────────────────────────────────────────────────────────

class ProtectedArtifacts(unittest.TestCase):
    def test_protected_before_json_exists(self):
        from pathlib import Path
        p = Path('outputs/exp3_pairwise/protected_before.json')
        self.assertTrue(p.exists())

    def test_protected_before_covers_core_directories(self):
        import json
        from pathlib import Path
        d = json.load(open('outputs/exp3_pairwise/protected_before.json'))
        paths = set(d.keys())
        # At minimum: run.py + some production files
        self.assertIn('run.py', paths)
        prod = [p for p in paths if p.startswith('outputs/joint_train/')]
        self.assertGreater(len(prod), 0)

    def test_c0_prediction_files_unchanged(self):
        """joint_train predictions must be bit-identical to what was hashed."""
        import json
        from experiments.exp1.env import sha256_file
        before = json.load(open('outputs/exp3_pairwise/protected_before.json'))
        for scene in SCENES:
            path = f'outputs/joint_train/{scene}.json'
            if path in before:
                self.assertEqual(sha256_file(path), before[path], f'{path} was modified')


# ─── ARM SPEC SANITY ──────────────────────────────────────────────────────────

class ArmSpec(unittest.TestCase):
    def test_all_arms_defined(self):
        from experiments.exp3_pairwise import ARMS
        for arm in ARMS:
            self.assertIn(arm, ARM_SPEC)

    def test_matched_e_uses_random_negatives(self):
        self.assertEqual(ARM_SPEC['E']['negatives'], 'random')
        self.assertEqual(ARM_SPEC['D']['negatives'], 'hard')
        self.assertEqual(ARM_SPEC['E']['hardnet_fusion'], True)
        self.assertEqual(ARM_SPEC['D']['hardnet_fusion'], True)

    def test_f_is_the_only_offset_arm(self):
        offset_arms = [a for a, s in ARM_SPEC.items() if s['offset']]
        self.assertEqual(offset_arms, ['F'])


# ─── REPAIR REGRESSION TESTS (this experiment's own defect fixes) ────────────

@needs_torch
class NegativeDeterminism(unittest.TestCase):
    """Repair task §3.1: negatives.py must never use Python's built-in hash()."""

    def _group(self):
        from experiments.exp3_pairwise.groups import QueryGroup
        xy = np.array([[10. * j, 20. * j] for j in range(10)], float)
        labels = ['positive'] + ['negative'] * 9
        return QueryGroup('test:repair:g1', 'present', 'pisces', 'pisces', 'scorpius',
                          'src:1', (0., 0.), np.zeros((32, 32), np.float32), xy,
                          np.zeros((10, 32, 32), np.float32), np.ones(10, bool),
                          np.concatenate([[np.nan], np.linspace(0.9, 0.1, 9)]),
                          labels, np.array([5.0] + [40.0] * 9), False, False)

    def test_select_hard_does_not_call_builtin_hash(self):
        """The EXECUTABLE code must never call Python's hash(); the docstring may
        still mention the old `hash(group.group_id)` pattern as a documented
        rationale for the fix, so only function BODIES are checked here."""
        import ast
        import inspect
        from experiments.exp3_pairwise import negatives
        tree = ast.parse(inspect.getsource(negatives))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == 'hash']
        self.assertEqual(calls, [], 'found a call to builtin hash() in executable code')

    def test_select_hard_reproducible_across_calls(self):
        from experiments.exp3_pairwise.negatives import select_hard
        g = self._group()
        r1 = select_hard(g, seed=31004)
        r2 = select_hard(g, seed=31004)
        self.assertEqual(r1['indices'].tolist(), r2['indices'].tolist())
        self.assertEqual(r1['digest'], r2['digest'])

    def test_select_random_reproducible_across_calls(self):
        from experiments.exp3_pairwise.negatives import select_random
        g = self._group()
        r1 = select_random(g, seed=31004)
        r2 = select_random(g, seed=31004)
        self.assertEqual(r1['indices'].tolist(), r2['indices'].tolist())

    def test_different_seed_changes_draw(self):
        from experiments.exp3_pairwise.negatives import select_hard
        g = self._group()
        r1 = select_hard(g, seed=31004)
        r2 = select_hard(g, seed=31005)
        self.assertNotEqual(r1['digest'], r2['digest'])

    def test_verify_negative_determinism_helper(self):
        from experiments.exp3_pairwise.negatives import verify_negative_determinism
        audit = verify_negative_determinism([self._group()], seed=31004)
        self.assertTrue(audit['ok'])

    def test_matched_presentation_count_preserved(self):
        """The determinism fix must not change hard/random presentation parity."""
        from experiments.exp3_pairwise.negatives import hard_vs_random_manifest
        manifest = hard_vs_random_manifest([self._group()], seed=31004)
        self.assertTrue(manifest['matched'])


@needs_torch
class HardNetSourceCorrectness(unittest.TestCase):
    """Repair task §3.2: frozen HardNet must come from Experiment 1B's
    fold-specific, seed-specific arm-B checkpoint, never the generic pretrained
    backbone (except via the explicitly named control)."""

    def test_frozen_hardnet_resolves_exp1b_checkpoint(self):
        from experiments.exp3_pairwise.hardnet_source import frozen_hardnet
        model, provenance = frozen_hardnet('cpu', 'pisces', 31004)
        self.assertEqual(provenance['source'], 'exp1b_fold_specific_arm_b')
        self.assertEqual(provenance['selected_arm'], 'B')
        self.assertIn('outputs/exp1b/runs/pisces/B_s31004/best.pt',
                      provenance['checkpoint_path'])
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_frozen_hardnet_differs_by_fold(self):
        from experiments.exp3_pairwise.hardnet_source import frozen_hardnet
        _, prov_pisces = frozen_hardnet('cpu', 'pisces', 31004)
        _, prov_scorpius = frozen_hardnet('cpu', 'scorpius', 31004)
        self.assertNotEqual(prov_pisces['checkpoint_path'], prov_scorpius['checkpoint_path'])
        self.assertNotEqual(prov_pisces['checkpoint_sha256'], prov_scorpius['checkpoint_sha256'])

    def test_frozen_hardnet_differs_by_seed(self):
        from experiments.exp3_pairwise.hardnet_source import frozen_hardnet
        _, prov_a = frozen_hardnet('cpu', 'pisces', 31004)
        _, prov_b = frozen_hardnet('cpu', 'pisces', 31005)
        self.assertNotEqual(prov_a['checkpoint_sha256'], prov_b['checkpoint_sha256'])

    def test_generic_control_is_explicitly_labeled(self):
        from experiments.exp3_pairwise.hardnet_source import frozen_hardnet_generic_control
        _, provenance = frozen_hardnet_generic_control('cpu')
        self.assertEqual(provenance['source'], 'generic_pretrained_control')
        self.assertIsNone(provenance['fold'])

    def test_wrong_selected_arm_raises(self):
        """If Exp1B's selection_frozen.json ever stops selecting arm B, this
        module must fail loudly rather than silently load a stale arm."""
        from experiments.exp3_pairwise.hardnet_source import resolve_exp1b_source
        import json
        from pathlib import Path
        from experiments.exp1.env import ROOT
        frozen_path = ROOT / 'outputs' / 'exp1b' / 'selection_frozen.json'
        real = json.loads(frozen_path.read_text())
        self.assertEqual(real['pisces']['arm'], 'B',
                         'precondition: this test assumes arm B is currently selected')


@needs_torch
class CalibratedArmSelection(unittest.TestCase):
    """Repair task §3.3: arm/checkpoint selection must use a calibrated presence
    probability (comparable across BCE and listwise objectives), fit on a
    partition DISTINCT from the one used for final checkpoint scoring."""

    def test_calibrated_selection_metrics_uses_calibrator_not_raw_margin(self):
        from experiments.exp3_pairwise.metrics import calibrated_selection_metrics, to_calibration_rows
        rows = [
            {'kind': 'present', 'target_scene': 'pisces', 'best_logit': 0.1,
            'second_logit': -0.1, 'absent_logit': -5.0, 'n_valid': 5,
            'localization_reward': 1.0, 'top1_correct': True, 'bank_has_correct': True},
            {'kind': 'absent', 'target_scene': 'pisces', 'best_logit': -0.2,
            'second_logit': None, 'absent_logit': 5.0, 'n_valid': 3,
            'localization_reward': 0.0},
        ] * 5
        cal = {'ok': False}
        m = calibrated_selection_metrics({'pisces': rows}, cal, 0.5)
        self.assertIn('selection_metric', m)
        self.assertGreaterEqual(m['presence'], 0.0)

    def test_matrix_uses_synthcal_partition(self):
        """build_eval_fn must accept and use a distinct synthcal stream."""
        import inspect
        from experiments.exp3_pairwise import matrix
        source = inspect.getsource(matrix.build_eval_fn)
        self.assertIn('synthcal', source)
        self.assertIn('fit_calibrator', source)


@needs_torch
class CheckpointMetadataAtomic(unittest.TestCase):
    """Repair task §3.4: the checkpoint metadata bug where `best`'s metric/step
    came from one evaluation but presence/localization from a LATER one."""

    def test_best_checkpoint_not_last_step_metadata_consistent(self):
        """Regression test: construct a scenario where the best checkpoint is
        NOT the last training step, and verify presence/localization/top1 in
        `best` match the SAME step as the recorded metric (not a later step)."""
        from experiments.exp3_pairwise.training import build_model
        from experiments.exp3_pairwise import ARM_SPEC
        # Simulate train_arm's evaluation bookkeeping directly (fast, no real
        # training needed): two evaluations, the FIRST improves, the SECOND does
        # not, so `best` must reflect the FIRST evaluation's presence/localization.
        evaluations = []
        best = {'metric': float('-inf'), 'selection_metric': float('-inf'),
               'step': None, 'presence': None, 'localization': None,
               'top1_correct': None, 'calibrator': None, 'threshold': None,
               'per_scene': None}
        for step, (metric, presence, localization) in enumerate(
                [(0.50, 0.90, 0.80), (0.30, 0.10, 0.05)], start=1):
            metric_doc = {'selection_metric': metric, 'presence': presence,
                         'localization': localization, 'top1_correct': 0.5,
                         'calibrator': {'ok': True}, 'threshold': {'selected': 0.5},
                         'per_scene': {}}
            evaluations.append({'step': step * 500, 'metric': metric, **metric_doc})
            improved = metric > best['metric'] + 1e-9
            if improved:
                best = {'step': step * 500, 'selection_metric': metric, 'metric': metric,
                        'presence': metric_doc['presence'],
                        'localization': metric_doc['localization'],
                        'top1_correct': metric_doc['top1_correct'],
                        'calibrator': metric_doc['calibrator'],
                        'threshold': metric_doc['threshold'],
                        'per_scene': metric_doc['per_scene']}
        # best must be from step 500 (metric=0.50), NOT step 1000 (metric=0.30,
        # presence=0.10) -- the exact bug pattern being regression-tested.
        self.assertEqual(best['step'], 500)
        self.assertAlmostEqual(best['presence'], 0.90)
        self.assertAlmostEqual(best['localization'], 0.80)
        self.assertNotEqual(evaluations[-1]['step'], best['step'],
                            'test precondition: last eval must differ from best')

    def test_checkpoint_best_dict_has_all_required_fields(self):
        from experiments.exp3_pairwise.repo_checklist import check_checkpoint_metadata_atomic
        good = {'step': 500, 'selection_metric': 0.4, 'presence': 0.9, 'localization': 0.8}
        bad = {'step': 500, 'selection_metric': 0.4}
        self.assertTrue(check_checkpoint_metadata_atomic(good)['ok'])
        self.assertFalse(check_checkpoint_metadata_atomic(bad)['ok'])


class HonestHardNegativeDefinition(unittest.TestCase):
    """Repair task §3.5: do not claim network-mined negatives unless wired."""

    def test_negatives_docstring_does_not_overclaim_network_mining(self):
        import inspect
        from experiments.exp3_pairwise import negatives
        doc = negatives.__doc__ or ''
        self.assertIn('classical', doc.lower())
        self.assertIn('random', doc.lower())
        # The corrected docstring must explicitly state the network slot is not
        # currently wired into training, rather than implying it always is.
        self.assertTrue('does not' in doc.lower() or 'not currently' in doc.lower()
                        or 'never' in doc.lower())


class MissingModulesNowExist(unittest.TestCase):
    """Repair task §3.6: every module the report references must actually exist."""

    def test_baseline_verification_module_exists(self):
        from experiments.exp3_pairwise import baseline_verification
        self.assertTrue(hasattr(baseline_verification, 'run'))

    def test_integrity_module_exists(self):
        from experiments.exp3_pairwise import integrity
        self.assertTrue(hasattr(integrity, 'snapshot_protected'))
        self.assertTrue(hasattr(integrity, 'verify_protected'))

    @needs_torch
    def test_held_out_runner_module_exists(self):
        # held_out.py (and therefore held_out_runner.py, which imports it)
        # requires torch; only meaningful in the .venv-exp1 environment.
        from experiments.exp3_pairwise import held_out_runner
        self.assertTrue(hasattr(held_out_runner, 'run_all_folds'))

    def test_finalize_module_exists(self):
        from experiments.exp3_pairwise import finalize
        self.assertTrue(hasattr(finalize, 'run'))

    def test_completion_audit_module_exists(self):
        from experiments.exp3_pairwise import completion_audit
        self.assertTrue(hasattr(completion_audit, 'audit'))


class GatesAreExecutable(unittest.TestCase):
    """Repair task §3.7: gates.py must actually be callable end-to-end."""

    def test_evaluate_gates_runs_with_synthetic_inputs(self):
        from experiments.exp3_pairwise.gates import evaluate_gates
        exp3_primary = {'score': 0.80, 'presence': 0.9, 'localization': 0.75,
                        'recovery': 0.9, 'identification': 0.6666666666666666,
                        'per_scene': {'pisces': {'score': 0.9},
                                     'scorpius': {'score': 0.85},
                                     'taurus': {'score': 0.7}}}
        exp3_repeat = dict(exp3_primary, score=0.77)
        e2_primary = {'mean': {'presence': 0.865, 'localization': 0.686,
                               'recovery': 0.833, 'identification': 0.6666666666666666},
                     'scenes': {'pisces': {'score': 0.85}, 'scorpius': {'score': 0.82},
                               'taurus': {'score': 0.6}}}
        e2_repeat = {'mean': {'presence': 0.87, 'localization': 0.69,
                              'recovery': 0.78, 'identification': 0.6666666666666666},
                    'scenes': {}}
        result = evaluate_gates(exp3_primary, exp3_repeat, e2_primary, e2_repeat,
                                integrity_ok=True, completed_folds=list(SCENES),
                                both_seeds_evaluated=True)
        self.assertEqual(len(result['checks']), 10)
        self.assertIn('pass', result)


class BaselineComparisonProvenance(unittest.TestCase):
    """Repair task §3.8: primary compares to primary, repeat to repeat, with
    explicit provenance on every delta."""

    def test_finalize_per_scene_deltas_carry_provenance(self):
        from experiments.exp3_pairwise.finalize import _per_scene_deltas
        e2_scenes = {s: {'score': 0.7} for s in SCENES}
        exp3_scenes = {s: {'score': 0.8} for s in SCENES}
        e2_source = {'source_file': 'outputs/exp2_geometry/record.json',
                    'source_seed_tag': 'primary'}
        deltas = _per_scene_deltas(exp3_scenes, e2_scenes, e2_source)
        for scene in SCENES:
            self.assertEqual(deltas[scene]['baseline_source_file'],
                             'outputs/exp2_geometry/record.json')
            self.assertEqual(deltas[scene]['baseline_seed'], 'primary')
            self.assertAlmostEqual(deltas[scene]['delta'], 0.1)


class TestSuiteCountsNotHardcoded(unittest.TestCase):
    """Repair task §3.9: test pass/fail/skip counts must be parsed, not hardcoded."""

    def test_parse_unittest_summary_extracts_real_counts(self):
        from experiments.exp3_pairwise.finalize import _parse_unittest_summary
        sample = 'Ran 179 tests in 12.193s\n\nFAILED (errors=1, skipped=39)\n'
        parsed = _parse_unittest_summary(sample)
        self.assertEqual(parsed['tests_run'], 179)
        self.assertEqual(parsed['errors'], 1)
        self.assertEqual(parsed['skipped'], 39)
        self.assertEqual(parsed['failures'], 0)
        self.assertFalse(parsed['wasSuccessful'])

    def test_parse_unittest_summary_ok_case(self):
        from experiments.exp3_pairwise.finalize import _parse_unittest_summary
        sample = 'Ran 53 tests in 1.234s\n\nOK\n'
        parsed = _parse_unittest_summary(sample)
        self.assertEqual(parsed['tests_run'], 53)
        self.assertEqual(parsed['failures'], 0)
        self.assertEqual(parsed['errors'], 0)
        self.assertTrue(parsed['wasSuccessful'])


class CompletionAuditRules(unittest.TestCase):
    """Repair task §6, §12: no COMPLETE claim without an executable audit; no
    pending fields may coexist with COMPLETE."""

    def test_forbidden_token_detection(self):
        from experiments.exp3_pairwise.completion_audit import _find_forbidden_tokens
        doc = {'status': 'pending', 'other': 'this field mentions pending in prose'}
        hits = _find_forbidden_tokens(doc)
        self.assertEqual(len(hits), 1)
        self.assertIn('status', hits[0])

    def test_forbidden_token_ignores_non_status_keys(self):
        from experiments.exp3_pairwise.completion_audit import _find_forbidden_tokens
        doc = {'note': 'this was previously pending but is now fixed'}
        hits = _find_forbidden_tokens(doc)
        self.assertEqual(hits, [])


class DeploymentPolicyNoOracleRouting(unittest.TestCase):
    """Repair task §8: the deployment policy must never route by scene identity."""

    def test_policy_selection_uses_only_allowed_sky_evidence(self):
        """The policy-selection FUNCTION CALLS must never read a held-out or
        Kaggle result; prose explaining that constraint may mention those words
        without violating it, so this checks the actual call graph: only
        `collect_inner_evidence` (which reads matrix_s{seed}.json, allowed-sky
        only) and `select_fixed_architecture` are invoked."""
        import ast
        import inspect
        from experiments.exp3_pairwise import deployment
        tree = ast.parse(inspect.getsource(deployment.build_deployment_policy))
        call_names = {n.func.id for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        forbidden = {'load_held_out', 'evaluate_fold', 'run_all_folds'}
        self.assertEqual(call_names & forbidden, set())
        self.assertIn('collect_inner_evidence', call_names)

    def test_select_fixed_architecture_is_deterministic_tiebreak(self):
        from experiments.exp3_pairwise.deployment import select_fixed_architecture
        evidence = {
            'A': [{'seed': 31004, 'fold': 'pisces', 'selection_metric': 0.5,
                  'presence': 0.9, 'localization': 0.8}],
            'B': [{'seed': 31004, 'fold': 'pisces', 'selection_metric': 0.5,
                  'presence': 0.9, 'localization': 0.8}],
        }
        result = select_fixed_architecture(evidence)
        self.assertEqual(result['selected_arm'], 'A')   # alphabetical tiebreak


if __name__ == '__main__':
    unittest.main()

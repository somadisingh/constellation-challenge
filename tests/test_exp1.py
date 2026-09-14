"""Required tests for Experiment 1 (plan §13).

Torch-dependent cases skip when torch is absent, so the repository's existing
`python -m unittest discover -s tests` still passes under `.venv`. Run the full set
with the experiment environment:

    OMP_NUM_THREADS=1 .venv-exp1/bin/python -m unittest tests.test_exp1 -v
"""
import unittest

import cv2
import numpy as np

from experiments.exp1 import SCENES, augment, banks, mining, pose as P, splits
from experiments.exp1.data import PATCH, PATCH_MID
from experiments.exp1.scoring import NEG_INF, summarise_query, AlignedSet

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

needs_torch = unittest.skipUnless(HAVE_TORCH, 'torch not installed in this env')


def band_limited(shape, seed=0, sigma=1.6):
    rs = np.random.default_rng(seed)
    return cv2.GaussianBlur(rs.uniform(0, 1, shape).astype(np.float32), (0, 0), sigma)


class Conventions(unittest.TestCase):
    """centre / rotation / scale conventions."""

    def test_round_trip(self):
        rs = np.random.default_rng(3)
        for angle in (0., 37., 180., 293.5):
            for scale in (.65, 1., 1.5):
                q = rs.uniform(0, 31, size=(50, 2))
                src = P.query_to_source(q, (500., 700.), angle, scale)
                back = P.source_to_query(src - np.array([500., 700.]), angle, scale)
                self.assertLess(np.abs(back - q).max(), 1e-9)

    def test_midpoint_maps_to_centre(self):
        for angle, scale in ((0., 1.), (77., 1.31), (250., .75)):
            src = P.query_to_source([[PATCH_MID, PATCH_MID]], (123.25, 456.75),
                                    angle, scale)[0]
            self.assertAlmostEqual(src[0], 123.25, places=9)
            self.assertAlmostEqual(src[1], 456.75, places=9)

    def test_matches_refine_warp_matrix(self):
        """refine.py's linear part must equal scale * R(-angle)."""
        for angle in (0., 33., 210.):
            for scale in (.8, 1.2):
                a = np.deg2rad(angle)
                c, s = scale * np.cos(a), scale * np.sin(a)
                self.assertLess(np.abs(np.array([[c, s], [-s, c]])
                                       - scale * P.rotation(-angle)).max(), 1e-12)

    def test_inverse_scale_direction(self):
        """A larger scale must read FURTHER out in the source, not nearer."""
        near = P.query_to_source([[31., 15.5]], (0., 0.), 0., 0.75)[0]
        far = P.query_to_source([[31., 15.5]], (0., 0.), 0., 1.33)[0]
        self.assertGreater(abs(far[0]), abs(near[0]))

    def test_rotation_direction_asymmetric(self):
        q = P.source_to_query([[10., 0.]], 90.0, 1.0)[0]
        self.assertAlmostEqual(q[0], PATCH_MID, places=9)
        self.assertAlmostEqual(q[1], PATCH_MID + 10.0, places=9)


class PoseWarpParity(unittest.TestCase):
    """actual candidate-pose warp parity against the classical sampler."""

    def test_aligned_candidate_reproduces_a_known_patch(self):
        from constellation.retrieval import normalize
        patch = band_limited((32, 32), seed=11)
        angle, scale, centre = 41.0, 1.15, (300.5, 412.25)
        yy, xx = np.mgrid[0:800, 0:800]
        off = np.stack([xx.ravel() - centre[0], yy.ravel() - centre[1]], axis=-1)
        qc = P.source_to_query(off, angle, scale)
        inb = (qc[:, 0] >= 0) & (qc[:, 0] <= 31) & (qc[:, 1] >= 0) & (qc[:, 1] <= 31)
        vals = np.zeros(len(qc), np.float32)
        vals[inb] = cv2.remap(patch, qc[inb, 0].astype(np.float32)[None, :],
                              qc[inb, 1].astype(np.float32)[None, :],
                              cv2.INTER_LINEAR).ravel()
        image = vals.reshape(800, 800)
        crop, mask = P.aligned_candidate(image, centre, (angle, scale), aa_factor=0.0)
        self.assertTrue(mask.all())
        self.assertGreater(P.masked_ncc(patch, crop, mask), 0.99)

        # The same pose read through the classical disc must also correlate.
        disc = np.array([(c, r) for r in range(-12, 13, 2) for c in range(-12, 13, 2)
                         if r * r + c * c <= 144], float)
        qd = P.source_to_query(disc, angle, scale)
        tmpl = cv2.remap(patch, qd[:, 0].astype(np.float32)[None, :],
                         qd[:, 1].astype(np.float32)[None, :], cv2.INTER_LINEAR).ravel()
        samp = cv2.remap(image, (centre[0] + disc[:, 0]).astype(np.float32)[None, :],
                         (centre[1] + disc[:, 1]).astype(np.float32)[None, :],
                         cv2.INTER_LINEAR).ravel()
        self.assertGreater((normalize(tmpl[None, :]) @ normalize(samp[None, :]).T).item(),
                           0.99)

    def test_out_of_image_pose_is_rejected_not_blackened(self):
        image = np.full((60, 60), 128.0, np.float32)
        crop, mask = P.aligned_candidate(image, (2.0, 2.0), (0.0, 1.33))
        self.assertFalse(mask.all())
        reps = P.SceneReps(image)
        q = np.full((32, 32), 128.0, np.float32)
        sel = P.select_pose(reps, q, (2.0, 2.0), (0.0, 1.33))
        self.assertTrue(sel['alignment_failed'])
        self.assertIsNone(sel['pose'])
        self.assertGreater(sel['rejected'], 0)

    def test_pose_trial_count_is_nine(self):
        self.assertEqual(len(P.pose_trials(10.0, 1.0)), 9)
        self.assertEqual(P.N_POSE_TRIALS, 9)


class Splits(unittest.TestCase):
    """no split footprint overlap; no held-out fitting."""

    def test_partitions_disjoint_including_footprints(self):
        check = splits.check_partition_disjoint(3000, 3000)
        self.assertTrue(check['ok'], check['problems'])
        self.assertLess(check['radius'], splits.MARGIN)

    def test_margin_covers_worst_case_sampling(self):
        self.assertLessEqual(splits.max_sampling_radius(), splits.MARGIN)

    def test_cell_partition_assignment(self):
        self.assertEqual([splits.partition_of(c) for c in range(9)],
                         ['fit'] * 6 + ['val'] * 2 + ['synthcal'])

    def test_fold_excludes_held_out_sky(self):
        for fold in SCENES:
            held, allowed = splits.fold_skies(fold)
            self.assertEqual(held, fold)
            self.assertNotIn(fold, allowed)
            self.assertEqual(len(allowed), 2)

    def test_bank_spacing_and_containment(self):
        bank = splits.build_source_bank('taurus', '.',
                                        {'fit': 300, 'val': 100, 'synthcal': 50})
        check = splits.check_bank(bank)
        self.assertTrue(check['ok'], check['problems'])

    def test_val_and_fit_centres_never_share_a_footprint(self):
        bank = splits.build_source_bank('taurus', '.',
                                        {'fit': 300, 'val': 100, 'synthcal': 50})
        from scipy.spatial import cKDTree
        fit = splits.partition_centres(bank, 'fit')
        val = splits.partition_centres(bank, 'val')
        d, _ = cKDTree(val).query(fit, k=1)
        self.assertGreater(d.min(), 2 * bank['max_sampling_radius'])


class Augmentation(unittest.TestCase):
    def test_synthesis_preserves_the_centre_contract(self):
        rs = np.random.default_rng(5)
        image = band_limited((400, 400), seed=2) * 255
        centre = (200.25, 199.5)
        pz = augment.sample_pose(rs)
        dg = augment.sample_degradation(rs)
        dg.update(blur_sigma=0.0, gain=1.0, offset=0.0, drift=0.0,
                  read_noise=0.0, shot_alpha=0.0, jpeg_quality=None)
        out = augment.synthesize_query(image, centre, pz, dg)
        self.assertTrue(out['ok'])
        # The undegraded synthesis must match a plain aligned read at the same pose.
        ref, mask = P.aligned_candidate(image.astype(np.float32), centre,
                                        (pz['angle'], pz['scale']))
        inner = np.zeros((32, 32), bool)
        inner[4:28, 4:28] = True
        self.assertGreater(P.masked_ncc(out['query'].astype(np.float32), ref, inner),
                           0.99)

    def test_degradation_only_touches_the_query_view(self):
        """The candidate view is a plain resample: no noise or JPEG is shared."""
        rs = np.random.default_rng(7)
        image = band_limited((400, 400), seed=3) * 255
        dg = augment.sample_degradation(rs)
        a = augment.synthesize_query(image, (200., 200.),
                                     {'angle': 10., 'scale': 1.0}, dg)
        b = augment.synthesize_query(image, (200., 200.),
                                     {'angle': 10., 'scale': 1.0},
                                     {**dg, 'noise_seed': dg['noise_seed'] + 1})
        self.assertFalse(np.array_equal(a['query'], b['query']))

    def test_constant_source_is_flagged_not_crashed(self):
        rs = np.random.default_rng(9)
        image = np.full((400, 400), 100.0, np.float32)
        dg = augment.sample_degradation(rs)
        dg.update(read_noise=0.0, shot_alpha=0.0, drift=0.0, jpeg_quality=None)
        out = augment.synthesize_query(image, (200., 200.),
                                       {'angle': 0., 'scale': 1.0}, dg)
        self.assertTrue(out['ok'])
        self.assertTrue(out['constant'])
        self.assertTrue(out['degenerate'])

    def test_stress_ranges_are_wider(self):
        main, stress = augment.Ranges(), augment.Ranges.stress()
        self.assertLess(stress.scale_log[0], main.scale_log[0])
        self.assertGreater(stress.scale_log[1], main.scale_log[1])
        self.assertGreater(stress.read_noise[1], main.read_noise[1])


class LabelPolicy(unittest.TestCase):
    """false-negative identity masks; ambiguity is ignored, not labelled."""

    def test_ignore_band_is_not_negative(self):
        self.assertEqual(mining.label_candidate(0.0, 4.0, 36.0), mining.POSITIVE)
        self.assertEqual(mining.label_candidate(4.0, 4.0, 36.0), mining.POSITIVE)
        self.assertEqual(mining.label_candidate(10.0, 4.0, 36.0), mining.IGNORE)
        self.assertEqual(mining.label_candidate(36.0, 4.0, 36.0), mining.IGNORE)
        self.assertEqual(mining.label_candidate(36.1, 4.0, 36.0), mining.NEGATIVE)

    def test_identity_mask_blocks_same_sky_neighbours_only(self):
        from experiments.exp1.pairs import identity_mask
        centres = np.array([[100., 100.], [110., 100.], [900., 900.]])
        scenes = ['a', 'a', 'a']
        mask = identity_mask(centres, scenes, centres, scenes, 36.0)
        self.assertTrue(mask[0, 1])          # 10px apart, same sky -> blocked
        self.assertFalse(mask[0, 2])         # far apart -> usable
        cross = identity_mask(centres, ['a', 'b', 'a'], centres, ['a', 'b', 'a'], 36.0)
        self.assertFalse(cross[0, 1])        # different sky is never the same source


class ScoreDirection(unittest.TestCase):
    """score-direction correctness; absent/empty handling."""

    def _aligned(self, k, xy=None):
        return AlignedSet(query_id='t', xy=(xy if xy is not None
                                            else np.zeros((k, 2))),
                          crops=np.zeros((k, 32, 32), np.float32),
                          ncc=np.zeros(k), poses=np.zeros((k, 2)),
                          admissible=np.ones(k, bool), low_info=np.zeros(k, bool),
                          pose_trials=9, rejected_poses=0,
                          query_raw=np.zeros((32, 32), np.float32))

    def test_higher_is_better_and_gap_is_positive(self):
        aligned = self._aligned(3, np.array([[0., 0.], [50., 0.], [90., 0.]]))
        out = summarise_query(np.array([0.2, 0.9, 0.5]), aligned,
                              truth_xy=(50., 0.))
        self.assertEqual(out['best_index'], 1)
        self.assertAlmostEqual(out['best_match_score'], 0.9)
        self.assertAlmostEqual(out['best_minus_second_score'], 0.4)
        self.assertTrue(out['top1_correct'])
        self.assertAlmostEqual(out['localization_reward'], 1.0)

    def test_empty_bank_returns_absent_without_a_score(self):
        out = summarise_query(np.zeros(0), self._aligned(0))
        self.assertTrue(out['empty_bank'])
        self.assertIsNone(out['best_match_score'])

    def test_single_candidate_uses_gap_zero_and_is_flagged(self):
        out = summarise_query(np.array([0.7]), self._aligned(1))
        self.assertEqual(out['best_minus_second_score'], 0.0)
        self.assertTrue(out['flagged'])

    def test_all_poses_rejected_is_attributed_not_scored(self):
        aligned = self._aligned(2)
        aligned.admissible[:] = False
        out = summarise_query(np.array([NEG_INF, NEG_INF]), aligned,
                              truth_xy=(0., 0.))
        self.assertTrue(out['alignment_failed_all'])
        self.assertEqual(out['localization_reward'], 0.0)

    def test_reward_matches_the_official_scorer(self):
        from constellation.contracts import reward
        aligned = self._aligned(1, np.array([[24., 0.]]))
        out = summarise_query(np.array([1.0]), aligned, truth_xy=(0., 0.))
        self.assertAlmostEqual(out['localization_reward'], float(reward(24.0)))


class Calibration(unittest.TestCase):
    """per-query absent calibration."""

    def _rows(self):
        rows = []
        rs = np.random.default_rng(1)
        for scene, n in (('a', 40), ('b', 40)):
            for i in range(n):
                present = i % 2 == 0
                base = 0.9 if present else 0.4
                rows.append({'scene': scene, 'present': present,
                             'best_match_score': base + rs.normal(0, .05),
                             'best_minus_second_score': (0.3 if present else 0.05),
                             'localization_reward': 1.0 if present else 0.0})
        return rows

    def test_calibrator_separates_and_threshold_is_selected(self):
        from experiments.exp1.calibration import (fit_calibrator, select_threshold,
                                                  apply_calibrator)
        from experiments.exp1.config import load_config
        cfg = load_config()
        rows = self._rows()
        cal = fit_calibrator(rows, cfg)
        self.assertTrue(cal['ok'])
        probs = apply_calibrator(cal, rows)
        present = np.array([r['present'] for r in rows])
        self.assertGreater(probs[present].mean(), probs[~present].mean())
        thr = select_threshold(cal, rows, cfg)
        self.assertIn(thr['selected'], cfg['calibration']['thresholds'])

    def test_unusable_query_gets_probability_zero(self):
        from experiments.exp1.calibration import fit_calibrator, apply_calibrator
        from experiments.exp1.config import load_config
        cfg = load_config()
        rows = self._rows()
        cal = fit_calibrator(rows, cfg)
        rows.append({'scene': 'a', 'present': False, 'best_match_score': None,
                     'best_minus_second_score': None, 'localization_reward': 0.0})
        self.assertEqual(apply_calibrator(cal, rows)[-1], 0.0)

    def test_constant_features_fall_back(self):
        from experiments.exp1.calibration import fit_calibrator
        from experiments.exp1.config import load_config
        rows = [{'scene': 'a', 'present': i % 2 == 0, 'best_match_score': 0.5,
                 'best_minus_second_score': 0.1, 'localization_reward': 0.0}
                for i in range(20)]
        cal = fit_calibrator(rows, load_config())
        self.assertFalse(cal['ok'])
        self.assertIn('fallback', cal)

    def test_never_reuses_the_classical_072_threshold(self):
        from experiments.exp1.config import load_config
        cfg = load_config()
        self.assertNotIn(0.72, cfg['calibration']['thresholds'])


class NoTorchImportPath(unittest.TestCase):
    """Alignment, the classical control and summaries must not require torch."""

    def test_low_info_threshold_is_consistent(self):
        from experiments.exp1 import scoring
        if HAVE_TORCH:
            from experiments.exp1.models import LOW_INFO_STD
            self.assertEqual(scoring.LOW_INFO_STD, LOW_INFO_STD)

    def test_pure_numpy_modules_import_without_torch(self):
        import subprocess
        import sys
        code = (
            'import sys, types\n'
            'sys.modules["torch"] = None\n'
            'import importlib\n'
            'for m in ("experiments.exp1.scoring", "experiments.exp1.pose",\n'
            '          "experiments.exp1.splits", "experiments.exp1.banks",\n'
            '          "experiments.exp1.augment", "experiments.exp1.calibration",\n'
            '          "experiments.exp1.evaluation"):\n'
            '    importlib.import_module(m)\n'
            'print("ok")\n')
        out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                             text=True)
        self.assertEqual(out.stdout.strip(), 'ok', out.stderr[-2000:])


class Banks(unittest.TestCase):
    """candidate-bank no oracle insertion; duplicate-query preservation."""

    def test_bank_traces_entirely_to_label_free_branch_caches(self):
        """No oracle insertion: every entry must come from a branch cache."""
        from experiments.exp1.data import query_records
        for scene in SCENES:
            bank = banks.build_bank(scene)
            check = banks.check_no_oracle(bank, query_records(scene, '.'))
            self.assertTrue(check['ok'], check['problems'])
            self.assertTrue(check['provenance_complete'])

    def test_every_query_is_preserved_including_duplicates(self):
        from experiments.exp1.data import load_scene
        for scene in SCENES:
            bank = banks.build_bank(scene)
            self.assertEqual(bank['n_queries'], len(load_scene(scene, '.')))
            self.assertEqual([q['index'] for q in bank['queries']],
                             list(range(bank['n_queries'])))

    def test_union_is_at_most_sixty_and_keeps_provenance(self):
        bank = banks.build_bank('taurus')
        for q in bank['queries']:
            self.assertLessEqual(q['n_candidates'], 60)
            for c in q['candidates']:
                self.assertTrue(c['branches'])
                self.assertTrue(c['provenance'])

    def test_union_recall_is_at_least_each_branch(self):
        from experiments.exp1.data import query_records
        for scene in SCENES:
            r = banks.branch_recall(scene, query_records(scene, '.'))
            for branch in ('fixed_coarse', 'fixed_ecc', 'adaptive'):
                self.assertGreaterEqual(r['union']['recall@12'] + 1e-12,
                                        r[branch]['recall@12'])


class ProductionUnchanged(unittest.TestCase):
    """C0 unchanged."""

    def test_c0_metrics_match_the_recorded_values(self):
        from experiments.exp1.env import read_json, ROOT
        m = read_json(ROOT / 'outputs' / 'joint_train' / 'metrics.json')
        self.assertAlmostEqual(m['mean']['score'], 0.7286878605463721, places=12)
        self.assertAlmostEqual(m['mean']['presence'], 0.717315544749591, places=12)
        self.assertAlmostEqual(m['mean']['localization'], 0.6495726495726496, places=12)
        self.assertAlmostEqual(m['mean']['recovery'], 0.8777777777777778, places=12)
        self.assertAlmostEqual(m['mean']['identification'], 0.6666666666666666, places=12)

    def test_exp1_does_not_import_into_production(self):
        import subprocess
        import sys
        out = subprocess.run(
            [sys.executable, '-c',
             'import constellation.pipeline, constellation.finalize, sys; '
             'print(any(m.startswith("experiments") for m in sys.modules))'],
            capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), 'False', out.stderr)


class Integration(unittest.TestCase):
    """full CSV output; membership and absent semantics."""

    def test_integration_builds_a_complete_scene_prediction(self):
        from experiments.exp1.evaluation import integrate
        rows = [
            {'query_id': 'x:0', 'best_xy': [100., 100.], 'empty_bank': False,
             'alignment_failed_all': False, 'flagged': False},
            {'query_id': 'x:1', 'best_xy': [900., 900.], 'empty_bank': False,
             'alignment_failed_all': False, 'flagged': False},
            {'query_id': 'x:2', 'best_xy': None, 'empty_bank': True,
             'alignment_failed_all': False, 'flagged': True},
        ]
        geometry = {'constellation': 'pisces', 'nodes': np.array([[105., 100.]]),
                    'source': 'test'}
        out = integrate('x', rows, np.array([0.9, 0.9, 0.9]), 0.5, geometry)
        patches = out['prediction'].patches
        self.assertEqual(len(patches), 3)
        self.assertEqual(patches[0][2], 1)        # within 18px of a frozen node
        self.assertEqual(patches[1][2], 0)        # far from every node
        self.assertIsNone(patches[2])             # empty bank -> absent
        self.assertEqual(out['prediction'].constellation, 'pisces')

    def test_absent_when_probability_below_threshold(self):
        from experiments.exp1.evaluation import integrate
        rows = [{'query_id': 'x:0', 'best_xy': [1., 1.], 'empty_bank': False,
                 'alignment_failed_all': False, 'flagged': False}]
        geometry = {'constellation': 'aries', 'nodes': np.empty((0, 2)),
                    'source': 'test'}
        out = integrate('x', rows, np.array([0.1]), 0.5, geometry)
        self.assertIsNone(out['prediction'].patches[0])

    def test_no_verified_nodes_gives_membership_zero(self):
        from experiments.exp1.evaluation import integrate
        rows = [{'query_id': 'x:0', 'best_xy': [1., 1.], 'empty_bank': False,
                 'alignment_failed_all': False, 'flagged': False}]
        geometry = {'constellation': 'aries', 'nodes': np.empty((0, 2)),
                    'source': 'test'}
        out = integrate('x', rows, np.array([0.9]), 0.5, geometry)
        self.assertEqual(out['prediction'].patches[0][2], 0)

    def test_submission_csv_round_trips(self):
        import csv
        import tempfile
        from pathlib import Path
        from constellation.contracts import write_submission, ScenePrediction
        from experiments.exp1.env import ROOT
        sample = ROOT / 'sample_submission.csv'
        rows = list(csv.DictReader(open(sample)))
        preds = {r['Id']: ScenePrediction(
            [(10.0, 20.0, 0)] * int(r['n_patches']), 'pisces') for r in rows}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'submission.csv'
            write_submission(preds, sample, out)
            got = list(csv.DictReader(open(out)))
        self.assertEqual(len(got), len(rows))
        self.assertEqual(list(got[0].keys()), list(rows[0].keys()))


@needs_torch
class TorchBehaviour(unittest.TestCase):
    """nonzero training update; train/eval restoration; constant input; parity."""

    def test_descriptor_is_unit_norm_and_finite_for_constant_input(self):
        from experiments.exp1.models import build_for_training, low_information
        model, _ = build_for_training('hardnet', pretrained=True, device='cpu')
        model.eval()
        x = torch.zeros(3, 1, 32, 32)
        x[1] = 0.5
        x[2] = torch.rand(1, 32, 32)
        with torch.no_grad():
            z = model(x)
        self.assertTrue(bool(torch.isfinite(z).all()))
        self.assertLess(float((z.norm(dim=1) - 1).abs().max()), 1e-5)
        flags = low_information(x)
        self.assertTrue(bool(flags[0]) and bool(flags[1]) and not bool(flags[2]))

    def test_training_step_updates_parameters(self):
        from experiments.exp1.models import build_for_training, pairwise_distances
        model, info = build_for_training('hardnet', pretrained=True, device='cpu')
        self.assertTrue(model.training)
        before = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        a = torch.rand(8, 1, 32, 32)
        p = torch.rand(8, 1, 32, 32)
        z = model(torch.cat([a, p]))
        d = pairwise_distances(z[:8], z[8:])
        loss = torch.relu(0.5 + torch.diagonal(d)
                          - (d + torch.eye(8) * 1e3).min(1).values).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        delta = max((q - r).abs().max().item()
                    for q, r in zip(model.parameters(), before))
        self.assertGreater(delta, 0.0)

    def test_bn_running_stats_frozen_for_pretrained_finetuning(self):
        from experiments.exp1.models import build_for_training
        model, info = build_for_training('hardnet', pretrained=True, device='cpu',
                                         freeze_bn=True)
        self.assertTrue(info['bn_frozen'])
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                self.assertFalse(module.training)

    def test_train_mode_restored_after_validation(self):
        from experiments.exp1.models import build_for_training, restore_train_mode
        model, _ = build_for_training('hardnet', pretrained=True, device='cpu')
        model.eval()
        self.assertFalse(model.training)
        restore_train_mode(model, freeze_bn=True)
        self.assertTrue(model.training)
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                self.assertFalse(module.training)

    def test_distance_matrix_equals_cdist(self):
        from experiments.exp1.models import pairwise_distances, descriptor_distance
        a = torch.nn.functional.normalize(torch.randn(20, 128), dim=1)
        b = torch.nn.functional.normalize(torch.randn(20, 128), dim=1)
        self.assertLess(float((pairwise_distances(a, b) - torch.cdist(a, b))
                              .abs().max()), 2e-3)
        self.assertLess(float((descriptor_distance(a, b)
                               - torch.diagonal(torch.cdist(a, b))).abs().max()), 2e-3)

    def test_anchor_without_valid_negative_is_omitted_not_self_matched(self):
        from experiments.exp1.training import triplet_loss
        z = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
        invalid = torch.ones(3, 3, dtype=torch.bool)      # everything blocked
        loss, stats = triplet_loss(z, z.clone(), invalid, [None, None, None],
                                   0.5, 1e-6)
        self.assertEqual(stats['anchors_without_negative'], 3)
        self.assertEqual(float(loss), 0.0)

    def test_pair_head_shapes_and_feature_construction(self):
        from experiments.exp1.models import PairHead
        head = PairHead(dim=128)
        zq = torch.nn.functional.normalize(torch.randn(5, 128), dim=1)
        zc = torch.nn.functional.normalize(torch.randn(5, 128), dim=1)
        self.assertEqual(PairHead.features(zq, zc).shape, (5, 256))
        self.assertEqual(head(zq, zc).shape, (5,))

    @unittest.skipUnless(HAVE_TORCH and torch.backends.mps.is_available(),
                         'MPS unavailable')
    def test_cpu_mps_parity_on_informative_patches(self):
        from experiments.exp1.preflight import cpu_mps_parity
        for name in ('hardnet', 'hynet', 'sosnet'):
            parity = cpu_mps_parity(name)
            self.assertGreaterEqual(parity['informative_cosine_min'], 0.999, name)
            self.assertEqual(parity['top1_rank_agreement'], 1.0, name)
            self.assertEqual(parity['zero_norm_descriptors']['mps'], 0, name)

    def test_checkpoint_resume_restores_state(self):
        import tempfile
        from pathlib import Path
        from experiments.exp1.models import build_for_training
        model, _ = build_for_training('hardnet', pretrained=True, device='cpu')
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'ck.pt'
            torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                        'step': 77, 'best': {'metric': 0.5, 'step': 50}}, path)
            other, _ = build_for_training('hardnet', pretrained=False, device='cpu')
            blob = torch.load(path, map_location='cpu', weights_only=False)
            other.load_state_dict(blob['model'])
            self.assertEqual(blob['step'], 77)
            for p, q in zip(model.parameters(), other.parameters()):
                self.assertLess((p - q).abs().max().item(), 1e-9)


if __name__ == '__main__':
    unittest.main()

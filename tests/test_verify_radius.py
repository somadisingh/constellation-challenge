"""Tests for the pose-admissible verification radius.

The invariant that matters is that a larger support never reads outside the patch: a
verifier that sampled invalid borders could score unrelated neighbourhoods highly.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.retrieval import (admissible_radius, _disc, _pose_templates,
                                     verify_adaptive, VERIFY_SCALES, VERIFY_ANGLES)
from constellation.pipeline import Config


class AdmissibleRadius(unittest.TestCase):
    def test_fixed_policy_is_the_frozen_twelve(self):
        for s in VERIFY_SCALES:
            self.assertEqual(admissible_radius(s, 'fixed'), 12)

    def test_adaptive_radius_tracks_the_scale(self):
        r = [admissible_radius(s) for s in VERIFY_SCALES]
        self.assertEqual(r, sorted(r))
        self.assertLess(admissible_radius(.75), 12)     # fixed 12 was inadmissible
        self.assertGreater(admissible_radius(1.33), 12)  # fixed 12 wasted valid area

    def test_every_adaptive_sample_stays_inside_the_patch(self):
        """No pose may read outside [0, 31]; asserted for all scales and angles."""
        for scale in VERIFY_SCALES:
            xx, yy = _disc(admissible_radius(scale), 2)
            for angle in VERIFY_ANGLES:
                a = np.deg2rad(angle)
                mx = 15.5 + (np.cos(a) * xx - np.sin(a) * yy) / scale
                my = 15.5 + (np.sin(a) * xx + np.cos(a) * yy) / scale
                self.assertTrue(((mx >= 0) & (mx <= 31)).all(),
                                f'scale {scale} angle {angle} left the patch in x')
                self.assertTrue(((my >= 0) & (my <= 31)).all(),
                                f'scale {scale} angle {angle} left the patch in y')

    def test_all_angles_admissible_under_adaptive_none_lost(self):
        for scale in VERIFY_SCALES:
            xx, yy = _disc(admissible_radius(scale), 2)
            templates, angles = _pose_templates(np.zeros((32, 32), np.float32),
                                                xx, yy, scale)
            self.assertEqual(len(angles), len(VERIFY_ANGLES))

    def test_fixed_policy_loses_angles_at_the_smallest_scale(self):
        """Documents the limitation the adaptive policy removes."""
        xx, yy = _disc(12, 2)
        _, angles = _pose_templates(np.zeros((32, 32), np.float32), xx, yy, .75)
        self.assertLess(len(angles), len(VERIFY_ANGLES))


class VerifyAdaptiveBehaviour(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.scene = rng.normal(0, 1, (400, 400)).astype(np.float32)
        # Plant a distinctive patch at a known location.
        self.truth = (200., 150.)
        blob = rng.normal(0, 1, (32, 32)).astype(np.float32)
        self.scene[134:166, 184:216] = blob
        self.patch = blob.copy()

    def test_recovers_a_planted_location(self):
        props = np.array([[200., 150.], [60., 60.], [300., 320.], [110., 250.]])
        out = verify_adaptive(self.scene, self.patch, props, keep=4)
        self.assertTrue(out)
        best = out[0]
        self.assertLess(np.hypot(best[0] - self.truth[0], best[1] - self.truth[1]), 3.)

    def test_returns_empty_for_no_candidates(self):
        self.assertEqual(verify_adaptive(self.scene, self.patch, np.zeros((0, 2))), [])

    def test_respects_keep_and_suppression(self):
        props = np.array([[200. + i, 150.] for i in range(30)], float)
        out = verify_adaptive(self.scene, self.patch, props, keep=3)
        self.assertLessEqual(len(out), 3)
        for i, a in enumerate(out):
            for b in out[i + 1:]:
                self.assertGreaterEqual(np.hypot(a[0] - b[0], a[1] - b[1]), 8.)

    def test_output_shape_matches_verify(self):
        props = np.array([[200., 150.], [60., 60.]])
        out = verify_adaptive(self.scene, self.patch, props, keep=2)
        for row in out:
            self.assertEqual(len(row), 5)      # x, y, score, angle, scale
        self.assertEqual(out, sorted(out, key=lambda v: -v[2]))


class ConfigDefaults(unittest.TestCase):
    def test_default_config_reproduces_the_frozen_verifier(self):
        c = Config()
        self.assertEqual(c.verify_radius, 'fixed')
        self.assertEqual(c.verify_rep, 'blur')
        self.assertEqual(c.alternatives, 20)


if __name__ == '__main__':
    unittest.main()

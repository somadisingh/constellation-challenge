"""Tests for the separation of ranking, calibration and seeding roles."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.slate import (QuerySlate, slate_from_candidates, slates_from_lists,
                                 with_scores)
from constellation.joint import build_pool, generation_points, recognize_joint
from constellation.geometry import consolidate
from constellation.reloc import AcceptAll, IndependentAppearance, apply_guard


def candidates(*rows):
    """(x, y, score, angle, scale) rows, score-descending as verify emits them."""
    return [(float(x), float(y), float(s), 0., 1.) for x, y, s in rows]


class SlateConstruction(unittest.TestCase):
    def test_roles_default_to_the_single_score(self):
        s = slate_from_candidates(candidates((10, 10, .9), (20, 20, .5)))
        self.assertTrue(np.array_equal(s.calib, s.rank))
        self.assertEqual(s.seed, 0)
        self.assertEqual(s.report_index, 0)
        self.assertEqual(s.presence_score, .9)
        self.assertAlmostEqual(s.ambiguity_gap, .4)

    def test_empty_slate_is_safe(self):
        s = slate_from_candidates([])
        self.assertEqual(len(s), 0)
        self.assertEqual(s.ambiguity_gap, float('inf'))

    def test_single_candidate_is_never_ambiguous(self):
        s = slate_from_candidates(candidates((10, 10, .5)))
        self.assertEqual(s.ambiguity_gap, float('inf'))

    def test_with_scores_preserves_identity(self):
        s = slate_from_candidates(candidates((10, 10, .9), (20, 20, .89)))
        t = with_scores(s, rank=[0., 1.], seed=1)
        self.assertTrue(np.array_equal(t.xy, s.xy))
        self.assertTrue(np.array_equal(t.calib, s.calib))   # calibration untouched
        self.assertEqual(t.report_index, 1)
        self.assertEqual(t.seed, 1)
        self.assertEqual(t.presence_score, .9)              # presence unchanged


class PoolMembership(unittest.TestCase):
    def setUp(self):
        # One clearly-won query, one near-tied query.
        self.confident = slate_from_candidates(
            candidates((10, 10, .90), (500, 500, .40), (600, 600, .30)))
        self.ambiguous = slate_from_candidates(
            candidates((20, 20, .90), (700, 700, .89), (800, 800, .88)))

    def test_confident_query_contributes_one_point(self):
        pool, tags, _, _ = build_pool([self.confident], [0], top_k=8, margin=.15, gap=.03)
        self.assertEqual(len(pool), 1)
        self.assertTrue(np.allclose(pool[0], [10, 10]))

    def test_ambiguous_query_contributes_several(self):
        pool, _, _, _ = build_pool([self.ambiguous], [0], top_k=8, margin=.15, gap=.03)
        self.assertEqual(len(pool), 3)

    def test_margin_excludes_distant_scores(self):
        s = slate_from_candidates(
            candidates((20, 20, .90), (700, 700, .89), (800, 800, .10)))
        pool, _, _, _ = build_pool([s], [0], top_k=8, margin=.15, gap=.03)
        self.assertEqual(len(pool), 2)      # .10 is more than .15 below .90

    def test_primary_contribution_survives_the_margin(self):
        """A query must never be left unrepresented in the pool."""
        s = slate_from_candidates(
            candidates((10, 10, .90), (500, 500, .40), (600, 600, .30)))
        reranked = with_scores(s, rank=[0., 1., .5])   # ranking prefers a weak candidate
        pool, _, _, _ = build_pool([reranked], [0], top_k=8, margin=.15, gap=.03)
        self.assertEqual(len(pool), 1)
        self.assertTrue(np.allclose(pool[0], [500, 500]))

    def test_rank_change_reorders_without_changing_membership(self):
        """pool_by='calib' isolates re-ordering from a membership change."""
        reranked = with_scores(self.ambiguous, rank=[0., 1., 0.])
        base, _, _, _ = build_pool([self.ambiguous], [0], top_k=2, margin=.15,
                                   gap=.03, pool_by='calib')
        new, _, _, _ = build_pool([reranked], [0], top_k=2, margin=.15,
                                  gap=.03, pool_by='calib')
        self.assertEqual({tuple(p) for p in base}, {tuple(p) for p in new})
        self.assertFalse(np.allclose(base, new))          # order did change

    def test_pool_by_rank_can_change_membership(self):
        reranked = with_scores(self.ambiguous, rank=[0., 0., 1.])
        base, _, _, _ = build_pool([self.ambiguous], [0], top_k=2, margin=.15,
                                   gap=.03, pool_by='rank')
        new, _, _, _ = build_pool([reranked], [0], top_k=2, margin=.15,
                                  gap=.03, pool_by='rank')
        self.assertNotEqual({tuple(p) for p in base}, {tuple(p) for p in new})

    def test_ambiguity_uses_calibration_not_ranking(self):
        """A ranking change must not turn a confident query into an ambiguous one."""
        reranked = with_scores(self.confident, rank=[0., 1., .5])
        pool, _, _, _ = build_pool([reranked], [0], top_k=8, margin=.15, gap=.03)
        self.assertEqual(len(pool), 1)      # still one point, not top_k
        self.assertEqual(reranked.ambiguity_gap, self.confident.ambiguity_gap)


class Seeding(unittest.TestCase):
    def test_seed_selects_the_generation_coordinate(self):
        s = slate_from_candidates(candidates((10, 10, .9), (900, 900, .89)))
        a = generation_points([s], [0])
        b = generation_points([with_scores(s, seed=1)], [0])
        self.assertTrue(np.allclose(a[0], [10, 10]))
        self.assertTrue(np.allclose(b[0], [900, 900]))

    def test_seed_drives_physical_star_grouping(self):
        """Two queries group together only if their seed anchors nearly coincide."""
        p = slate_from_candidates(candidates((100, 100, .9), (101, 100, .89)))
        q = slate_from_candidates(candidates((101, 100, .9), (900, 900, .89)))
        anchors = np.array([p.seed_xy(), q.seed_xy()])
        _, groups = consolidate(anchors)
        self.assertEqual(groups[0], groups[1])
        moved = with_scores(q, seed=1)
        anchors = np.array([p.seed_xy(), moved.seed_xy()])
        _, groups = consolidate(anchors)
        self.assertNotEqual(groups[0], groups[1])


class BackwardCompatibility(unittest.TestCase):
    def test_raw_lists_and_slates_agree(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(200, 2800, size=(9, 2))
        lists = [candidates((x, y, .9 - .01 * i)) for i, (x, y) in enumerate(pts)]
        patterns = {'alpha': pts[:6] + rng.normal(0, 1., (6, 2))}
        a = recognize_joint(lists, patterns)
        b = recognize_joint(slates_from_lists(lists), patterns)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])


class RelocationGuards(unittest.TestCase):
    def setUp(self):
        self.slates = [slate_from_candidates(candidates((10, 10, .9), (900, 900, .89)))]

    def test_accept_all_keeps_every_move(self):
        kept, rejected = apply_guard({0: (900., 900.)}, self.slates, AcceptAll())
        self.assertEqual(kept, {0: (900., 900.)})
        self.assertEqual(rejected, {})

    def test_rejected_move_falls_back_to_the_original(self):
        guard = IndependentAppearance({0: np.array([1., 0.])},
                                      {0: np.array([[10., 10.], [900., 900.]])})
        kept, rejected = apply_guard({0: (900., 900.)}, self.slates, guard)
        self.assertEqual(kept, {})
        self.assertIn(0, rejected)
        # Presence is unaffected: the query still has a coordinate to report.
        self.assertEqual(self.slates[0].presence_score, .9)

    def test_independent_guard_accepts_a_supported_move(self):
        guard = IndependentAppearance({0: np.array([0., 1.])},
                                      {0: np.array([[10., 10.], [900., 900.]])})
        kept, _ = apply_guard({0: (900., 900.)}, self.slates, guard)
        self.assertEqual(kept, {0: (900., 900.)})

    def test_zero_length_move_is_never_guarded(self):
        guard = IndependentAppearance({0: np.array([1., 0.])},
                                      {0: np.array([[10., 10.], [900., 900.]])})
        kept, rejected = apply_guard({0: (10., 10.)}, self.slates, guard)
        self.assertEqual(kept, {0: (10., 10.)})
        self.assertEqual(rejected, {})


if __name__ == '__main__':
    unittest.main()

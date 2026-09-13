import unittest
import numpy as np
from lab.branch_agreement import select, predict


def branch(stage, rows):
    return {'stage':stage,'ids':[0], 'diag':{'hypotheses':[
        {'name':n,'score':s,'support':4,'nodes':[[10.,10.]],'pairs':[(0,0)]}
        for n,s in rows], 'relocation':{'pool':[[10.,10.]],'tags':[0], 'group_of_query':{'0':0}}}}


class BranchAgreement(unittest.TestCase):
    def test_baseline_and_ties(self):
        bs=[branch('refined',[('a',5)]),branch('coarse',[('b',6)])]
        self.assertEqual(select(bs,'baseline')[0]['name'],'b')
        self.assertEqual(select(bs[::-1],'baseline')[0]['name'],'b')

    def test_consensus_can_recover_second_choice(self):
        bs=[branch('refined',[('a',5),('c',4.9)]),branch('coarse',[('b',5),('c',4.9)])]
        self.assertEqual(select(bs,'mean_regret')[0]['name'],'c')
        self.assertEqual(select(bs,'agreement_1')[0]['name'],'c')

    def test_stability_checks_node_positions(self):
        bs=[branch('refined',[('a',5),('c',4.9)]),branch('coarse',[('b',5),('c',4.9)])]
        bs[1]['diag']['hypotheses'][1]['nodes']=[[100.,100.]]
        self.assertEqual(select(bs,'stable_1')[0]['name'],'a')

    def test_id_only_preserves_locations_and_absence(self):
        bs=[branch('refined',[('a',5)]),branch('coarse',[('b',6)])]
        qs=[[[11.,11.,.9]],[[12.,12.,.5]]]
        a=predict(qs,bs,'baseline');b=predict(qs,bs,'refined',True)
        self.assertEqual(a.patches,b.patches)
        self.assertEqual(b.constellation,'a');self.assertIsNone(b.patches[1])

    def test_empty_and_single_branch(self):
        self.assertIsNone(select([],'rrf'))
        self.assertEqual(select([branch('coarse',[('a',3)])],'rrf')[0]['name'],'a')

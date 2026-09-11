import unittest
import numpy as np
from constellation.geometry import consolidate,recognize
from constellation.references import extract_patterns

class GeometryTests(unittest.TestCase):
    def test_nontransitive_groups_and_close_stars(self):
        p,g=consolidate(np.array([[0,0],[1,0],[2.9,0],[5,0],[10,0]]))
        self.assertEqual(len(p),3);self.assertEqual(g,[0,0,0,1,2])
    def test_no_evidence(self):
        self.assertEqual(recognize([],{'a':np.array([[0,0],[1,0]])})[0],'unknown')
    def test_reflection_affine_missing_duplicate_clutter(self):
        rng=np.random.default_rng(9)
        template=rng.uniform(0,1,(10,2));wrong=rng.uniform(0,1,(8,2))
        true=template@np.array([[-1200,100],[80,900]])+[1900,800]
        pts=np.vstack([true[:8],true[:1],rng.uniform(0,3000,(8,2)),wrong[:3]*300+[200,2100]])
        patterns={'target':template,'distractor':wrong}
        name,m,d=recognize(pts,patterns,cap=4000)
        self.assertEqual(name,'target');self.assertGreaterEqual(len(m),8)
        name2,m2,d2=recognize(pts,dict(reversed(list(patterns.items()))),cap=4000)
        self.assertEqual(name,name2);self.assertEqual(m,m2)
    def test_catalog(self):
        patterns=extract_patterns('patterns')
        self.assertEqual(len(patterns),48)
        self.assertTrue(all(len(p)>=2 for p in patterns.values()))

if __name__=='__main__':unittest.main()

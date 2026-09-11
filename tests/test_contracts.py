import unittest
from constellation.contracts import ScenePrediction as P, evaluate

class ScorerTests(unittest.TestCase):
    def score(self,p,t): return evaluate({'s':p},{'s':t})['scenes']['s']
    def test_perfect_and_wrong_class(self):
        t=P([(100,100,1),(300,300,0),None],'a')
        self.assertEqual(self.score(t,t)['score'],1)
        self.assertAlmostEqual(self.score(P(t.patches,'b'),t)['score'],.7)
    def test_ramp(self):
        t=P([(100,100,1),None],'a')
        for delta,r in [(12,1),(24,.5),(36,0)]:
            m=self.score(P([(100+delta,100,0),None],'a'),t)
            self.assertEqual(m['localization'],r); self.assertEqual(m['recovery'],r)
    def test_duplicate_and_extra(self):
        t=P([(0,0,1),(100,100,1),None],'a')
        self.assertEqual(self.score(P([(0,0,1),(0,0,1),None],'a'),t)['recovery'],.5)
        self.assertEqual(self.score(P([(0,0,0),(100,100,0),(500,500,0)],'a'),t)['recovery'],1)
    def test_greedy_not_optimal(self):
        t=P([(0,0,1),(25,0,1)],'a')
        p=P([(10,0,0),(-11,0,0)],'a')
        self.assertEqual(self.score(p,t)['recovery'],.5)

if __name__=='__main__': unittest.main()

class SerializationTests(unittest.TestCase):
    def test_order_padding_and_repeated_queries(self):
        import tempfile,csv
        from pathlib import Path
        from constellation.contracts import write_submission,read_truth
        with tempfile.TemporaryDirectory() as folder:
            template=Path(folder)/'sample.csv';out=Path(folder)/'out.csv'
            template.write_text('Id,n_patches,patch_03,patch_01,patch_02,constellation\nrenamed,2,-1,-1,-1,unknown\n')
            pred={'renamed':P([(10,20,1),(10,20,1)],'foo')}
            write_submission(pred,template,out)
            self.assertEqual(read_truth(out)['renamed'].patches,pred['renamed'].patches)
            with out.open() as f:row=next(csv.DictReader(f))
            self.assertEqual(row['patch_03'],'-1')

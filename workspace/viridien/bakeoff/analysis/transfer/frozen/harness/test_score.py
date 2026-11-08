import unittest
from .score import score

def row(a,b,label,cluster=0,**kw):
    return dict(top_y=a,bottom_y=b,lithology=label,cluster_id=cluster,**kw)

class ScoreTests(unittest.TestCase):
    def test_perfect(self):
        gt=[row(0,50,'Shale'),row(50,100,'Sand stone',1)]
        pred=[row(0,50,' SHALE '),row(50,100,'sandstone',1)]
        result=score(pred,gt,(0,100))
        for k in ('score','row_acc','boundary_f1','cluster_ari'): self.assertEqual(result[k],1)

    def test_shifted(self):
        gt=[row(0,50,'a'),row(50,100,'b',1)]
        near=[row(0,53,'a'),row(53,100,'b',1)]
        far=[row(0,60,'a'),row(60,100,'b',1)]
        self.assertAlmostEqual(score(near,gt,(0,100))['score'],.97)
        self.assertEqual(score(far,gt,(0,100))['boundary_f1'],0)

    def test_blanks(self):
        gt=[row(20,40,'a')]
        self.assertEqual(score(gt+[row(60,70,'bad')],gt,(0,100))['score'],1)
        self.assertEqual(score([],gt,(0,100))['score'],0)
        self.assertEqual(score([],[],(0,100))['score'],1)

    def test_one_to_one(self):
        gt=[row(0,50,'a'),row(50,100,'b',1)]
        pred=[row(0,48,'a'),row(48,52,'b',1),row(52,100,'b',1)]
        self.assertAlmostEqual(score(pred,gt,(0,100))['boundary_f1'],2/3)

    def test_depth_and_missing_columns(self):
        gt=[row(0,50,'a',bottom_depth=100),row(50,100,'b',1,top_depth=100)]
        pred=[row(0,50,'a',bottom_depth=103),row(50,100,'b',1,top_depth=103)]
        self.assertEqual(score(pred,gt,(0,100))['depth_mae_ft'],3)
        self.assertIsNone(score([row(0,50,'a')],gt,(0,100))['depth_mae_ft'])

    def test_window_and_no_internal_boundaries(self):
        gt=[row(0,100,'a')]
        self.assertEqual(score(gt,gt,(20,80))['score'],1)

    def test_silver_csv_without_depth_columns(self):
        import tempfile
        from pathlib import Path
        from . import io
        with tempfile.NamedTemporaryFile(mode='w',suffix='.csv',dir=io.ROOT/'out'/'_cache',delete=False) as f:
            f.write('top_y,bottom_y,lithology\n0,50,Shale\n50,100,Sandstone\n')
            path=Path(f.name)
        try:
            rows=io.load_rows(path)
            result=score(rows,rows,(0,100))
            self.assertEqual(result['score'],1)
            self.assertIsNone(result['depth_mae_ft'])
        finally:path.unlink()

if __name__=='__main__': unittest.main()

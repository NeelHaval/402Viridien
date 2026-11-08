import json
import tempfile
import unittest
from pathlib import Path

from viridian402.data import load_manifest,sha256
from viridian402.synthetic import generate_demo,generate_synthetic,STRESSES


class SyntheticTests(unittest.TestCase):
    def test_seed_reproducibility_and_lineage(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            pa=generate_demo(a,seed=41,n=2);pb=generate_demo(b,seed=41,n=2)
            self.assertEqual(pa.read_bytes(),pb.read_bytes())
            da=load_manifest(pa);db=load_manifest(pb)
            self.assertEqual([r['image_sha256'] for r in da['records']],[r['image_sha256'] for r in db['records']])
            self.assertTrue(all(r['lineage']['parent_split']==r['split'] for r in da['records']))

    def test_real_crop_stresses_preserve_parent_split(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=generate_demo(root/'parent',n=1)
            # Fixture takes the role of a labelled real crop source for this isolated construction test.
            manifest=json.loads(parent.read_text());record=manifest['records'][0]
            record['source']='human';record['group']='well-1';record['split']='verify';parent.write_text(json.dumps(manifest))
            for stress in STRESSES:
                child=load_manifest(generate_synthetic(parent,root/stress,n=1,height=128,stress=stress))['records'][0]
                self.assertEqual(child['split'],'verify');self.assertEqual(child['group'],'well-1')
                self.assertEqual(child['lineage']['parent_image_sha256'],sha256(root/'parent'/record['image']))
                self.assertEqual(child['lineage']['origin'],'real_interval_crops')

    def test_reject_synthetic_only_parent(self):
        with tempfile.TemporaryDirectory() as d:
            parent=generate_demo(Path(d)/'parent',n=1)
            with self.assertRaisesRegex(ValueError,'nonsynthetic'):
                generate_synthetic(parent,Path(d)/'child')

    def test_parent_bed_crossing_window_is_clipped(self):
        with tempfile.TemporaryDirectory() as d:
            from viridian402.data import load_intervals
            root=Path(d);parent=generate_demo(root/'parent',n=1)
            manifest=json.loads(parent.read_text());record=manifest['records'][0]
            bed=load_intervals(root/'parent'/record['truth'])[0]
            record.update(source='human',window=[bed['top_y']+2,bed['bottom_y']-2])
            parent.write_text(json.dumps(manifest))
            child=load_manifest(generate_synthetic(parent,root/'child',n=1,height=128))['records'][0]
            self.assertEqual(len(child['lineage']['crops']),1)
            crop=child['lineage']['crops'][0]['box']
            self.assertGreaterEqual(crop[1],record['window'][0])
            self.assertLessEqual(crop[3],record['window'][1])


if __name__=='__main__':unittest.main()

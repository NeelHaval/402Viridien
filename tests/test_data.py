import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from viridian402.data import load_manifest, load_intervals, read_rgb, save_intervals, validate_intervals
from viridian402.synthetic import generate_demo
from viridian402.groups import normalize_group


class DataTests(unittest.TestCase):
    def test_cross_task_group_aliases(self):
        for old,new in [('challenge:d-582','well:d582'),('challenge:MC765-P011','well:mc765_p011'),
                        ('challenge:60811405800300','api:6081140580'),('challenge:60811405800500','api:6081140580'),
                        ('challenge:Shell-08852-KE001','document:shell08852ke001'),
                        ('challenge:Callaway 3-33 mudlog to TD.PDF','document:callaway333')]:
            self.assertEqual(normalize_group(old),new)
        with tempfile.TemporaryDirectory() as d:
            path=generate_demo(d,n=1);data=json.loads(path.read_text());record=data['records'][0]
            record['group']='challenge:60811405800300'
            record['lineage']['parent_group']='api:6081140580'
            path.write_text(json.dumps(data))
            normalized=load_manifest(path)['records'][0]
            self.assertEqual(normalized['group'],'api:6081140580')
            self.assertEqual(normalized['source_group_id'],'challenge:60811405800300')
    def test_intervals_reject_nonfinite_fractional_overlap_and_sort(self):
        valid={'top_y':1,'bottom_y':4,'lithology':'Shale','cluster_id':0}
        for invalid in ({**valid,'top_y':1.5},{**valid,'top_y':float('nan')},
                        {**valid,'bottom_y':1},{**valid,'top_depth':float('inf'),'bottom_depth':9}):
            with self.assertRaises(ValueError): validate_intervals([invalid])
        with self.assertRaises(ValueError): validate_intervals([valid,valid])
        with self.assertRaisesRegex(ValueError,'cluster_id'):
            validate_intervals([{'top_y':1,'bottom_y':4,'lithology':'Shale'}])
        with self.assertRaises(ValueError): validate_intervals([{**valid,'top_y':5,'bottom_y':7},valid])
        self.assertEqual(validate_intervals([{**valid,'base_depth':20,'top_depth':10}])[0]['bottom_depth'],20)

    def test_manifest_leakage_partial_truth_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            path=generate_demo(d,n=1); data=json.loads(path.read_text()); record=data['records'][0]
            normalized=load_manifest(path)['records'][0]
            self.assertTrue(Path(normalized['image']).is_absolute())
            self.assertEqual(len(normalized['truth_sha256']),64)
            record.pop('truth_window')
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'truth_window'): load_manifest(path)
            record['truth_window']=record['window']
            duplicate={**record,'id':'duplicate','split':'test','group':'different'}
            duplicate['lineage']={**record['lineage'],'parent_split':'test','parent_group':'different'}
            data['records'].append(duplicate);path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'duplicate image'): load_manifest(path)

    def test_empty_truth_window_is_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            path=generate_demo(d,n=1);data=json.loads(path.read_text());truth=Path(d)/data['records'][0]['truth']
            save_intervals(truth,[])
            with self.assertRaisesRegex(ValueError,'no truth-filled'):load_manifest(path)

    def test_header_bounds_and_window_caps(self):
        with tempfile.TemporaryDirectory() as d:
            path=generate_demo(d,n=1);data=json.loads(path.read_text());record=data['records'][0]
            annotation=Path(d)/record['annotation'];ann=json.loads(annotation.read_text());ann['width']=99
            annotation.write_text(json.dumps(ann))
            with self.assertRaisesRegex(ValueError,'dimensions'):load_manifest(path)
            image=Path(d)/record['image']
            with self.assertRaisesRegex(ValueError,'max_window'):read_rgb(image,(0,0,100,100),max_window_pixels=9)
            with self.assertRaises(MemoryError):read_rgb(image,(0,0,10,10),max_decode_bytes=1)

    def test_native_decode_budget_accounts_for_16bit(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'16.png';Image.fromarray(np.ones((30,30),dtype=np.uint16)).save(path)
            with self.assertRaises(MemoryError):read_rgb(path,(0,0,3,3),max_decode_bytes=1000)


if __name__=='__main__': unittest.main()

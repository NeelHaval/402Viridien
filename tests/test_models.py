import unittest
import tempfile

from viridian402.data import load_manifest,validate_intervals
from viridian402.models import resolve_model
from viridian402.synthetic import generate_demo


class ModelTests(unittest.TestCase):
    def test_baseline_pixels_unknowns_and_supplied_legend(self):
        with tempfile.TemporaryDirectory() as d:
            record=load_manifest(generate_demo(d,n=1))['records'][0]
            request={k:record[k] for k in ('id','image','annotation','window')}
            baseline=resolve_model('builtin:baseline')
            rows=validate_intervals(baseline(request,{}),window=record['window'])
            self.assertGreater(len(rows),2)
            self.assertEqual({r['lithology'] for r in rows},{'Unknown'})
            self.assertGreater(len({r['cluster_id'] for r in rows}),1)
            diagnostic=baseline({**request,'legend':record['legend'],'depth_calibration':record['depth_calibration']},{})
            self.assertTrue(all(r['lithology'] in {'Shale','Siltstone','Sandstone'} for r in diagnostic))
            self.assertTrue(all('top_depth' in r for r in diagnostic))

    def test_plugins_and_invalid_config(self):
        self.assertTrue(callable(resolve_model('viridian402.models:baseline')))
        with self.assertRaises(ValueError):resolve_model('builtin:does-not-exist')


if __name__=='__main__':unittest.main()

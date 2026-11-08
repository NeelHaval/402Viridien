"""Runner integration checks with offline strategy and shared preprocessing stubs."""
import json
from pathlib import Path
import uuid
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from . import io, run


class RunnerArgsTests(unittest.TestCase):
    def test_external_items_truth_override_and_tagged_outputs(self):
        with self.subTest('external evaluation'):
            root = io.ROOT/'out/_tests'/('runner_'+uuid.uuid4().hex)
            root.mkdir(parents=True)
            gt = root/'truth'
            gt.mkdir()
            io.save_rows(gt/'outside.csv', [dict(top_y=0, bottom_y=30, lithology='Rock', cluster_id=0)])
            items = root/'items.json'
            items.write_text(json.dumps([dict(log='outside', y0=0, y1=10, tier='external'),
                                         dict(log='outside', y0=20, y1=30, tier='external')]))
            module = SimpleNamespace(__name__='strategies.test', NAME='Test', PAID_API=None,
                run=lambda log, window, ctx: [dict(top_y=window[0], bottom_y=window[1], lithology='Rock')])
            def open_log(name):
                return io.Log(Image.new('RGB', (10, 30), 'white'),
                              {'interpreted_lithology_column': [0, 0, 10, 30]}, name)
            rendered = []
            def render(log, rows, output, **kw):
                rendered.append((output, kw['gt_rows']))
            calib = SimpleNamespace(to_depth=lambda y: y)
            with patch.object(io, 'ROOT', root), patch.object(io, 'open_log', side_effect=open_log), \
                 patch.object(run, '_strategy', return_value=module), \
                 patch.object(run.legend, 'parse', return_value=[]), \
                 patch.object(run.depth, 'calibrate', return_value=calib), \
                 patch.object(run, '_truth', side_effect=AssertionError('--gt must be overridden')), \
                 patch.object(run.viz, 'render', side_effect=render):
                run.main(['--strategy', 'test', '--items', str(items), '--gt-dir', str(gt),
                          '--gt', 'nonexistent', '--out-tag', 'external'])
            out = root/'out/test/external'
            data = json.loads((out/'metrics.json').read_text())
            self.assertEqual(data['mean']['row_acc'], 1)
            self.assertEqual(data['mean']['n_scored'], 2)
            self.assertEqual(len(list(out.glob('*.csv'))), 2)
            self.assertEqual(len({p for p, rows in rendered}), 2)
            self.assertTrue(all(p.parent == out and rows[0]['lithology'] == 'Rock' for p, rows in rendered))
            self.assertFalse((root/'out/test/metrics.json').exists())
            with patch.object(io, 'ROOT', root), patch.object(io, 'open_log', side_effect=open_log), \
                 patch.object(run, '_strategy', return_value=module), \
                 patch.object(run.legend, 'parse', side_effect=ValueError('OCR failed')):
                run.main(['--strategy', 'test', '--items', str(items), '--gt-dir', str(gt),
                          '--out-tag', 'failed_preprocessing'])
            failed = json.loads((root/'out/test/failed_preprocessing/metrics.json').read_text())
            self.assertEqual(failed['mean']['crashes'], 2)
            self.assertEqual(failed['mean']['n_scored'], 2)
            self.assertEqual(failed['mean']['row_acc'], 0)

    def test_tag_cannot_escape_strategy_directory(self):
        for tag in ('..', '../outside', 'a/b', 'a\\b'):
            with self.assertRaises(SystemExit):
                run.main(['--strategy', '_dummy', '--out-tag', tag])


if __name__ == '__main__':unittest.main()

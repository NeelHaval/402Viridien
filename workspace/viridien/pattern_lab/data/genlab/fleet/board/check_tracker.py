"""Tracker and Progress regressions; all fixtures live inside the board folder."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


t = module('tracker_check', HERE.parents[1] / 'tracker.py')
b = module('board_check', HERE / 'board.py')


class TrackerChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tracker-test-', dir=HERE)
        self.root = Path(self.temp.name).resolve()
        self.g = self.root / 'data/genlab'
        self.g.mkdir(parents=True)
        self.patches = [patch.object(t, 'ROOT', self.root), patch.object(t, 'G', self.g),
                        patch.object(t, 'HIST', self.g / 'tracker.jsonl')]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        assert HERE.resolve() in self.root.parents
        self.temp.cleanup()

    def write(self, relative, obj):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding='utf-8')
        return path

    def test_numeric_latest_pending_and_version_specific_audit(self):
        self.write('data/genlab/datasets/latest.json', dict(version='v9'))
        self.write('data/genlab/datasets/v9/eval.json', dict(status='passed'))
        ev = dict(closed_set_within_document={'matcher': {'augmented': dict(accuracy=20, chance=10, n=12)}},
                  discriminator={'augmented': dict(accuracy=60, n=24)})
        self.write('data/genlab/datasets/v10/eval.json', ev)
        report = self.g / 'datasets/diag/v9-report.md'
        report.parent.mkdir()
        report.write_text('- audit_dataset.py v9: PASS zero test samples', encoding='utf-8')
        folder, _, promoted, gates = t.dataset_evidence()
        self.assertEqual((folder.name, promoted), ('v10', 'v9'))
        self.assertEqual([g['status'] for g in gates], ['fail', 'pass', 'pending'])
        ev['closed_set_within_document']['matcher']['augmented']['accuracy'] = 20.01
        self.write('data/genlab/datasets/v10/eval.json', ev)
        self.assertEqual(t.dataset_evidence()[3][0]['status'], 'pass')
        ev.pop('closed_set_within_document')
        self.write('data/genlab/datasets/v10/eval.json', ev)
        self.assertEqual(t.dataset_evidence()[3][0]['status'], 'pending')

    def test_no_samples_never_pass(self):
        g = t.metric_gate('No evidence', 0, '0 leaks', 0, True)
        self.assertEqual((g['status'], g['score']), ('pending', 0))
        self.assertFalse(t.number(float('nan')))

    def test_markdown_fallback_does_not_confuse_cross_log_task(self):
        folder = self.g / 'datasets/v1'
        folder.mkdir(parents=True)
        (folder / 'eval.md').write_text('''# v1: closed-set within-document legend recognition
| Model / variant | Fake n | Real n | Correct | Top-1 % | Chance % |
| matcher / augmented | 100 | 12 | 8 | 66.667 | 10 |
## Cross-log name generalisation
| Model / variant | Fake n | Real n | Correct | Top-1 % | Chance % |
| matcher / augmented | 100 | 12 | 0 | 0 | 1 |
## Real versus fake discriminator
| Variant / source | Train n | Held real n | Held fake n | Accuracy % (50 is chance) |
| augmented | 100 | 12 | 12 | 60 |
''', encoding='utf-8')
        gates = t.dataset_evidence()[3]
        self.assertEqual([g['status'] for g in gates], ['pass', 'pass', 'pending'])
        self.assertEqual(gates[0]['value'], '66.667%')
        self.assertEqual(gates[1]['n'], 24)

    def test_owner_row_basis_unions_overlaps_and_excludes_test(self):
        def row(top, bottom, lid=1, split='train'):
            return dict(document_id='a', box_xyxy=[1, top, 9, bottom], split=split,
                        accepted=True, targets=[dict(legend_id=lid)])
        result = t.owner_row_basis([row(10, 30), row(20, 40), row(40, 60, 2), row(0, 200, split='test')],
                                   [dict(doc_id='a', legend_id=1)])
        self.assertEqual((result['rows'], result['covered_rows'], result['boxes']), (50, 30, 3))

    def test_old_history_and_first_label_baseline(self):
        h = self.g / 'tracker.jsonl'
        h.write_text('{"at":1,"pct":{"overall":0.99},"labels":10}\ninvalid\n{"at":2,"labels":20}\n{', encoding='utf-8')
        self.assertEqual(len(t.history()), 2)
        self.write('data/normalized/current.json', dict(records=[dict(split='train')] * 12))
        with patch.object(t, 'get', return_value=None), patch.object(t, 'jobs', return_value=None):
            r = t.reading()
        self.assertEqual((r['labels_baseline'], r['labels_growth']), (10, 2))
        self.assertEqual(r['pct']['overall'], 0)
        self.assertFalse(r['sources']['jobs'])

    def test_good_uses_quality_and_counts_unverified(self):
        board = dict(training_total=4, training_counts={'In the lab': 4},
                     documents=[dict(doc_id='doc', training_allowed=True)], entries=[])
        lab = dict(cards=[dict(doc_id='doc', realism=dict(crops=2, median=.95)),
                          dict(doc_id='doc', realism=dict(crops=1, median=.8)),
                          dict(doc_id='doc', real=[[1, 2, 3, 4]], realism=None),
                          dict(doc_id='doc', realism=None)])
        def get(url):
            return board if ':8782' in url else lab if ':8781' in url else None
        with patch.object(t, 'get', side_effect=get), patch.object(t, 'jobs', return_value={}):
            r = t.reading()
        legend = r['streams']['legend_training']
        self.assertEqual(legend['built'], 1)
        self.assertEqual(legend['good'], .625)  # mean(100% coverage, 25% scored pass)
        self.assertEqual((legend['gates'][1]['unverified'], legend['gates'][1]['scored']), (1, 2))
        self.assertAlmostEqual(r['pct']['overall'], sum(r['pct'][k]*w for k, w in t.WEIGHTS.items()) / sum(t.WEIGHTS.values()))

    def test_assembler_ignores_prefill_and_tiny_fixtures(self):
        fidelity = dict(validation=dict(metrics=dict(a=dict(status='pass', value=.01)),
                                       noisy_prefill_diagnostic=dict(metrics=dict(a=dict(status='fail', value=1)))),
                        test=dict(metrics=dict(b=dict(status='warn', value=None))))
        self.write('data/genlab/assemble/bench/runs/starter/fidelity.json', fidelity)
        self.write('data/genlab/assemble/bench/runs/tiny/realism.json', dict(balanced_accuracy=.5, n_real=100, n_synthetic=100))
        self.write('data/genlab/assemble/bench/runs/starter/detector.json', dict(human_all=dict(rows=40, accuracy_gap=-.25)))
        gates, _, _ = t.assembler_evidence()
        self.assertEqual(gates[0]['value'], {'pass': 1, 'warn': 1})
        self.assertEqual(gates[0]['score'], .5)
        self.assertEqual(gates[1]['status'], 'pending')
        self.assertEqual(gates[2]['status'], 'fail')

    def test_label_next_mapping_priority_and_alternative_guess(self):
        data = self.root / 'data/genlab'
        data.mkdir(parents=True, exist_ok=True)
        self.write('data/genlab/candidates/a.json', dict(doc_id='a', name='Example.PDF'))
        prefill = self.root / 'bakeoff/labeltool/data/prefill.jsonl'
        prefill.parent.mkdir(parents=True)
        prefill.write_text(json.dumps(dict(log='Example', top_y=123, bottom_y=245,
            suggestion='Clay', suggestions=[dict(label='Sand', tier='high', confidence=.99, source='machine')]))+'\n', encoding='utf-8')
        def entry(lid, label, note):
            return dict(doc_id='a', doc_name='Example.PDF', legend_id=lid, label=label,
                        training_allowed=True, state='Failed', note=note)
        board = dict(entries=[entry(1, 'Sand', 'One course, no vertical recurrence'), entry(2, 'Clay', 'Wrong hue')])
        lab = dict(coverage=dict(labels=[dict(label='Sand', all_entries=2, estimated_share_of_column_rows=.2),
                                        dict(label='Clay', all_entries=1, estimated_share_of_column_rows=.5)]))
        with patch.object(b, 'DATA', data), patch.object(b, 'ROOT', self.root / 'pattern_lab'):
            result = b.label_next(board, lab)
        self.assertEqual(result[0]['label'], 'Sand')
        self.assertEqual(result[0]['estimated_share'], .1)
        self.assertEqual(result[0]['interval']['top_y'], 123)
        self.assertTrue(result[0]['interval']['alternative'])

    def test_read_only_report_containment(self):
        data = self.root / 'data/genlab'
        self.write('data/genlab/assemble/stats.json', {})
        report = data / 'assemble/report.html'
        report.write_text('report', encoding='utf-8')
        with patch.object(b, 'DATA', data):
            self.assertEqual(b.assembler_file('/assembler/report.html'), report)
            for route in ('/assembler/../tracker.py', '/assembler/%2e%2e/tracker.py', '/assembler/stats.json'):
                with self.assertRaises(ValueError):
                    b.assembler_file(route)

    def test_tracker_background_does_not_block_progress_reads(self):
        folder = self.root / 'board'
        folder.mkdir()
        (self.g / 'tracker.jsonl').write_text('{"schema":2,"at":1}\n', encoding='utf-8')
        started, release = threading.Event(), threading.Event()
        class EndLoop(BaseException):
            pass
        def sleep(seconds):
            if seconds == 300:
                raise EndLoop()
        def run(*args, **kwargs):
            started.set()
            release.wait(3)
            return SimpleNamespace(returncode=0, stdout='{}')
        with patch.object(b, 'HERE', folder), patch.object(b, 'DATA', self.g), \
             patch.object(b.time, 'sleep', side_effect=sleep), patch.object(b.subprocess, 'run', side_effect=run) as command:
            board = b.Board()
            def worker():
                try:
                    board.track()
                except EndLoop:
                    pass
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(started.wait(2))
                begin = time.monotonic()
                for _ in range(30):
                    board.get_progress()
                self.assertLess(time.monotonic()-begin, .2)
                self.assertEqual(command.call_count, 1)
            finally:
                release.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TrackerChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(('PASS' if result.wasSuccessful() else 'FAIL') + ': tracker quality, legacy history, label ranking and report containment')
    sys.exit(0 if result.wasSuccessful() else 1)

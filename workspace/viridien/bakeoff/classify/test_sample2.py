"""Offline only: fake readers and classifiers, no network or child processes."""
import json
from pathlib import Path
import shutil
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from classify import sample2 as s


@contextmanager
def workspace_temp():
    # tempfile's private mode=0700 directory gets an unusable ACL in this sandbox.
    root = s.ROOT / 'classify' / 'out' / 'test_scratch' / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        assert root.resolve().is_relative_to((s.ROOT / 'classify' / 'out' / 'test_scratch').resolve())
        shutil.rmtree(root)


def fixture(families=7, logs=4):
    rows, boxes = [], {}
    for family in range(families):
        for log in range(logs):
            name = f'family{family}_well{log:02d}'
            boxes[name] = [0, 0, 100, 100000]
            for tier in s.TIERS:
                for height in (16, 48, 144):
                    for i in range(12):
                        identifier = f'{name}_{tier}_{height}_{i}'
                        rows.append(dict(id=identifier, log=name, family=f'F{family}', tier=tier,
                                         top_y=100+i*200, bottom_y=100+i*200+height))
    return rows, boxes


class SamplingTests(unittest.TestCase):
    def test_counts_tiers_heights_and_log_coverage(self):
        rows, boxes = fixture()
        selected, audit = s.select(rows, boxes)
        self.assertEqual(len(selected), 420)
        for family in audit:
            self.assertEqual(family['n'], 60)
            self.assertEqual(family['logs_covered'], 4)
            self.assertEqual(set(family['per_log'].values()), {15})
            self.assertEqual(family['tiers'], dict(low=20, medium=20, high=20))
            self.assertEqual(set(family['heights'].values()), {20})
        for group in s.sheets(selected):
            self.assertLessEqual(len(group), 8)
            self.assertEqual(len({r['log'] for r in group}), 1)

    def test_determinism_independent_of_queue_order(self):
        rows, boxes = fixture(families=1)
        a = s.select(rows, boxes)
        self.assertEqual(a, s.select(list(reversed(rows)), boxes))
        self.assertNotEqual(a[0], s.select(rows, boxes, seed=123)[0])

    def test_three_logs_shortfall_and_missing_tiers(self):
        rows, boxes = fixture(families=1, logs=3)
        rows = [r for r in rows if r['tier'] == 'low']
        selected, audit = s.select(rows, boxes)
        self.assertEqual(len(selected), 45)
        self.assertEqual(audit[0]['shortfall'], 15)
        self.assertEqual(audit[0]['tiers'], {'low': 45})

    def test_coverage_priority_including_sparse_log(self):
        rows, boxes = fixture(families=1, logs=18)
        rows = [r for r in rows if r['log'] != 'family0_well17'] + [rows[-1]]
        selected, audit = s.select(rows, boxes)
        self.assertEqual(len(selected), 60)
        self.assertEqual(audit[0]['logs_covered'], 18)
        self.assertEqual(audit[0]['tiers'], dict(low=20, medium=20, high=20))
        self.assertLessEqual(max(audit[0]['per_log'].values()), 15)

    def test_exclusions_and_exact_margin(self):
        def row(identifier, log, a, b):
            return dict(id=identifier, log=log, family='F', tier='low', top_y=a, bottom_y=b)
        rows = [row('short', 'l', 100, 115), row('top', 'l', 29, 60),
                row('bottom', 'l', 900, 971), row('gold', 'log06', 100, 140),
                row('exact_top', 'l', 30, 46), row('exact_bottom', 'l', 954, 970)]
        selected, _ = s.select(rows, {'l': [0, 0, 20, 1000], 'log06': [0, 0, 20, 1000]})
        self.assertEqual({r['id'] for r in selected}, {'exact_top', 'exact_bottom'})

    def test_agreement_filter_and_denominators(self):
        rows = [dict(patch_id=str(i), family='F', log='log') for i in range(6)]
        answers = {'0': dict(S='Shale', T='Shale'), '1': dict(S='unknown', T='unknown'),
                   '2': dict(S='Shale', T='Sand'), '3': dict(S='Absent', T='Absent'),
                   '4': dict(S='Shale'), '5': dict(S='shale', T='Shale')}
        accepted, audit, stats = s.agreement_filter(rows, answers, {'log': ['Shale', 'Sand']})
        self.assertEqual([r['patch_id'] for r in accepted], ['0'])
        self.assertEqual(stats[0]['n_paired'], 5)
        self.assertEqual(stats[0]['agreement_rate'], .2)
        self.assertEqual(stats[0]['logs_covered'], 1)

    def test_terra_subset_exact_and_deterministic(self):
        rows, boxes = fixture()
        selected, _ = s.select(rows, boxes)
        ids = s.terra_subset(selected)
        self.assertEqual(len(ids), 105)
        self.assertEqual(ids, s.terra_subset(list(reversed(selected))))
        for family in range(7):
            self.assertEqual(sum(r['patch_id'] in ids for r in selected if r['family'] == f'F{family}'), 15)

    def test_native_sheet_context_brackets_and_legend(self):
        import numpy as np
        from PIL import Image
        from harness import io
        rows = [dict(patch_id='p', log='L', top_y=100, bottom_y=160)]
        log = SimpleNamespace(name='L', boxes={'interpreted_lithology_column': [0, 0, 32, 1000]}, close=Mock())
        calls = []
        def crop(log, box, a=None, b=None):
            calls.append((box, a, b))
            return np.full((50, 60, 3) if isinstance(box, str) else (b-a, 32, 3), 123, dtype=np.uint8)
        with workspace_temp() as root, patch.object(s, 'OUT', root), patch.object(io, 'open_log', return_value=log), \
             patch.object(io, 'read_rgb', side_effect=crop), \
             patch.object(s, 'legend_entries', return_value=[SimpleNamespace(label='Shale')]):
            png, prompt, vocab = s.prepare_sheet(rows)
            with Image.open(png) as image:
                self.assertEqual(image.getpixel((24, 50)), (123, 123, 123))
                self.assertEqual(image.getpixel((9, 90)), (255, 0, 0))
                self.assertEqual(image.getpixel((9, 149)), (255, 0, 0))
                self.assertEqual(image.getpixel((300, 30)), (123, 123, 123))
            self.assertIn(([0, 0, 32, 1000], 60, 200), calls)
            self.assertIn('"p"', prompt)
            self.assertEqual(vocab, ['Shale'])
        log.close.assert_called_once()


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = workspace_temp()
        self.root = self.temporary.__enter__()
        self.ledger = self.root / 'calls.json'
        self.png = self.root / 'sheet.png'
        self.png.write_bytes(b'fake image for content hashing')
        self.runner = Mock(return_value=SimpleNamespace(returncode=0, stdout='{"p":"Shale"}', stderr=''))

    def tearDown(self):
        self.temporary.__exit__(None, None, None)

    def reader(self, cap=None):
        return s.Reader(s.CallBudget(self.ledger, cap), self.root / 'cache', self.runner)

    def test_cache_free_even_at_cap_and_model_independence(self):
        reader = self.reader(2)
        s.dump(self.ledger, dict(limit=220, calls=[dict(number=i+1) for i in range(218)]))
        _, hit = reader.invoke(self.png, 'prompt', *s.MODELS['S'], 'key')
        self.assertFalse(hit)
        reader.invoke(self.png, 'prompt', *s.MODELS['T'], 'key')
        _, hit = self.reader(0).invoke(self.png, 'prompt', *s.MODELS['S'], 'key')
        self.assertTrue(hit)
        with self.assertRaises(s.CallCap):
            reader.invoke(self.png, 'new prompt', *s.MODELS['S'], 'key')
        self.assertEqual(self.runner.call_count, 2)
        ledger = s.read(self.ledger)
        self.assertEqual(len(ledger['calls']), 220)
        self.assertTrue(ledger['calls'][-1]['ok'])
        cmd = self.runner.call_args.args[0]
        self.assertIn('read-only', cmd)
        self.assertIn('features.shell_tool=false', cmd)
        self.assertIn('codex.js', ' '.join(cmd))

    def test_failed_launch_cached_no_retry_and_no_secrets(self):
        self.runner.side_effect = PermissionError('api_key=do-not-store-this-secret')
        with self.assertRaises(s.ReaderFailure):
            self.reader().invoke(self.png, 'prompt', *s.MODELS['S'], 'key')
        with self.assertRaises(s.ReaderFailure):
            self.reader().invoke(self.png, 'prompt', *s.MODELS['S'], 'key')
        self.assertEqual(self.runner.call_count, 1)
        self.assertEqual(len(s.read(self.ledger)['calls']), 1)
        for file in self.root.rglob('*.json'):
            self.assertNotIn('do-not-store', file.read_text())

    def test_threaded_reservations_never_exceed_cap(self):
        s.dump(self.ledger, dict(limit=220, calls=[dict(number=i+1) for i in range(217)]))
        def reserve(i):
            try:
                return s.CallBudget(self.ledger).reserve(str(i), 'model', 'effort', str(i))['number']
            except s.CallCap:
                return None
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(reserve, range(12)))
        self.assertEqual(sorted(v for v in results if v), [218, 219, 220])
        self.assertEqual(len(s.read(self.ledger)['calls']), 220)

    def test_label_stops_and_saves_partial_pair_at_call_cap(self):
        rows = [dict(patch_id='p', family='F', log='log')]
        def make_sheet(group):
            return self.png, 'prompt', ['Shale']
        with patch.object(s, 'OUT', self.root), patch.object(s, 'REFERENCE', self.root / 'refs.jsonl'):
            stats = s.label(rows, self.reader(1), workers=1, make_sheet=make_sheet)
            self.assertEqual(stats[0]['n_paired'], 0)
            self.assertEqual(s.jsonl(s.REFERENCE), [])
            self.assertIn('cap', s.read(self.root / 'sample2_status.json')['status'])
            # S is cached; a second invocation only spends one T call.
            stats = s.label(rows, self.reader(1), workers=1, make_sheet=make_sheet)
            self.assertEqual(stats[0]['n_agreed'], 1)
            self.assertEqual(len(s.jsonl(s.REFERENCE)), 1)
        self.assertEqual(self.runner.call_count, 2)

    def test_dry_run_never_calls_reader_or_writes_sample(self):
        rows, boxes = fixture(families=1)
        with patch.object(s, 'jsonl', return_value=rows), patch.object(s, 'column_boxes', return_value=boxes), \
             patch.object(s, 'LEDGER', self.ledger), patch.object(s, 'write_jsonl') as write, \
             patch.object(s.Reader, 'invoke', side_effect=AssertionError('model call')), patch('builtins.print'):
            s.main(['--dry-run', '--label', '--bench'])
        write.assert_not_called()

    def test_shared_benchmark_failure_is_counted_once(self):
        from classify.api import APIs
        from PIL import Image
        self.runner.side_effect = PermissionError('blocked')
        apis = object.__new__(APIs)
        apis.sample2_reader = self.reader()
        apis.events = []
        apis.save_event = lambda event: apis.events.append(event)
        for _ in range(2):
            record, event = apis.sample2_terra_request(Image.new('RGB', (10, 10)), 'prompt',
                self.root / 'bench.png', dict(key='key', classifier='terra'))
        self.assertEqual(sum(e['calls'] for e in apis.events), 1)
        self.assertEqual(len(s.read(self.ledger)['calls']), 1)
        self.assertEqual(self.runner.call_count, 1)
        self.assertTrue(event['cache_hit'])

    def test_original_labeller_shares_the_same_cap(self):
        from labelling import llm_label
        s.dump(self.ledger, dict(limit=220, calls=[dict(number=i+1) for i in range(219)]))
        with patch.object(llm_label, 'OUT', self.root), patch.object(llm_label, 'BASE', self.root), \
             patch.object(llm_label, 'BUDGET', s.CallBudget(self.ledger)), \
             patch.object(llm_label.subprocess, 'run', self.runner), patch('builtins.print'):
            answer = llm_label.invoke(self.png, 'prompt', 'S', 'old_labeller')
            self.assertEqual(answer, {'p': 'Shale'})
            self.assertEqual(llm_label.invoke(self.png, 'prompt', 'S', 'old_labeller'), answer)
            with self.assertRaises(s.CallCap):
                self.reader().invoke(self.png, 'new prompt', *s.MODELS['T'], 'sample2')
        self.assertEqual(self.runner.call_count, 1)
        self.assertEqual(len(s.read(self.ledger)['calls']), 220)


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from classify import bench
        cls.b = bench

    def test_wilson_and_winner_tie_rule(self):
        import pandas as pd
        low, high = self.b.wilson(5, 10)
        self.assertAlmostEqual(low, .2365930905)
        self.assertAlmostEqual(high, .7634069095)
        rows = []
        for family, score in [('win', .5), ('tie', .95)]:
            for name, p in [('a', 1.), ('b', score)]:
                lo, hi = self.b.wilson(round(p*60), 60)
                rows.append(dict(source='SAMPLE2', family=family, classifier=name, n_patches=60,
                                 logs_covered=4, n_scored=60, top1=p, top1_ci_low=lo, top1_ci_high=hi))
        choices = {r['family']: r for r in self.b.family_choices(pd.DataFrame(rows))}
        self.assertEqual(choices['win']['classifier'], 'a')
        self.assertEqual(choices['tie']['classifier'], 'tie')

    def test_benchmark_terra_only_subsample_and_cache(self):
        import numpy as np
        b = self.b
        with workspace_temp() as root:
            patches = [dict(patch_id=f'p{i}', lithology='Shale', original_truth='Shale', top_y=i*30,
                           bottom_y=i*30+20, patch_y0=i*30, patch_y1=i*30+20, path='mock') for i in range(20)]
            case = dict(info=dict(source='SAMPLE2', family='F', key='mock', log='L', truth_sha256='abc'),
                        patches=patches, vocab=['Shale', 'Sand'])
            def predictions(case, *args):
                return [np.array([.8, .2]) for _ in case['patches']], [dict(seconds=.1, calls=.125, status='ok') for _ in case['patches']]
            apis = SimpleNamespace(retry_failed=False, sample2_terra_ids={f'p{i}' for i in range(0, 20, 4)},
                                   terra=Mock(side_effect=predictions), jev=Mock(side_effect=predictions))
            local = Mock(side_effect=lambda case, name: ([np.array([.8, .2]) for _ in case['patches']], {}))
            with patch.object(b, 'OUT', root), patch.object(b, 'classify', local), patch.object(b, 'release'), patch('builtins.print'):
                rows = b.run_case(case, apis)
                cached = b.run_case(case, apis)
            self.assertEqual(len([r for r in rows if r['classifier'] == 'terra']), 5)
            self.assertEqual(len(rows), 20*7+5)
            self.assertEqual(len(cached), len(rows))
            apis.terra.assert_called_once()
            summary = b.summarize(rows)
            terra = summary[summary.classifier == 'terra'].iloc[0]
            self.assertEqual(terra.n_patches, 5)
            self.assertEqual(terra.logs_covered, 1)
            self.assertLess(terra.top1_ci_low, 1)
            self.assertEqual(terra.top1, 1)
            # A quota skip made no blocked request and is eligible to resume.
            cachepath = root / 'prediction_cache' / 'mock_terra.json'
            cache = s.read(cachepath)
            for row in cache['rows']:
                row.update(predicted='', status='Shared 220-call cap or --max-calls reached')
            s.dump(cachepath, cache)
            apis.terra.reset_mock()
            with patch.object(b, 'OUT', root), patch.object(b, 'classify', local), patch('builtins.print'):
                resumed = b.run_case(case, apis)
            apis.terra.assert_called_once()
            self.assertTrue(all(r['predicted'] for r in resumed if r['classifier'] == 'terra'))

    def test_report_and_outputs_preserve_old_sources_with_sample2_first(self):
        import pandas as pd
        from classify import report
        b = self.b
        rows = []
        for source in ('TRUE', 'CONSENSUS', 'SAMPLE2'):
            for i, correct in enumerate((True, False)):
                rows.append(dict(source=source, family='F', classifier='fft_hatch', patch_id=source+str(i),
                    key=source, log=source+'log', truth='Shale', original_truth='Shale',
                    predicted='Shale' if correct else 'Sand', correct=correct, top3=True, rank_of_truth=1 if correct else 2,
                    margin=.2, seconds=.1, calls=0, peak_rss_gb=1, peak_gpu_gb=0))
        with workspace_temp() as root:
            manifest = root / 'manifest.json'
            s.dump(manifest, {'old': dict(family='F')})
            s.dump(root / 'sample2_plan.json', dict(families=[dict(family='F', n=60, logs_covered=4)]))
            apis = SimpleNamespace(events=[])
            with patch.object(b, 'OUT', root), patch.object(b, 'MANIFEST', manifest), \
                 patch.object(report, 'OUT', root), patch.object(report, 'HERE', root), \
                 patch.object(report, 'MANIFEST', manifest), patch('builtins.print'):
                b.outputs(rows, [], apis, dict(stage='offline_mock'))
            summary = pd.read_csv(root / 'summary.csv')
            self.assertEqual(summary.iloc[0].source, 'SAMPLE2')
            for source in ('TRUE', 'CONSENSUS', 'SAMPLE2'):
                cell = summary[(summary.source == source) & (summary.classifier == 'fft_hatch')].iloc[0]
                self.assertEqual(cell.n_patches, 2)
                self.assertEqual(cell.top1, .5)
                self.assertEqual(cell.logs_covered, 1)
                self.assertLess(cell.top1_ci_low, .5)
            page = (root / 'report.html').read_text(encoding='utf-8')
            self.assertLess(page.index('<h2>SAMPLE2</h2>'), page.index('<h2>TRUE</h2>'))
            self.assertIn('95% CI', page)
            self.assertIn('Which classifier per family?', page)


if __name__ == '__main__':
    unittest.main()

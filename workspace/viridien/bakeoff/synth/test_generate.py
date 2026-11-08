"""Offline integration tests. Run: python -B -m unittest synth.test_generate -v"""
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from harness import io, score
from . import io_synth
from .bench_synth import classifier_case, confidence, discover, prediction_rows
from .generate import HERE, allocate, generate
from .mock import build_mock
from .nuisances import PROBABILITIES, apply_nuisances, draw_nuisances
from .offline import Replay, offline_guard
from .statistics import read_json, split_sources
from .textures import fit_width, minimum_cut, quilt


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = HERE / 'out/tests'
        test_root.mkdir(parents=True, exist_ok=True)
        # pathlib.mkdir preserves inherited workspace ACLs on this Windows host;
        # tempfile.mkdtemp's private-directory ACL is unreadable in the sandbox.
        cls.temporary = (test_root / ('generate_' + uuid.uuid4().hex[:12])).resolve()
        cls.temporary.mkdir()
        if not cls.temporary.is_relative_to(HERE):
            raise RuntimeError('Invalid test workspace')
        cls.catalog, cls.approved = build_mock(cls.temporary / 'mock')
        cls.first = cls.temporary / 'first'
        cls.second = cls.temporary / 'second'
        # Both real source logs must be exercised; each source belongs to only
        # one pool. Exact repeated generation is an end-to-end determinism check.
        for output in (cls.first, cls.second):
            generate(cls.catalog, cls.approved, output, n_logs=2, seed=51,
                     min_height=2000, max_height=2050, clean_fraction=0)
        cls.names = io_synth.list_logs(cls.first, 'all')

    @classmethod
    def tearDownClass(cls):
        target = cls.temporary.resolve()
        if target.is_relative_to((HERE / 'out/tests').resolve()):
            shutil.rmtree(target)

    def test_determinism_by_seed(self):
        for name in self.names:
            first_meta = read_json(self.first / name / 'meta.json')
            self.assertEqual(first_meta, read_json(self.second / name / 'meta.json'))
            for filename in ('truth.csv', 'boxes.json', first_meta['image_file'], 'image_clean.png'):
                self.assertEqual((self.first / name / filename).read_bytes(),
                                 (self.second / name / filename).read_bytes())

    def test_truth_exactly_tiles_column(self):
        for name in self.names:
            meta = read_json(self.first / name / 'meta.json')
            rows = io_synth.load_rows(self.first / name / 'truth.csv')
            self.assertEqual(rows[0]['top_y'], 0)
            self.assertEqual(rows[-1]['bottom_y'], meta['height'])
            for row in rows:
                self.assertGreaterEqual(row['bottom_y'] - row['top_y'], 6)
                self.assertAlmostEqual(row['top_depth'], round(meta['calibration']['a'] * row['top_y'] +
                                                             meta['calibration']['b'], 1))
            for left, right in zip(rows, rows[1:]):
                self.assertEqual(left['bottom_y'], right['top_y'])
                if left['lithology'] == right['lithology']:
                    self.assertIn(right['top_y'], meta['nuisances']['boundary_lines'].get('rows', []))

    def test_no_anisotropic_stretch(self):
        for name in self.names:
            for bed in read_json(self.first / name / 'meta.json')['beds']:
                for source in bed['source_crops']:
                    self.assertEqual(source['scale_x'], source['scale_y'])
                    ratio = source['fitted_size'][0] / source['source_size'][0]
                    if abs(ratio - 1) <= .15:
                        self.assertEqual(source['scale_x'], 1)
                        self.assertFalse(source['isotropic_exception'])
                    else:
                        self.assertTrue(source['isotropic_exception'])

    def test_crop_tile_width_and_exception(self):
        rgb = np.random.default_rng(1).integers(40, 250, (75, 100, 3), dtype=np.uint8)
        for width in (85, 90, 110, 115, 150):
            fitted, meta = fit_width(rgb, width, np.random.default_rng(2))
            self.assertEqual(fitted.shape[1], width)
            self.assertEqual(meta['scale_x'], meta['scale_y'])
            if width == 150:
                self.assertEqual(fitted.shape[0], round(75 * 1.5))
            else:
                self.assertEqual(fitted.shape[0], 75)

    def test_source_rails_do_not_phase_into_interior(self):
        rgb = np.full((40, 100, 3), 255, np.uint8)
        rgb[:, :5] = 0
        rgb[:, -4:] = 0
        fitted, transform = fit_width(rgb, 100, np.random.default_rng(5))
        self.assertTrue(np.all(fitted == 255))
        self.assertEqual(transform['rail_trim_left'], 5)
        self.assertEqual(transform['rail_trim_right'], 4)

    def test_holdout_disjoint_by_source_log(self):
        split = read_json(self.first / 'split.json')
        train, test = set(split['train_logs']), set(split['test_logs'])
        self.assertTrue(train)
        self.assertTrue(test)
        self.assertFalse(train & test)
        for pools in split['patterns'].values():
            self.assertFalse({c['log'] for c in pools['train']} & {c['log'] for c in pools['test']})
        for name in self.names:
            meta = read_json(self.first / name / 'meta.json')
            allowed = train if meta['pool'] == 'train' else test
            self.assertIn(meta['donor_log'], allowed)
            for bed in meta['beds']:
                self.assertTrue(all(c['log'] in allowed for c in bed['source_crops']))
        self.assertEqual(read_json(self.first / 'fitted_models.json')['excluded_source_logs'], split['test_logs'])

    def test_global_holdout_across_patterns_and_singletons(self):
        approved = {'one': [dict(log='a'), dict(log='b')],
                    'two': [dict(log='b'), dict(log='c')], 'singleton': [dict(log='s')]}
        split = split_sources(approved, 1)
        self.assertFalse(set(split['train_logs']) & set(split['test_logs']))
        self.assertIn('s', split['train_logs'])
        self.assertIn('singleton', split['unavailable_test_patterns'])

    def test_approved_pattern_coverage(self):
        counts = read_json(self.first / 'run.json')['pattern_beds']
        self.assertTrue(all(n >= 5 for n in counts.values()))

    def test_adapter_roundtrip_through_harness_score(self):
        original_open, original_root = io.open_log, io.ROOT
        with io_synth.patched(self.first, 'test', runtime=self.temporary / 'runtime'):
            names = io.list_logs()
            self.assertEqual(len(names), 1)
            log = io.open_log(names[0])
            try:
                rgb = io.read_rgb(log, 'interpreted_lithology_column')
                rows = io.load_rows(self.first / names[0] / 'truth.csv')
                result = score.score(rows, rows, (0, rgb.shape[0]))
                self.assertEqual(result['row_acc'], 1)
                self.assertEqual(result['boundary_f1'], 1)
                self.assertEqual(result['cluster_ari'], 1)
                self.assertEqual(result['depth_mae_ft'], 0)
            finally:
                log.close()
        self.assertIs(io.open_log, original_open)
        self.assertEqual(io.ROOT, original_root)

    def test_existing_strategy_and_harness_runner_interface(self):
        from harness import legend, depth, run
        with io_synth.patched(self.first, 'test', runtime=self.temporary / 'runtime'):
            name = io.list_logs()[0]
            log = io.open_log(name)
            try:
                entries = legend.parse(log)
                calib = depth.calibrate(log)
                self.assertTrue(entries)
                self.assertEqual(calib.method, 'synthetic-linear-known')
                module = run._strategy('s01')
                window = (0, log.boxes['interpreted_lithology_column'][3])
                result = module.run(log, window, dict(legend=entries, calib=calib, device='cpu',
                                    cache_dir=self.temporary / 'cache'))
                predictions = run._clean(result[0], window, calib)
                truth = run._truth(name, 'gold', window)
                self.assertTrue(np.isfinite(score.score(predictions, truth, window)['row_acc']))
            finally:
                log.close()

    def test_donor_legend_is_copied_without_scaling(self):
        for name in self.names:
            log = io_synth.open_log(name, self.first)
            try:
                actual = io_synth.read_rgb(log, 'lithology_legend')
                with Image.open(self.first / name / 'legend.png') as image:
                    expected = np.asarray(image.convert('RGB'))
                # The original 1-bit donor is unchanged by binary page encoding.
                np.testing.assert_array_equal(actual, expected)
            finally:
                log.close()

    def test_nuisances_recorded_per_log_and_sources_per_bed(self):
        for name in self.names:
            meta = read_json(self.first / name / 'meta.json')
            self.assertEqual(set(meta['nuisances']), set(PROBABILITIES))
            for nuisance in meta['nuisances'].values():
                self.assertIn('probability', nuisance)
                self.assertIsInstance(nuisance['selected'], bool)
                self.assertIsInstance(nuisance['applied'], bool)
            for bed in meta['beds']:
                self.assertIn('source_crops', bed)

    def test_explicit_blank_rows_and_all_image_nuisance_paths(self):
        rng = np.random.default_rng(20)
        flags = draw_nuisances(rng, mono=True, overrides={k: 1. for k in PROBABILITIES})
        beds = [dict(top_y=0, bottom_y=60, lithology='Shale'),
                dict(top_y=60, bottom_y=120, lithology='Sandstone'),
                dict(top_y=120, bottom_y=140, lithology='')]
        rgb = rng.integers(30, 240, (140, 80, 3), dtype=np.uint8)
        assets = dict(name='unit', overlays=[(np.zeros((8, 9, 3), np.uint8), [0, 0, 9, 8])],
                      grid=np.zeros((1, 80, 3), np.uint8), grid_k=20, grid_source_y=15)
        result = apply_nuisances(rgb, beds, assets, flags, rng)
        self.assertTrue(np.all(result[120:] == 255))
        self.assertTrue(flags['blank_gaps']['applied'])
        self.assertTrue(flags['binarisation']['applied'])
        self.assertLess(abs(flags['skew']['degrees']), .3)
        self.assertTrue(set(np.unique(result)).issubset({0, 255}))
        truth = [dict(r, cluster_id=i) for i, r in enumerate(beds)]
        self.assertEqual(score.score(truth[:-1], truth, (0, 140))['row_acc'], 1)

    def test_minimum_error_seam_is_connected_and_optimal(self):
        error = np.full((4, 6), 20.)
        error[2] = 0
        seam = minimum_cut(error)
        np.testing.assert_array_equal(seam, np.full(6, 2))
        self.assertTrue(np.all(np.abs(np.diff(seam)) <= 1))

    def test_allocation_total_and_minimum(self):
        models = {'a': dict(filled_share=.81), 'b': dict(filled_share=.09)}
        result = allocate(models, 20)
        self.assertEqual(sum(result.values()), 20)
        self.assertEqual(result, {'a': 15, 'b': 5})
        self.assertGreater(result['a'], result['b'])
        self.assertTrue(all(n >= 2 for n in result.values()))
        with self.assertRaises(ValueError):
            allocate(models, 3)

    def test_transition_smoothing_and_no_self_transitions(self):
        model = next(iter(read_json(self.first / 'fitted_models.json')['types'].values()))
        matrix = np.asarray(model['transitions'])
        self.assertTrue(np.all(matrix.diagonal() == 0))
        np.testing.assert_allclose(matrix.sum(axis=1), 1)
        self.assertAlmostEqual(sum(model['shares']), 1)
        self.assertTrue(all(t['p1'] <= t['p99'] for t in model['thickness'].values()))

    def test_classifier_grid_has_no_truth_contacts(self):
        name = self.names[0]
        log = io_synth.open_log(name, self.first)
        try:
            case = classifier_case(log, [], 32)
            self.assertTrue(all('lithology' not in p for p in case['patches']))
            self.assertEqual([p['top_y'] for p in case['patches']], list(range(0, case['patches'][-1]['bottom_y'], 32)))
            self.assertEqual(discover()[1], ['colour', 'fft_hatch', 'dinov2', 'siglip', 'cnn_synth', 'hybrid'])
        finally:
            log.close()

    def test_offline_guard_blocks_network_and_processes(self):
        import socket
        import subprocess
        with offline_guard():
            with self.assertRaises(RuntimeError):
                socket.create_connection(('example.invalid', 80))
            with self.assertRaises(RuntimeError):
                subprocess.Popen(['never-launched'])

    def test_replay_cap_and_ci_singleton(self):
        replay = Replay(self.temporary / 'empty', cap=2)
        for i in range(4):
            self.assertIsNone(replay.get('test', dict(i=i)))
        self.assertEqual(len(replay.events), 2)
        self.assertEqual(confidence([.4], np.random.default_rng(0)), (.4, None, None))


if __name__ == '__main__':
    unittest.main()

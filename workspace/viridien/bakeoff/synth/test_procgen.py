"""Offline procedural measurements/rendering regression tests.

python -B -m unittest synth.test_procgen -v
NCC uses complete native source crops against a larger generated field, at all
translations. Threshold .985 allows shared regular motifs but rejects near
pixel-identical crops. Correlation alone is not a proof of pixel independence;
the separate blocked-image-I/O test enforces the renderer's input contract.
"""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
from .common import HOME, read_json, local_file, write_json
from .texstats import load_stats, texture_stats, aggregate, boundary_stats, ink_mask
from .procgen import render_pattern, render_column


class ProceduralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_stats()
        cls.patterns = cls.data['patterns']

    def test_every_pattern_deterministic_native_rgb(self):
        for pid in self.patterns:
            with self.subTest(pid=pid):
                a = render_pattern(pid, 73, 83, 101)
                np.testing.assert_array_equal(a, render_pattern(pid, 73, 83, 101))
                self.assertEqual(a.shape, (73, 83, 3))
                self.assertEqual(a.dtype, np.uint8)
                if self.patterns[pid]['scan_mode'] == '1-bit':
                    self.assertTrue(set(np.unique(a)).issubset({0, 255}))

    def test_seed_changes_realizations(self):
        for pid in ('d-582-1-bit-grid-7096e7/shale', 'mc765-colour-grid-96c56f/clays'):
            self.assertFalse(np.array_equal(render_pattern(pid, 200, 200, 1), render_pattern(pid, 200, 200, 2)))

    def test_sparse_measured_dots_are_not_erased(self):
        pid = 'gulf-3-panel-c1fffe/clayshale'
        self.assertGreater(self.patterns[pid]['horizontal_ink']['n'], 0)
        rgb = render_pattern(pid, 180, 80, 5)
        self.assertGreater(float(ink_mask(rgb)[:, 12:-12].mean()), .0001)

    def test_d582_observed_hooks_keep_their_vertical_limbs(self):
        from .procgen import motif
        pid = 'd-582-1-bit-grid-7096e7/clay'
        self.assertEqual(motif(self.patterns[pid]), 'hooks')
        stats = texture_stats(render_pattern(pid, 160, 246, 5))
        self.assertGreater(stats['component_height']['median'], 4)

    def test_renderer_does_not_read_any_source_images(self):
        # Stats are already cached; the public APIs must work when real image
        # access is unavailable. Also verifies no legend swatch is pasted.
        with patch('PIL.Image.open', side_effect=AssertionError('Source pixel read')), \
             patch('harness.io.read_rgb', side_effect=AssertionError('Dataset pixel read')):
            for pid in self.patterns:
                render_pattern(pid, 41, 70, 3)
            pid = next(iter(self.patterns))
            render_column(dict(width=80, beds=[dict(pid=pid, height=20), dict(pid=pid, height=1), dict(pid=pid, height=15)]), 3)

    def test_max_ncc_against_every_source_crop(self):
        index = read_json(HOME/'data/sheets/index.json')
        paths = self.data.get('inputs', {}).get('crop_files') or sorted({c['file'] for p in index.values() for c in p['candidates'].values()})
        crops = []
        for path in paths:
            with Image.open(local_file(path)) as image:
                rgb = np.asarray(image.convert('RGB'))
            crops.append((path, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))
        height = max(len(rgb) for _, rgb in crops)+40
        width = max(rgb.shape[1] for _, rgb in crops)+40
        report = []
        # Every source crop is checked against representative generated fields
        # from every family, plus both of the principal hatch/dot motifs.
        selected, families = [], set()
        for pid, p in self.patterns.items():
            if p['family'] not in families:
                selected.append(pid)
                families.add(p['family'])
        selected = sorted(set(selected+['d-582-1-bit-grid-7096e7/siltstone', 'mc765-colour-grid-96c56f/sandmss']))
        for pid in selected:
            generated = cv2.cvtColor(render_pattern(pid, height, width, 701), cv2.COLOR_RGB2GRAY)
            best, source = -1., None
            for path, crop in crops:
                if float(crop.std()) < .01:
                    continue  # NCC of a constant patch is mathematically undefined.
                ncc = float(cv2.matchTemplate(generated, crop, cv2.TM_CCOEFF_NORMED).max())
                if ncc > best:
                    best, source = ncc, path
            report.append(dict(pid=pid, max_ncc=best, source=source))
        write_json(HOME/'out/procgen_ncc.json', dict(threshold=.985, source_crops=len(crops), patterns=report))
        self.assertLess(max(r['max_ncc'] for r in report), .985, sorted(report, key=lambda r: -r['max_ncc'])[:3])

    def test_stats_round_trip_measured_hatches_and_colour(self):
        pids = ['d-582-1-bit-grid-7096e7/'+label for label in ('shale', 'siltstone', 'sandstone')]
        pids += ['mc765-colour-grid-96c56f/clays', 'mc765-colour-grid-96c56f/sandmss',
                 'geoservices-grid-6a0f0a/clayshale', 'one-off-layout-cdfeda/claystone',
                 'mc761-mineral-lithology-dbf40d/sandstone', 'texas-rock-types-box-32e585/shale']
        report = []
        for pid in pids:
            target = self.patterns[pid]
            measured = aggregate([texture_stats(render_pattern(pid, 240, round(target['width']), seed)) for seed in range(12)])
            errors = {}
            for stat, tolerance in [('horizontal_ink', .35), ('horizontal_gap', .3), ('row_spacing', .2)]:
                reference, actual = target[stat]['median'], measured[stat]['median']
                error = abs(actual-reference)/max(1, reference)
                errors[stat] = error
                with self.subTest(pid=pid, stat=stat):
                    self.assertLessEqual(error, tolerance)
            ink_error = abs(measured['ink_share']-target['ink_share'])/max(.01, target['ink_share'])
            delta = float(np.linalg.norm(np.array(measured['lab_mean'])-target['lab_mean']))
            with self.subTest(pid=pid, stat='ink share'):
                self.assertLessEqual(ink_error, .35)
            with self.subTest(pid=pid, stat='Lab mean'):
                self.assertLessEqual(delta, 12)
            report.append(dict(pid=pid, errors=errors, ink_share_error=ink_error, delta_lab=delta))
        write_json(HOME/'out/procgen_roundtrip.json', report)

    def test_known_run_lengths_and_colour_binary_detection(self):
        rgb = np.full((100, 120, 3), 255, np.uint8)
        for y in range(4, 96, 10):
            for x in range(3, 110, 15): rgb[y:y+2, x:x+8] = 0
        stats = texture_stats(rgb)
        self.assertEqual(stats['horizontal_ink']['median'], 8)
        self.assertEqual(stats['horizontal_gap']['median'], 7)
        self.assertEqual(stats['vertical_ink']['median'], 2)
        self.assertEqual(stats['row_spacing']['median'], 10)
        neon = rgb.copy()
        neon[rgb[..., 0] == 255] = [0, 255, 0]
        self.assertEqual(texture_stats(neon)['scan_mode'], 'colour')
        self.assertEqual(stats['scan_mode'], '1-bit')

    def test_column_exact_beds_and_one_row_sliver(self):
        pids = ['mc765-colour-grid-96c56f/clays', 'mc765-colour-grid-96c56f/sandmss']
        spec = dict(width=37, borders=False, grid=False, beds=[dict(pid=pids[0], height=31),
                    dict(pid=pids[1], height=1), dict(pid=pids[0], height=48)])
        a = render_column(spec, 9)
        self.assertEqual(a.shape, (80, 37, 3))
        np.testing.assert_array_equal(a, render_column(spec, 9))
        # Saturated yellow occupies only the requested one-row bed. No fade.
        yellow = (a[..., 0] > 150) & (a[..., 1] > 150) & (a[..., 2] < 80)
        self.assertGreater(yellow[31].mean(), .5)
        self.assertEqual(int(yellow[:31].sum()+yellow[32:].sum()), 0)

    def test_boundary_sharpness_against_observed_family_contacts(self):
        spec = dict(width=70, grid=False, borders=False, beds=[
            dict(pid='mc765-colour-grid-96c56f/clays', height=48),
            dict(pid='mc765-colour-grid-96c56f/sandmss', height=48)])
        widths = [boundary_stats(render_column(spec, seed), 48)['profile_width'] for seed in range(20)]
        widths = [v for v in widths if v is not None]
        real = self.data['families']['MC765 colour grid']['boundary']['profile_width']
        self.assertGreaterEqual(len(widths), 18)
        self.assertLessEqual(float(np.median(widths)), max(1, real['p90']))

    def test_invalid_columns_rejected(self):
        pid = next(iter(self.patterns))
        for beds in ([], [dict(pid=pid, height=0)], [dict(pid=pid, top_y=2, bottom_y=8)]):
            with self.assertRaises(ValueError): render_column(dict(width=20, beds=beds), 0)
        with self.assertRaises(KeyError): render_pattern('unknown/rock', 20, 20, 0)
        with self.assertRaises(ValueError): render_pattern(pid, 0, 20, 0)

    def test_comparison_native_images_and_lookalikes(self):
        from .compare import choose_patterns, stat_rows
        selected, extras, pairs = choose_patterns(self.data)
        self.assertEqual(len(selected), 20)
        self.assertEqual(len(set(selected)), 20)
        self.assertEqual(len(extras), 4)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len({self.patterns[pid]['family'] for pid in selected}), 7)
        stats = next(iter(self.patterns.values()))
        self.assertFalse(any(r['off'] for r in stat_rows(stats, stats, [1], [1])))

    def test_generate_procedural_and_quilt_modes_keep_pool_separation(self):
        import shutil
        import uuid
        from .generate import generate
        from .mock import build_mock
        root = (HOME/'out/tests'/('proc_switch_'+uuid.uuid4().hex[:12])).resolve()
        root.mkdir(parents=True)
        try:
            catalog, approved = build_mock(root/'mock')
            for mode in ('procedural', 'quilt'):
                output = root/mode
                generate(catalog, approved, output, n_logs=2, seed=11, min_height=2000,
                         max_height=2001, texture=mode)
                split = read_json(output/'split.json')
                for path in output.glob('*/meta.json'):
                    meta = read_json(path)
                    allowed = set(split[meta['pool']+'_logs'])
                    for bed in meta['beds']:
                        if not bed['pid']: continue
                        self.assertEqual(bed['texture'], mode)
                        if mode == 'procedural':
                            self.assertFalse(bed['source_crops'])
                            self.assertTrue(set(bed['parameter_source_logs']) <= allowed)
                        else:
                            self.assertTrue(bed['source_crops'])
        finally:
            if root.is_relative_to((HOME/'out/tests').resolve()): shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()

"""Checks of challenge export schemas and source validation."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from strategies import challenge_export as exporter

OUT = Path(__file__).resolve().parents[1] / 'out'


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=OUT)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.csv'
        self.rows = [
            dict(top_y='2', bottom_y='10', lithology='Clay / Shale', cluster_id='01', top_depth='2.5', bottom_depth='4.5'),
            dict(top_y='12', bottom_y='20', lithology='Shale', cluster_id='2', top_depth='5.0', bottom_depth='8.0'),
        ]
        self.write_source()

    def tearDown(self):
        self.temp.cleanup()

    def write_source(self):
        with self.source.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=exporter.INPUT_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def test_formats_round_trip_and_manifest(self):
        manifest = exporter.export(self.source, self.root / 'result', image_height=20)
        self.assertEqual(manifest['source_sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(manifest['rows'], 2)
        for variant, fields in [('notebook', exporter.NOTEBOOK_FIELDS), ('slide', exporter.SLIDE_FIELDS)]:
            path = self.root / 'result' / manifest[variant]['file']
            with path.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, list(fields))
                actual = list(reader)
            if variant == 'slide':
                for row in actual:
                    row['bottom_depth'] = row.pop('base_depth')
            self.assertEqual(actual, self.rows)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest[variant]['sha256'])
        saved = json.loads((self.root / 'result' / 'source.manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(saved, manifest)

    def test_missing_nonfinite_depth_and_bad_coordinates_fail(self):
        for field, value in [('bottom_depth', ''), ('top_depth', 'NaN'),
                             ('bottom_depth', 'inf'), ('top_y', '2.5')]:
            with self.subTest(field=field, value=value):
                original = self.rows[0][field]
                self.rows[0][field] = value
                self.write_source()
                with self.assertRaises(ValueError):
                    exporter.read_rows(self.source)
                self.rows[0][field] = original

    def test_overlap_missing_cluster_and_image_bounds_fail(self):
        self.rows[1]['top_y'] = '9'
        self.write_source()
        with self.assertRaises(ValueError):
            exporter.read_rows(self.source)
        self.rows[1]['top_y'] = '12'
        self.rows[1]['cluster_id'] = ' '
        self.write_source()
        with self.assertRaises(ValueError):
            exporter.read_rows(self.source)
        self.rows[1]['cluster_id'] = 'abc'
        self.write_source()
        with self.assertRaises(ValueError):
            exporter.read_rows(self.source)
        self.rows[1]['cluster_id'] = '2'
        self.write_source()
        with self.assertRaises(ValueError):
            exporter.read_rows(self.source, image_height=19)

if __name__ == '__main__':
    unittest.main(verbosity=2)

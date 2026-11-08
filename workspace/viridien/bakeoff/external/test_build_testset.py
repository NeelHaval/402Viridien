"""Outside-set geometry and truth tests; run with python -I."""
import csv
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('build_testset', Path(__file__).with_name('build_testset.py'))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class BuilderTests(unittest.TestCase):
    def test_windows_are_bounded_nonoverlapping_and_spread(self):
        self.assertEqual(builder.windows([10, 50, 30, 7550]), [(50, 6050), (6050, 7550)])
        self.assertEqual(builder.windows([10, 50, 30, 18050]),
                         [(50, 6050), (6050, 12050), (12050, 18050)])
        wins = builder.windows([10, 100, 30, 100100])
        self.assertEqual(len(wins), 3)
        self.assertEqual((wins[0][0], wins[-1][1]), (100, 100100))
        self.assertTrue(all(b-a == 6000 for a, b in wins))
        self.assertTrue(all(b <= c for (a, b), (c, d) in zip(wins, wins[1:])))

    def test_truth_shifts_pixels_but_preserves_depth_and_exact_labels(self):
        out = builder.ROOT/'external_data/kgs/tests/builder_truth'
        out.mkdir(parents=True, exist_ok=True)
        fields = builder.io.FIELDS
        rows = [dict(top_y=0, bottom_y=5, lithology='(blank)', cluster_id=-1, top_depth=0, bottom_depth=5),
                dict(top_y=5, bottom_y=15, lithology='Rock A', cluster_id=3, top_depth=5.25, bottom_depth=15.25),
                dict(top_y=15, bottom_y=20, lithology='A / B', cluster_id=4, top_depth=15.25, bottom_depth=20.25),
                dict(top_y=20, bottom_y=20.1, lithology='Rock A', cluster_id=3, top_depth=20.25, bottom_depth=20.35)]
        with (out/'truth.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        truth, excluded, collapsed = builder.truth_rows(out,
            {'legend_entries': [dict(kind='rock', label='Rock A')]}, 100, [10, 100, 30, 130])
        self.assertEqual(len(truth), 1)
        self.assertEqual((truth[0]['top_y'], truth[0]['bottom_y']), (105, 115))
        self.assertEqual(truth[0]['top_depth'], 5.25)
        self.assertEqual(truth[0]['lithology'], 'Rock A')
        self.assertEqual((len(excluded), collapsed), (2, 1))

    def test_inherited_legend_contains_label_pixels(self):
        import fitz
        out = builder.ROOT/'external_data/kgs/tests/builder_legend'
        out.mkdir(parents=True, exist_ok=True)
        pdf = out/'source.pdf'
        with fitz.open() as doc:
            page = doc.new_page(width=300, height=200)
            page.draw_rect(fitz.Rect(20, 40, 60, 60), color=(0, 0, 0), fill=(1, 0, 0))
            page.insert_text((75, 55), 'RockA', fontsize=12)
            doc.save(pdf)
        meta = dict(page=2, render={'px_per_pt': 2}, legend_entries=[
            dict(kind='rock', label='RockA', page=1, bbox_pt=[20, 40, 60, 60],
                 label_bbox_pt=[75, 42, 110, 58])])
        crop, local, page = builder.legend_crop(out, meta, pdf, (600, 400))
        self.assertIsNone(local)
        self.assertEqual(page, 1)
        self.assertGreater(crop.width, 180)
        import numpy as np
        self.assertTrue((np.asarray(crop)[:, 115:].min(axis=2) < 100).any())


if __name__ == '__main__':unittest.main(verbosity=2)

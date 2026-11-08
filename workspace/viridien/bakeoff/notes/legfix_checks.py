"""Targeted layout regressions: python -X utf8 -m notes.legfix_checks."""
import unittest
from unittest.mock import patch

import numpy as np

from harness import legend_v2 as parser, legend, io


def word(x, y, text, width=120):
    return (np.asarray([[x, y], [x+width, y], [x+width, y+18], [x, y+18]], float), text, .99)


class LayoutChecks(unittest.TestCase):
    def test_heading_variants(self):
        for text in ['LITHOLOGY SYMBOL', 'LITHOLOGYSYMBOL', 'Lithology symbols', 'LITHOLOGICAL SYMBOLS']:
            self.assertEqual(parser._section(text), 'lithology')
        for text in ['SHOW & DRILLING SYMBOL', 'Show and drilling', 'SHOW&DRILLINGSYMBOL']:
            self.assertEqual(parser._section(text), 'shows')
        self.assertIsNone(parser._section('Dolomitic limestone'))

    def fixture(self, flat=False, ruled=False):
        rgb = np.full((250, 1000, 3), 255, np.uint8)
        words = [word(12, 5, 'LITHOLOGYSYMBOL', 160), word(410, 5, 'MODIFIERS'),
                 word(810, 5, 'SHOW&DRILLINGSYMBOL', 180)]
        boxes = []
        xs = [10, 210, 410, 610, 810] if flat else [10, 410, 810]
        for y, names in [(55, ['Shale']*len(xs)), (155, ['Dry']*len(xs))]:
            for x, name in zip(xs, names):
                boxes.append((x, y, x+45, y+35))
                rgb[y:y+35, x:x+45] = (0, 200, 0)
                words.append(word(x+50, y+8, name, 85))
        if ruled:
            rgb[35:, 398:400] = 0
            rgb[35:, 798:800] = 0
        return rgb, boxes, words

    def test_flat_banner(self):
        rgb, boxes, words = self.fixture(flat=True)
        self.assertEqual(parser._layout(rgb, boxes, words)[0], 'flat')
        self.assertFalse(parser._rock('Gas_Good'))
        self.assertFalse(parser._rock('Salted water'))
        self.assertTrue(parser._rock('Sylvite'))

    def test_dividers_override_flat_inventory(self):
        rgb, boxes, words = self.fixture(flat=True, ruled=True)
        self.assertEqual(parser._layout(rgb, boxes, words)[0], 'panels')

    def test_flat_with_short_technical_row(self):
        rgb, boxes, words = self.fixture(flat=True)
        boxes = boxes[:8]
        words = words[:11]
        self.assertEqual(parser._layout(rgb, boxes, words)[0], 'flat')

    def test_real_panels_and_cache(self):
        # Even identical rock labels in Modifier/Show panels must stay false.
        rgb, boxes, words = self.fixture(ruled=True)
        log = io.Log(None, {'lithology_legend': [100, 200, 1100, 450]}, 'synthetic_panels')
        directory = io.ROOT/'out'/'_cache'/'legend_v2_checks'
        directory.mkdir(parents=True, exist_ok=True)
        (directory/(log.name+'.json')).unlink(missing_ok=True)
        with patch.object(parser, 'CACHE', directory), \
             patch.object(parser.io, 'read_rgb', return_value=rgb), \
             patch.object(parser.v1, 'ocr', return_value=words), \
             patch.object(parser, '_boxes', return_value=(boxes, False)), \
             patch.object(parser, '_refine_words', return_value=words):
            entries = parser.parse(log)
            self.assertEqual([e.is_lithology for e in entries[:3]], [True, False, False])
            self.assertIsInstance(entries[0], legend.LegendEntry)
            self.assertEqual(entries[0].swatch_box, [110, 255, 155, 290])
            with patch.object(parser.io, 'read_rgb', side_effect=AssertionError('cache miss')):
                cached = parser.parse(log)
            self.assertEqual([(e.label, e.is_lithology) for e in entries],
                             [(e.label, e.is_lithology) for e in cached])
            for a, b in zip(entries, cached):
                np.testing.assert_array_equal(a.swatch, b.swatch)


if __name__ == '__main__':
    unittest.main()

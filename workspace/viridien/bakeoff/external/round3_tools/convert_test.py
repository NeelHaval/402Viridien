"""Offline converter checks, including original extractor/builder regressions.

Run: python -I -m pytest -q -p no:cacheprovider bakeoff/external/round3_tools/convert_test.py
All original-test scratch output is redirected to /tmp (never another lane).
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fitz
import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


converter = load('convert_round3_test_module', HERE/'convert_round3.py')
original = load('original_extractor_tests', HERE.parent/'test_kgs_extract.py')
builder_tests = load('original_builder_tests', HERE.parent/'test_build_testset.py')
scratch = tempfile.TemporaryDirectory(prefix='round3-convert-tests-')
original.kgs.OUTPUT_ROOT = Path(scratch.name)/'extractor'
original.TEST_OUT = original.kgs.OUTPUT_ROOT/'tests'
builder_tests.builder.ROOT = Path(scratch.name)/'builder'


@unittest.skipUnless(original.kgs.SAMPLE.exists(), 'Original KGS PDF fixture is absent from VM mirror')
class TestExtractorSample(original.SampleTests):
    pass


class TestExtractorRobustness(original.RobustnessTests):
    def setUp(self):
        super().setUp()
        # Fixed test resolution avoids host-specific analysis CSV defaults and
        # keeps the synthetic metric example below the original render cap.
        self.resolution = patch.object(original.kgs, 'default_resolution', return_value=(.1, 'offline test'))
        self.resolution.start()
        self.addCleanup(self.resolution.stop)

    @unittest.skipUnless(original.kgs.SAMPLE.exists(), 'Original KGS PDF fixture is absent from VM mirror')
    def test_batch_summary_one_row_and_contact_sheet(self):
        super().test_batch_summary_one_row_and_contact_sheet()

    def test_nonlog_pages_skip_before_graphics_and_summary_is_short(self):
        fixture = Path(scratch.name)/'fixture'/'mini.pdf'
        fixture.parent.mkdir(exist_ok=True)
        with fitz.open() as doc:
            original.miniature(doc)
            doc.save(fixture)
        with patch.object(original.kgs, 'SAMPLE', fixture):
            super().test_nonlog_pages_skip_before_graphics_and_summary_is_short()


class TestBuilder(builder_tests.BuilderTests):
    pass


def test_render_budget_and_draft_boxes():
    with fitz.open() as doc:
        page = doc.new_page(width=300, height=700)
        page.insert_text((105, 125), 'Depth', fontsize=8)
        page.insert_text((151, 125), 'Lithology', fontsize=8)
        for x in (100, 135, 145, 190):
            page.draw_line((x, 130), (x, 690))
        image, scale = converter.render_pdf_page(page)
        assert image.width <= 2001
        assert image.width*image.height < 32_020_000
        annotation = converter.draft_vector_boxes(page, scale, 'test', image.size)
        assert annotation['draft'] is True
        assert set(annotation['bounding_boxes']) == {'depth_column', 'interpreted_lithology_column'}
        image.close()


def test_ambiguous_or_landscape_headers_do_not_get_boxes():
    with fitz.open() as doc:
        page = doc.new_page(width=700, height=300)
        page.insert_text((100, 100), 'Depth Lithology')
        assert converter.draft_vector_boxes(page, 1, 'test', (700,300)) is None
    with fitz.open() as doc:
        page = doc.new_page(width=300, height=700)
        assert converter.draft_vector_boxes(page, 1, 'test', (300,700)) is None


def test_mississippi_key_requires_positive_same_viewer():
    docs = [('mississippi', {'has_legend':True}, Path('known-001.png')),
            ('mississippi', {'has_legend':None}, Path('unknown-002.png')),
            ('elsewhere', {'has_legend':True}, Path('elsewhere-001.png'))]
    assert set(converter.raster_groups(docs)) == {'known'}


def test_report_selection_excludes_nonlog_pages():
    pages = converter.pages_for_document('nwt_orogo', {}, Path('02.pdf'))
    assert 1 not in pages and 64 not in pages and 99 not in pages
    assert 65 in pages and 98 in pages and 100 in pages and 118 in pages


def test_untrusted_path_escape_is_rejected(tmp_path):
    import json
    (tmp_path/'source').mkdir()
    (tmp_path/'source'/'triage.json').write_text(json.dumps({'documents':[{'path':'../escape.pdf'}]}))
    with pytest.raises(ValueError, match='escapes'):
        list(converter.documents(tmp_path))

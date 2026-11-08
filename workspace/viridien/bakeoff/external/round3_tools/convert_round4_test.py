"""Round4 native-scan and shared-manifest regression tests."""
import importlib.util
import json
from pathlib import Path

import fitz
import pytest

spec = importlib.util.spec_from_file_location('convert_round4', Path(__file__).with_name('convert_round4.py'))
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_append_preserves_previous_bytes_and_rejects_changed_schema(tmp_path):
    path = tmp_path/'manifest.csv'
    original = b'id,notes\r\nold,"quoted, note"\r\n'
    path.write_bytes(original)
    converter.append_csv(path, [{'id': 'new', 'notes': 'two\nlines'}], ['id', 'notes'])
    assert path.read_bytes().startswith(original)
    assert converter.read_csv(path)[-1] == {'id': 'new', 'notes': 'two\nlines'}
    before = path.read_bytes()
    with pytest.raises(ValueError, match='header'):
        converter.append_csv(path, [{'other': 'x'}], ['other'])
    assert path.read_bytes() == before


def test_native_scan_does_not_shrink_tiny_pdf_placement():
    with fitz.open() as doc:
        page = doc.new_page(width=612, height=792)
        raw = bytes([0, 255])*100*100
        pix = fitz.Pixmap(fitz.csGRAY, 100, 200, raw, False)
        page.insert_image(fitz.Rect(250, 300, 260, 320), pixmap=pix)
        with converter.scan_image(doc, page) as image:
            assert image.size == (100, 200)
            assert image.convert('L').tobytes() == raw


def test_reflected_scan_preserves_pdf_appearance():
    with fitz.open() as doc:
        page = doc.new_page(width=200, height=200)
        raw = bytes([0]*10+[255]*10)*20
        pix = fitz.Pixmap(fitz.csGRAY, 20, 20, raw, False)
        page.insert_image(fitz.Rect(50, 50, 150, 150), pixmap=pix)
        # Reflect the image placement across its vertical centre.
        content = page.get_contents()[-1]
        data = doc.xref_stream(content)
        doc.update_stream(content, data.replace(b'100 0 0 100 50 50 cm', b'-100 0 0 100 150 50 cm'))
        assert page.get_image_info()[0]['transform'][0] < 0
        with converter.scan_image(doc, page) as image:
            assert image.getpixel((3, 10)) > 240
            assert image.getpixel((16, 10)) < 15


def test_triage_cannot_escape_input_root(tmp_path, monkeypatch):
    monkeypatch.setattr(converter, 'ROOT', tmp_path)
    inputs = tmp_path/'round4'
    family = inputs/'bw_hatch'
    family.mkdir(parents=True)
    (family/'triage.json').write_text(json.dumps({'sources': [{'documents': [{'path': '../outside.pdf'}]}]}))
    with pytest.raises(ValueError, match='escapes'):
        list(converter.documents(inputs))

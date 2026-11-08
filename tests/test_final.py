"""Exercise the real command, including failure isolation and output contracts."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont


def test_generated_page(tmp_path):
    images = tmp_path / 'images'
    boxes = tmp_path / 'boxes'
    images.mkdir()
    boxes.mkdir()
    image = Image.new('RGB', (700, 1000), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('DejaVuSans.ttf', 30)
    draw.rectangle((30, 50, 80, 75), fill='#808080', outline='black', width=2)
    draw.text((95, 45), 'Shale', font=font, fill='black')
    draw.rectangle((400, 150, 460, 950), fill='#808080', outline='black')
    image.save(images / 'tiny.png')
    (boxes / 'tiny.json').write_text(json.dumps({'bounding_boxes': {
        'lithology_legend': [10, 10, 350, 140],
        'interpreted_lithology_column': [400, 150, 460, 950]}}), encoding='utf-8')
    # A second page has no annotation. It must not stop the valid page.
    image.save(images / 'missing.jpg')
    out = tmp_path / 'out'
    result = subprocess.run([sys.executable, '-m', 'pipeline.final', '--images', str(images),
                             '--boxes', str(boxes), '--out', str(out), '--workers', '2'],
                            capture_output=True, text=True, timeout=180)
    manifest = json.loads((out / 'official_manifest.json').read_text(encoding='utf-8'))
    docs = {d['page']: d for d in manifest['documents']}
    assert result.returncode == 1, result.stdout + result.stderr
    assert docs['missing']['status'] == 'failed'
    assert docs['tiny']['status'] == 'ok', (out / 'tiny/worker.log').read_text(encoding='utf-8')
    assert docs['tiny']['depth_status'] == 'unavailable'
    assert docs['tiny']['reason']
    with (out / 'tiny/csv/tiny.official.raw.notebook.csv').open(encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['top_depth', 'bottom_depth', 'lithology', 'top_y', 'bottom_y', 'cluster_id']
        rows = list(reader)
    assert rows and all(r['top_depth'] == r['bottom_depth'] == '' for r in rows)
    with Image.open(out / 'tiny/tiny.official.full-column.png') as rendered:
        assert rendered.height == 870
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    timed = tmp_path / 'timeout'
    result = subprocess.run([sys.executable, '-m', 'pipeline.final', '--images', str(images),
                             '--boxes', str(boxes), '--out', str(timed), '--timeout', '0.001'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    failed = json.loads((timed / 'official_manifest.json').read_text(encoding='utf-8'))
    assert len(failed['documents']) == 2
    assert all(d['status'] == 'failed' and 'timed out' in d['reason'] for d in failed['documents'])

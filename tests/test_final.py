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
    font = ImageFont.load_default(size=30)
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
    assert (out / 'tiny/tiny.official.raw.notebook.csv').read_bytes() == (out / 'tiny/csv/tiny.official.raw.notebook.csv').read_bytes()
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
    assert all(d['status'] == 'failed' and d['failure_type'] in ('timeout', 'missing_annotation') for d in failed['documents'])


import pytest
from pipeline.final import discover, bounded_log, InputError


@pytest.mark.parametrize('suffix', ['.tif', '.tiff', '.TIF', '.png', '.jpg', '.jpeg'])
def test_nested_layout(tmp_path, suffix):
    images = tmp_path / 'images'
    boxes = tmp_path / 'bbox_annotation'
    (images / 'nested').mkdir(parents=True)
    (boxes / 'nested').mkdir(parents=True)
    (images / 'nested' / ('page ü space' + suffix)).touch()
    (boxes / 'nested/page ü space.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'labels.json').write_text('MUST NOT READ', encoding='utf-8')
    (boxes / 'extra.json').write_text('{}', encoding='utf-8')
    jobs, extras = discover(images, boxes)
    assert len(jobs) == 1 and jobs[0][1].name == 'page ü space.json'
    assert extras == ['extra.json']


def test_ambiguous_and_missing(tmp_path):
    images, boxes = tmp_path / 'images', tmp_path / 'bbox_annotation'
    images.mkdir()
    for name in ['a', 'b']:
        (boxes / name).mkdir(parents=True)
        (boxes / name / 'p.json').write_text('{}', encoding='utf-8')
    (images / 'p.png').touch()
    (images / 'missing.png').touch()
    jobs, _ = discover(images, boxes)
    assert {job[3] for job in jobs} == {'ambiguous_annotation', 'missing_annotation'}


@pytest.mark.parametrize('mode', ['I;16', 'CMYK', 'RGB'])
def test_tiff_modes_and_frames(tmp_path, mode):
    from pipeline import run
    from harness import io
    path = tmp_path / 'multi.tiff'
    image = Image.new(mode, (20, 30))
    image.save(path, save_all=True, append_images=[image])
    previous = io._crop
    try:
        log = bounded_log(path, {}, io)
        assert log.size == (20, 30)
        assert io._crop(log, (0, 0, 10, 10)).mode == 'RGB'
        with pytest.raises(InputError, match='outside'):
            io._crop(log, (-1, 0, 10, 10))
        log.close()
    finally:
        io._crop = previous


@pytest.mark.parametrize('option', ['--data', '--images', 'positional', 'cwd'])
def test_root_cli(tmp_path, option):
    (tmp_path / 'images').mkdir()
    (tmp_path / 'bbox_annotation').mkdir()
    Image.new('RGB', (2, 2)).save(tmp_path / 'images/ü space.png')
    command = [sys.executable, '-m', 'pipeline.final']
    if option == 'positional':
        command += [str(tmp_path)]
    elif option != 'cwd':
        command += [option, str(tmp_path)]
    command += ['--out', str(tmp_path / 'out')]
    import os
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run(command, cwd=tmp_path if option == 'cwd' else None,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stderr
    doc = json.loads((tmp_path / 'out/official_manifest.json').read_text(encoding='utf-8'))['documents'][0]
    assert doc['failure_type'] == 'missing_annotation'

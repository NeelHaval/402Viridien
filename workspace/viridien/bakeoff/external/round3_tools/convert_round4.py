"""Append round4 family-search scans without changing round3 products.

Run with python -I, passing --inputs and --out. Triage and PDFs are data.
Existing source URLs are deliberately not reconverted across rounds.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '3'
sys.dont_write_bytecode = True

import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
Image.MAX_IMAGE_PIXELS = 100_000_000
FIELDS = ['id', 'source', 'well', 'page', 'source_url', 'layout_family_guess',
          'has_legend', 'notes']


def safe(value):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', value).strip('_')


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def append_csv(path, rows, fields):
    """One append, retaining existing bytes and rejecting incompatible headers."""
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            if next(csv.reader(stream)) != fields:
                raise ValueError(f'Incompatible shared CSV header: {path}')
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def documents(inputs):
    for triage in sorted(inputs.glob('*/triage.json')):
        data = json.loads(triage.read_text(encoding='utf-8'))
        for source in data.get('sources', []):
            for doc in source.get('documents', []):
                path = (ROOT / doc['path']).resolve()
                if not path.is_relative_to(inputs.resolve()):
                    raise ValueError('Input path escapes round4')
                yield triage.parent.name, source, doc, path


def scan_image(doc, page):
    """Decode the single substantial scan at native resolution, without RGB expansion.

    A one-pixel background image is allowed. Reject pages with added text,
    drawings or multiple substantive images. Render PDF image masks and
    reflections at native scan resolution to retain their actual appearance.
    """
    images = [i for i in page.get_image_info(xrefs=True) if i['width']*i['height'] > 1]
    if len(images) != 1 or page.get_text().strip() or page.get_drawings() or page.rotation:
        raise ValueError('Page is not a simple single-image scan')
    info = images[0]
    if info['width'] * info['height'] > 100_000_000:
        raise ValueError('Native scan exceeds 100 MP budget')
    ref = info['xref']
    if not ref or any(doc.xref_get_key(ref, k)[0] != 'null' for k in ('SMask', 'Mask')):
        raise ValueError('Soft/external mask or inline scan requires review')
    a, b, c, d, _, _ = info['transform']
    if not a or not d or abs(b) > .001 or abs(c) > .001:
        raise ValueError('Rotated scan requires review')
    bounds = fitz.Rect(info['bbox'])
    if not (page.rect + (-.01, -.01, .01, .01)).contains(bounds):
        raise ValueError('Clipped scan requires review')
    if info['colorspace'] == 0 or a < 0 or d < 0:
        scale = max(info['width']/bounds.width, info['height']/bounds.height)
        if bounds.width*bounds.height*scale*scale > 100_000_000:
            raise ValueError('Native render exceeds 100 MP budget')
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=bounds,
                             colorspace=fitz.csGRAY, alpha=False, annots=False)
        image = Image.frombytes('L', (pix.width, pix.height), pix.samples)
        image.info['conversion_method'] = 'Native-resolution PDF render of scan extent, retaining image-mask decode and reflection'
        return image
    with Image.open(BytesIO(doc.extract_image(ref)['image'])) as original:
        image = original.copy()
        image.info['conversion_method'] = 'Native embedded scan exported losslessly'
        return image


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def preview(image, label, path):
    # Header and a logged-body excerpt remain useful when the full log is tall.
    panel = Image.new('RGB', (480, 790), 'white')
    draw = ImageDraw.Draw(panel)
    draw.text((10, 10), label, fill='black', font=font(15))
    draw.text((10, 34), f'{image.width} x {image.height} native pixels', fill='black', font=font(13))
    for y, top, title in ((80, 0, 'Header / legend'),
                           (435, image.height//2, 'Middle of log')):
        draw.text((10, y-20), title, fill='black', font=font(14))
        height = min(image.height-top, round(image.width*320/460))
        with image.crop((0, top, image.width, top+height)) as crop:
            crop.thumbnail((460, 320))
            panel.paste(crop.convert('RGB'), (10, y))
    panel.save(path)
    panel.close()


def main():
    if not sys.flags.isolated:
        raise SystemExit('Use python -I')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    inputs, out = args.inputs.resolve(), args.out.resolve()
    if inputs != ROOT/'external_untrusted/round4' or out != ROOT/'external_unlabelled':
        parser.error('Expected round4 inputs and lane-owned unlabelled output')
    for sub in ('images', 'thumbnails'):
        (out/sub).mkdir(parents=True, exist_ok=True)
    existing = read_csv(out/'manifest.csv')
    truth = read_csv(ROOT/'external_tests_round3/manifest.csv')
    by_url = {}
    for r in existing + truth:
        by_url.setdefault(r['source_url'], []).append(r)
    rows, audit, failures = [], [], []
    families = {p.parent.name for p in inputs.glob('*/triage.json')}
    for family, source, record, path in documents(inputs):
        item = dict(family=family, path=str(path.relative_to(ROOT)))
        if float(record.get('similarity') or 0) < 2 or not record.get('has_lith_column'):
            audit.append(dict(item, status='excluded', reason='Similarity below 2 or no lithology column'))
            continue
        if record['url'] in by_url:
            audit.append(dict(item, status='existing', ids=[r['id'] for r in by_url[record['url']]],
                              reason='Source URL already converted; round3 products retained'))
            continue
        try:
            if 'vector' in record.get('raster_or_vector', '').lower():
                raise ValueError('New vector source requires exact extraction and truth packaging')
            if not record.get('has_legend'):
                raise ValueError('No confirmed source legend')
            with fitz.open(path) as pdf:
                for page in pdf:
                    name = f'ul_r4_{safe(family)}_{safe(path.parent.name)}_{safe(path.stem)}_p{page.number+1:03d}'
                    dest = out/'images'/f'{name}.tif'
                    if dest.exists():
                        raise ValueError(f'Refusing to overwrite {name}')
                    image = scan_image(pdf, page)
                    try:
                        image.save(dest, compression='tiff_adobe_deflate')
                        thumb = out/'thumbnails'/f'{name}.png'
                        preview(image, path.stem, thumb)
                        with Image.open(dest) as check:
                            if check.size != image.size or check.mode != image.mode:
                                raise ValueError('TIFF dimensions or mode changed')
                            check.load()
                            if hashlib.sha256(check.tobytes()).digest() != hashlib.sha256(image.tobytes()).digest():
                                raise ValueError('TIFF pixels changed')
                        notes = (record.get('notes', '') + ' ' + image.info['conversion_method'] + '; '
                                 'PDF page whitespace omitted. No reliable semantic box detector; no draft boxes. '
                                 'Licence: ' + source.get('license_terms', 'Unverified.'))
                        rows.append(dict(id=name, source=path.parent.name, well=path.stem.rsplit('_mud', 1)[0],
                                         page=page.number+1, source_url=record['url'], layout_family_guess=family,
                                         has_legend=True, notes=notes))
                        audit.append(dict(item, status='exported', id=name, dimensions=list(image.size),
                                          mode=image.mode, pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
                        print(name, image.size, image.mode, flush=True)
                    finally:
                        image.close()
        except Exception as exc:
            failures.append(dict(item, error=f'{type(exc).__name__}: {exc}'))
    append_csv(out/'manifest.csv', rows, FIELDS)
    previous_families = read_csv(out/'families.csv')
    counts = Counter(r['layout_family_guess'] for r in rows)
    family_rows = [dict(round='round4', family=f, logs=len({r['well'] for r in rows if r['layout_family_guess']==f}),
                        pages=counts[f]) for f in sorted(families)
                   if counts[f] or not any(r.get('round')=='round4' and r.get('family')==f for r in previous_families)]
    append_csv(out/'families.csv', family_rows, ['round', 'family', 'logs', 'pages'])
    all_round4 = [r for r in read_csv(out/'manifest.csv') if r['id'].startswith('ul_r4_')]
    for family in sorted(families):
        members = [r for r in all_round4 if r['layout_family_guess']==family]
        sheet = Image.new('RGB', (960, max(100, 45+790*((len(members)+1)//2))), 'white')
        ImageDraw.Draw(sheet).text((12, 10), f'Round 4: {family} ({len(members)} new pages)', fill='black', font=font(18))
        for i, row in enumerate(members):
            with Image.open(out/'thumbnails'/f'{row["id"]}.png') as thumb:
                sheet.paste(thumb, ((i%2)*480, 45+(i//2)*790))
        sheet.save(out/f'contact_sheet_r4_{family}.png')
        sheet.close()
    summary = dict(new_logs=len({r['well'] for r in all_round4}), new_pages=len(all_round4),
                   appended_this_run=len(rows),
                   unlabelled_pages_by_family={f:sum(r['layout_family_guess']==f for r in all_round4) for f in sorted(families)}, new_truth_logs=0, new_truth_windows=0,
                   new_truth_intervals=0, failures=failures, documents=audit)
    (out/'round4_conversion.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='documents'}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

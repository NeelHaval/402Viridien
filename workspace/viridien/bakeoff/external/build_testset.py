"""Build the outside challenge dataset from exact PDF tile extractions.

Run from the workspace: python -I bakeoff/external/build_testset.py
No strategy inputs contain truth. Ambiguous image aliases are masked from truth,
never resolved by guessing. PDF depth fits are retained in feet, including metric
sources converted by the extractor. Source geometry and limitations are audited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import colorsys
import csv
import html
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BAKEOFF = ROOT / 'bakeoff'
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, str(BAKEOFF))
from harness import io, legend

MANIFEST_FIELDS = ['id', 'source', 'well', 'page', 'px_per_ft', 'rock_type_count',
                   'truth_intervals', 'legend_parse_labels_matched/total', 'source_url', 'notes']
URLS = {
    '386825': 'https://www.kgs.ku.edu/PRS/Scans/kcc_logs_2016/386825.pdf',
    '303738': 'https://www.kgs.ku.edu/PRS/Scans/kcc_logs_2013/303738.pdf',
    '516448': 'https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/kcc_logs_2022/516448.pdf',
    'doc1191374': 'https://kgsimages.blob.core.windows.net/kolar/filer_1/public/2014-02/solar.documents.released_pdf/doc_id%3D1191374.pdf',
    '4304150011': 'https://oilgas.ogm.utah.gov/wellfiles/041/4304150011.pdf',
    '1107520036_Barlow2-14_MUD_5in_20211011_PTS': 'https://ogcc.idaho.gov/wp-content/uploads/1107520036_Barlow2-14_MUD_5in_20211011_PTS.pdf',
    'barlow3-14': 'https://ogcc.idaho.gov/wp-content/uploads/2022.11.10_1107520040_Barlow3-14_MUD_PTS.pdf',
    'irvin1-19': 'https://ogcc.idaho.gov/wp-content/uploads/2022.10.17_1107520039_Irvin1-19_MUD_PTS.pdf',
    'flat-bay-2': 'https://www.gov.nl.ca/em/files/publications-energy-fwr-vulcan-minerals-flat-bay-2.pdf',
    'flat-bay-3': 'https://www.gov.nl.ca/em/files/publications-energy-fwr-vulcan-minerals-flat-bay-3.pdf',
    'red-brook-1': 'https://www.gov.nl.ca/em/files/publications-energy-fwr-red-brook-1-final-well-report.pdf',
    'storm': 'https://www.gov.nl.ca/iet/files/publications-energy-fwr-storm-final-well-report.pdf',
}
WELLS = {'386825': '386825', '303738': 'Marcotte1-12', '516448': 'EmmaWard1',
         'doc1191374': 'ForsythB2-17', '4304150011': '4304150011',
         '1107520036_Barlow2-14_MUD_5in_20211011_PTS': 'Barlow2-14'}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def safe(value):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', value).strip('_')


def signature(path):
    if not path.exists():return None
    with path.open('rb') as f:return hashlib.file_digest(f, 'sha256').hexdigest()


def windows(box):
    a, b = box[1], box[3]
    height = b-a
    n = min(3, math.ceil(height/6000))
    if n == 1:return [(a, b)]
    if height <= 18000:
        return [(start, min(start+6000, b)) for start in range(a, b, 6000)]
    return [(round(a+i*(height-6000)/(n-1)), round(a+i*(height-6000)/(n-1))+6000)
            for i in range(n)]


def discover():
    """Use extraction manifests so stale files from rejected rerenders are ignored."""
    roots = [ROOT/'external_data/kgs/386825', ROOT/'external_data/kgs/round2',
             ROOT/'external_data/kgs/round3']
    for root in roots:
        for path in sorted(root.rglob('extraction.json')):
            extraction = json.loads(path.read_text(encoding='utf-8'))
            key = path.parent.name
            source = ('utah' if key == '4304150011' else
                      'idaho' if key.startswith('110752') or key in ('barlow3-14', 'irvin1-19') else
                      'nl' if key in ('flat-bay-2', 'flat-bay-3', 'red-brook-1', 'storm') else 'kgs')
            for page in extraction['pages']:
                if page['status'] != 'ok':continue
                folder = path.parent / f"page_{page['page']:03d}"
                if (path.parent/'meta.json').exists():folder = path.parent
                yield source, key, folder, Path(extraction['pdf'])


def legend_crop(folder, meta, pdf, size):
    """Crop original source pixels, including every rock swatch and label glyph."""
    rocks = [e for e in meta['legend_entries'] if e['kind'] == 'rock']
    pages = sorted({e['page'] for e in rocks})
    if len(pages) != 1:raise ValueError(f'Expected one inherited rock legend: {pages}')
    source_page = pages[0]
    entries = [e for e in rocks if e['page'] == source_page]
    boxes = [e[k] for e in entries for k in ('bbox_pt', 'label_bbox_pt')]
    rect = [min(b[0] for b in boxes)-3, min(b[1] for b in boxes)-18,
            max(b[2] for b in boxes)+3, max(b[3] for b in boxes)+3]
    scale = meta['render']['px_per_pt']
    if source_page == meta['page']:
        with Image.open(folder/'image.png') as image:
            box = [max(0, math.floor(rect[0]*scale)), max(0, math.floor(rect[1]*scale)),
                   min(image.width, math.ceil(rect[2]*scale)), min(image.height, math.ceil(rect[3]*scale))]
            return image.crop(box).convert('RGB'), box, source_page
    import fitz
    with fitz.open(pdf) as doc:
        page = doc[source_page-1]
        rect = fitz.Rect(rect) & page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect,
                              colorspace=fitz.csRGB, alpha=False)
        crop = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    return crop, None, source_page


def truth_rows(folder, meta, shift, box):
    with (folder/'truth.csv').open(encoding='utf-8', newline='') as f:
        original = list(csv.DictReader(f))
    exact_labels = {e['label'] for e in meta['legend_entries'] if e['kind'] == 'rock'}
    rows, excluded, collapsed = [], [], 0
    for row in original:
        if row['lithology'] == '(blank)':continue
        if row['lithology'] not in exact_labels:
            excluded.append(dict(row, reason='Identical legend pixels have multiple label aliases'))
            continue
        a = max(box[1], round(float(row['top_y']))+shift)
        b = min(box[3], round(float(row['bottom_y']))+shift)
        if b <= a:
            collapsed += 1
            excluded.append(dict(row, reason='Subpixel interval occupies no integer raster rows'))
            continue
        row.update(top_y=a, bottom_y=b, cluster_id=int(row['cluster_id']))
        # Preserve the original vector depths; moving a legend changes no depth.
        for field in ('top_depth', 'bottom_depth'):row[field] = float(row[field])
        rows.append(row)
    for first, second in zip(rows, rows[1:]):
        if first['bottom_y'] > second['top_y']:raise ValueError('Overlapping truth')
    return rows, excluded, collapsed


def colour(cluster):
    return tuple(round(c*255) for c in colorsys.hsv_to_rgb((cluster*.61803398875)%1, .50, .86))


def font(size):
    try:return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)
    except OSError:return ImageFont.load_default(size=size)


def readable_legend(crop, meta):
    """Reflow complete original swatch/label pairs, without cutting their text."""
    entries = [e for e in meta['legend_entries'] if e['kind'] == 'rock']
    scale = meta['render']['px_per_pt']
    bounds = [e[k] for e in entries for k in ('bbox_pt', 'label_bbox_pt')]
    origin_x = max(0, math.floor((min(b[0] for b in bounds)-3)*scale))
    origin_y = max(0, math.floor((min(b[1] for b in bounds)-18)*scale))
    panels = []
    for e in entries:
        def original_pixels(box):
            rect = [max(0, math.floor(box[0]*scale)-origin_x),
                    max(0, math.floor(box[1]*scale)-origin_y),
                    min(crop.width, math.ceil(box[2]*scale)-origin_x),
                    min(crop.height, math.ceil(box[3]*scale)-origin_y)]
            return crop.crop(rect)
        swatch = original_pixels(e['bbox_pt'])
        label = original_pixels(e['label_bbox_pt'])
        part = Image.new('RGB', (swatch.width+8+label.width, max(swatch.height, label.height)), 'white')
        part.paste(swatch, (0, (part.height-swatch.height)//2))
        part.paste(label, (swatch.width+8, (part.height-label.height)//2))
        factor = min(1., 212/part.width)
        panels.append(part.resize((max(1, round(part.width*factor)), max(1, round(part.height*factor))), Image.Resampling.LANCZOS))
    row_h = max(p.height for p in panels)+7
    result = Image.new('RGB', (436, math.ceil(len(panels)/2)*row_h), 'white')
    for i, part in enumerate(panels):result.paste(part, ((i%2)*222, (i//2)*row_h))
    return result


def card(out, name, rows, record, crop, meta):
    log = io.open_log(name)
    try:
        box = log.boxes['interpreted_lithology_column']
        # Show an actual logged interval rather than a long leading blank.
        a = max(box[1], rows[0]['top_y']-20)
        b = min(box[3], a+1500)
        rgb = io.read_rgb(log, box, y0=a, y1=b)
    finally:log.close()
    thumb = readable_legend(crop, meta)
    column_y = 34+thumb.height+36
    target_h = max(1, round(rgb.shape[0]*100/rgb.shape[1]))
    width, height = 460, column_y+target_h+140
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width-1, height-1), outline='#bdbdbd')
    draw.text((12, 10), 'SOURCE LEGEND', font=font(15), fill='black')
    canvas.paste(thumb, (12, 34))
    draw.text((12, column_y-30), f'Column rows {a}..{b}  |  truth clusters', font=font(15), fill='black')
    im = Image.fromarray(rgb).resize((100, target_h), Image.Resampling.NEAREST)
    canvas.paste(im, (14, column_y))
    draw.rectangle((125, column_y, 164, column_y+target_h-1), outline='#bdbdbd')
    for row in rows:
        top, bottom = max(a, row['top_y']), min(b, row['bottom_y'])
        if bottom <= top:continue
        y0, y1 = column_y+round((top-a)*target_h/(b-a)), column_y+round((bottom-a)*target_h/(b-a))
        draw.rectangle((126, y0, 163, max(y0, y1-1)), fill=colour(row['cluster_id']))
    labels = sorted({(r['cluster_id'], r['lithology']) for r in rows})
    for i, (cluster, label) in enumerate(labels[:26]):
        yy = column_y+5+i*22
        draw.rectangle((180, yy, 192, yy+12), fill=colour(cluster))
        draw.text((198, yy-3), label[:30], font=font(15), fill='black')
    lines = textwrap.wrap(name, 47)+[
        f"{record['source']} p{record['page']} | {len(rows)} intervals | {len(labels)} rocks",
        f"{record['px_per_ft']:.2f} px/ft | legend match {record['legend_parse_labels_matched/total']}"]
    for i, line in enumerate(lines):draw.text((12, column_y+target_h+15+i*21), line, font=font(16), fill='black')
    canvas.save(out/'gallery'/(name+'.png'))
    return canvas


def build(out):
    out = out.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must stay in the workspace')
    for subdir in ('images', 'bbox_annotation', 'truth', 'gallery'):
        (out/subdir).mkdir(parents=True, exist_ok=True)
    os.environ['LITHOLOGY_DATASET'] = str(out)
    io.DATASET = out
    manifest, items, audit, cards, skipped = [], [], [], [], []
    ocr_by_pixels = {}
    for source, key, folder, pdf in discover():
        meta = json.loads((folder/'meta.json').read_text(encoding='utf-8'))
        annotation = json.loads((folder/'boxes.json').read_text(encoding='utf-8'))
        well = WELLS.get(key, key)
        paginated = folder.name.startswith('page_')
        name = 'ext_'+source+'_'+safe(well)+(f"_p{meta['page']:03d}" if paginated else '')
        tiff = out/'images'/(name+'.tif')
        annotation_path = out/'bbox_annotation'/(name+'.json')
        previous = (signature(tiff), signature(annotation_path))
        if any(r['id'] == name for r in manifest):raise ValueError(f'Duplicate id: {name}')
        with Image.open(folder/'image.png') as original:
            crop, local_box, legend_page = legend_crop(folder, meta, pdf, original.size)
            shift = 0 if local_box is not None else crop.height+30
            image = Image.new('RGB', (max(original.width, crop.width+20), original.height+shift), 'white')
            image.paste(original.convert('RGB'), (0, shift))
        boxes = {}
        for kind in ('depth_column', 'interpreted_lithology_column'):
            x0, y0, x1, y1 = annotation['bounding_boxes'][kind]
            boxes[kind] = [max(0, x0), max(0, y0)+shift,
                           min(image.width, x1), min(image.height, y1+shift)]
        if shift:
            image.paste(crop, (10, 0))
            boxes['lithology_legend'] = [10, 0, 10+crop.width, crop.height]
        else:boxes['lithology_legend'] = local_box
        rows, excluded, collapsed = truth_rows(folder, meta, shift, boxes['interpreted_lithology_column'])
        if not rows:
            skipped.append(dict(id=name, reason='No unambiguous, nonblank integer truth rows', excluded=excluded))
            image.close()
            continue
        for kind, box in boxes.items():
            if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
                raise ValueError(f'{name}: invalid {kind}: {box}')
        image.save(tiff, compression='tiff_adobe_deflate')
        write_json(annotation_path, dict(image_name=name+'.tif',
                   width=image.width, height=image.height, bounding_boxes=boxes))
        image.close()
        io.save_rows(out/'truth'/(name+'.csv'), rows)
        # Invalidate only this generated log's OCR cache after rebuilding pixels.
        cache = BAKEOFF/'out/_cache/legend'
        if previous != (signature(tiff), signature(annotation_path)):
            for path in cache.glob(name+'.*'):path.unlink()
            for path in (BAKEOFF/'out/_cache/depth').glob(name+'.*'):path.unlink()
        # Continuation pages share byte-identical legend pixels. Reuse only OCR
        # detections; parse() still discovers swatches and full-image boxes anew.
        ocr_key = (crop.size, hashlib.sha256(crop.tobytes()).hexdigest())
        cache_path = cache/(name+'.json')
        if not cache_path.exists() and ocr_key in ocr_by_pixels:
            cache.mkdir(parents=True, exist_ok=True)
            write_json(cache_path, {'ocr': ocr_by_pixels[ocr_key]})
        log = io.open_log(name)
        try:
            if log.mode != 'RGB':raise ValueError('Expected RGB TIFF')
            for box in boxes.values():
                for y0 in range(box[1], box[3], 4000):
                    rgb = io.read_rgb(log, box, y0=y0, y1=min(box[3], y0+4000))
                    if rgb.shape[2] != 3:raise ValueError('Invalid RGB strip')
            entries = legend.parse(log)
        finally:log.close()
        ocr_by_pixels[ocr_key] = json.loads(cache_path.read_text(encoding='utf-8'))['ocr']
        parsed = sorted({e.label for e in entries if e.is_lithology})
        labels = sorted({r['lithology'] for r in rows})
        matched = sorted(set(labels) & set(parsed))
        missing = sorted(set(labels)-set(parsed))
        notes = []
        if shift:notes.append(f'Original legend from p{legend_page} pasted above page; y shift={shift}')
        if meta.get('mixed_lithology_rows'):
            notes.append(f"{len(meta['mixed_lithology_rows'])} mixed-width intervals: exact dominant-width label; ties alphabetical")
        aliases = len(excluded)-collapsed
        if aliases:notes.append(f'{aliases} ambiguous-alias intervals excluded from scoring')
        if collapsed:notes.append(f'{collapsed} subpixel intervals occupy no integer rows')
        if meta['depth_fit'].get('source_unit') == 'm':notes.append('Metric source; truth depths converted to ft (m/0.3048)')
        notes.append(f"Depth OLS RMSE={meta['depth_fit']['rmse_ft']:.4g} ft; raster edges rounded to nearest row")
        if meta.get('known_limits'):notes.extend(meta['known_limits'])
        if missing:notes.append('Legend parse missing: '+ ' | '.join(missing))
        record = dict(id=name, source=source, well=well, page=meta['page'],
                      px_per_ft=1/meta['render']['ft_per_px'], rock_type_count=len(labels),
                      truth_intervals=len(rows), source_url=URLS.get(key, ''), notes='; '.join(notes))
        record['legend_parse_labels_matched/total'] = f'{len(matched)}/{len(labels)}'
        manifest.append(record)
        for a, b in windows(boxes['interpreted_lithology_column']):
            items.append(dict(log=name, y0=a, y1=b, tier='external'))
        audit.append(dict(id=name, extraction=str(folder.relative_to(ROOT)), source_pdf=str(pdf),
                          legend_source_page=legend_page, y_shift=shift, truth_labels=labels,
                          parsed_labels=parsed, matched=matched, missing=missing, excluded=excluded,
                          source_sha256=meta['pdf_sha256']))
        cards.append(card(out, name, rows, record, crop, meta))
        # Save a usable dataset even if a later source fails validation.
        write_json(out/'eval_set.json', items)
        write_json(out/'audit.json', dict(logs=audit, skipped=skipped))
        print(f'{name}: {len(rows)} intervals; legend {len(matched)}/{len(labels)}', flush=True)
    with (out/'manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    columns = min(5, len(cards))
    if not columns:raise ValueError('No exact truth logs found')
    heights = [max(c.height for c in cards[i:i+columns])+10 for i in range(0, len(cards), columns)]
    gallery = Image.new('RGB', (columns*470+10, sum(heights)+10), 'white')
    y = 10
    for row, row_height in enumerate(heights):
        for col, im in enumerate(cards[row*columns:(row+1)*columns]):gallery.paste(im, (10+col*470, y))
        y += row_height
    gallery.save(out/'gallery.png')
    body = ''.join(f'<figure><a href="gallery/{r["id"]}.png"><img src="gallery/{r["id"]}.png" '
                   f'alt="{html.escape(r["id"])}"></a><figcaption>{html.escape(r["notes"])}</figcaption></figure>'
                   for r in manifest)
    (out/'gallery.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Outside strip log test set</title><style>body{background:white;color:#222;font:15px Arial;'
        'margin:18px}main{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:460px;'
        'border:1px solid #bbb}img{width:100%;height:auto}figcaption{padding:12px;line-height:1.4}</style>'
        '<h1>Outside strip log test set</h1><p>Original legends and column pixels, beside truth cluster colours. '
        'Mixed columns use the largest visible width; ambiguous aliases are excluded. '
        'All depth fields use feet. See manifest.csv and audit.json for limitations.</p><main>'+body+'</main></html>',
        encoding='utf-8')
    summary = dict(logs=len(manifest), windows=len(items),
                   logs_by_source=dict(Counter(r['source'] for r in manifest)),
                   windows_by_source=dict(Counter(next(r['source'] for r in manifest if r['id']==i['log']) for i in items)),
                   legend_labels_matched=sum(len(r['matched']) for r in audit),
                   legend_labels_total=sum(len(r['truth_labels']) for r in audit),
                   excluded_intervals=sum(len(r['excluded']) for r in audit), skipped_logs=len(skipped))
    write_json(out/'build_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    if not sys.flags.isolated:raise SystemExit('Use python -I: source PDFs are untrusted data')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'external_tests')
    parser.add_argument('--gallery-only', action='store_true', help='Refresh cards from existing challenge images')
    args = parser.parse_args()
    if args.gallery_only:refresh_gallery(args.out)
    else:build(args.out)


def refresh_gallery(out):
    out = out.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must stay in the workspace')
    io.DATASET = out
    os.environ['LITHOLOGY_DATASET'] = str(out)
    with (out/'manifest.csv').open(encoding='utf-8', newline='') as f:manifest = list(csv.DictReader(f))
    audit = {r['id']: r for r in json.loads((out/'audit.json').read_text(encoding='utf-8'))['logs']}
    cards = []
    for record in manifest:
        name = record['id']
        record['px_per_ft'] = float(record['px_per_ft'])
        log = io.open_log(name)
        try:crop = Image.fromarray(io.read_rgb(log, 'lithology_legend'))
        finally:log.close()
        meta = json.loads((ROOT/audit[name]['extraction']/'meta.json').read_text(encoding='utf-8'))
        cards.append(card(out, name, io.load_rows(out/'truth'/(name+'.csv')), record, crop, meta))
    columns = min(5, len(cards))
    heights = [max(c.height for c in cards[i:i+columns])+10 for i in range(0, len(cards), columns)]
    gallery = Image.new('RGB', (columns*470+10, sum(heights)+10), 'white')
    y = 10
    for row, row_height in enumerate(heights):
        for col, im in enumerate(cards[row*columns:(row+1)*columns]):gallery.paste(im, (10+col*470, y))
        y += row_height
    gallery.save(out/'gallery.png')
    print(f'Refreshed {len(cards)} readable cards; gallery {gallery.width} x {gallery.height}', flush=True)


if __name__ == '__main__':main()

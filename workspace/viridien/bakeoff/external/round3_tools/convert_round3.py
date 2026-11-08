"""Convert round-three triage data; invoke with python -I and explicit paths.

Only trusted repository code is imported. Downloaded files and JSON are data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
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
Image.MAX_IMAGE_PIXELS = 160_000_000


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def safe(value):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', value).strip('_')


def documents(inputs):
    for triage in sorted(inputs.glob('*/triage.json')):
        data = json.loads(triage.read_text(encoding='utf-8'))
        for doc in data.get('documents', []):
            path = (ROOT / doc['path']).resolve()
            if not path.is_relative_to(inputs.resolve()):
                raise ValueError('Input escapes round3: ' + str(path))
            yield triage.parent.name, doc, path


def extract(inputs, out):
    out = out.resolve()
    kgs = module('round3_kgs', ROOT/'bakeoff/external/kgs_extract.py')
    kgs.OUTPUT_ROOT = out/'extraction'
    records = []
    for source, doc, path in documents(inputs):
        if path.suffix.lower() != '.pdf' or not (
            doc.get('born_digital_wellsight') or
            ('vector' in doc.get('raster_or_vector', '') and doc.get('has_lith_column'))):
            continue
        dest = kgs.OUTPUT_ROOT/source/safe(path.stem)
        try:
            if (dest/'extraction.json').exists():
                pages = json.loads((dest/'extraction.json').read_text())['pages']
            else:
                pages = kgs.extract_pdf(path, dest)
            counts = Counter(p.get('reason', 'ok') for p in pages)
            records.append(dict(source=source, path=doc['path'], extraction=str(dest.relative_to(ROOT)),
                                pages=[{k:v for k,v in p.items() if k != 'meta'} for p in pages]))
            print(source, path.name, dict(counts), flush=True)
        except Exception as exc:
            records.append(dict(source=source, path=doc['path'], error=f'{type(exc).__name__}: {exc}'))
            print(source, path.name, records[-1]['error'], flush=True)
        write_json(out/'extraction_audit.json', records)
    return records


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def font(size=16):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def contact_sheet(paths, out, title):
    # Read only one source at a time; keep the combined sheet phone-readable.
    columns, cw, ch = 3, 320, 400
    count = max(1, len(paths))
    sheet = Image.new('RGB', (columns*cw, 50+math.ceil(count/columns)*ch), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill='black', font=font(18))
    for i, (label, path) in enumerate(paths):
        x, y = (i % columns)*cw, 50+(i//columns)*ch
        caption = label if len(label) <= 42 else label[:19]+'...'+label[-20:]
        draw.text((x+8, y+4), caption, fill='black', font=font(12))
        with Image.open(path) as im:
            im.thumbnail((cw-16, ch-38))
            sheet.paste(im.convert('RGB'), (x+8, y+30))
    if not paths:
        draw.text((12, 80), 'No exact truth pages. See extraction_audit.json.', fill='black', font=font())
    sheet.save(out)
    sheet.close()


LICENCE = 'Publicly downloadable source; no explicit reuse licence verified in triage. Retain source attribution.'


def package_truth(inputs, out):
    builder = module('round3_builder', ROOT/'bakeoff/external/build_testset.py')
    builder.font = font
    builder.io.DATASET = out
    os.environ['LITHOLOGY_DATASET'] = str(out)
    for sub in ('images', 'bbox_annotation', 'truth', 'gallery'):
        (out/sub).mkdir(parents=True, exist_ok=True)
    documents_by_path = {str(path.relative_to(ROOT)): (source, doc, path)
                         for source, doc, path in documents(inputs)}
    manifest, windows, audit, cards, successful = [], [], [], [], set()
    extraction = json.loads((out/'extraction_audit.json').read_text())
    for record in extraction:
        source, doc, pdf = documents_by_path[record['path']]
        for page in record.get('pages', []):
            if page['status'] != 'ok':
                continue
            # Ignore extractor-recorded absolute paths so copied datasets can
            # be rebuilt unchanged on Windows or a different Linux checkout.
            base = ROOT/record['extraction']
            folder = base/f'page_{page["page"]:03d}'
            if not (folder/'meta.json').exists():
                folder = base
            meta = json.loads((folder/'meta.json').read_text())
            annotation = json.loads((folder/'boxes.json').read_text())
            well = 'Mirror_Lake_P-20' if source == 'nwt_orogo' and pdf.stem == '02' else pdf.stem
            name = f'r3_{safe(source)}_{safe(well)}_p{meta["page"]:03d}'
            with Image.open(folder/'image.png') as original:
                crop, local_box, legend_page = builder.legend_crop(folder, meta, pdf, original.size)
                shift = crop.height+30 if local_box is None else 0
                image = Image.new('RGB', (max(original.width, crop.width+20), original.height+shift), 'white')
                image.paste(original, (0, shift))
            boxes = {}
            for kind in ('depth_column', 'interpreted_lithology_column'):
                x0, y0, x1, y1 = annotation['bounding_boxes'][kind]
                boxes[kind] = [max(0, x0), max(0, y0)+shift, min(image.width, x1), min(image.height, y1+shift)]
            if shift:
                image.paste(crop, (10, 0))
                boxes['lithology_legend'] = [10, 0, 10+crop.width, crop.height]
            else:
                boxes['lithology_legend'] = local_box
            rows, excluded, collapsed = builder.truth_rows(folder, meta, shift, boxes['interpreted_lithology_column'])
            if not rows:
                image.close()
                continue
            for box in boxes.values():
                if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
                    raise ValueError(f'Invalid box: {name}: {box}')
            image.save(out/'images'/f'{name}.tif', compression='tiff_adobe_deflate')
            write_json(out/'bbox_annotation'/f'{name}.json', dict(image_name=name+'.tif', width=image.width,
                       height=image.height, bounding_boxes=boxes))
            image.close()
            builder.io.save_rows(out/'truth'/f'{name}.csv', rows)
            labels = sorted({r['lithology'] for r in rows})
            notes = [f'Inherited source legend p{legend_page}; y shift={shift}.',
                     f'{len(excluded)} excluded alias/subpixel intervals ({collapsed} collapsed).',
                     'Legend OCR matching not run during conversion.',
                     f'Depth fit RMSE={meta["depth_fit"]["rmse_ft"]:.5g} ft.',
                     *meta.get('known_limits', []), LICENCE]
            row = dict(id=name, source=source, well=well, page=meta['page'],
                       px_per_ft=1/meta['render']['ft_per_px'], rock_type_count=len(labels),
                       truth_intervals=len(rows), source_url=doc['url'], notes=' '.join(notes),
                       licence_note=LICENCE)
            row['legend_parse_labels_matched/total'] = 'not_run'
            manifest.append(row)
            windows.extend(dict(log=name, y0=a, y1=b, tier='external_round3')
                           for a, b in builder.windows(boxes['interpreted_lithology_column'])
                           if any(r['top_y'] < b and r['bottom_y'] > a for r in rows))
            audit.append(dict(id=name, extraction=str(folder.relative_to(ROOT)), source_pdf=record['path'],
                              source_sha256=meta['pdf_sha256'], legend_source_page=legend_page,
                              y_shift=shift, excluded=excluded, truth_labels=labels))
            card = builder.card(out, name, rows, row, crop, meta)
            card.close()
            crop.close()
            cards.append((name, out/'gallery'/f'{name}.png'))
            successful.add((record['path'], meta['page']))
            print(name, len(rows), 'truth intervals', flush=True)
    write_csv(out/'manifest.csv', manifest, builder.MANIFEST_FIELDS+['licence_note'])
    write_json(out/'eval_set.json', windows)
    write_json(out/'audit.json', dict(logs=audit))
    contact_sheet(cards, out/'gallery.png', 'Round 3: source legends and exact truth')
    summary = dict(logs=len(manifest), source_documents=len({r['source_pdf'] for r in audit}), windows=len(windows),
                   truth_intervals=sum(r['truth_intervals'] for r in manifest), by_source={})
    for source in sorted({r['source'] for r in manifest}):
        chosen = [r for r in manifest if r['source'] == source]
        ids = {r['id'] for r in chosen}
        summary['by_source'][source] = dict(logs=len(chosen), wells=len({r['well'] for r in chosen}),
            windows=sum(w['log'] in ids for w in windows), truth_intervals=sum(r['truth_intervals'] for r in chosen))
    write_json(out/'build_summary.json', summary)
    return successful


def raster_groups(docs):
    """Associate Mississippi continuation pages only with a positively reviewed key."""
    return {re.sub(r'-\d{3}$', '', path.stem): doc for source, doc, path in docs
            if source == 'mississippi' and doc.get('has_legend') is True}


def pages_for_document(source, doc, path):
    if source == 'nwt_orogo' and path.stem == '02':
        return list(range(65, 99)) + list(range(100, 119))
    return list(range(1, int(doc.get('pages', 1))+1))


def render_pdf_page(page):
    # <=32 million RGB pixels: bounded independently of untrusted page sizes.
    area = page.rect.width*page.rect.height
    if not math.isfinite(area) or area <= 0:
        raise ValueError('Invalid page dimensions')
    scale = min(200/72, 2000/page.rect.width, math.sqrt(32_000_000/area))
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    return Image.frombytes('RGB', (pix.width, pix.height), pix.samples), scale


def draft_vector_boxes(page, scale, name, size):
    """A conservative semantic detector: printed headers and enclosing long rules.

    No guesses based solely on column order; require both depth/lithology headers.
    Reject landscape pages, multiple candidate headings and fragmented rules.
    """
    if page.rotation or page.rect.width > page.rect.height:
        return None
    words = page.get_text('words')
    lith = [w for w in words if w[4].lower() in ('lithology', 'lith') and w[1] < page.rect.height*.4]
    depth = [w for w in words if w[4].lower() in ('depth', 'm.depth') and w[1] < page.rect.height*.4]
    if len(lith) != 1 or len(depth) != 1:
        return None
    # get_drawings can be huge on Epoch pages; call only for sparse WellSight headers.
    if len(page.read_contents()) > 400_000:
        return None
    borders = []
    for drawing in page.get_drawings():
        for item in drawing['items']:
            if item[0] == 'l':
                a, b = item[1:3]
                if abs(a.x-b.x) < .01 and abs(a.y-b.y) > page.rect.height*.45:
                    borders.append((a.x, min(a.y,b.y), max(a.y,b.y)))
    boxes = {}
    for kind, word in [('depth_column', depth[0]), ('interpreted_lithology_column', lith[0])]:
        cx = (word[0]+word[2])/2
        lefts = [b for b in borders if b[0] < cx and b[1] < word[3]+40]
        rights = [b for b in borders if b[0] > cx and b[1] < word[3]+40]
        if not lefts or not rights:
            return None
        left, right = max(lefts), min(rights)
        box = [round(left[0]*scale), round(max(left[1],right[1],word[3])*scale),
               round(right[0]*scale), round(min(left[2],right[2])*scale)]
        if not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
            return None
        boxes[kind] = box
    if boxes['depth_column'][2] > boxes['interpreted_lithology_column'][0]:
        return None
    return dict(image_name=name+'.tif', width=size[0], height=size[1], draft=True,
                detector='printed header words plus enclosing long vector rules', bounding_boxes=boxes,
                notes='Draft track boxes only; legend remains on referenced source page. Manual review required.')


def package_unlabelled(inputs, out, successful):
    for sub in ('images', 'bbox_annotation', 'thumbnails'):
        (out/sub).mkdir(parents=True, exist_ok=True)
    docs = list(documents(inputs))
    groups = raster_groups(docs)
    manifest, audit, failures, contacts = [], [], [], []
    for source, doc, path in docs:
        group = re.sub(r'-\d{3}$', '', path.stem)
        has_key = doc.get('has_legend') is True or (source == 'mississippi' and group in groups)
        if not doc.get('is_log') or not doc.get('has_lith_column') or not has_key:
            audit.append(dict(path=doc['path'], status='excluded', reason='No confirmed lithology column and associated legend'))
            continue
        pdf = None
        try:
            if path.suffix.lower() == '.pdf':
                pdf = fitz.open(path)
            for page_number in pages_for_document(source, doc, path):
                if (doc['path'], page_number) in successful:
                    continue
                name = f'ul_r3_{safe(source)}_{safe(path.stem)}_p{page_number:03d}'
                dest = out/'images'/f'{name}.tif'
                family = doc.get('layout_guess', 'other')
                notes = [doc.get('notes', ''), LICENCE]
                if pdf is not None:
                    page = pdf[page_number-1]
                    image, scale = render_pdf_page(page)
                    notes.append(f'PDF rendered at {scale*72:.2f} dpi, capped at 2000 px wide / 32 MP; source rotation retained.')
                    physical_page = page_number
                else:
                    with Image.open(path) as original:
                        if original.width*original.height > 32_000_000:
                            raise ValueError('Raster input exceeds 32 MP conversion budget')
                        image = original.copy()
                    physical_page = int(path.stem.rsplit('-',1)[-1]) if source == 'mississippi' else page_number
                    scale = None
                try:
                    annotation = None
                    if pdf is not None and 'vector' in doc.get('raster_or_vector',''):
                        annotation = draft_vector_boxes(page, scale, name, image.size)
                    image.save(dest, compression='tiff_adobe_deflate')
                    if annotation:
                        write_json(out/'bbox_annotation'/f'{name}.json', annotation)
                    image.thumbnail((310, 360))
                    thumb = out/'thumbnails'/f'{name}.png'
                    image.convert('RGB').save(thumb)
                finally:
                    image.close()
                if source == 'mississippi':
                    well = re.search(r'API ([0-9]+)', doc.get('notes',''))
                    well = well.group(1) if well else group
                    notes.append(f'Associated key: {groups[group]["path"]}. Page belongs to viewer log {group}.')
                elif source == 'nwt_orogo':
                    well = {'01':'Cameron_J-74','02':'Mirror_Lake_P-20'}.get(path.stem,path.stem)
                else:
                    well = path.stem
                notes.append(f'Legend source page(s): {doc.get("legend_page")}. Boxes: '+('draft' if annotation else 'not reliably detected'))
                row = dict(id=name, source=source, well=well, page=physical_page, source_url=doc['url'],
                           layout_family_guess=family, has_legend=True, notes=' '.join(notes))
                manifest.append(row)
                contacts.append((name, thumb))
            print(source, path.name, 'unlabelled exported', flush=True)
        except Exception as exc:
            failures.append(dict(path=doc['path'], error=f'{type(exc).__name__}: {exc}'))
            print('FAILED', path.name, failures[-1]['error'], flush=True)
        finally:
            if pdf is not None:
                pdf.close()
        write_csv(out/'manifest.csv', manifest, ['id','source','well','page','source_url','layout_family_guess','has_legend','notes'])
    write_json(out/'audit.json', dict(excluded=audit, failures=failures))
    contact_sheet(contacts, out/'contact_sheet.png', 'Round 3: unlabelled pages (keys may be on a companion page)')
    # Small per-source sheets are easier to navigate than a single long overview.
    for source in sorted({r['source'] for r in manifest}):
        contact_sheet([(label,path) for label,path in contacts if label.startswith('ul_r3_'+source+'_')],
                      out/f'contact_sheet_{source}.png', source)
    summary = dict(pages=len(manifest), by_family=dict(Counter(r['layout_family_guess'] for r in manifest)),
                   by_source=dict(Counter(r['source'] for r in manifest)), failures=failures,
                   draft_boxes=len(list((out/'bbox_annotation').glob('*.json'))))
    write_json(out/'build_summary.json', summary)
    return summary


def package(inputs, truth_out, unlabelled_out):
    inputs, truth_out, unlabelled_out = inputs.resolve(), truth_out.resolve(), unlabelled_out.resolve()
    successful = package_truth(inputs, truth_out)
    summary = package_unlabelled(inputs, unlabelled_out, successful)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    if not sys.flags.isolated:
        raise SystemExit('Use python -I')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--truth-out', type=Path, required=True)
    parser.add_argument('--unlabelled-out', type=Path, required=True)
    parser.add_argument('--stage', choices=['extract', 'package', 'all'], default='all')
    args = parser.parse_args()
    if args.truth_out.resolve() != ROOT/'external_tests_round3' or args.unlabelled_out.resolve() != ROOT/'external_unlabelled':
        parser.error('Outputs must be the lane-owned dataset roots')
    args.truth_out.mkdir(parents=True, exist_ok=True)
    if args.stage in ('extract', 'all'):
        extract(args.inputs, args.truth_out)
    if args.stage in ('package', 'all'):
        package(args.inputs, args.truth_out, args.unlabelled_out)


if __name__ == '__main__':
    main()

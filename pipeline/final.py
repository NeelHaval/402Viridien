"""Batch S19 inference and final exports. Each page has its own process."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def page(image, annotation, folder, allow_assumed=False):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '2'
    from . import run as setup
    from PIL import Image, ImageDraw, ImageFont
    from harness import io, depth
    from ml.combine.runtime import run
    Image.MAX_IMAGE_PIXELS = None
    model_root = Path(__file__).parent / 's19'
    pins = json.loads((model_root / 'models/sha256.json').read_text(encoding='utf-8'))
    for name, digest in pins.items():
        if sha(model_root / name) != digest:
            raise ValueError('Model hash mismatch: ' + name)
    data = json.loads(annotation.read_text(encoding='utf-8'))
    boxes = data.get('bounding_boxes', data)
    log = io.Log(Image.open(image), boxes, image.stem)
    if log.image.format != 'TIFF':
        log._strip_backend = False
    # Put all mutable depth caches under this page's output directory.
    io.ROOT = folder / 'cache'
    doc = dict(page=image.stem, image=str(image), annotation=str(annotation),
               image_sha256=sha(image), annotation_sha256=sha(annotation),
               dimensions=list(log.size), model_config_hashes=pins,
               source_hashes={str(p.relative_to(Path(__file__).parent)): sha(p)
                              for p in Path(__file__).parent.rglob('*.py')},
               depth_status='unavailable', reason='', unit_conversion=False)
    try:
        box = list(map(int, boxes['interpreted_lithology_column']))
        doc['full_column_bounds'] = box
        calib = None
        try:
            calib = depth.calibrate(log)
            receipt = json.loads((io.ROOT / 'out/_cache/depth' / (log.name + '.rapidocr.json')).read_text(encoding='utf-8'))
            doc['calibration'] = vars(calib).copy()
            doc['unit_status'] = receipt.get('unit', 'unknown')
            doc['unit_text'] = receipt.get('unit_text', [])
            if '+m-to-ft' in calib.method:
                calib = depth.Calib(calib.a / depth.METRES_TO_FEET, calib.b / depth.METRES_TO_FEET,
                                    calib.n_points, calib.rmse_px, calib.method.replace('+m-to-ft', '+m'))
            if doc['unit_status'] not in ('ft', 'm') and not allow_assumed:
                raise ValueError('Printed depth unit is not verified')
            if calib.n_points < 3 or calib.rmse_px > 4:
                raise ValueError('Depth fit needs >=3 anchors and <=4 pixel RMSE')
            if not all(math.isfinite(calib.to_depth(y)) and calib.to_depth(y) >= 0 for y in (box[1], box[3])):
                raise ValueError('Invalid depth at column edge')
            doc['depth_status'] = 'calibrated' if doc['unit_status'] in ('ft', 'm') else 'ft-assumed'
            doc['output_calibration'] = vars(calib).copy()
        except Exception as exc:
            doc['reason'] = str(exc)
            calib = None
        # Segmentation is independent of depth. Use the pinned OCR fit directly,
        # as the official exporter did, without the runtime's secondary refit.
        rows = run(log, (box[1], box[3]), {'calib': None, 'device': 'cpu', 'cache': str(folder / 'cache')})
        if not rows:
            raise ValueError('S19 returned no intervals')
        for row in rows:
            for edge in ('top', 'bottom'):
                row[edge + '_depth'] = round(calib.to_depth(row[edge + '_y']), 1) if calib else ''
        prefix = folder / (image.stem + '.official')
        raw = Path(str(prefix) + '.raw.csv')
        fields = ['top_y', 'bottom_y', 'lithology', 'cluster_id', 'top_depth', 'bottom_depth']
        write_csv(raw, fields, rows)
        csvdir = folder / 'csv'
        csvdir.mkdir(exist_ok=True)
        write_csv(csvdir / (image.stem + '.official.raw.notebook.csv'),
                  ['top_depth', 'bottom_depth', 'lithology', 'top_y', 'bottom_y', 'cluster_id'], rows)
        write_csv(csvdir / (image.stem + '.official.raw.slide.csv'),
                  ['top_y', 'bottom_y', 'lithology', 'cluster_id', 'top_depth', 'base_depth'],
                  [dict(r, base_depth=r['bottom_depth']) for r in rows])
        for output in csvdir.glob('*.csv'):
            shutil.copyfile(output, folder / output.name)
        x0, y0, x1, y1 = box
        header = 70
        tx = max(150, x1-x0+30)
        width = max(1000, tx+800)
        canvas = Image.new('RGB', (width, y1-y0+header), 'white')
        canvas.paste(io._crop(log, tuple(box)), (8, header))
        draw = ImageDraw.Draw(canvas)
        small = ImageFont.load_default(size=8)
        draw.text((8, 5), image.stem, fill='black')
        draw.text((8, 27), 'Depth: ' + doc['depth_status'] + '; unit: ' + doc.get('unit_status', 'unknown'), fill='black')
        draw.text((8, 48), doc['reason'], fill='#a02020')
        for row in rows:
            a, b = int(row['top_y']), int(row['bottom_y'])
            yy, bb = header+a-y0, header+b-y0
            draw.line((8, yy, tx, yy), fill='#d02020')
            draw.line((8, bb, tx, bb), fill='#d02020')
            draw.rectangle((tx, yy, width-5, max(yy, bb-1)), fill='#e5ebf2', outline='#657987')
            label = f"y[{a},{b}) depth[{row['top_depth']},{row['bottom_depth']}] {doc.get('unit_status', '')} {row['lithology']} cluster={row['cluster_id']}"
            draw.text((tx+4, yy), label, font=small, fill='black')
        canvas.save(Path(str(prefix) + '.full-column.png'), compress_level=1)
        doc['rows'] = len(rows)
        doc['status'] = 'ok'
        for output in sorted(folder.rglob('*')):
            if output.is_file() and output.name != 'worker.log' and 'cache' not in output.relative_to(folder).parts:
                write(output.with_suffix(output.suffix + '.sidecar.json'),
                      dict(page=image.stem, sha256=sha(output), depth_status=doc['depth_status'],
                           unit_status=doc.get('unit_status', 'unknown'), reason=doc['reason']))
        doc['outputs'] = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*')
                          if p.is_file() and p.name != 'worker.log' and 'cache' not in p.relative_to(folder).parts}
        import rapidocr_onnxruntime
        doc['ocr_model_hashes'] = {p.name: sha(p) for p in (Path(rapidocr_onnxruntime.__file__).parent / 'models').glob('*.onnx')}
        weights = Path.home() / '.EasyOCR/model/english_g2.pth'
        if weights.exists():
            doc['ocr_model_hashes'][weights.name] = sha(weights)
        return doc
    finally:
        log.close()


def launch(image, args):
    start = time.monotonic()
    folder = args.out / image.stem
    folder.mkdir(parents=True, exist_ok=True)
    annotation = args.boxes / (image.stem + '.json')
    command = [sys.executable, '-m', 'pipeline.final', '--page', str(image),
               '--boxes', str(annotation), '--out', str(folder)]
    if args.allow_assumed_units:
        command.append('--allow-assumed-units')
    try:
        with (folder / 'worker.log').open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
        if result.returncode:
            raise RuntimeError('Worker failed; see ' + str(folder / 'worker.log'))
        doc = json.loads((folder / 'document.sidecar.json').read_text(encoding='utf-8'))
    except Exception as exc:
        doc = dict(page=image.stem, rows=0, status='failed', depth_status='unavailable', reason=str(exc),
                   image_sha256=sha(image), annotation_sha256=sha(annotation) if annotation.is_file() else None)
    doc['seconds'] = round(time.monotonic()-start, 3)
    write(folder / 'document.sidecar.json', doc)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--boxes', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--allow-assumed-units', action='store_true', help='Reproduce provisional official numeric depths even when units are assumed')
    parser.add_argument('--page', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.page:
        write(args.out / 'document.sidecar.json', page(args.page, args.boxes, args.out, args.allow_assumed_units))
        return 0
    if not args.images or args.workers < 1 or args.timeout <= 0:
        parser.error('--images, positive --workers and positive --timeout are required')
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'))
    if not images:
        parser.error('No supported images found')
    if len({p.stem for p in images}) != len(images):
        parser.error('Duplicate image stems would overwrite outputs')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Output directory must be empty; use a new directory for each batch')
    args.out.mkdir(parents=True, exist_ok=True)
    docs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(launch, image, args) for image in images]
        for future in concurrent.futures.as_completed(futures):
            doc = future.result()
            docs.append(doc)
            print('Finished ' + doc['page'] + ': ' + doc['status'], flush=True)
    manifest = dict(version='s19-final-v1', documents=sorted(docs, key=lambda d: d['page']),
                    complete=all(d['status'] == 'ok' for d in docs),
                    depth_complete=all(d['depth_status'] == 'calibrated' for d in docs))
    manifest['files'] = {str(p.relative_to(args.out)): sha(p) for p in args.out.rglob('*')
                         if p.is_file() and 'cache' not in p.relative_to(args.out).parts and p.name != 'official_manifest.json'}
    write(args.out / 'official_manifest.json', manifest)
    print('page\trows\tdepth status\tseconds', flush=True)
    for doc in manifest['documents']:
        print(f"{doc['page']}\t{doc['rows']}\t{doc['depth_status']}\t{doc['seconds']}", flush=True)
    return 0 if manifest['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

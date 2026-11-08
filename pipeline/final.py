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
import warnings


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


class InputError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def discover(images, boxes):
    pictures = sorted(p for p in images.rglob('*') if p.is_file() and
                      p.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'))
    annotations = sorted(p for p in boxes.rglob('*') if p.is_file() and p.suffix.lower() == '.json')
    jobs, used = [], set()
    for image in pictures:
        relative = image.relative_to(images)
        exact = [p for p in annotations if p.relative_to(boxes).with_suffix('') == relative.with_suffix('')]
        matches = exact or [p for p in annotations if p.stem == image.stem]
        annotation = matches[0] if len(matches) == 1 else None
        if annotation:
            used.add(annotation)
        jobs.append((image, annotation, relative.with_suffix('').as_posix(),
                     'ambiguous_annotation' if len(matches) > 1 else 'missing_annotation'))
    counts = {}
    for job in jobs:
        counts[job[2].casefold()] = counts.get(job[2].casefold(), 0) + 1
    jobs = [(im, ann, key + im.suffix.lower() if counts[key.casefold()] > 1 else key, error)
            for im, ann, key, error in jobs]
    return jobs, [str(p.relative_to(boxes)) for p in annotations if p not in used]


def bounded_log(image, boxes, io):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 1_000_000_000
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', Image.DecompressionBombWarning)
        source = Image.open(image)
    log = io.Log(source, boxes, image.stem)
    original_crop = io._crop
    def probe(item):
        import tifffile
        item._strip_backend = False
        try:
            item._tiff = tifffile.TiffFile(image)
            page = item._tiff.pages[0]
            if (not page.is_tiled and page.planarconfig == 1 and page.photometric in (0,1,2,3)
                    and max(page.databytecounts) <= 64_000_000
                    and page.rowsperstrip * page.imagewidth * max(1, page.samplesperpixel) * page.dtype.itemsize <= 64_000_000):
                page.decode(None, 0)
                item._strip_backend = True
        except (ImportError, ValueError, NotImplementedError):
            pass
    def crop(item, box):
        x0, y0, x1, y1 = map(int, box)
        if not (0 <= x0 < x1 <= item.size[0] and 0 <= y0 < y1 <= item.size[1]):
            raise InputError('invalid_bounds', 'Crop is outside the image: ' + str(box))
        if (x1-x0)*(y1-y0) > 32_000_000:
            raise InputError('crop_limit', 'Crop exceeds 32 million pixels')
        if item._strip_backend is None:
            probe(item) if item.image.format == 'TIFF' else setattr(item, '_strip_backend', False)
        bytes_per_pixel = {'1': 1, 'L': 1, 'P': 1, 'I;16': 2, 'I;16B': 2, 'I;16L': 2}.get(item.mode, 4)
        decoded_bytes = item.size[0]*item.size[1]*bytes_per_pixel
        if not item._strip_backend and decoded_bytes > 256_000_000:
            raise InputError('decode_limit', 'Full-page decoding exceeds 256 MB estimated raster budget; supported TIFF strips required')
        return original_crop(item, box)
    io._crop = crop
    return log


def page(image, annotation, folder, allow_assumed=False):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '2'
    from . import run as setup
    from PIL import Image, ImageDraw, ImageFont
    from harness import io, depth
    from ml.combine.runtime import run
    model_root = Path(__file__).parent / 's19'
    pins = json.loads((model_root / 'models/sha256.json').read_text(encoding='utf-8'))
    for name, digest in pins.items():
        if sha(model_root / name) != digest:
            raise ValueError('Model hash mismatch: ' + name)
    data = json.loads(annotation.read_text(encoding='utf-8'))
    boxes = data.get('bounding_boxes', data)
    log = bounded_log(image, boxes, io)
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
        if (box[2]-box[0])*(box[3]-box[1]) > 32_000_000:
            raise InputError('crop_limit', 'Column exceeds 32 million pixels')
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
        preview_scale = min(1.0, 32_000_000 / (width * (y1-y0+header)))
        canvas = Image.new('RGB', (width, max(header+1, int((y1-y0)*preview_scale)+header)), 'white')
        column = io._crop(log, tuple(box))
        if preview_scale < 1:
            column = column.resize((column.width, max(1, int(column.height*preview_scale))))
        canvas.paste(column, (8, header))
        doc['preview_vertical_scale'] = preview_scale
        doc['frame_index'] = 0
        draw = ImageDraw.Draw(canvas)
        small = ImageFont.load_default(size=8)
        draw.text((8, 5), image.stem, fill='black')
        draw.text((8, 27), 'Depth: ' + doc['depth_status'] + '; unit: ' + doc.get('unit_status', 'unknown'), fill='black')
        draw.text((8, 48), doc['reason'], fill='#a02020')
        for row in rows:
            a, b = int(row['top_y']), int(row['bottom_y'])
            yy, bb = header+int((a-y0)*preview_scale), header+int((b-y0)*preview_scale)
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


def launch(job, args):
    image, annotation, page_id, match_error = job
    start = time.monotonic()
    folder = args.out / page_id
    if folder.exists() and not args.force:
        receipt = folder / 'document.sidecar.json'
        if receipt.is_file():
            try:
                cached = json.loads(receipt.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                cached = {}
            if (cached.get('status') == 'ok' and annotation and
                    cached.get('image_sha256') == sha(image) and
                    cached.get('annotation_sha256') == sha(annotation) and
                    cached.get('allow_assumed_units') == args.allow_assumed_units):
                return cached
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'pipeline.final', '--page', str(image),
               '--boxes', str(annotation), '--out', str(folder)]
    if args.allow_assumed_units:
        command.append('--allow-assumed-units')
    try:
        if annotation is None:
            raise InputError(match_error, 'No unique bounding-box JSON for ' + str(image))
        with (folder / 'worker.log').open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
        if result.returncode and not (folder / 'document.sidecar.json').exists():
            raise RuntimeError('Worker failed; see ' + str(folder / 'worker.log'))
        doc = json.loads((folder / 'document.sidecar.json').read_text(encoding='utf-8'))
    except Exception as exc:
        doc = dict(page=image.stem, rows=0, status='failed', depth_status='unavailable', reason=str(exc),
                   image_sha256=sha(image), annotation_sha256=sha(annotation) if annotation and annotation.is_file() else None,
                   failure_type=getattr(exc, 'code', 'timeout' if isinstance(exc, subprocess.TimeoutExpired) else type(exc).__name__))
    doc['allow_assumed_units'] = args.allow_assumed_units
    doc['page'] = page_id
    doc['seconds'] = round(time.monotonic()-start, 3)
    write(folder / 'document.sidecar.json', doc)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--boxes', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('root', type=Path, nargs='?')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--force', action='store_true', help='Replace per-page outputs; default resumes successful pages with unchanged inputs')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--allow-assumed-units', action='store_true', help='Reproduce provisional official numeric depths even when units are assumed')
    parser.add_argument('--page', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.selftest:
        import tempfile
        sample = Path(__file__).resolve().parents[1] / 'tests/assets/final-sample'
        with tempfile.TemporaryDirectory(prefix='s19-selftest-') as directory:
            result = subprocess.run([sys.executable, '-m', 'pipeline.final',
                '--data', str(sample), '--out', directory, '--workers', '1', '--timeout', '50'], timeout=55)
            manifest = json.loads((Path(directory) / 'official_manifest.json').read_text(encoding='utf-8'))
            print('SELFTEST ' + ('PASS' if manifest['complete'] else 'FAIL'), flush=True)
            return result.returncode
    if not args.out:
        parser.error('--out is required')
    if args.page:
        try:
            doc = page(args.page, args.boxes, args.out, args.allow_assumed_units)
        except Exception as exc:
            doc = dict(page=args.page.stem, rows=0, status='failed', depth_status='unavailable',
                       reason=str(exc), failure_type=getattr(exc, 'code', type(exc).__name__))
        write(args.out / 'document.sidecar.json', doc)
        return 0 if doc['status'] == 'ok' else 1
    root = args.data or args.root
    if root is None and args.images and (args.images / 'images').is_dir():
        root = args.images
    if root is None and args.images is None:
        root = Path.cwd()
    if root:
        args.images = root / 'images'
        args.boxes = args.boxes or root / 'bbox_annotation'
    if not args.images or not args.boxes or args.workers < 1 or args.timeout <= 0:
        parser.error('Supply --data or --images/--boxes, positive workers and timeout')
    if not args.images.is_dir() or not args.boxes.is_dir():
        parser.error('Image and bounding-box directories must exist')
    jobs, extra = discover(args.images, args.boxes)
    if not jobs:
        parser.error('No supported images found')
    ids = [job[2] for job in jobs]
    if len({name.casefold() for name in ids}) != len(ids):
        parser.error('Image paths have duplicate stems; rename conflicting files')
    args.out.mkdir(parents=True, exist_ok=True)
    docs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(launch, image, args) for image in jobs]
        for future in concurrent.futures.as_completed(futures):
            doc = future.result()
            docs.append(doc)
            print('Finished ' + doc['page'] + ': ' + doc['status'], flush=True)
    manifest = dict(version='s19-final-v2', extra_annotations=extra, documents=sorted(docs, key=lambda d: d['page']),
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

"""Validated, portable manifests and bounded image access.

Coordinates are full-image pixels with exclusive bottoms/right edges. A Pillow
crop may decode an entire TIFF: the fallback is therefore guarded by estimated
native decode bytes. Optional tifffile can decode supported individual strips.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

FIELDS = ['top_y', 'bottom_y', 'lithology', 'cluster_id', 'top_depth', 'bottom_depth']
SOURCES = {'human', 'synthetic', 'external_labelled', 'external_unlabelled',
           'challenge_unlabelled', 'organizer', 'silver'}
SPLITS = {'train', 'dev', 'test', 'verify'}
Image.MAX_IMAGE_PIXELS = None  # explicit byte/window limits below replace Pillow's heuristic


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an integer')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be an integer') from error
    if not math.isfinite(number) or int(number) != number:
        raise ValueError(f'{name} must be a finite integer')
    return int(number)


def validate_intervals(rows: list[dict], window=None) -> list[dict]:
    """Return normalized rows; reject unsorted/overlapping and nonfinite values."""
    result, previous = [], None
    for index, original in enumerate(rows):
        row = dict(original)
        for key in ('top_y', 'bottom_y'):
            row[key] = _int(row.get(key), f'row {index} {key}')
        a, b = row['top_y'], row['bottom_y']
        if a < 0 or b <= a:
            raise ValueError(f'row {index} must have positive height and nonnegative top')
        if previous is not None and a < previous:
            raise ValueError(f'row {index}: intervals must be sorted and non-overlapping')
        if window is not None and not (window[0] <= a < b <= window[1]):
            raise ValueError(f'row {index} outside window {window}')
        if not isinstance(row.get('lithology'), str) or not row['lithology'].strip():
            raise ValueError(f'row {index}: lithology must be nonempty text')
        row['cluster_id'] = _int(row.get('cluster_id'), f'row {index} cluster_id (required)')
        # The slide deck calls this base_depth; the shared scorer calls it bottom_depth.
        if row.get('base_depth') not in (None, ''):
            if row.get('bottom_depth') not in (None, '') and float(row['base_depth']) != float(row['bottom_depth']):
                raise ValueError('conflicting base_depth and bottom_depth')
            row['bottom_depth'] = row.pop('base_depth')
        for key in ('top_depth', 'bottom_depth'):
            if row.get(key) in (None, ''):
                row.pop(key, None)
            else:
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f'row {index} {key} must be finite')
        if ('top_depth' in row) != ('bottom_depth' in row):
            raise ValueError(f'row {index}: both depth endpoints are required together')
        result.append(row)
        previous = b
    return result


def load_intervals(path: str | Path, window=None) -> list[dict]:
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    return validate_intervals(rows, window=window)


def save_intervals(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(validate_intervals(rows))


def load_boxes(path: str | Path | dict, image_size=None) -> dict:
    data = dict(path) if isinstance(path, dict) else json.loads(Path(path).read_text(encoding='utf-8-sig'))
    boxes = data.get('bounding_boxes', data)
    if not isinstance(boxes, dict) or 'interpreted_lithology_column' not in boxes:
        raise ValueError('annotation needs interpreted_lithology_column')
    width, height = image_size or (data.get('width'), data.get('height'))
    if width is None or height is None:
        raise ValueError('image dimensions required to validate bounding boxes')
    if data.get('width', width) != width or data.get('height', height) != height:
        raise ValueError('annotation dimensions disagree with image header')
    normalized = {}
    for name, raw in boxes.items():
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValueError(f'invalid bounding box {name}')
        x0, y0, x1, y1 = [_int(v, f'{name} coordinate') for v in raw]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f'bounding box {name} outside image dimensions')
        normalized[name] = [x0, y0, x1, y1]
    return normalized


def load_manifest(path: str | Path, check_leakage=True) -> dict:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('records'), list):
        raise ValueError('expected schema_version=1 and records list')
    ids, hashes, groups, hash_cache = set(), {}, {}, {}
    records = []
    for original in manifest['records']:
        record = dict(original)
        rid = record.get('id')
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError(f'invalid or duplicate record id: {rid!r}')
        ids.add(rid)
        if record.get('source') not in SOURCES or record.get('split') not in SPLITS:
            raise ValueError(f'{rid}: invalid source or split')
        if not isinstance(record.get('group'), str) or not record['group']:
            raise ValueError(f'{rid}: group required for leakage detection')
        if not isinstance(record.get('tags', []), list):
            raise ValueError(f'{rid}: tags must be a list')
        for key in ('image', 'annotation', 'truth'):
            value = record.get(key)
            if value is None:
                if key != 'truth':
                    raise ValueError(f'{rid}: {key} required')
                record[key] = None
                continue
            file_path = (path.parent / value).resolve()
            if not file_path.is_file():
                raise ValueError(f'{rid}: missing {key}: {file_path}')
            record[key] = str(file_path)
        with Image.open(record['image']) as image:
            size = image.size
        boxes = load_boxes(record['annotation'], size)
        column = boxes['interpreted_lithology_column']
        window = record.get('window')
        if not isinstance(window, (tuple, list)) or len(window) != 2:
            raise ValueError(f'{rid}: explicit window required (including partial truths)')
        window = [_int(v, 'window') for v in window]
        if not (column[1] <= window[0] < window[1] <= column[3]):
            raise ValueError(f'{rid}: window outside lithology bounding box')
        record['window'] = window
        if record['truth']:
            if not record.get('label_provenance'):
                raise ValueError(f'{rid}: labelled data require label_provenance')
            truth = load_intervals(record['truth'], window=(0, size[1]))
            if not any(row['top_y'] < window[1] and row['bottom_y'] > window[0] for row in truth):
                raise ValueError(f'{rid}: labelled scoring window contains no truth-filled pixels')
            coverage = record.get('truth_window')
            if coverage is None:
                raise ValueError(f'{rid}: truth_window required; partial truth must be explicit')
            coverage = [_int(v, 'truth_window') for v in coverage]
            if len(coverage) != 2 or not (0 <= coverage[0] <= window[0] < window[1] <= coverage[1] <= size[1]):
                raise ValueError(f'{rid}: scoring window outside declared truth coverage')
            record['truth_window'] = coverage
        elif record.get('source') in {'human', 'organizer', 'external_labelled', 'silver', 'synthetic'}:
            raise ValueError(f'{rid}: labelled source needs truth; use an unlabelled source instead')
        if 'unlabelled' in record['source'] and record['truth']:
            raise ValueError(f'{rid}: unlabelled sources cannot carry truth')
        if record['image'] not in hash_cache:
            hash_cache[record['image']] = sha256(record['image'])
        digest = hash_cache[record['image']]
        if record.get('image_sha256') and digest != record['image_sha256']:
            raise ValueError(f'{rid}: image checksum mismatch')
        record['image_sha256'] = digest
        for key in ('annotation', 'truth'):
            actual = sha256(record[key]) if record[key] else None
            if record.get(key+'_sha256') and record[key+'_sha256'] != actual:
                raise ValueError(f'{rid}: {key} checksum mismatch')
            record[key+'_sha256'] = actual
        if check_leakage:
            for store, key, noun in ((groups, record['group'], 'group'), (hashes, digest, 'duplicate image')):
                if key in store and store[key] != record['split']:
                    raise ValueError(f'{rid}: {noun} leakage across splits')
                store[key] = record['split']
            lineage = record.get('lineage', {})
            if record['source'] == 'synthetic':
                if not lineage or lineage.get('parent_group') != record['group'] or lineage.get('parent_split') != record['split']:
                    raise ValueError(f'{rid}: synthetic must inherit parent_group and parent_split')
        records.append(record)
    by_id = {r['id']: r for r in records}
    for record in records:
        parent = by_id.get(record.get('lineage', {}).get('parent_id'))
        if check_leakage and parent and (parent['split'] != record['split'] or parent['group'] != record['group']):
            raise ValueError(f"{record['id']}: synthetic parent crosses split/group")
    return {**manifest, 'records': records, '_manifest_path': str(path)}


validate_manifest = load_manifest


def read_rgb(image_path, box, *, max_window_pixels=12_000_000, max_decode_bytes=512_000_000) -> np.ndarray:
    """Read a bounded RGB crop; reject excessive fallback decode allocations.

    tifffile strip decoding is used when installed and supported. Pillow's crop
    is a full-image decoding fallback, never described as strip-safe.
    """
    box = tuple(_int(v, 'crop') for v in box)
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0 or (x1-x0)*(y1-y0) > max_window_pixels:
        raise ValueError('empty crop or max_window_pixels exceeded')
    with Image.open(image_path) as image:
        width, height = image.size
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('crop outside image')
        bytes_per_pixel = {'1': 1, 'P': 1, 'L': 1, 'I': 4, 'F': 4, 'I;16': 2,
                           'I;16L': 2, 'I;16B': 2}.get(image.mode, len(image.getbands()))
        native_bytes = width * height * bytes_per_pixel
        if Path(image_path).suffix.lower() in ('.tif', '.tiff'):
            try:
                import tifffile
                with tifffile.TiffFile(image_path) as tif:
                    page = tif.pages[0]
                    if not page.is_tiled and page.planarconfig == 1 and page.photometric in (0, 1, 2, 3) and page.rowsperstrip <= 4000:
                        page.decode(None, 0)  # unsupported codec fails before allocating output
                        output = np.empty((y1-y0, x1-x0, 3), np.uint8)
                        for index in range(y0//page.rowsperstrip, (y1-1)//page.rowsperstrip+1):
                            tif.filehandle.seek(page.dataoffsets[index])
                            segment, where, _ = page.decode(tif.filehandle.read(page.databytecounts[index]), index)
                            start = int(where[2]); a, b = max(y0, start), min(y1, start+segment.shape[1])
                            data = segment[0, a-start:b-start, x0:x1]
                            if page.photometric == 3:
                                data = (page.colormap >> 8).astype(np.uint8).T[data[:, :, 0]]
                            elif data.shape[-1] == 1:
                                if data.dtype == bool:
                                    data = data.astype(np.uint8)*255
                                else:
                                    data = np.clip(data, 0, 255).astype(np.uint8)
                                if page.photometric == 0:
                                    data = 255-data
                                data = np.repeat(data, 3, axis=2)
                            output[a-y0:b-y0] = data[:, :, :3]
                        return output
            except (ImportError, ValueError, NotImplementedError, OSError):
                pass
        if native_bytes > max_decode_bytes:
            raise MemoryError(f'Pillow may decode {native_bytes} native bytes; max_decode_bytes={max_decode_bytes}. Install tifffile/imagecodecs for supported strip decoding or pre-tile the input.')
        return np.asarray(image.crop(box).convert('RGB')).copy()

"""Export saved pattern_lab window reviews. Do not change source labels."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from viridian402.data import load_boxes, load_manifest, read_rgb, save_intervals, sha256, validate_intervals
from viridian402.groups import normalize_group

MAX_PREPARED_RGB_BYTES=128*1024**2
# Preserve the existing 16 MiB single-document allowance while bounding the
# combined source JSON retained as per-window review provenance. Repeated
# documents are charged again because their parsed objects are retained again.
MAX_PREPARED_JSON_BYTES=32*1024**2
MAX_WINDOWS=1000


def read_json(path):
    return _json_snapshot(path)[0]


def _json_snapshot(path, *, budget=None):
    limit=min(16*1024*1024,budget[0]) if budget is not None else 16*1024*1024
    with Path(path).open('rb') as stream:
        raw=stream.read(limit+1)
    if len(raw)>limit:
        if budget is not None and budget[0]<=16*1024*1024:
            raise ValueError('Review source JSON exceeds the cumulative 32 MiB export budget; export smaller batches')
        raise ValueError('Review JSON exceeds the 16 MiB supported budget')
    if budget is not None:budget[0]-=len(raw)
    def nonfinite(value):raise ValueError('Review JSON must contain finite values')
    value=json.loads(raw.decode('utf-8-sig'),parse_constant=nonfinite)
    if not isinstance(value,dict):raise ValueError('Review JSON must be an object')
    return value,hashlib.sha256(raw).hexdigest()


def _source_path(root,value):
    if not isinstance(value,str) or '\\' in value or ':' in value:
        raise ValueError('Source files must use portable relative paths')
    relative=PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('Source file escapes the exchange directory')
    lexical=root.joinpath(*relative.parts)
    resolved=lexical.resolve()
    if not resolved.is_relative_to(root) or lexical.absolute()!=resolved:
        raise ValueError('Source file escapes or traverses a symlink')
    return resolved


def _new_destination(value):
    lexical=Path(value).absolute()
    for part in (lexical,*lexical.parents):
        if part.is_symlink() or getattr(part,'is_junction',lambda:False)():
            raise ValueError('Output must not traverse a symlink or junction')
    if lexical.exists():raise ValueError('output must be a new directory; keep previous exports')
    return lexical.resolve()


def _document_schema(document):
    for key in ('id','image_name'):text(document.get(key),'document '+key)
    for key in ('width','height'):
        if type(document.get(key)) is not int or not 0<document[key]<2**63:
            raise ValueError('Document dimensions must be positive integers')
    width,height=document['width'],document['height']
    if not isinstance(document.get('boxes'),dict):raise ValueError('Document boxes must be an object')
    column=box(document['boxes'].get('interpreted_lithology_column'),'column')
    if column[2]>width or column[3]>height:raise ValueError('Column box is outside the page')
    if not isinstance(document.get('legend'),list):raise ValueError('Document legend must be a list')
    legends={}
    for entry in document['legend']:
        if not isinstance(entry,dict) or type(entry.get('id')) is not int or not -(2**63)<=entry['id']<2**63:
            raise ValueError('Legend IDs must be signed 64-bit integers')
        if entry['id'] in legends:raise ValueError('Legend IDs must be unique')
        legends[entry['id']]=text(entry.get('label_printed'),'legend label')
        if 'box' in entry:
            swatch=box(entry['box'],'legend box')
            if swatch[2]>width or swatch[3]>height:raise ValueError('Legend box is outside the page')
    if not isinstance(document.get('segments'),list):raise ValueError('Document segments must be a list')
    ids=set()
    for segment in document['segments']:
        if not isinstance(segment,dict):raise ValueError('Each segment must be an object')
        sid=text(segment.get('id'),'segment ID')
        if len(sid)>100 or sid in ids:raise ValueError('Segment IDs must be unique and at most 100 characters')
        ids.add(sid)
        geometry=box(segment.get('box'),'segment')
        if geometry[2]>width or geometry[3]>height:raise ValueError('Segment box is outside the page')
        if segment.get('status') not in ('suggested','approved','rejected') or type(segment.get('approved')) is not bool or segment['approved']!=(segment['status']=='approved'):
            raise ValueError('Segment status and approved flag disagree')
        labels=segment.get('legend_ids',[])
        if not isinstance(labels,list) or any(type(label) is not int for label in labels) or len(set(labels))!=len(labels):
            raise ValueError('Segment legend_ids must be a list of unique integer IDs')
    return column,legends


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def box(value, name):
    if (not isinstance(value, list) or len(value) != 4
            or any(type(v) is not int for v in value)
            or not (0 <= value[0] < value[2] and 0 <= value[1] < value[3])):
        raise ValueError(f'{name}: use four integer page coordinates')
    return value


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name}: text is required')
    return value


def export_windows(windows_file, output, catalog=None):
    """Return a validated manifest. Images must be existing native window crops.

    CSV coordinates are local to the exported PNG. Page offsets and unchanged
    source segments are in each record's provenance. No depth is inferred.
    """
    windows_file, output = Path(windows_file).resolve(), _new_destination(output)
    json_budget=[MAX_PREPARED_JSON_BYTES]
    exchange,exchange_hash = _json_snapshot(windows_file,budget=json_budget)
    if type(exchange.get('schema_version')) is not int or exchange['schema_version'] != 1 or not isinstance(exchange.get('windows'),list) or not exchange['windows']:
        raise ValueError('expected schema_version=1 and a nonempty windows list')
    if len(exchange['windows'])>MAX_WINDOWS:raise ValueError(f'Review export supports at most {MAX_WINDOWS} windows')
    parents = {}
    catalog_hash=None
    if catalog:
        # Check split metadata before the loader can open any truth file.
        raw_catalog,catalog_hash = _json_snapshot(catalog,budget=json_budget)
        if not isinstance(raw_catalog.get('records'),list) or any(not isinstance(r,dict) or r.get('split') not in {'train', 'dev'} for r in raw_catalog['records']):
            raise ValueError('use a development catalog; test/verify truth must stay closed')
        parents = {r['id']: r for r in load_manifest(catalog)['records']}
    prepared, ids = [], set()
    prepared_bytes=0
    for item in exchange['windows']:
        if not isinstance(item,dict):raise ValueError('Each review window must be an object')
        rid = item['id']
        if (not isinstance(rid, str) or len(rid)>120 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', rid)
                or rid.casefold() in ids or rid.endswith('.') or re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',rid,re.I)):
            raise ValueError('window IDs must be unique safe file names')
        ids.add(rid.casefold())
        parent = dict(item['parent'])
        for key in ('id', 'well', 'group'):
            text(parent.get(key), f'parent {key}')
        if parent.get('split') not in {'train', 'dev'}:
            raise ValueError('review windows must keep a train/dev parent split')
        if not isinstance(parent.get('exposure_history'), list) or not parent['exposure_history']:
            raise ValueError('parent exposure_history must be a nonempty list')
        if catalog:
            known = parents.get(parent['id'])
            if known is None:
                raise ValueError(f'{rid}: parent ID is absent from catalog')
            if normalize_group(parent['group']) != known['group'] or parent['split'] != known['split']:
                raise ValueError(f'{rid}: parent group/split differs from catalog')
            if parent.get('image_sha256') not in (None, known['image_sha256']):
                raise ValueError(f'{rid}: parent image hash differs from catalog')
            parent['image_sha256'] = known['image_sha256']
            parent['catalog_exposure'] = {k: known[k] for k in
                ('tags', 'label_provenance', 'exposure_history', 'fresh_holdout') if k in known}
        document_path = _source_path(windows_file.parent,item['document'])
        image_path = _source_path(windows_file.parent,item['image'])
        document,document_hash = _json_snapshot(document_path,budget=json_budget)
        column,legends=_document_schema(document)
        if document['id'] != item['document_id']:
            raise ValueError(f'{rid}: document ID differs from saved document')
        if catalog:
            with Image.open(known['image']) as image:
                if image.size != (document['width'], document['height']) or Path(known['image']).name != document['image_name']:
                    raise ValueError(f'{rid}: saved document differs from parent image')
                if load_boxes(known['annotation'],image.size)['interpreted_lithology_column']!=column:
                    raise ValueError(f'{rid}: saved column differs from parent annotation')
            if known.get('well') and known['well'] != parent['well']:
                raise ValueError(f'{rid}: parent well differs from catalog')
        x0, y0, x1, y1 = box(item['box'], rid)
        if not (column[0] == x0 < x1 == column[2]
                and column[1] <= y0 < y1 <= column[3]
                and x1 <= document['width'] and y1 <= document['height']):
            raise ValueError(f'{rid}: window must span the page lithology column')
        with Image.open(image_path) as image:
            if image.size != (x1-x0, y1-y0) or image.width * image.height > 12_000_000:
                raise ValueError(f'{rid}: image must be a bounded native window crop')
        hashes = {'document_sha256': document_hash, 'window_image_sha256': sha256(image_path)}
        for key, actual in hashes.items():
            if item.get(key) != actual:
                raise ValueError(f'{rid}: missing or changed {key}')
        rgb_bytes=(x1-x0)*(y1-y0)*3
        if prepared_bytes+rgb_bytes>MAX_PREPARED_RGB_BYTES:
            raise ValueError('Prepared RGB crops exceed the 128 MiB export budget; export smaller batches')
        pixels=read_rgb(image_path,(0,0,x1-x0,y1-y0))
        prepared_bytes+=rgb_bytes
        if sha256(image_path)!=hashes['window_image_sha256']:raise ValueError('Window image changed while being read')
        if catalog:
            parent_pixels=read_rgb(known['image'],(x0,y0,x1,y1))
            if not np.array_equal(pixels,parent_pixels):raise ValueError(f'{rid}: window pixels do not match the catalog parent crop')
            hashes['parent_crop_rgb_sha256']=hashlib.sha256(pixels.tobytes()).hexdigest()
        review = item.get('review')
        if review is not None:
            if not isinstance(review,dict) or review.get('kind') != 'human' or review.get('document_sha256') != hashes['document_sha256']:
                raise ValueError(f'{rid}: human review must name this saved document hash')
            for key in ('reviewer', 'reviewed_at'):
                text(review.get(key), f'review {key}')
            try:
                if datetime.fromisoformat(review['reviewed_at'].replace('Z','+00:00')).utcoffset() is None:raise ValueError()
            except ValueError as exc:raise ValueError('Review reviewed_at must be an ISO timestamp with a timezone') from exc
        rows, segments = [], []
        for segment in document['segments']:
            sx0, sy0, sx1, sy1 = box(segment['box'], 'segment')
            if sy1 <= y0 or sy0 >= y1 or sx1 <= x0 or sx0 >= x1:
                continue
            reason = None
            if not review:
                reason = 'no human review attestation'
            elif segment.get('status') != 'approved' or segment.get('approved') is not True:
                reason = 'not accepted by reviewer'
            elif segment.get('flags') or segment.get('uncertain') or segment.get('unknown'):
                reason = 'unresolved uncertainty'
            elif sx0 != x0 or sx1 != x1 or len(segment.get('legend_ids', [])) != 1:
                reason = 'lateral mixture or missing single legend ID'
            else:
                lid = segment['legend_ids'][0]
                if type(lid) is not int or lid not in legends:
                    raise ValueError(f'{rid}: accepted segment has an invalid legend ID')
                rows.append({'top_y': max(y0, sy0)-y0, 'bottom_y': min(y1, sy1)-y0,
                             'lithology': legends[lid], 'cluster_id': lid})
            segments.append({'segment': segment, 'truth_exported': reason is None, 'exclusion_reason': reason})
        rows.sort(key=lambda row: row['top_y'])
        validate_intervals(rows, window=(0, y1-y0))
        record = {'id': rid, 'image': f'images/{rid}.png', 'annotation': f'annotations/{rid}.json',
                  'truth': f'truth/{rid}.csv' if rows else None,
                  'source': 'human' if rows else 'external_unlabelled',
                  'group': parent['group'], 'split': parent['split'], 'well': parent['well'],
                  'window': [0, y1-y0], 'tags': ['visible_development', 'review_window'],
                  'fresh_holdout': False, 'depth_unit': 'unknown', 'depth_datum': 'unknown',
                  'lineage': {'parent_id': parent['id'], 'parent_group': parent['group'],
                              'parent_split': parent['split'], 'parent_image_sha256': parent.get('image_sha256'),
                              'parent_crop_pixels_verified':bool(catalog)},
                  'exposure_history': parent['exposure_history'] + [{'kind': 'review_window_export', 'fresh_holdout': False}],
                  'original_coordinates': {'box': item['box'], 'page_size': [document['width'], document['height']],
                      'document_id': document['id'], 'image_name': document['image_name'],
                      'page_x': 'local_x + box[0]', 'page_y': 'local_y + box[1]'},
                  'review_provenance': f'provenance/{rid}.json', 'source_hashes': hashes}
        if rows:
            record.update(truth_window=[0, y1-y0], label_provenance={**review, 'kind': 'human',
                          'source': record['review_provenance'], 'partial_truth': True})
        provenance = {'parent': parent, 'review': review, 'source_hashes': hashes,
                      'document_id': document['id'], 'revision': document.get('revision'),
                      'updated_at': document.get('updated_at'), 'source_box': item['box'],
                      'legend': document['legend'], 'depth_calibration': document.get('depth_calibration', {}),
                      'segments': segments, 'note': 'Only accepted, unambiguous full-width beds enter CSV. Gaps have no truth.'}
        prepared.append((record, provenance, rows, pixels))
    # Validate in a temporary directory. A failed export must not leave a bundle.
    destination = output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.review-export-', dir=destination.parent) as staging:
        return _write_export(Path(staging), destination, prepared, exchange, exchange_hash, catalog_hash, parents)


def _write_export(output, destination, prepared, exchange, exchange_hash, catalog_hash, parents):
    for record, provenance, rows, pixels in prepared:
        target = output / record['image']
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(target)
        width, height = record['original_coordinates']['box'][2]-record['original_coordinates']['box'][0], record['window'][1]
        write_json(output / record['annotation'], {'width': width, 'height': height,
                   'bounding_boxes': {'interpreted_lithology_column': [0, 0, width, height]}})
        if rows:
            save_intervals(output / record['truth'], rows)
        write_json(output / record['review_provenance'], provenance)
        for key in ('image', 'annotation', 'truth'):
            record[key+'_sha256'] = sha256(output / record[key]) if record[key] else None
    records = [entry[0] for entry in prepared]
    manifest = output / 'all.json'
    write_json(manifest, {'schema_version': 1, 'name': exchange.get('name', 'review-windows'),
                         'fresh_holdout': False, 'records': records})
    load_manifest(manifest)
    if catalog_hash:
        # Use the shared validator for cross-file group, hash and lineage checks.
        combined = output / '.combined-validation.json'
        write_json(combined, {'schema_version': 1, 'records': list(parents.values()) + records})
        try:
            load_manifest(combined)
        finally:
            combined.unlink()
    write_json(output / 'source-snapshot.json', {'schema_version': 1, 'exchange_sha256': exchange_hash,
               'development_catalog_sha256': catalog_hash,
               'catalog_validation': 'passed' if catalog_hash else 'not_run', 'fresh_holdout': False,
               'catalog_crop_pixels':'passed' if catalog_hash else 'not_run',
               'source_hashes': {r['id']: r['source_hashes'] for r in records},
               'limits': ['No full-page image copy.', 'No depth inference.',
                          'No unseen verification claim.', 'No release upload.']})
    _new_destination(destination)
    output.rename(destination)
    return destination / 'all.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', required=True, help='Window exchange JSON')
    parser.add_argument('--output', required=True, help='New export directory')
    parser.add_argument('--catalog', help='Existing train/dev combined catalog; do not supply sealed truth')
    args = parser.parse_args()
    manifest = export_windows(args.windows, args.output, args.catalog)
    data = read_json(manifest)
    print(json.dumps({'manifest': str(manifest), 'records': len(data['records']),
                      'human_truth_records': sum(r['truth'] is not None for r in data['records']),
                      'fresh_holdout': False}))


if __name__ == '__main__':
    main()

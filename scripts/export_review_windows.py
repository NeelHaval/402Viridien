"""Export saved pattern_lab window reviews. Do not change source labels."""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from viridian402.data import load_manifest, save_intervals, sha256, validate_intervals
from viridian402.groups import normalize_group


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


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
    windows_file, output = Path(windows_file).resolve(), Path(output).resolve()
    exchange = read_json(windows_file)
    if exchange.get('schema_version') != 1 or not exchange.get('windows'):
        raise ValueError('expected schema_version=1 and a nonempty windows list')
    parents = {}
    if catalog:
        # Check split metadata before the loader can open any truth file.
        raw_catalog = read_json(catalog)
        if any(r.get('split') not in {'train', 'dev'} for r in raw_catalog['records']):
            raise ValueError('use a development catalog; test/verify truth must stay closed')
        parents = {r['id']: r for r in load_manifest(catalog)['records']}
    if output.exists():
        raise ValueError('output must be a new directory; keep previous exports')
    prepared, ids = [], set()
    for item in exchange['windows']:
        rid = item['id']
        if not isinstance(rid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', rid) or rid in ids:
            raise ValueError('window IDs must be unique safe file names')
        ids.add(rid)
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
        document_path = (windows_file.parent / item['document']).resolve()
        image_path = (windows_file.parent / item['image']).resolve()
        document = read_json(document_path)
        if document['id'] != item['document_id']:
            raise ValueError(f'{rid}: document ID differs from saved document')
        if catalog:
            with Image.open(known['image']) as image:
                if image.size != (document['width'], document['height']) or Path(known['image']).name != document['image_name']:
                    raise ValueError(f'{rid}: saved document differs from parent image')
            if known.get('well') and known['well'] != parent['well']:
                raise ValueError(f'{rid}: parent well differs from catalog')
        x0, y0, x1, y1 = box(item['box'], rid)
        column = box(document['boxes']['interpreted_lithology_column'], 'column')
        if not (column[0] == x0 < x1 == column[2]
                and column[1] <= y0 < y1 <= column[3]
                and x1 <= document['width'] and y1 <= document['height']):
            raise ValueError(f'{rid}: window must span the page lithology column')
        with Image.open(image_path) as image:
            if image.size != (x1-x0, y1-y0) or image.width * image.height > 12_000_000:
                raise ValueError(f'{rid}: image must be a bounded native window crop')
        hashes = {'document_sha256': sha256(document_path), 'window_image_sha256': sha256(image_path)}
        for key, actual in hashes.items():
            if item.get(key) != actual:
                raise ValueError(f'{rid}: missing or changed {key}')
        review = item.get('review')
        if review is not None:
            if review.get('kind') != 'human' or review.get('document_sha256') != hashes['document_sha256']:
                raise ValueError(f'{rid}: human review must name this saved document hash')
            for key in ('reviewer', 'reviewed_at'):
                text(review.get(key), f'review {key}')
        legends = {e['id']: e['label_printed'] for e in document['legend']}
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
                              'parent_split': parent['split'], 'parent_image_sha256': parent.get('image_sha256')},
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
        prepared.append((record, provenance, rows, image_path))
    # Validate in a temporary directory. A failed export must not leave a bundle.
    destination = output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.review-export-', dir=destination.parent) as staging:
        return _write_export(Path(staging), destination, prepared, exchange, windows_file, catalog, parents)


def _write_export(output, destination, prepared, exchange, windows_file, catalog, parents):
    for record, provenance, rows, image_path in prepared:
        target = output / record['image']
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            image.convert('RGB').save(target)
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
    if catalog:
        # Use the shared validator for cross-file group, hash and lineage checks.
        combined = output / '.combined-validation.json'
        write_json(combined, {'schema_version': 1, 'records': list(parents.values()) + records})
        try:
            load_manifest(combined)
        finally:
            combined.unlink()
    write_json(output / 'source-snapshot.json', {'schema_version': 1, 'exchange_sha256': sha256(windows_file),
               'development_catalog_sha256': sha256(catalog) if catalog else None,
               'catalog_validation': 'passed' if catalog else 'not_run', 'fresh_holdout': False,
               'source_hashes': {r['id']: r['source_hashes'] for r in records},
               'limits': ['No full-page image copy.', 'No depth inference.',
                          'No unseen verification claim.', 'No release upload.']})
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

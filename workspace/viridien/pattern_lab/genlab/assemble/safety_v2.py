"""Version two generator boundaries: native spans, final masks and cache receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def spans(value):
    """Validate and union half-open native page-row spans."""
    result = []
    for a, b in sorted(value):
        if not isinstance(a, int) or not isinstance(b, int) or a < 0 or b <= a:
            raise ValueError('Expected positive half-open integer page spans')
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def row_selection(top, bottom, intervals):
    rows = np.zeros(bottom-top, bool)
    for a, b in spans(intervals):
        lo, hi = max(a, top), min(b, bottom)
        if lo < hi:
            rows[lo-top:hi-top] = True
    return rows


def refresh_masks(folder, mask=None):
    """Intersect existing bed identities with final support; never relabel rails."""
    folder = Path(folder)
    if mask is None:
        with Image.open(folder/'mask.png') as im:
            mask = np.asarray(im).copy()
    file = folder/'row-mask.npy'
    if file.exists():
        row = np.load(file, allow_pickle=False).reshape(-1).copy()
    elif (folder/'row-mask.png').exists():
        with Image.open(folder/'row-mask.png') as im:
            row = np.asarray(im).reshape(-1).copy()
    else:
        row = np.full(len(mask), 255, np.uint8)
        for y, pixels in enumerate(mask):
            known = np.unique(pixels[pixels != 255])
            if len(known) == 1:
                row[y] = known[0]
    if row.shape != (len(mask),):
        raise ValueError('Row-mask length differs from final mask')
    row[~np.any(mask != 255, axis=1)] = 255
    np.save(file, row, allow_pickle=False)
    Image.fromarray(row[:, None]).save(folder/'row-mask.png')
    return row


def restore_column(folder, rgb, truth, doc, windows):
    """Apply reviewed native spans even when a quilt column is called directly."""
    folder = Path(folder)
    x0, y0, x1, y1 = doc['boxes']['interpreted_lithology_column']
    no_data = spans(doc.get('native_no_data_spans', []))
    reserved = spans(doc.get('reserved_spans', []))
    with Image.open(folder/'mask.png') as im:
        mask = np.asarray(im).copy()
    restored = np.zeros(mask.shape, bool)
    for a, b in no_data:
        for top in range(max(a, y0), min(b, y1), 512):
            bottom = min(b, y1, top+512)
            rgb[top-y0:bottom-y0] = windows.get(doc, [x0, top, x1, bottom])
            restored[top-y0:bottom-y0] = True
            mask[top-y0:bottom-y0] = 255
    hidden = row_selection(y0, y1, reserved)
    rgb[hidden] = 255
    mask[hidden] = 255
    restored[hidden] = False
    provenance = folder/'pixel-sources.npz'
    if provenance.exists():
        with np.load(provenance) as source:
            maps = {k: source[k].copy() for k in source.files}
        yy, xx = np.indices(mask.shape)
        maps['source_index'][restored] = -1
        maps['page_y'][restored] = (yy+y0)[restored]
        maps['page_x'][restored] = (xx+x0)[restored]
        for value in maps.values():
            value[hidden] = -2
        np.savez_compressed(provenance, **maps)
    Image.fromarray(rgb).save(folder/'column.png')
    Image.fromarray(mask).save(folder/'mask.png')
    refresh_masks(folder, mask)
    truth.update(native_no_data_spans=no_data, reserved_spans=reserved,
                 exposure=dict(reserved_spans=reserved, retained_reserved_spans=[]))
    (folder/'truth.json').write_text(json.dumps(truth, allow_nan=False), encoding='utf8')
    return rgb, truth


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def cache_binding(inputs, model, parameters, renderer):
    payload = dict(schema='generator-cache-v2',
                   inputs={str(Path(p).resolve()): file_hash(p) for p in sorted(map(Path, inputs))},
                   model=model, parameters=parameters,
                   renderer={str(Path(p).resolve()): file_hash(p) for p in sorted(map(Path, renderer))})
    # A JSON round trip also rejects accidental nonportable model values.
    payload = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False, default=str))
    return dict(payload=payload, sha256=hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest())


def cache_valid(folder, binding):
    folder = Path(folder)
    receipt = folder/'cache-v2.json'
    if not receipt.exists():
        return False
    saved = json.loads(receipt.read_text(encoding='utf8'))
    return (saved.get('binding') == binding and bool(saved.get('outputs')) and
            all((folder/p).is_file() and file_hash(folder/p) == sha
                for p, sha in saved['outputs'].items()))


def seal_cache(folder, binding):
    folder = Path(folder)
    outputs = {p.relative_to(folder).as_posix(): file_hash(p)
               for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'cache-v2.json'}
    (folder/'cache-v2.json').write_text(json.dumps(dict(binding=binding, outputs=outputs),
                                                sort_keys=True), encoding='utf8')


def tick_mapping(anchors):
    """Fit native printed tick centres, never printed text baselines."""
    y = np.array([p['page_y'] for p in anchors], float)
    ft = np.array([p['depth_ft'] for p in anchors], float)
    if len(y) < 2 or not np.isfinite(y).all() or not np.isfinite(ft).all():
        raise ValueError('Two finite printed ticks required')
    rate, intercept = np.polyfit(ft, y, 1)
    residual = float(np.max(abs(y-(rate*ft+intercept))))
    if rate <= 0 or residual > 1:
        raise ValueError('Printed tick fit failed')
    return dict(rows_per_ft=float(rate), page_intercept=float(intercept),
                coordinate='MD', confidence='verified', whole_page_verified=False,
                valid_page_span=[float(y.min()), float(y.max())],
                max_anchor_residual_px=residual, anchors=anchors,
                source='review-v1/format printed-depth-anchors.json; native tick centres')


def physical_duration_eligible(record, mapping):
    scope = mapping.get('valid_page_span')
    return bool(mapping.get('confidence') in ('verified', 'verified-local') and
                mapping.get('rows_per_ft', 0) > 0 and scope and
                scope[0] <= record['top'] < record['bottom'] <= scope[1])

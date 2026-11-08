"""Export existing harness rows in the two published challenge formats.

This is an export-only adapter: it performs no inference, grading or relabeling.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

INPUT_FIELDS = ('top_y', 'bottom_y', 'lithology', 'cluster_id', 'top_depth', 'bottom_depth')
NOTEBOOK_FIELDS = ('top_depth', 'bottom_depth', 'lithology', 'top_y', 'bottom_y', 'cluster_id')
SLIDE_FIELDS = ('top_y', 'bottom_y', 'lithology', 'cluster_id', 'top_depth', 'base_depth')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path, image_height=None):
    """Validate full-image pixel intervals and preserve input field strings."""
    if image_height is not None and image_height <= 0:
        raise ValueError('image_height must be positive')
    rows = []
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or any(field not in fields for field in INPUT_FIELDS):
            raise ValueError('CSV must have unique headers and all harness interval fields')
        previous_bottom = 0
        for line, row in enumerate(reader, 2):
            if None in row or any(row.get(field) is None for field in INPUT_FIELDS):
                raise ValueError(f'Line {line}: malformed CSV row')
            try:
                top, bottom = int(row['top_y']), int(row['bottom_y'])
            except ValueError as exc:
                raise ValueError(f'Line {line}: pixel coordinates must be integers') from exc
            if top < 0 or bottom <= top or top < previous_bottom:
                raise ValueError(f'Line {line}: intervals must be sorted and nonoverlapping with positive height')
            if image_height is not None and bottom > image_height:
                raise ValueError(f'Line {line}: interval exceeds image height')
            for field in ('top_depth', 'bottom_depth'):
                try:
                    value = float(row[field])
                except ValueError as exc:
                    raise ValueError(f'Line {line}: {field} must be a finite number') from exc
                if not math.isfinite(value):
                    raise ValueError(f'Line {line}: {field} must be a finite number')
            for field in ('lithology', 'cluster_id'):
                if not row[field].strip():
                    raise ValueError(f'Line {line}: {field} must be nonempty')
            try:
                int(row['cluster_id'])
            except ValueError as exc:
                raise ValueError(f'Line {line}: cluster_id must be an integer') from exc
            rows.append({field: row[field] for field in INPUT_FIELDS})
            previous_bottom = bottom
    return rows


def _write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def export(source, out_dir, image_height=None):
    """Write explicit notebook and slide CSVs plus an export manifest."""
    source, out_dir = Path(source), Path(out_dir)
    # Hash and parse the same input bytes. Refuse a concurrently changed source.
    before = _sha256(source)
    rows = read_rows(source, image_height=image_height)
    if _sha256(source) != before:
        raise ValueError('Source CSV changed during validation')
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook = out_dir / (source.stem + '.notebook.csv')
    slide = out_dir / (source.stem + '.slide.csv')
    manifest_path = out_dir / (source.stem + '.manifest.json')
    if source.resolve() in {notebook.resolve(), slide.resolve(), manifest_path.resolve()}:
        raise ValueError('Source and output paths must differ')
    _write_csv(notebook, NOTEBOOK_FIELDS, rows)
    _write_csv(slide, SLIDE_FIELDS,
               [dict(row, base_depth=row['bottom_depth']) for row in rows])
    manifest = {
        'source_csv': str(source.resolve()), 'source_sha256': before, 'rows': len(rows),
        'schema_policy': 'two explicit formats; lower-depth alias only; no relabeling or scoring',
        'coordinate_policy': 'nonnegative full-image pixel coordinates; sorted, nonoverlapping, bottom exclusive',
        'image_height_checked': image_height,
        'notebook': {'file': notebook.name, 'fields': list(NOTEBOOK_FIELDS), 'sha256': _sha256(notebook)},
        'slide': {'file': slide.name, 'fields': list(SLIDE_FIELDS), 'sha256': _sha256(slide)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Existing harness interval CSV')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--image-height', type=int, help='Also verify that bottom_y is within the full image')
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.out_dir, args.image_height), indent=2))


if __name__ == '__main__':
    main()

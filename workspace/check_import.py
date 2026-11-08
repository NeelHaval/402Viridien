"""Check the source inventory and optional fixed release ZIP files."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    errors = []
    paths = set()
    python_files = 0
    for row in manifest['files']:
        relative = Path(row['path'])
        path = (ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or row['path'] in paths:
            errors.append('Invalid or duplicate path: ' + row['path'])
            continue
        paths.add(row['path'])
        if not path.is_file():
            errors.append('Missing file: ' + row['path'])
            continue
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            errors.append('Changed file: ' + row['path'])
        if path.suffix == '.py':
            python_files += 1
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=row['path'])
            except (SyntaxError, UnicodeError) as error:
                errors.append(f'Syntax error: {row["path"]}: {error}')
    bundles = 0
    if args.bundle_dir:
        catalog = json.loads((ROOT/'viridien/data/catalog.json').read_text(encoding='utf-8'))
        for asset in catalog['bundles'].values():
            path = args.bundle_dir/asset['file']
            if not path.is_file() or path.stat().st_size != asset['bytes'] or digest(path) != asset['sha256']:
                errors.append('Invalid bundle: ' + asset['file'])
            bundles += 1
    for error in errors[:30]:
        print('ERROR', error)
    print(f'Files: {len(paths)}; Python syntax: {python_files}; bundles: {bundles}; errors: {len(errors)}')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())

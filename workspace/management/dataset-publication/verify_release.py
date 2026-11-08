"""Check a downloaded dataset release. Uses only the Python standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()


def verify(root, archives=False):
    manifest = json.loads((root / 'DATASET.json').read_text(encoding='utf-8'))
    checked = 0
    handles = {}
    try:
        for name, info in manifest['files'].items():
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe path: ' + name)
            if archives:
                bundle = info['bundle']
                if bundle not in handles:
                    handles[bundle] = zipfile.ZipFile(root / bundle)
                archive = handles[bundle]
                if archive.getinfo(name).file_size != info['bytes']:
                    raise ValueError('Size mismatch: ' + name)
                with archive.open(name) as stream:
                    actual = digest(stream)
            else:
                source = root / rel
                if source.stat().st_size != info['bytes']:
                    raise ValueError('Size mismatch: ' + name)
                with source.open('rb') as stream:
                    actual = digest(stream)
            if actual != info['sha256']:
                raise ValueError('Hash mismatch: ' + name)
            checked += 1
        records = manifest['records']
        for record in records:
            for key in ('image', 'annotation', 'truth', 'ignore'):
                if record.get(key) and record[key] not in manifest['files']:
                    raise ValueError('Unlisted input: ' + record['id'] + ' ' + key)
        print(f'RELEASE_CHECK_OK files={checked} records={len(records)} '
              f'ready={sum(bool(x["ready"]) for x in records)} '
              f'archives={len(handles)}')
    finally:
        for archive in handles.values():
            archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--archives', action='store_true', help='Check ZIP files before extraction')
    args = parser.parse_args()
    verify(args.root.resolve(), args.archives)

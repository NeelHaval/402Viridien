"""Check and install the shared workspace. Uses the Python standard library."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'viridien/data'
ALLOWED = {
    'real': ('viridien/data/real/',),
    'lab': ('viridien/pattern_lab/data/',),
    'generated': ('viridien/pattern_lab/data/genlab/',),
    'synthetic-v5': ('viridien/archives/synthetic-v5/',),
    'models': ('viridien/bakeoff/ml/', 'viridien/bakeoff/emil/'),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def audit_real(manifest):
    """Reject split leaks and quality holds that could enter training."""
    errors = []
    records = manifest['records']
    ids = [r['sample_id'] for r in records]
    if len(ids) != len(set(ids)):
        errors.append('Duplicate sample IDs')
    groups = {}
    test_pixels = {r['pixel_sha256'] for r in records if r['split'] == 'test'}
    seed = manifest['split_policy']['seed']
    for row in records:
        ident = row['sample_id']
        group = row['source_group']
        split = row['split']
        if split not in {'train', 'test'}:
            errors.append(f'{ident}: unknown split')
        if group in groups and groups[group] != split:
            errors.append(f'{group}: one well occurs in both splits')
        groups[group] = split
        expected = 'test' if int(hashlib.sha256((seed+':'+group).encode()).hexdigest()[:8], 16) % 5 == 0 else 'train'
        if split != expected:
            errors.append(f'{ident}: fixed split policy changed')
        eligible = row['quality'] == 'accepted' and not row['quality_issues']
        if row['usable_for_generation'] != (eligible and split == 'train'):
            errors.append(f'{ident}: invalid generator permission')
        if row['usable_for_test'] != (eligible and split == 'test'):
            errors.append(f'{ident}: invalid test permission')
        if row['usable_for_generation'] and row['pixel_sha256'] in test_pixels:
            errors.append(f'{ident}: exact test pixels enter training')
        if row['kind'] != 'real':
            errors.append(f'{ident}: synthetic record in real data')
        x0, y0, x1, y1 = row['box_xyxy']
        width, height = row['source_size']
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            errors.append(f'{ident}: invalid source coordinates')
    return errors


def verify(require_assets=False):
    manifest = read(DATA / 'real/manifest.json')
    errors = audit_real(manifest)
    records = manifest['records']
    for split in ('train', 'test', 'needs-review'):
        actual = [json.loads(line) for line in (DATA/'real'/(split+'.jsonl')).read_text(encoding='utf-8').splitlines() if line]
        expected = [r for r in records if (r['quality'] == 'review' if split == 'needs-review' else r['quality'] == 'accepted' and r['split'] == split)]
        if actual != expected:
            errors.append(f'{split}.jsonl differs from the fixed manifest')
    sources = read(DATA / 'real/generator-sources.json')
    if sources != [r for r in records if r['usable_for_generation']]:
        errors.append('Generator sources include missing or withheld records')
    assets = {r[k] for r in records for k in ('native_png', 'normalized_png', 'valid_mask_png')}
    missing = [name for name in assets if not (DATA/'real'/name).is_file()]
    if require_assets and missing:
        errors.append(f'{len(missing)} real image assets are not installed')
    if not missing:
        from PIL import Image
        for row in records:
            with Image.open(DATA/'real'/row['native_png']) as image:
                image = image.convert('RGB')
                value = hashlib.sha256(f'{image.width}x{image.height}:'.encode()+image.tobytes()).hexdigest()
                if value != row['pixel_sha256']:
                    errors.append(f"{row['sample_id']}: native pixel hash mismatch")
    if errors:
        for error in errors[:30]:
            print('FAIL', error)
        print(f'FAILED: {len(errors)} data errors')
        return 1
    print('Real snapshot:', manifest['snapshot'])
    print('Records:', len(records), 'split counts:', dict(Counter(r['split'] for r in records)))
    print('Images:', f'{len(missing)} not installed; use download real' if missing else f'{len(assets)} checked')
    print('PASS: real data splits, quality holds, source bounds and generator permissions')
    return 0


def archive_members(archive, bundle, root):
    """Validate every member before writing any file."""
    result = []
    seen = set()
    root = root.resolve()
    for item in archive.infolist():
        name = item.filename
        path = PurePosixPath(name)
        if (name in seen or '\\' in name or path.is_absolute() or '..' in path.parts
                or any(':' in part for part in path.parts)
                or not any(name.startswith(prefix) for prefix in ALLOWED[bundle])
                or stat.S_ISLNK(item.external_attr >> 16)):
            raise ValueError('Invalid archive member: '+name)
        seen.add(name)
        target = root.joinpath(*path.parts).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Archive member leaves the repository')
        if item.is_dir():
            continue
        if target.exists():
            if sha256(target) != hashlib.sha256(archive.read(item)).hexdigest():
                raise ValueError('Existing file differs; keep your work and use a fresh clone: '+name)
        result.append((item, target))
    return result


def download(bundle, archive_path=None):
    catalog = read(DATA / 'catalog.json')
    asset = catalog['bundles'][bundle]
    with tempfile.TemporaryDirectory(prefix='viridien-download-') as folder:
        path = Path(archive_path).resolve() if archive_path else Path(folder)/asset['file']
        if not archive_path:
            if not shutil.which('gh'):
                raise ValueError('Install GitHub CLI and run gh auth login, or supply --archive PATH')
            subprocess.run(['gh', 'release', 'download', catalog['tag'],
                            '--repo', catalog.get('repository', 'NeelHaval/viridian402'),
                            '--pattern', asset['file'],
                            '--dir', folder], check=True)
        if path.stat().st_size != asset['bytes'] or sha256(path) != asset['sha256']:
            raise ValueError('Archive checksum differs from data/catalog.json')
        with zipfile.ZipFile(path) as archive:
            members = archive_members(archive, bundle, ROOT)
            for item, target in members:
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
        print(f'Installed {bundle}: {len(members)} files; SHA-256 checked')
        print('Data status:', asset['status'])
    return 0


def doctor():
    print('Python:', sys.version.split()[0])
    missing = []
    for module in ['numpy', 'PIL', 'cv2', 'scipy', 'sklearn', 'pandas', 'rapidocr_onnxruntime', 'cloudpickle']:
        installed = importlib.util.find_spec(module) is not None
        print(module+':', 'installed' if installed else 'missing')
        if not installed:
            missing.append(module)
    dataset = os.environ.get('LITHOLOGY_DATASET')
    if dataset:
        base = Path(dataset).expanduser()
        print('Source scans:', 'found' if (base/'images').is_dir() and (base/'bbox_annotation').is_dir() else 'path is not a challenge dataset')
    else:
        print('Source scans: set LITHOLOGY_DATASET for original-log work; shared outside tests do not need it')
    print('Lab seed:', 'installed' if (ROOT/'viridien/pattern_lab/data/documents.json').exists() else 'use download lab')
    print('GitHub CLI:', 'installed' if shutil.which('gh') else 'missing; browser downloads also work')
    return 1 if missing else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    check = sub.add_parser('verify')
    check.add_argument('--require-assets', action='store_true')
    fetch = sub.add_parser('download')
    fetch.add_argument('bundle', choices=ALLOWED)
    fetch.add_argument('--archive', help='Use an already downloaded ZIP file')
    args = parser.parse_args()
    try:
        if args.command == 'doctor':
            return doctor()
        if args.command == 'verify':
            return verify(args.require_assets)
        return download(args.bundle, args.archive)
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print('ERROR:', error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

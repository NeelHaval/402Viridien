"""Freeze the approved dataset selections into private GitHub release assets."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('DATASET_RELEASE_OUT', str(HERE / 'dataset-review-20261108-v1')))
REVIEW = ROOT / 'management/dataset-review'
SYNTH = ROOT / 'viridien_work/pattern_lab/data/genlab/release/synth-v1'
LIMIT = 1_700_000_000


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def local(value):
    p = Path(str(value).replace('\\', '/'))
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    p.relative_to(ROOT)
    if any('vault' in part.casefold() or 'sealed' in part.casefold() for part in p.parts):
        raise ValueError('Excluded source: ' + str(p))
    return p


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files, records, archives, counters = {}, [], {}, Counter()
    metadata = {}
    cached_json = {}
    for path in sorted(OUT.glob('*.zip')):
        group, number = path.stem.rsplit('-', 1)
        total = 0
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.filename in files:
                    raise ValueError('Duplicate archived source: ' + info.filename)
                h = hashlib.sha256()
                data = bytearray() if info.filename.endswith('.json') and info.file_size < 30_000_000 else None
                with archive.open(info) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(chunk)
                        if data is not None:
                            data.extend(chunk)
                files[info.filename] = dict(bytes=info.file_size, sha256=h.hexdigest(), bundle=path.name)
                if data is not None:
                    cached_json[info.filename] = bytes(data)
                total += info.file_size
        counters[group] = max(counters[group], int(number))
        archives[group] = [zipfile.ZipFile(path, 'a', zipfile.ZIP_DEFLATED,
                                          compresslevel=1, allowZip64=True), total, path.name]
        print(f'REUSE_OK {path.name}', flush=True)

    def add(path, group, expected=None):
        path = local(path)
        rel = path.relative_to(ROOT).as_posix()
        if rel in files:
            if expected and files[rel]['sha256'] != expected:
                raise ValueError('Conflicting source hash: ' + rel)
            return rel
        before = path.stat()
        if group not in archives or archives[group][1] + before.st_size > LIMIT:
            if group in archives:
                archives[group][0].close()
            counters[group] += 1
            name = f'{group}-{counters[group]:02d}.zip'
            archives[group] = [zipfile.ZipFile(OUT / name, 'x', zipfile.ZIP_DEFLATED,
                                               compresslevel=1, allowZip64=True), 0, name]
        archive, _, name = archives[group]
        h = hashlib.sha256()
        with path.open('rb') as source, archive.open(rel, 'w', force_zip64=True) as dest:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                h.update(chunk)
                dest.write(chunk)
        actual = h.hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during snapshot: ' + rel)
        if expected and expected != actual:
            raise ValueError('Source hash mismatch: ' + rel)
        files[rel] = dict(bytes=before.st_size, sha256=actual, bundle=name)
        archives[group][1] += before.st_size
        if path.suffix == '.json' and before.st_size < 30_000_000:
            metadata[rel] = actual
        return rel

    def snapshot_json(path, group='instructions'):
        # Read and archive the exact same revision, even if its owner is still active.
        rel = local(path).relative_to(ROOT).as_posix()
        data = cached_json.get(rel)
        if data is None:
            data = path.read_bytes()
        add(path, group, hashlib.sha256(data).hexdigest())
        return json.loads(data.decode('utf-8'))

    try:
        synth = snapshot_json(SYNTH / 'eval-manifest.json', 'synthetic')
        inventory = snapshot_json(REVIEW / 'inputs/synthetic-inventory.json', 'synthetic')
        if len(synth['pages']) != inventory['summary']['records']:
            raise ValueError('Synthetic page count mismatch')
        if sum(bool(p['ready']) for p in synth['pages']) != inventory['summary']['ready']:
            raise ValueError('Synthetic readiness mismatch')
        for page in synth['pages']:
            source = local(page['image']).with_suffix('.manifest.json')
            detail = snapshot_json(source, 'synthetic')
            record = dict(page, selection='synthetic', split='diagnostic',
                          truthSource=('AI-derived base-page labels; blocked for construction-truth scoring'
                                       if page['family'] == 'MC765' else
                                       'synthetic construction; see page manifest'), humanConfirmed=False)
            record['window'] = [detail['column_box'][1], detail['column_box'][3]]
            for key, suffix in [('image', '.tif'), ('annotation', '.json'),
                                ('truth', '_truth.csv'), ('ignore', '_ignore.csv')]:
                record[key] = add(page[key], 'synthetic', detail['sha256'][suffix])
            record['pageManifest'] = source.relative_to(ROOT).as_posix()
            records.append(record)
        for path in sorted(SYNTH.glob('*.json')):
            if path.name not in {Path(p['annotation']).name for p in synth['pages']}:
                add(path, 'synthetic')
        add(SYNTH / 'README.md', 'synthetic')
        for name in ['convert_v1.py', 'convert_v1.r1.py', 'validate_v1.py', 'make_inventory.py']:
            path = SYNTH.parent / name
            if path.is_file():
                add(path, 'instructions')
        print('SNAPSHOT synthetic=' + str(len(synth['pages'])), flush=True)

        frozen = REVIEW / 'freezes/labeller-first-20261108'
        human = snapshot_json(frozen / 'inputs/labeller-inventory.json', 'human-training')
        boxes = 0
        for page in human['records']:
            record = {k: page.get(k) for k in ['id', 'family', 'groupId', 'baseWell', 'window']}
            record.update(selection='human-crops', split='train', ready=True,
                          truthSource='partial human-approved lithology boxes', humanConfirmed=True,
                          diagnosticOnly=True, boundaryTruth=False, depthTruth=False)
            for key in ['image', 'annotation']:
                record[key] = add(page[key], 'human-training', page['sourceHashes'][key])
            for key in ['truth', 'ignore']:
                path = frozen / 'labeller-assets' / key / (page['id'] + '.json')
                record[key] = add(path, 'human-training')
            truth = read(ROOT / record['truth'])
            boxes += len(truth['boxes'])
            records.append(record)
        if len(human['records']) != 27 or boxes != 458:
            raise ValueError(f'Unexpected human subset: pages={len(human["records"])} boxes={boxes}')
        print(f'SNAPSHOT human={len(human["records"])} boxes={boxes}', flush=True)

        outside_counts = {}
        outside_meta = snapshot_json(REVIEW / 'inputs/outside-inventory.json')
        outside_by_id = {r['id']: r for r in outside_meta['records']}
        for name in ['ext_dev', 'ext_shared_verify', 'ext_test', 'mirror_lake',
                     'djs_idaho_test', 'frozen130_unlabelled']:
            packet = snapshot_json(REVIEW / 'inputs/outside-manifests' / (name + '.json'))
            is_test = name in ['ext_test', 'mirror_lake', 'djs_idaho_test']
            group = ('outside-heldout-inputs' if is_test else
                     'outside-unlabelled' if name == 'frozen130_unlabelled' else 'outside-development')
            for key, hash_key in [('itemsFile', 'itemsSha256'), ('manifestCsv', 'manifestSha256')]:
                add(packet[key], group, packet[hash_key])
            for case in packet['cases']:
                record = dict(case, selection=name, split=packet['split'], usage=packet['usage'],
                              humanConfirmed=False, heldOut=is_test,
                              diagnosticOnly=(name == 'frozen130_unlabelled'),
                              truthSource=('none' if name == 'frozen130_unlabelled' else
                                           'programmatic source-log extraction; not human-certified'))
                source_meta = outside_by_id[case['id']]
                for key in ['family', 'baseWell', 'groupId', 'lineage', 'knownIssues',
                            'window', 'boxes', 'labelsHeldOut', 'showTruthInReview', 'modelReadableTruth']:
                    record[key] = source_meta.get(key)
                for key in ['image', 'annotation', 'truth']:
                    if case.get(key):
                        target = 'outside-heldout-references' if is_test and key == 'truth' else group
                        record[key] = add(case[key], target, case[key + 'Sha256'])
                records.append(record)
            outside_counts[name] = len(packet['cases'])
            print('SNAPSHOT ' + name + '=' + str(len(packet['cases'])), flush=True)

        # Retain the existing evaluation instructions and adapters as source snapshots.
        for path in sorted((REVIEW / 'packs').glob('*')):
            if path.is_file() and path.suffix in ['.py', '.json', '.md']:
                add(path, 'instructions')
        for name in ['labeller-inventory.json', 'review-suggestions.json', 'outside-validation.json']:
            add(REVIEW / 'inputs' / name, 'instructions')
        for archive, _, _ in archives.values():
            archive.close()

        manifest = dict(schema='viridian402-portable-dataset-v1',
                        createdUtc=datetime.now(timezone.utc).isoformat(),
                        repository='NeelHaval/viridian402', release='dataset-review-20261108-v1',
                        sourceThread='t3-thread://v2/3ae12678-d93c-42ac-ad9f-a7be072f7f7c/71567cbf-512f-43b8-8223-2d117d7bf7f5',
                        pathsRelativeTo='directory containing DATASET.json',
                        summary=dict(syntheticPages=len(synth['pages']),
                                     syntheticReady=sum(bool(p['ready']) for p in synth['pages']),
                                     syntheticBlocked=sum(not p['ready'] for p in synth['pages']),
                                     humanTrainingPages=len(human['records']), humanApprovedBoxes=boxes,
                                     outside=outside_counts, sealedWindowsIncluded=0),
                        rules=['Do not train or tune on held-out sets.',
                               'Keep held-out reference archives outside the model input directory.',
                               'Score only records with ready=true.',
                               'Apply ignore masks before scoring.',
                               'Report synthetic results by family and base well; no pooled accuracy.',
                               'Synthetic and programmatic labels are not human-confirmed truth.',
                               'Sparse human labels provide no full boundary or depth truth.',
                               'DJS: saved depth calibration unavailable on 24 of 25 windows.',
                               'Unlabelled set: saved clean depth unavailable on 32 of 130 pages.',
                               'Source reuse licences are not verified; retain private access.'],
                        records=records, files=files)
        write(OUT / 'DATASET.json', manifest)
        shutil.copyfile(HERE / 'verify_release.py', OUT / 'verify_release.py')
        shutil.copyfile(HERE / 'README.md', OUT / 'README.md')
        shutil.copyfile(Path(__file__), OUT / 'build_release.py')
        assets = []
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                assets.append(dict(name=path.name, bytes=path.stat().st_size, sha256=sha(path)))
        write(OUT / 'ASSETS.json', dict(assets=assets))
        checksum_paths = sorted(p for p in OUT.iterdir() if p.is_file())
        (OUT / 'SHA256SUMS.txt').write_text(''.join(sha(p) + '  ' + p.name + '\n'
                                                  for p in checksum_paths), encoding='utf-8')
        print('BUILD_OK records=' + str(len(records)) + ' files=' + str(len(files)) +
              ' assets=' + str(len(list(OUT.iterdir()))), flush=True)
    finally:
        for archive, _, _ in archives.values():
            archive.close()


if __name__ == '__main__':
    main()

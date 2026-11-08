"""Upload a verified snapshot, compare GitHub digests, then publish the release."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from verify_release import verify

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('DATASET_RELEASE_OUT', str(HERE / 'dataset-review-20261108-v1')))
REPO = 'NeelHaval/viridian402'
TAG = 'dataset-review-20261108-v1'
TARGET = 'a41654266c270cce2cca8baee6f6632b54a55196'


def gh(*args):
    result = subprocess.run(['gh', *args], capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'GitHub command failed ({result.returncode}): {result.stderr.strip()}')
    return result.stdout


def release_info():
    info = json.loads(gh('release', 'view', TAG, '--repo', REPO, '--json', 'apiUrl'))
    return json.loads(gh('api', info['apiUrl']))


def main():
    verify(ROOT, archives=True)
    manifest = json.loads((ROOT / 'DATASET.json').read_text(encoding='utf-8'))
    files = sorted(path for path in ROOT.iterdir() if path.is_file())
    hashes = {}
    for path in files:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        hashes[path.name] = dict(sha256=h.hexdigest(), bytes=path.stat().st_size)
    summary = manifest['summary']
    notes = ('Private dataset snapshot from the dataset review work.\n\n'
             f'- Synthetic: {summary["syntheticPages"]} pages; '
             f'{summary["syntheticReady"]} ready and {summary["syntheticBlocked"]} blocked.\n'
             '- Human crop subset: 27 training pages and 458 approved lithology boxes.\n'
             '- Outside references: 136 labelled windows and 130 unlabelled diagnostic pages.\n'
             '- Permanently sealed images and labels: excluded.\n\n'
             'Download DATASET.json, README.md, SHA256SUMS.txt and the required ZIP files. '
             'The manifest uses relative paths and pins every source file by SHA-256. '
             'Held-out inputs and held-out reference labels are separate downloads.\n\n'
             'The current synthetic selection blocks five empty Gulf pages and eight '
             'MC765 pages with AI-derived base-page boundaries. Earlier 77-page notes '
             'are superseded. Score only ready records and apply ignore masks. '
             'Synthetic and programmatic labels are not human-confirmed truth. '
             'Keep held-out references out of model inputs. No new inference was run.\n\n'
             'Source data reuse licences remain unverified. Repository access remains private.\n')
    notes_path = HERE / 'release-notes.md'
    notes_path.write_text(notes, encoding='utf-8')
    repo = json.loads(gh('repo', 'view', REPO, '--json', 'isPrivate'))
    if not repo['isPrivate']:
        raise ValueError('The destination repository is not private')
    existing = subprocess.run(['gh', 'release', 'view', TAG, '--repo', REPO,
                               '--json', 'isDraft'], capture_output=True, text=True, encoding='utf-8')
    if existing.returncode == 0:
        if not json.loads(existing.stdout)['isDraft']:
            raise ValueError('An already published release cannot be overwritten')
        print('DRAFT_RESUMED ' + TAG, flush=True)
    elif 'release not found' in existing.stderr:
        gh('release', 'create', TAG, '--repo', REPO, '--target', TARGET, '--draft',
           '--prerelease', '--title', 'Dataset review: synthetic, human crops and outside references',
           '--notes-file', str(notes_path))
        print('DRAFT_CREATED ' + TAG, flush=True)
    else:
        raise RuntimeError(existing.stderr.strip())
    initial = release_info()
    present = {asset['name']: asset for asset in initial['assets']}

    def upload(path):
        previous = present.get(path.name)
        if (previous and previous['state'] == 'uploaded' and
                previous.get('digest') == 'sha256:' + hashes[path.name]['sha256']):
            print('UPLOAD_REUSED ' + path.name, flush=True)
            return
        print(f'UPLOAD_START {path.name} bytes={path.stat().st_size}', flush=True)
        gh('release', 'upload', TAG, str(path), '--repo', REPO, '--clobber')
        print('UPLOAD_OK ' + path.name, flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(upload, files))
    remote = release_info()
    assets = {asset['name']: asset for asset in remote['assets']}
    if set(assets) != set(hashes):
        raise ValueError('Uploaded asset names do not match the local snapshot')
    verified = []
    for name, expected in hashes.items():
        actual = assets[name]
        if (actual['size'] != expected['bytes'] or
                actual.get('digest') != 'sha256:' + expected['sha256'] or
                actual['state'] != 'uploaded'):
            raise ValueError('Uploaded digest or size mismatch: ' + name)
        verified.append(dict(name=name, bytes=actual['size'], digest=actual['digest']))
    gh('release', 'edit', TAG, '--repo', REPO, '--draft=false', '--prerelease', '--latest=false')
    published = release_info()
    if published['draft'] or not published['published_at']:
        raise ValueError('The release is not published')
    receipt = dict(status='published', url=published['html_url'], tag=TAG, target=TARGET,
                   checkedUtc=datetime.now(timezone.utc).isoformat(),
                   privateRepository=True, assets=verified, summary=summary,
                   checks=dict(archiveFileHashes=True, uploadedAssetDigests=True,
                               sourceRecords=len(manifest['records']),
                               sourceFiles=len(manifest['files']), inferenceRun=False))
    (HERE / 'publication-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(f'PUBLICATION_OK assets={len(verified)} records={len(manifest["records"])} '
          f'files={len(manifest["files"])} private=true url={published["html_url"]}', flush=True)


if __name__ == '__main__':
    main()

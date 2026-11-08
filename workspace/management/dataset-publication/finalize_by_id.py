"""Finish the existing upload using the draft release ID."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = Path('S:/CodexArtifacts/hacktheburgh/dataset-review-20261108-v1')
API = 'repos/NeelHaval/viridian402/releases/438114990'


def gh(*args):
    result = subprocess.run(['gh', *args], capture_output=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


checksums = {}
for line in (ROOT / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
    digest, name = line.split('  ', 1)
    checksums[name] = digest
for name in ['DATASET.json', 'ASSETS.json', 'README.md', 'build_release.py', 'verify_release.py']:
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksums[name], name
assets = json.loads((ROOT / 'ASSETS.json').read_text(encoding='utf-8'))['assets']
expected = {item['name']: item for item in assets}
for name in ['ASSETS.json', 'SHA256SUMS.txt']:
    path = ROOT / name
    expected[name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
manifest = json.loads((ROOT / 'DATASET.json').read_text(encoding='utf-8'))
proof = Path('S:/CodexArtifacts/hacktheburgh/final-publish.log').read_text(encoding='utf-8')
assert 'RELEASE_CHECK_OK files=1214 records=375 ready=362 archives=7' in proof
assert len(manifest['records']) == 375 and len(manifest['files']) == 1214
assert sum(bool(r['ready']) for r in manifest['records']) == 362
assert all(not r['ready'] and 'AI-derived' in r['truthSource']
           for r in manifest['records'] if r.get('family') == 'MC765')
remote = gh('api', API)
actual = {item['name']: item for item in remote['assets']}
assert set(actual) == set(expected)
for name, item in expected.items():
    assert actual[name]['state'] == 'uploaded', name
    assert actual[name]['size'] == item['bytes'], name
    assert actual[name]['digest'] == 'sha256:' + item['sha256'], name
assert gh('repo', 'view', 'NeelHaval/viridian402', '--json', 'isPrivate')['isPrivate']
print(f'UPLOAD_DIGEST_CHECK_OK assets={len(actual)}', flush=True)
request = HERE / 'publish-request.json'
request.write_text(json.dumps(dict(draft=False, prerelease=True, make_latest='false')), encoding='utf-8')
gh('api', '--method', 'PATCH', API, '--input', str(request))
published = gh('api', API)
assert not published['draft'] and published['published_at']
assert published['tag_name'] == 'dataset-review-20261108-v1'
receipt = dict(status='published', url=published['html_url'], tag=published['tag_name'],
               privateRepository=True, checkedUtc=datetime.now(timezone.utc).isoformat(),
               assets=[dict(name=name, bytes=a['size'], digest=a['digest']) for name, a in actual.items()],
               summary=manifest['summary'], checks=dict(archiveFileHashes=True,
               uploadedAssetDigests=True, sourceRecords=375, sourceFiles=1214, inferenceRun=False))
(HERE / 'publication-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(f'PUBLICATION_OK assets={len(actual)} records=375 files=1214 private=true url={published["html_url"]}')

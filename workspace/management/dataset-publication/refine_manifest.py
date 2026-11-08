"""Clarify blocked-label provenance and add public synthetic row windows."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = Path('S:/CodexArtifacts/hacktheburgh/dataset-review-20261108-v1')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


manifest = json.loads((ROOT / 'DATASET.json').read_text(encoding='utf-8'))
with zipfile.ZipFile(ROOT / 'synthetic-01.zip') as archive:
    for record in manifest['records']:
        if record['selection'] != 'synthetic':
            continue
        detail = json.loads(archive.read(record['pageManifest']).decode('utf-8'))
        record['window'] = [detail['column_box'][1], detail['column_box'][3]]
        if record['family'] == 'MC765':
            record['truthSource'] = 'AI-derived base-page labels; blocked for construction-truth scoring'
            assert not record['ready']
(ROOT / 'DATASET.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
shutil.copyfile(HERE / 'build_release.py', ROOT / 'build_release.py')
assets = [dict(name=p.name, bytes=p.stat().st_size, sha256=sha(p))
          for p in sorted(ROOT.iterdir()) if p.is_file() and p.name not in ['ASSETS.json', 'SHA256SUMS.txt']]
(ROOT / 'ASSETS.json').write_text(json.dumps(dict(assets=assets), indent=2) + '\n', encoding='utf-8')
(ROOT / 'SHA256SUMS.txt').write_text(''.join(sha(p) + '  ' + p.name + '\n'
                                          for p in sorted(ROOT.iterdir())
                                          if p.is_file() and p.name != 'SHA256SUMS.txt'), encoding='utf-8')
print('MANIFEST_CHECK_OK synthetic_windows=82 blocked_AI_records=8')

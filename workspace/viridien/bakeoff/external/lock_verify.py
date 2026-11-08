"""Lock the verify wells away (encrypted) so nobody, human or agent, can look at them.

    python -I bakeoff/external/lock_verify.py            # dry run: list what would be locked
    python -I bakeoff/external/lock_verify.py --lock     # encrypt + delete plaintext
    python -I bakeoff/external/lock_verify.py --unlock   # restore (final verification only)

Anything under viridien_work whose path mentions a verify well or verify log id
is packed (images, boxes, truth, Emil copies, per-window results, source PDFs),
together with eval_set_verify.json and the pre-split eval_set.json.
After locking, eval_set.json is rewritten to dev+test only.
The key lives OUTSIDE the project folder: ~/.lithology_verify.key (keep it out of git).
"""
import csv, io, json, sys, tarfile
from pathlib import Path
from cryptography.fernet import Fernet

WORK = Path(__file__).resolve().parents[2]  # viridien_work
ROOT = WORK / 'external_tests'
VAULT = WORK / 'external_vault'
KEY = Path.home() / '.lithology_verify.key'
SKIP = {'external_vault', '.git', '__pycache__'}


def targets():
    splits = json.load(open(ROOT / 'splits.json', encoding='utf-8'))
    wells = splits['summary']['verify']['wells']
    ids = splits['logs']['verify']
    srcs = [r['source_url'].rsplit('/', 1)[-1] for r in csv.DictReader(open(ROOT / 'manifest.csv', encoding='utf-8'))
            if r['id'] in ids and r['source_url']]
    needles = [n.lower() for n in wells + ids + srcs if n]
    hits = []
    for p in WORK.rglob('*'):
        if not p.is_file() or SKIP & set(p.relative_to(WORK).parts):
            continue
        name = p.name.lower()
        if any(n in name for n in needles):
            hits.append(p)
    hits += [ROOT / n for n in ('eval_set_verify.json', 'eval_set.json', 'manifest.csv',
                                'gallery.png', 'gallery.html', 'overview.html')]
    return sorted(set(h for h in hits if h.exists())), splits


def lock():
    files, splits = targets()
    if not KEY.exists():
        KEY.write_bytes(Fernet.generate_key())
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for f in files:
            tar.add(f, arcname=str(f.relative_to(WORK)))
    VAULT.mkdir(exist_ok=True)
    (VAULT / 'verify.tar.gz.enc').write_bytes(Fernet(KEY.read_bytes()).encrypt(buf.getvalue()))
    (VAULT / 'README.txt').write_text(
        f'{len(files)} files of the verify wells {splits["summary"]["verify"]["wells"]} are encrypted here.\n'
        'Do not unlock until the final verification run. Key: ~/.lithology_verify.key\n', encoding='utf-8')
    check = Fernet(KEY.read_bytes()).decrypt((VAULT / 'verify.tar.gz.enc').read_bytes())
    with tarfile.open(fileobj=io.BytesIO(check), mode='r:gz') as tar:
        n = sum(m.isfile() or m.islnk() for m in tar.getmembers())
    if n != len(files):
        sys.exit(f'vault check failed ({n} != {len(files)}); nothing deleted')
    dev_test = (json.load(open(ROOT / 'eval_set_dev.json', encoding='utf-8'))
                + json.load(open(ROOT / 'eval_set_test.json', encoding='utf-8')))
    rows = list(csv.DictReader(open(ROOT / 'manifest.csv', encoding='utf-8')))
    for f in files:
        f.unlink()
    with open(ROOT / 'manifest.csv', 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(r for r in rows if r['id'] not in splits['logs']['verify'])
    json.dump(dev_test, open(ROOT / 'eval_set.json', 'w', encoding='utf-8'), indent=1)
    print(f'locked {len(files)} files -> {VAULT / "verify.tar.gz.enc"}; eval_set.json now dev+test ({len(dev_test)} windows)')


def unlock():
    data = Fernet(KEY.read_bytes()).decrypt((VAULT / 'verify.tar.gz.enc').read_bytes())
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        tar.extractall(WORK, filter='data')
    print('restored verify files; eval_set.json is the full pre-split set again')


if __name__ == '__main__':
    if '--lock' in sys.argv:
        lock()
    elif '--unlock' in sys.argv:
        unlock()
    else:
        files, _ = targets()
        for f in files:
            print(f.relative_to(WORK).as_posix())
        print(len(files), 'files would be locked')

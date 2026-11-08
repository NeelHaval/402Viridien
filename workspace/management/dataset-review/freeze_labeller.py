"""Keep the first owner-approved sparse truth and suggestion snapshot."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'freezes' / 'labeller-first-20261108'


def main():
    if (DEST / 'snapshot.json').exists():
        saved = json.loads((DEST / 'snapshot.json').read_text(encoding='utf-8'))
        print('FROZEN_SUBSET_EXISTS files=' + str(len(saved['files'])))
        return
    files = [ROOT / 'inputs' / name for name in ('labeller-inventory.json', 'review-suggestions.json')]
    for folder in ('truth', 'ignore'):
        files.extend(sorted((ROOT / 'labeller-assets' / folder).rglob('*')))
    hashes = {}
    for source in files:
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != hashlib.sha256(source.read_bytes()).hexdigest():
            raise RuntimeError('Source changed during freeze: ' + str(relative))
        hashes[str(relative).replace('\\', '/')] = digest
    inventory = json.loads((DEST / 'inputs/labeller-inventory.json').read_text(encoding='utf-8'))
    suggestions = json.loads((DEST / 'inputs/review-suggestions.json').read_text(encoding='utf-8'))
    summary = inventory['summary']
    snapshot = {'source': 'First owner-approved labeller export', 'summary': summary,
                'inventoryRecords': len(inventory['records']),
                'suggestionRecords': len(suggestions['records']), 'files': hashes,
                'note': 'Source path references retain their original meaning. Frozen sparse truth and ignore files are copied here; model suggestions are not truth.'}
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'snapshot.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    print(f'FROZEN_SUBSET_OK pages={len(inventory["records"])} human_boxes={summary["readyHumanBoxes"]} suggestions={len(suggestions["records"])} files={len(hashes)}')


if __name__ == '__main__':
    main()

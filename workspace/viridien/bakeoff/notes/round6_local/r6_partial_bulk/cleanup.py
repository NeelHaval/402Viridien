from pathlib import Path
import csv
root=Path('viridien_work/external_unlabelled/round6/r6_partial_bulk').resolve();images=root/'images'
ids={r['id'] for r in csv.DictReader((root/'manifest.csv').open(encoding='utf-8'))};count=0
for p in images.glob('*.png'):
 if p.stem not in ids:
  assert p.resolve().parent==images
  p.unlink();count+=1
print('Removed',count,'duplicate images absent from the manifest.')

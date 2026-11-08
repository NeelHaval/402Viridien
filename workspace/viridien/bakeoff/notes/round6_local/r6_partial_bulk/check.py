from pathlib import Path
import json,csv,hashlib,re
from urllib.parse import unquote
from PIL import Image
R=Path('viridien_work/external_untrusted/round6/r6_partial_bulk');O=Path('viridien_work/external_unlabelled/round6/r6_partial_bulk');N=Path('viridien_work/bakeoff/notes/round6_local/r6_partial_bulk')
seen=set()
for p in [Path('viridien_work/external_unlabelled/manifest.csv'),Path('viridien_work/external_unlabelled/round5/manifest.csv')]:
 for row in csv.DictReader(p.open(encoding='utf-8-sig')):seen.add(unquote(row['source_url']).lower())
rows=list(csv.DictReader((O/'manifest.csv').open(encoding='utf-8')));errors=[];hashes=set()
if len(list((O/'images').glob('*.png')))!=len(rows):errors.append('image count differs from manifest')
docs=json.loads((N/'review_index.json').read_text(encoding='utf-8'));decs=json.loads((N/'decisions.json').read_text(encoding='utf-8'))
expected=set()
for i,x in enumerate(docs):
 dec=decs.get(x['url'],decs.get(str(i),{}))
 if dec.get('similarity',0)>=2:expected.update((x['url'],int(pn)) for pn in dec.get('accepted_pages',[1]))
covered={(r['source_url'],int(r['page'])) for r in rows}
covered.update((r['url'],int(r['page'])) for r in json.loads((O/'deduplication.json').read_text(encoding='utf-8')))
if expected-covered:errors.append('selected pages missing from output or duplicate record')
for row in rows:
 p=O/'images'/(row['id']+'.png');data=p.read_bytes();h=hashlib.sha256(data).hexdigest()
 if h!=row['image_sha256']:errors.append('image hash '+row['id'])
 if h in hashes:errors.append('duplicate '+row['id'])
 hashes.add(h)
 with Image.open(p) as im:
  if (im.width,im.height)!=(int(row['width']),int(row['height'])):errors.append('dimensions '+row['id'])
 if unquote(row['source_url']).lower() in seen:errors.append('existing URL '+row['id'])
 if re.search(r'barlow|marcotte|red.brook|emma.?ward|flat.bay|irvin.?1.?19|forsyth',unquote(row['source_url']),re.I):errors.append('excluded '+row['id'])
 if int(row['similarity'])<2:errors.append('score '+row['id'])
size=sum(p.stat().st_size for base in [R,O,N] for p in base.rglob('*') if p.is_file())
if size>2000000000:errors.append('lane size')
result=dict(pages_checked=len(rows),lane_bytes=size,errors=errors)
(N/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('PASS' if not errors else 'FAIL',len(rows),'pages;',size,'bytes;',len(errors),'errors')
raise SystemExit(bool(errors))

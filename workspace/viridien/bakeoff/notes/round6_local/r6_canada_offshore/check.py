import sys,json,csv,hashlib
from pathlib import Path
from PIL import Image
b=Path(sys.argv[1]);lane='r6_canada_offshore';out=b/'external_unlabelled/round6'/lane;raw=b/'external_untrusted/round6'/lane
t=json.loads((raw/'triage.json').read_text(encoding='utf-8'));rows=list(csv.DictReader((out/'manifest.csv').open(encoding='utf-8')))
assert len(rows)==54
for r in rows:
 f=out/'images'/(r['id']+'.png')
 with Image.open(f) as im:assert im.size==(int(r['width']),int(r['height']));im.verify()
 assert hashlib.sha256(f.read_bytes()).hexdigest()==r['image_sha256']
for f in (out/'bbox_annotation').glob('*.json'):
 x=json.loads(f.read_text(encoding='utf-8'));w,h=x['image_size']
 for box in x['bounding_boxes'].values():assert 0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h
docs=t['sources'][0]['documents'];assert len(docs)==16
print(f'PASS: {len(docs)} unique documents; {sum(x["pages"] for x in docs)} source pages; {len(rows)} valid images; 53 box files; 0 labelled windows')

import sys,json,urllib.request,hashlib,collections
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from genlab import learn,candidates
OUT=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def norm(s):return ''.join(s.casefold().split()).removesuffix('.pdf').removesuffix('.tif')
policy=read(ROOT/'data/normalized/split-policy.json')
def train(d):return int(hashlib.sha256((policy['seed']+':'+policy['document_groups'][d]).encode()).hexdigest()[:8],16)%5!=0
inv=read(OUT/'inventory.json');focus={e['key']:e for e in inv['entries']};state=read(OUT/'state-snapshot.json');cards={c['key']:c for c in state['cards']}
names={norm(r['name']):r['doc_id'] for r in (read(p) for p in (ROOT/'data/genlab/candidates').glob('*.json') if len(p.stem)==16)}
legend=json.load(urllib.request.urlopen('http://127.0.0.1:8771/api/legend?v=2'))
missing=set();pairs=[];heldout=[]
for group in legend['groups']:
    for item in group['items']:
        members=[]
        for name,lid,label in item['members']:
            did=names.get(norm(name))
            if not did:missing.add(name);continue
            members.append((did+':'+str(lid),label))
        for key,label in members:
            if key not in focus:continue
            rgb,_=learn.clean(np.asarray(Image.open(OUT/f"swatch-{key.replace(':','-')}.png").convert('RGB')))
            for dk,dl in members:
                if dk==key or dk.split(':')[0]==key.split(':')[0]:continue
                if not train(dk.split(':')[0]):heldout.append((key,dk));continue
                c=cards.get(dk)
                if not c or not c.get('tile'):continue
                tile=np.asarray(Image.open(ROOT/'data/genlab'/c['tile']).convert('RGB'));d=dict(tile=tile,bg=learn.background(tile))
                pairs.append(dict(target=key,donor=dk,label=dl,source=c['source'],tile=c['tile'],rgb=candidates.pixel_match(rgb,d,distance=20),dice=candidates.symbol_match(rgb,d),rows=focus[key]['estimated_rows']))
(OUT/'lookalike-results.json').write_text(json.dumps(dict(pairs=pairs,heldout_pairs_excluded=heldout,unmapped_names=sorted(missing)),indent=2),encoding='utf-8')
print('pairs',len(pairs),'targets',len(set(p['target'] for p in pairs)),'excluded test',len(heldout),'unmapped',missing)
for p in pairs:print(p)

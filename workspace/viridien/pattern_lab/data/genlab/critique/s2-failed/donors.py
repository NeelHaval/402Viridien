import sys,json,hashlib,collections
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from genlab import learn,candidates
OUT=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
inv=read(OUT/'inventory.json');state=read(OUT/'state-snapshot.json');policy=read(ROOT/'data/normalized/split-policy.json')
def train(d):return int(hashlib.sha256((policy['seed']+':'+policy['document_groups'][d]).encode()).hexdigest()[:8],16)%5!=0
pool=[]
for c in state['cards']:
    if not train(c['doc_id']) or not c.get('tile'):continue
    with Image.open(ROOT/'data/genlab'/c['tile']) as im:tile=np.asarray(im.convert('RGB'))
    if tile.shape[0]*tile.shape[1]>200000:continue
    pool.append((c,dict(tile=tile,bg=learn.background(tile),kind='tile')))
results=[]
for e in inv['entries']:
    raw=np.asarray(Image.open(OUT/f"swatch-{e['key'].replace(':','-')}.png").convert('RGB'))
    rgb,_=learn.clean(raw);bg=learn.background(rgb);hits=[]
    for c,d in pool:
        if c['doc_id']==e['doc_id'] or candidates.name_key(c['label'])!=candidates.name_key(e['label']):continue
        dice=candidates.symbol_match(rgb,d)
        dist=float(np.linalg.norm(bg-d['bg']))
        match=candidates.pixel_match(rgb,d,distance=20) if dist<12 else 0
        hits.append(dict(donor=c['key'],source=c['source'],size=list(d['tile'].shape[:2]),dice=round(dice,5),rgb=round(match,5),bg_distance=round(dist,2),tile=c['tile'],real=c.get('real'),real_doc_id=c.get('real_doc_id',c['doc_id'])))
    hits.sort(key=lambda h:(h['rgb']>=.92 and h['dice']>=.8,h['dice']),reverse=True)
    results.append(dict(key=e['key'],label=e['label'],rows=e['estimated_rows'],hits=hits[:5],strict=any(h['rgb']>=.92 and h['dice']>=.8 and h['source'] in ('human','machine-verified') for h in hits),broad_strict=any(h['rgb']>=.92 and h['dice']>=.8 for h in hits),geometry=any(h['dice']>=.8 for h in hits)))
(OUT/'donor-results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('COUNTS',len(results),'strict',sum(r['strict'] for r in results),'broad strict',sum(r['broad_strict'] for r in results),'geometry',sum(r['geometry'] for r in results))
for r in results:
    if r['strict'] or r['broad_strict'] or r['geometry'] or r['rows']:print(r['key'],r['label'],'strict',r['strict'],'broad',r['broad_strict'],'best',r['hits'][:1])

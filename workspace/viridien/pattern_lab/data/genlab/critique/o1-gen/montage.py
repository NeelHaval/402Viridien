import json, sys, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, '.')
from genlab.realism import aligned
V = 'data/genlab/datasets/v4/'
st = json.load(open(V+'stats.json', encoding='utf-8'))
cls = {c['class_id']: c for c in st['classes']}
out = json.load(open('data/genlab/critique/o1-gen/match.json'))
seen=set(); rows=[]
for o in out:
    if o['ok'] or o['cid'] in seen: continue
    seen.add(o['cid'])
    rgb = np.asarray(Image.open(V+o['file']).convert('RGB'))[:, 3:-3]
    panels=[rgb]
    for k in (o['cid'], o['best']):
        t = np.asarray(Image.open(V+cls[k]['model_file']).convert('RGB'))
        f,_ = aligned(t, rgb.astype(float)); panels.append(f)
        raw = np.asarray(Image.open(V+cls[k]['tile_provenance']['raw_file']).convert('RGB')); panels.append(raw)
    h = max(p.shape[0] for p in panels); gap=np.full((h,6,3),255,np.uint8)
    row = np.concatenate(sum([[np.pad(p,((0,h-p.shape[0]),(0,0),(0,0)),constant_values=255),gap] for p in panels],[]),1)
    rows.append((o['cid']+' '+o['name']+' | real | true-tile aligned | true legend raw | best-other aligned | best-other legend raw ('+o['best']+')', row))
W=max(r.shape[1] for _,r in rows)
canvas=[]
for t,r in rows:
    lab=np.full((14,W,3),255,np.uint8); im=Image.fromarray(lab); ImageDraw.Draw(im).text((2,1),t,fill=(0,0,0))
    canvas += [np.asarray(im), np.pad(r,((0,0),(0,W-r.shape[1]),(0,0)),constant_values=255), np.full((8,W,3),255,np.uint8)]
img=Image.fromarray(np.concatenate(canvas)); img=img.resize((img.width*2,img.height*2),Image.NEAREST)
img.save('data/genlab/critique/o1-gen/val-failures.png'); print(img.size)

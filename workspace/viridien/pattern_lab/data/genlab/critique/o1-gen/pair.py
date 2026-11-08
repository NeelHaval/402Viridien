import json, sys, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0,'.')
from genlab.realism import aligned
V='data/genlab/datasets/v4/'
st=json.load(open(V+'stats.json',encoding='utf-8')); cls={c['class_id']:c for c in st['classes']}
real=json.load(open(V+'real.json',encoding='utf-8'))
ids=sys.argv[2:]; rows=[]
for cid in ids:
    r=next(x for x in real if x['class_id']==cid)
    R=np.asarray(Image.open(V+r['file']).convert('RGB'))[:120]
    T=np.asarray(Image.open(V+cls[cid]['model_file']).convert('RGB'))
    F,_=aligned(T,R.astype(float))
    raw=np.asarray(Image.open(V+cls[cid]['tile_provenance']['raw_file']).convert('RGB')) if 'tile_provenance' in cls[cid] else T
    ps=[R,F,raw]; h=max(p.shape[0] for p in ps)
    row=np.concatenate(sum([[np.pad(p,((0,h-p.shape[0]),(0,0),(0,0)),constant_values=255),np.full((h,8,3),200,np.uint8)] for p in ps],[]),1)
    rows.append(row)
W=max(r.shape[1] for r in rows)
img=np.concatenate([np.pad(r,((0,10),(0,W-r.shape[1]),(0,0)),constant_values=255) for r in rows])
Image.fromarray(img).resize((W*3,img.shape[0]*3),Image.NEAREST).save(sys.argv[1])

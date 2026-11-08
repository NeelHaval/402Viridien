"""Native-pixel contact sheets from bounded saved crops, never full TIFFs."""
import sys, os, json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[4]; OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
idx=read('data/normalized/current.json'); st=read('data/genlab/datasets/v5/stats.json')
state=read('data/genlab/state.json'); noise=read('data/genlab/assemble/noise/measurements.json')
def load(path):
 with Image.open(path) as im:
  assert im.width*im.height<4000000
  return np.asarray(im.convert('RGB')).copy()
def repeat(tile,h,w):return tile[np.arange(h)[:,None]%len(tile),np.arange(w)[None,:]%tile.shape[1]]
def sheet(name,items):
 width=950; height=sum(max(a.shape[0] for _,a in arr)+60 for _,arr in items)+20
 im=Image.new('RGB',(width,height),'#ededed');d=ImageDraw.Draw(im);y=10
 for title,arr in items:
  d.text((10,y),title,fill='black');y+=24
  for i,(label,a) in enumerate(arr):
   x=15+i*300;d.text((x,y),label,fill='black');im.paste(Image.fromarray(a),(x,y+18))
  y+=max(a.shape[0] for _,a in arr)+36
 path=OUT/name;im.save(path);print('IMAGE',path,im.size,flush=True)
r=next(r for r in idx['records'] if r['document_id']=='28112eef4c3c645a' and 'duplicate_of_test' in r['quality_issues'])
t=next(t for t in idx['records'] if t['split']=='test' and t['pixel_sha256']==r['pixel_sha256'])
a=load(ROOT/'data/normalized'/r['native_png']);b=load(ROOT/'data/normalized'/t['native_png'])
c=next(c for c in st['classes'] if c['class_id']=='28112eef4c3c645a:20')
tile=load(ROOT/'data/genlab/datasets/v5'/c['model_file'])
items=[('Exact cross-well duplicate; every image at 1 source pixel = 1 display pixel',[
 ('TRAIN 28112eef4c3c645a shale',a),('TEST eb4e39e664edd11d shale',b),('V5 rendered training tile',repeat(tile,*a.shape[:2]))])]
bad=next(r for r in idx['records'] if 'legend_fill_mismatch' in r['quality_issues'])
a2=load(ROOT/'data/normalized'/bad['native_png'])
c=next(c for c in st['classes'] if c['class_id']==bad['document_id']+':'+str(bad['targets'][0]['legend_id']))
tile=load(ROOT/'data/genlab/datasets/v5'/c['model_file'])
items.append(('Quality-rejected box admitted to training augmentation: '+bad['document_id'],[
 ('Real legend_fill_mismatch crop',a2),('V5 rendered training tile',repeat(tile,*a2.shape[:2]))]))
sheet('native-quality.png',items)
print('DUPLICATE',json.dumps(dict(equal=bool(np.array_equal(a,b)),shape=a.shape,colours=len(np.unique(a.reshape(-1,3),axis=0)),white_fraction=float((a.min(2)>240).mean()),train_box=r['box_xyxy'],test_box=t['box_xyxy'])),flush=True)
# Validation crop demonstrably used as a noise source; use existing cache only.
for m in noise['measurements']:
 if m['source_doc']!='b9ba393ab59601f2':continue
 crop=ROOT/'data/genlab/crops'/m['source_doc']/('%d_%d_%d_%d.png'%tuple(m['box']))
 card=next((c for c in state['cards'] if c['key']==m['class_id']),None)
 if crop.exists() and card and (ROOT/'data/genlab'/card['tile']).exists():
  a=load(crop)[:240];tile=load(ROOT/'data/genlab'/card['tile'])
  sheet('native-validation.png',[(m['class_id']+' validation well used by ASM noise fitter',[
    ('Real crop used for residual fitting',a),('Generated repeat, native scale',repeat(tile,*a.shape[:2])),
    ('Stored noise mask (same origin)',load(ROOT/'data/genlab/assemble/noise'/m['mask'])[:240])])])
  print('VALIDATION_SOURCE',json.dumps(m),flush=True);break

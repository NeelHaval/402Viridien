import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import model
from genlab import learn,candidates,paths,render
OUT=Path(__file__).resolve().parent
paths.CROPS=OUT/'crops'
snap=json.loads((OUT/'snapshot.json').read_text())
cards=snap['cards']
store=model.Store()
results={}

def load(stem): return np.asarray(Image.open(OUT/stem).convert('RGB'))
def tile(c): return np.asarray(Image.open(ROOT/'data/genlab'/c['tile']).convert('RGB'))

# Quantify overlap in all eight cards advertised as independently held out.
state=json.loads((ROOT/'data/genlab/state.json').read_text())
overlap=[]
for c in state['cards']:
    if not (c.get('realism') or {}).get('held_out'): continue
    if c['source']=='machine-verified':
        report=json.loads((ROOT/'data/genlab/candidates'/f'{c["doc_id"]}.json').read_text())
        entry=next(e for e in report['entries'] if e['legend_id']==c['legend_id'])
        exs=candidates.examples(store.get(c['doc_id']),entry)
    else:
        doc=store.get(c['doc_id'])
        exs=[dict(id=e['id'],box=e['box']) for e in doc['segments']
             if e.get('status')=='approved' and e.get('legend_ids')==[c['legend_id']]]
    used=set(c['sources'])|{x['id'] for x in c.get('suspects',[])}
    train=[e for e in exs if e['id'] in set(c['sources'])]
    held=sorted([e for e in exs if e['id'] not in used],key=lambda e:-(e['box'][3]-e['box'][1]))[:24]
    measurements=[]
    for e in held:
        a=e['box']; rows=set()
        for t in train:
            b=t['box']
            if min(a[2],b[2])>max(a[0],b[0]):
                rows.update(range(max(a[1],b[1]),min(a[3],b[3])))
        measurements.append(dict(id=e['id'],box=a,overlap_rows=len(rows),
                                 fraction=round(len(rows)/(a[3]-a[1]),5)))
    overlap.append(dict(key=c['key'],source=c['source'],reported=c['realism'],held=measurements))
results['held_overlap']=overlap

# Distinguish symbol loss from good background agreement.
blanks=[]
for i,c in enumerate(cards):
    a=tile(c)
    r=load(f'{i+1:02d}-{c["doc_id"]}-{c["legend_id"]}-real0.png')
    r,_=learn.clean(r)
    blank=np.full_like(a,learn.background(a).astype(np.uint8))
    data=dict(tile=blank,bg=learn.background(blank))
    blanks.append(dict(key=c['key'],black_ink_fraction=round(float((a.max(2)<80).mean()),5),
       blank_pixel=round(candidates.pixel_match(r,data),5),
       blank_ink_dice=round(candidates.symbol_match(r,data),5)))
results['blank_baselines']=blanks

# Native repeat consistency, without assuming that a short period is correct.
curves=[]
for n in [2,5]:
    c=cards[n-1]; arr=tile(c)
    raw=load(f'{n:02d}-{c["doc_id"]}-{c["legend_id"]}-real0.png')
    rgb,_=learn.clean(raw)
    curve=[]
    for lag in ([6,12,18,24,30,36] if n==2 else [10,20,40,80]):
        if len(rgb)<=lag:continue
        a,b=rgb[:-lag].astype(float),rgb[lag:].astype(float)
        m1,m2=a.max(2)<80,b.max(2)<80
        curve.append(dict(lag=lag,pixel=round(float((np.linalg.norm(a-b,axis=2)<45).mean()),5),
            dice=round(float(2*(m1&m2).sum()/max(1,m1.sum()+m2.sum())),5)))
    curves.append(dict(key=c['key'],lag_curve=curve,
       tile_colours=np.unique(arr.reshape(-1,3),axis=0).tolist() if n==5 else 'continuous',
       real_colours=np.unique(rgb.reshape(-1,3),axis=0).tolist() if n==5 else 'continuous'))
results['repeat_curves']=curves

# Render the gray rail without the renderer's intentional frame, across phases.
c=cards[0]; a=tile(c)
data=dict(kind='tile',tile=a,anchored=False,phase=(0,0))
phase_rows=[]
views=[]
for seed in range(8):
    rgb=render.layer(data,250,310,0,120,np.random.default_rng(seed))
    columns=[x for x in range(1,59) if (rgb[:,x].max(1)<150).mean()>.95]
    phase_rows.append(dict(seed=seed,interior_rail_columns=columns))
    views.append(Image.fromarray(rgb).resize((180,360),Image.Resampling.NEAREST))
results['random_phase_rail']=phase_rows
sheet=Image.new('RGB',(8*190,400),'#e8e8e8')
draw=ImageDraw.Draw(sheet)
for i,v in enumerate(views):
    sheet.paste(v,(i*190,30));draw.text((i*190+4,4),f'seed {i}',fill='black')
sheet.save(OUT/'rail-phases.png')

# A separate equal-scale close-up of the three concrete human-tile defects.
sheet=Image.new('RGB',(1650,900),'#e8e8e8');draw=ImageDraw.Draw(sheet)
for row,n in enumerate([1,2,5]):
    c=cards[n-1]
    raw=load(f'{n:02d}-{c["doc_id"]}-{c["legend_id"]}-real0.png')
    rgb=np.asarray(Image.open(ROOT/'data/genlab'/c['samples'][0]).convert('RGB'))
    t=tile(c)
    draw.text((4,row*300+4),c['key'],fill='black')
    for j,(title,x) in enumerate([('real',raw),('tile',t),('generated',rgb)]):
        small=Image.fromarray(x[:48,:100])
        sheet.paste(small.resize((small.width*5,small.height*5),Image.Resampling.NEAREST),(j*550,row*300+40))
        draw.text((j*550+4,row*300+20),title,fill='black')
sheet.save(OUT/'human-defects.png')
(OUT/'diagnostics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=1))

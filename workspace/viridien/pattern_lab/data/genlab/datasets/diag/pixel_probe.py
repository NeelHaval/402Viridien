import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from genlab.dataset import *
from genlab.build import _write
f=ROOT/'v1'
classes={r['class_id']:r for r in read(f/'stats.json')['classes']}
truth=[r for r in read(f/'real.json') if r['split']=='validation']
records=[]
for r in truth:
    meta=classes.get(r['class_id'])
    if not meta:
        records.append(dict(class_id=r['class_id'],annotation_id=r['annotation_id'],match=None,reason='No v1 fake for accepted class'))
        continue
    tile=np.asarray(Image.open(f/meta['model_file']).convert('RGB'))
    rgb=np.asarray(Image.open(f/r['file']).convert('RGB'))
    data=dict(kind='tile',tile=tile,bg=learn.background(tile))
    value=candidates.pixel_match(rgb,data)
    records.append(dict(class_id=r['class_id'],annotation_id=r['annotation_id'],match=value))
_write(ROOT/'diag/pixel-matches.json',records)
print('PASS true-class cyclic pixel probe',len([r for r in records if r['match'] is not None]),'median',np.median([r['match'] for r in records if r['match'] is not None]))
sink=[m for m in classes.values() if m['name']=='anhydrite' and m['split']=='train']
sheet=Image.new('RGB',(900,330),'#dedede');draw=ImageDraw.Draw(sheet)
for i,m in enumerate(sink):
    tile=np.asarray(Image.open(f/m['model_file']).convert('RGB'))
    rgb=render.layer(dict(kind='tile',tile=tile,anchored=False),0,m['width'],0,240,np.random.default_rng(1))
    x=10+i*140
    sheet.paste(Image.fromarray(rgb),(x,50))
    draw.text((x,10),m['doc'][:12],fill='black')
    draw.text((x,28),f"{m['legend_id']}: white={100*(rgb.min(2)>=240).mean():.1f}%",fill='black')
sheet.save(ROOT/'diag/anhydrite-training-prototypes.png')
print('anhydrite training document classes',len(sink))

"""Bounded window reads; all rendered evidence stays beside this script."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import model
from genlab.assemble.noise import Windows

OUT=Path(__file__).parent
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sheet(name, panels, captions):
    widths=[max(180,p.width) for p in panels]
    canvas=Image.new('RGB',(sum(widths)+30*(len(panels)+1),max(p.height for p in panels)+85),'#eeeeee')
    draw=ImageDraw.Draw(canvas); x=30
    for p,caption,width in zip(panels,captions,widths):
        draw.text((x,8),caption,fill='black')
        canvas.paste(p,(x,75));x+=width+30
    canvas.save(OUT/name)
    print('IMAGE',name,canvas.size)

did='d368760651ded938'
folder=ROOT/'data/genlab/assemble/out/starter/page-texas-rock-types-box-d368760651ded938-01'
t=read(folder/'truth.json'); gap=next(l for l in t['layers'] if l['class_id']=='__gap__')
a=gap['top']; b=min(gap['bottom'],a+512)
with Image.open(folder/'column.png') as im: fake=im.crop((0,a,t['width'],b)).convert('RGB')
with Image.open(folder/'mask.png') as im: mask=np.asarray(im.crop((0,a,t['width'],b)))
with Image.open(ROOT/'data/genlab/assemble/noise'/gap['source_file']) as im: source=im.crop((0,0,t['width'],b-a)).convert('RGB')
sheet('gap-truth.png',[source,fake,Image.fromarray(np.where(mask[...,None]==0,[255,70,70],[255,255,255]).astype('uint8'))],
      ['Real source\nunlabelled\n1 px = 1 px','Synthetic\n__gap__\n1 px = 1 px','Mask zero\nshown red\n1 px = 1 px'])
print('GAP_PIXEL_CHECK',{'window_rows':[a,b],'zero_mask_fraction':float((mask==0).mean()),'nonwhite_fraction':float((np.asarray(fake).min(2)<240).mean())})

store=model.Store();doc=store.get(did);windows=Windows()
x0,y0,x1,y1=doc['boxes']['interpreted_lithology_column'];box=[x0-180,y0+a,x1+180,y0+b]
real=Image.fromarray(windows.get(doc,box));windows.close()
page=Image.new('RGB',real.size,'white')
for band in read(folder/'page-bands.json')['bands']:
    lo=max(box[1],band['top']);hi=min(box[3],band['bottom'])
    if hi<=lo:continue
    with Image.open(folder/band['file']) as im:
        crop=im.crop((box[0],lo-band['top'],box[2],hi-band['top'])).convert('RGB')
        page.paste(crop,(0,lo-box[1]))
sheet('page-context.png',[real,page],['Real page, original native pixels','Composed page, same coordinates and scale'])
ra=np.asarray(real);pa=np.asarray(page);outside=np.ones(ra.shape[:2],bool);outside[:,180:180+x1-x0]=False
print('CONTEXT',{'box':box,'outside_column_equal_fraction':float((ra[outside]==pa[outside]).all(1).mean()),'column_equal_fraction':float((ra[:,180:180+x1-x0]==pa[:,180:180+x1-x0]).all(2).mean()),'depth_calibration':doc.get('depth_calibration')})

stats=read(ROOT/'data/genlab/assemble/stats.json');cards={c['key']:c for c in read(ROOT/'data/genlab/state.json')['cards']}
families={d:k[7:] for k,v in stats['levels'].items() if k.startswith('family:') for d in v['document_ids']}
norm=lambda s:''.join(s.casefold().split())
subs=[]
for cid,c in stats['classes'].items():
    did=c['doc_id'];family=families[did]
    if cid in cards:continue
    matches=sorted(k for k,v in cards.items() if families.get(v['doc_id'])==family and norm(v['label'])==norm(c['name']))
    evidence=any((cid if level.startswith('doc:') else c['norm']) in block.get('thickness',{}) for level,block in stats['levels'].items() if level in ('doc:'+did,'family:'+family))
    if matches and evidence:subs.append((cid,matches[0],c['name'],family))
print('NAME_ONLY_SUBSTITUTES',len(subs),subs)

from genlab.assemble.noise import load_model
from genlab import render, learn
real_index=read(ROOT/'data/genlab/datasets/v5/real.json')
for target,donor in [('16c9064b18b77b61:3','5709fe9e86ce942b:7'),('7966a10fba6fa984:11','3515004b6f643c87:11')]:
    did,lid=target.split(':');doc=store.get(did)
    entry=next(e for e in doc['legend'] if str(e['id'])==lid)
    windows=Windows(); swatch=windows.get(doc,entry['box']);windows.close()
    w=doc['boxes']['interpreted_lithology_column'][2]-doc['boxes']['interpreted_lithology_column'][0]
    synthetic=render.layer(load_model(cards[donor]),0,w,0,160,np.random.default_rng(42))
    panels=[Image.fromarray(swatch),Image.fromarray(synthetic)];labels=['Target legend\n'+target,'Name-only donor\n'+donor]
    rows=[r for r in real_index if r['class_id']==target]
    if rows:
        with Image.open(ROOT/'data/genlab/datasets/v5'/rows[0]['file']) as im:panels.append(im.convert('RGB'))
        labels.append('Real accepted crop\n'+target)
    sheet('substitute-'+did+'.png',panels,labels)
    bg1=learn.background(swatch); bg2=learn.background(synthetic)
    print('SUBSTITUTE_BACKGROUND',target,donor,'legend_rgb',bg1.tolist(),'donor_rgb',bg2.tolist(),'RGB_L2',float(np.linalg.norm(bg1-bg2)))

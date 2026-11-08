import sys,json,collections,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from genlab import learn,candidates
import model
OUT=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
inv=read(OUT/'inventory.json'); focus={e['key']:e for e in inv['entries']}
images=model.Images();docs={};results=[];panels=[]
def crop(did,box):
    cache=ROOT/'data/genlab/crops'/did/('%d_%d_%d_%d.png'%tuple(box))
    if cache.exists():
        with Image.open(cache) as im:return np.asarray(im.convert('RGB'))
    if did not in docs:docs[did]=read(ROOT/'data/seeds'/f'{did}.json')
    return np.asarray(images.crop(docs[did],box).convert('RGB'))
def sw(key):return np.asarray(Image.open(OUT/f"swatch-{key.replace(':','-')}.png").convert('RGB'))
def data(tile):return dict(kind='tile',tile=tile,bg=learn.background(tile))
def repeat(tile,h=160,w=100):return tile[np.arange(h)[:,None]%len(tile),np.arange(w)[None,:]%tile.shape[1]]
def components(mask):
    seen=np.zeros(mask.shape,bool);out=[]
    for y,x in zip(*np.where(mask)):
        if seen[y,x]:continue
        todo=[(y,x)];seen[y,x]=1;pts=[]
        while todo:
            a,b=todo.pop();pts.append((a,b))
            for da in (-1,0,1):
                for db in (-1,0,1):
                    c,d=a+da,b+db
                    if 0<=c<len(mask) and 0<=d<mask.shape[1] and mask[c,d] and not seen[c,d]:seen[c,d]=1;todo.append((c,d))
        yy,xx=zip(*pts);out.append(dict(box=[min(xx),min(yy),max(xx)+1,max(yy)+1],area=len(pts)))
    return out
def stochastic(key):
    rgb,_=learn.clean(sw(key));mask=rgb.max(2)<80;comps=components(mask)
    complete=[c for c in comps if c['box'][0]>0 and c['box'][1]>0 and c['box'][2]<rgb.shape[1] and c['box'][3]<len(rgb)]
    glyphs=[]
    for c in complete:
        x0,y0,x1,y1=c['box'];glyphs.append(rgb[y0:y1,x0:x1])
    outputs=[]
    for seed in (7,11):
        rng=np.random.default_rng(seed);out=np.empty((160,100,3),np.uint8);out[:]=learn.background(rgb);occupied=np.zeros(out.shape[:2],bool)
        for attempt in range(5000):
            if occupied.mean()>=mask.mean():break
            glyph=glyphs[int(rng.integers(len(glyphs)))];gh,gw=glyph.shape[:2]
            y=int(rng.integers(1,160-gh));x=int(rng.integers(1,100-gw));gm=glyph.max(2)<80
            if occupied[max(0,y-1):y+gh+1,max(0,x-1):x+gw+1].any():continue
            patch=out[y:y+gh,x:x+gw];patch[gm]=glyph[gm];occupied[y:y+gh,x:x+gw]|=gm
        outputs.append(out)
    stats=dict(key=key,strategy='stochastic glyph bootstrap; hypothesis only',source_shape=list(rgb.shape),source_components=len(comps),complete_components=len(complete),component_areas=[c['area'] for c in complete],source_ink=float(mask.mean()),generated_ink=[float((o.max(2)<80).mean()) for o in outputs],seed_outputs_distinct=not np.array_equal(*outputs))
    results.append(stats);panels.append((f'{key}: stochastic glyphs (unvalidated)',[('real swatch',rgb),('seed 7',outputs[0]),('seed 11',outputs[1])]))
    for i,o in enumerate(outputs):Image.fromarray(o).save(OUT/f"trial-{key.replace(':','-')}-{i}.png")

# 1. Same-palette donor, with target and independently located donor crop.
key='952caeadf1d8467c:7';raw=sw(key);rgb,_=learn.clean(raw)
donor=next(c for c in read(OUT/'state-snapshot.json')['cards'] if c['key']=='aeb414a4367bafd4:7')
tile=np.asarray(Image.open(ROOT/'data/genlab'/donor['tile']).convert('RGB'));d=data(tile)
box=donor['real'][0];real=crop(donor['doc_id'],box);rc,_=learn.clean(real)
long_box=[1090,208692,1145,208756];long_real=crop(donor['doc_id'],long_box)
target_probes=[];target_crops=[]
doc=read(ROOT/'data/seeds'/f"{key.split(':')[0]}.json");x0,y0,x1,y1=doc['boxes']['interpreted_lithology_column']
for g in focus[key]['guess_intervals']:
    b=[x0+2,g['top_y'],x1-2,g['bottom_y']];r=crop(doc['id'],b);c,_=learn.clean(r)
    target_probes.append(dict(box=b,match=candidates.pixel_match(c,d)));target_crops.append(r)
results.append(dict(key=key,strategy='training donor transfer',donor=donor['key'],source=donor['source'],target_rgb=candidates.pixel_match(rgb,d,distance=20),target_dice=candidates.symbol_match(rgb,d),donor_crop_match=candidates.pixel_match(rc,d),donor_crop_height=len(real),period_y=len(tile),expanded_donor_box=long_box,expanded_donor_match=candidates.pixel_match(long_real,d),expanded_donor_dice=candidates.symbol_match(long_real,d),expanded_period35_overlap_match=float((np.linalg.norm(long_real[35:].astype(float)-long_real[:-35],axis=2)<20).mean()),target_probes=target_probes))
panels.append((f'{key}: longer training donor evidence verifies the 35px period',[('target swatch',rgb),('cited donor 22 rows',real),('expanded donor 64 rows',long_real),('generated donor',repeat(tile)),('target guessed crop',target_crops[0])]))

# 2. Conventional brick spacing is not identified by a cropped field of view.
key='16c9064b18b77b61:5';rgb,_=learn.clean(sw(key));h,w=rgb.shape[:2]
tiles=[]
for width in (w,w*2):
    tile=np.empty((h,width,3),np.uint8);tile[:]=np.median(rgb,axis=1).astype(np.uint8)[:,None,:];tile[:,:w]=rgb;tiles.append(tile)
target_probes=[];target_crops=[]
doc=read(ROOT/'data/seeds'/f"{key.split(':')[0]}.json");x0,y0,x1,y1=doc['boxes']['interpreted_lithology_column']
for g in focus[key]['guess_intervals']:
    b=[x0+2,g['top_y'],x1-2,min(g['bottom_y'],g['top_y']+160)];r=crop(doc['id'],b);target_crops.append(r)
    target_probes.append(dict(box=b,matches=[candidates.pixel_match(learn.clean(r)[0],data(t)) for t in tiles]))
results.append(dict(key=key,strategy='two spacing hypotheses, not accepted tiles',tile_sizes=[list(t.shape[:2]) for t in tiles],swatch_matches=[candidates.pixel_match(rgb,data(t)) for t in tiles],target_probes=target_probes))
panels.append((f'{key}: both spacing hypotheses fit the same real swatch',[('real swatch',rgb),('hypothesis width '+str(w),repeat(tiles[0],160,150)),('hypothesis width '+str(2*w),repeat(tiles[1],160,150)),('guessed column crop',target_crops[0])]))

# 3-4. Bootstrap complete observed glyphs without imposing a repeat lattice.
stochastic('28112eef4c3c645a:21');stochastic('57b1857d99779870:12')

# 5. Rejected Sand: fit the first listed column crop; score the remaining four.
key='5709fe9e86ce942b:6';p=read(ROOT/'data/genlab/proposals/5709fe9e86ce942b/legend-6.json')
proposal=np.asarray(Image.open(ROOT/'data/genlab/proposals/5709fe9e86ce942b'/p['tile_png']).convert('RGB'))
reals=[learn.clean(crop(p['doc_id'],b))[0] for b in p['column_boxes']]
from generator._texture import estimate_period
estimates=[estimate_period(r,1) for r in reals]
# A 14 px horizontal hypothesis is explicitly recorded in the maker's notes.
ref=reals[0];tile=ref[:35,:14].copy()
results.append(dict(key=key,strategy='column-derived diagnostic; not owner labelled',proposal_size=list(proposal.shape[:2]),column_hypothesis_size=list(tile.shape[:2]),horizontal_period_estimates=estimates,proposal_rgb=[candidates.pixel_match(r,data(proposal)) for r in reals],column_hypothesis_rgb=[candidates.pixel_match(r,data(tile)) for r in reals],proposal_ink_dice=[candidates.symbol_match(r,data(proposal)) for r in reals],column_hypothesis_ink_dice=[candidates.symbol_match(r,data(tile)) for r in reals],fit_crop=0,held_out_crops=[1,2,3,4]))
panels.append((f'{key}: rejected Sand, one fit crop then independent listed crops',[('real fit column',reals[0]),('real check column',reals[2][:160]),('rejected 42px tile',repeat(proposal)),('14px column hypothesis',repeat(tile))]))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
for i,(title,items) in enumerate(panels):
    height=max(im.shape[0] for _,im in items)*3+95
    width=max(1100,sum(max(im.shape[1]*3,185)+20 for _,im in items))
    sheet=Image.new('RGB',(width,height),'#dddddd');draw=ImageDraw.Draw(sheet);draw.text((10,8),title,font=font,fill='black');x=10
    for label,im in items:
        draw.text((x,38),label,font=font,fill='black');image=Image.fromarray(im);image=image.resize((image.width*3,image.height*3),Image.Resampling.NEAREST);sheet.paste(image,(x,68));x+=max(image.width,185)+20
    sheet.save(OUT/f'test-{i+1}.png')
if images.current:images.current.close()
(OUT/'test-results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))

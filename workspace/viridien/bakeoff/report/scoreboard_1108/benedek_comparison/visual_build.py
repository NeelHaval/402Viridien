from pathlib import Path
import sys,json,csv,hashlib,html,base64
import numpy as np
from PIL import Image,ImageDraw,ImageFont
BASE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(BASE))
from harness import io
ROOT=BASE/'out/benedek_compare_newtest'
OUT=Path(__file__).resolve().parent
IMG=OUT/'pred_images';IMG.mkdir(exist_ok=True)
records=json.loads((ROOT/'predictions_remote.json').read_text(encoding='utf-8'))['records']
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
groups={}
for r in records:groups.setdefault(r['case_id'],{})[r['strategy']]=r
manifest={'truth_available':False,'selection':'Three fixed 768 px windows: start, middle, end. No labels used for selection.','images':[],'errors':[],'missing':[]}
sections=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for case,models in groups.items():
 if not all(s in models for s in ('s19_combine','s24_benedek')):
  manifest['missing'].append(case);continue
 rec=models['s19_combine'];image_path=ROOT/rec['image'];ann_path=ROOT/rec['annotation']
 ann=json.loads(ann_path.read_text(encoding='utf-8'));log=io.Log(Image.open(image_path),ann['bounding_boxes'],rec['log'])
 hashes={'image':sha(image_path),'annotation':sha(ann_path)}
 rows={}
 for s in ('s19_combine','s24_benedek'):
  path=ROOT/(case+'_'+s+'.csv');rows[s]=io.load_rows(path);hashes[s]=sha(path)
 y0,y1=rec['window'];span=min(768,y1-y0)
 starts=sorted(set([y0,(y0+y1-span)//2,y1-span]))
 cards=[]
 for i,start in enumerate(starts):
  end=start+span;rgb=io.read_rgb(log,'interpreted_lithology_column',start,end)
  canvas=Image.new('RGB',(1024,span+100),'white');d=ImageDraw.Draw(canvas)
  d.text((12,8),rec['log'],font=font,fill='black');d.text((12,30),f'Original pixels y={start}..{end}; truth unavailable',font=font,fill='black')
  crop=Image.fromarray(rgb);width=max(1,min(180,round(crop.width*span/crop.height)));crop=crop.resize((width,span))
  canvas.paste(crop,(12,80));d.text((12,58),'Source column',font=small,fill='black')
  for col,s in enumerate(('s19_combine','s24_benedek')):
   x=215+col*400;d.text((x,58),'S19 ours' if col==0 else 'Benedek d39919f',font=font,fill='black')
   hits=[r for r in rows[s] if r['top_y']<end and r['bottom_y']>start]
   label_y=80
   for r in hits:
    a=max(start,r['top_y']);b=min(end,r['bottom_y']);ya=80+a-start;yb=80+b-start
    colour=tuple(85+(v%135) for v in hashlib.sha256(r['lithology'].encode()).digest()[:3])
    d.rectangle((x,ya,x+34,max(ya,yb-1)),fill=colour,outline='black')
    ty=max(ya,label_y);label_y=ty+30
    if ty+28>span+80:continue
    text=r['lithology'][:43];depth=''
    if 'top_depth' in r and 'bottom_depth' in r:depth=f" depth {r['top_depth']}–{r['bottom_depth']}"
    d.line((x+35,ya,x+43,ty),fill='#555555')
    d.text((x+46,ty),text,font=small,fill='black');d.text((x+46,ty+14),f"y {r['top_y']}–{r['bottom_y']}"+depth,font=small,fill='#444444')
  path=IMG/f'{case}_{i+1}.png';canvas.save(path)
  manifest['images'].append({'case':case,'window':[start,end],'source_shape':list(rgb.shape),'output_size':list(canvas.size),'file':str(path.relative_to(OUT)),'hashes':hashes})
  encoded=base64.b64encode(path.read_bytes()).decode();cards.append(f'<img alt="Source column and two model predictions at rows {start} to {end}" src="data:image/png;base64,{encoded}">')
 log.close()
 sections.append('<section><h2>'+html.escape(rec['log'])+'</h2><p>Saved inference window '+str(rec['window'])+'. Three fixed excerpts below. Labels show complete predicted interval bounds. Short intervals can have labels displaced or omitted to prevent overlap.</p>'+''.join(cards)+'</section>')
manifest['completed_pairs']=len(sections);manifest['rendered_windows']=len(manifest['images']);manifest['rendered_pixels_per_source']=sum(x['window'][1]-x['window'][0] for x in manifest['images'])
(OUT/'visual_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
page='<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>New test predictions</title><style>body{font:16px Arial,sans-serif;margin:20px auto;max-width:1060px;padding:0 12px;color:#17212a;background:#fff}img{max-width:100%;height:auto;border:1px solid #bbb;margin:12px 0}p{line-height:1.5}section{border-top:2px solid #ccc;margin-top:24px}h1{font-size:28px}h2{font-size:20px}</style><h1>New Drive test: saved model predictions</h1><p><strong>Truth is unavailable. These images do not measure accuracy.</strong> Source pixels appear beside saved S19 and Benedek predictions. No inference, training, or tuning was done for this report. Colours distinguish names; they do not show correctness. An empty model column means that the saved CSV has no predicted interval in that excerpt. Depth values use the existing calibration and its original units.</p><p>'+str(len(sections))+' completed pairs; '+str(len(manifest['images']))+' source excerpts. All three pages have saved outputs for both models.</p>'+''.join(sections)+'</html>'
(OUT/'predictions_report.html').write_text(page,encoding='utf-8')
assert all(max(x['output_size'])<=1024 for x in manifest['images'])
print(f"PASS: {len(sections)} pairs, {len(manifest['images'])} annotated images, max edge 1024 px; HTML embeds all images")

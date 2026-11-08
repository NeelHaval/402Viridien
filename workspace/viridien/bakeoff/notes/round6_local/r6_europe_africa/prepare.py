from pathlib import Path
import re,csv,json,fitz
from PIL import Image,ImageDraw
Image.MAX_IMAGE_PIXELS=None
root=Path('C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work')
p=root/'bakeoff/notes/round6_local/r6_europe_africa/sample.py'
s=p.read_text(encoding='utf-8').replace('import os,json,time,requests,fitz','import os,json,time,re,requests,fitz');a=s.index('urls=list(');b=s.index('\n(out/',a)
s=s[:a]+'''urls=list(dict.fromkeys(u for u in re.findall(r'https://factpages\.sodir\.no/pbl/[^\s<>"\x27]+?\.[pP][dD][fF]',r.text) if 'completion_log' in u.lower() or 'mud_log' in u.lower()))'''+s[b:];p.write_text(s,encoding='utf-8')
rows=list(csv.DictReader((root/'analysis/label_shares_per_log.csv').open(encoding='utf-8')));ev={x['log'] for x in json.loads((root/'bakeoff/eval_set.json').read_text(encoding='utf-8'))}
seen=set();ims=[]
imgs=root.parent/'viridien_challenge/Viridien_HTB_challenge/hack27/project/dataset/images'
for r in rows:
 if r['log'] in ev or r['family'] in seen:continue
 seen.add(r['family']);print(r['family'],r['log'])
 if 'Gulf' in r['family'] or 'Geoservices' in r['family']:
  files=[f for f in imgs.iterdir() if f.stem==r['log'] or f.name==r['log']]
  if files:
   im=Image.open(files[0]);im.thumbnail((450,850));ims.append((r['family']+' (non-eval)',im.copy()))
for f in (root/'external_untrusted/round6/r6_europe_africa/sodir').glob('*preview.png'):
 im=Image.open(f);im.thumbnail((450,850));ims.append((f.stem,im.copy()))
canvas=Image.new('RGB',(450*len(ims),900),'white');draw=ImageDraw.Draw(canvas)
for j,(lab,im) in enumerate(ims):canvas.paste(im,(450*j,45));draw.text((450*j+5,5),lab[:65],fill='black')
canvas.save(root/'bakeoff/notes/round6_local/r6_europe_africa/contact_sheet.png')
canvas.thumbnail((1024,1024));canvas.save(root/'bakeoff/notes/round6_local/r6_europe_africa/contact_sheet_preview.png')

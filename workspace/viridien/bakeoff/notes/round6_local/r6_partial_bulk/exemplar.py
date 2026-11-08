from pathlib import Path
import json,csv,fitz
from PIL import Image,ImageDraw
N=Path('viridien_work/bakeoff/notes/round6_local/r6_partial_bulk');base=Path('viridien_challenge/Viridien_HTB_challenge/hack27/project/dataset/images')
ev={x['log'] for x in json.loads(Path('viridien_work/bakeoff/eval_set.json').read_text(encoding='utf-8'))}
names=[x['log'] for x in csv.DictReader(Path('viridien_work/analysis/label_shares_per_log.csv').open(encoding='utf-8')) if x['family']=='D-582 1-bit grid' and x['log'] not in ev]
p=next(p for p in base.iterdir() if p.stem in names)
Image.MAX_IMAGE_PIXELS=None
im=Image.open(p);im=im.crop((0,4000,im.width,13000)).convert('RGB');im.thumbnail((480,850));sheet=Image.new('RGB',(1024,1024),'white');dr=ImageDraw.Draw(sheet);sheet.paste(im,(10,85));dr.text((10,10),'Non-eval D-582 exemplar: y=4000..13000',fill='black')
sample=json.loads((N/'samples.json').read_text(encoding='utf-8'))[1];d=fitz.open(sample['path']);p=d[0];z=min(480/p.rect.width,850/p.rect.height);pix=p.get_pixmap(matrix=fitz.Matrix(z,z));im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);sheet.paste(im,(530,85));dr.text((530,10),'New Oregon mud log: score 2/3',fill='black');sheet.save(N/'contact_sheet.jpg');print('Contact sheet saved; exemplar',str(p))

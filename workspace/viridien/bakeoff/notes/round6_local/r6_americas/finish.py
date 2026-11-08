import pathlib,sys,json,csv,hashlib
import fitz
from PIL import Image,ImageDraw
root=pathlib.Path(sys.argv[1]);base=root/'external_untrusted/round6/r6_americas';notes=root/'bakeoff/notes/round6_local/r6_americas'
rows=json.loads((base/'sample_index.json').read_text(encoding='utf-8'))
evals={r['log'] for r in json.loads((root/'bakeoff/eval_set.json').read_text(encoding='utf-8'))}
with (root/'analysis/label_shares_per_log.csv').open(encoding='utf-8',newline='') as f:
 names=list(dict.fromkeys(r['log'] for r in csv.DictReader(f) if r['family']=='Gulf 3-panel' and r['log'] not in evals))
imgs=root.parent/'viridien_challenge/Viridien_HTB_challenge/hack27/project/dataset/images'
ex=next(p for n in names for p in imgs.glob(n+'.*'))
Image.MAX_IMAGE_PIXELS=None
im=Image.open(ex).convert('RGB');im=im.crop((0,0,im.width,min(im.height,im.width*2)));im.thumbnail((300,390))
sheet=Image.new('RGB',(1024,980),'white');draw=ImageDraw.Draw(sheet)
draw.text((15,8),'Gulf reference and public source samples. No full mud log found.',fill='black')
draw.text((15,34),'NON-eval Gulf reference',fill='black');sheet.paste(im,(15,65))
panels=[r for r in rows if r.get('pages') and r['source'] in ['mexico','trinidad','colombia']][:5]
for j,r in enumerate(panels,1):
 x=(j%3)*340+15;y=(j//3)*465+34
 d=fitz.open(root/r['path']);page=d[0];pix=page.get_pixmap(matrix=fitz.Matrix(0.7,0.7));p=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);p.thumbnail((310,380))
 draw.text((x,y),r['source']+' / '+pathlib.Path(r['path']).name[:31],fill='black');draw.text((x,y+17),'Score 0: inventory / report / access rules',fill='black');sheet.paste(p,(x,y+40));d.close()
sheet.save(notes/'contact_sheet.png')
# Review every page of the Mexican promotional presentation in one sheet.
r=next(r for r in rows if r.get('path','').endswith('shcp-sener-cnh.pdf'));d=fitz.open(root/r['path']);thumb=Image.new('RGB',(1000,1000),'white');dr=ImageDraw.Draw(thumb)
for i,p in enumerate(d):
 pix=p.get_pixmap(matrix=fitz.Matrix(.3,.3));im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.thumbnail((235,120));x=(i%4)*250;y=(i//4)*140;dr.text((x+5,y+2),f'Page {i+1}',fill='black');thumb.paste(im,(x+5,y+19))
thumb.save(notes/'mexico_presentation_sheet.png')
print('Reference:',ex.name,'excluded eval:',ex.stem in evals)
print('PDF documents:',sum(bool(r.get('pages')) for r in rows),'PDF pages:',sum(r.get('pages') or 0 for r in rows))
print('Sheets written; inspect before scoring.')

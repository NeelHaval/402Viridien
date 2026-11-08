from pathlib import Path
import json,fitz,hashlib,io,math,os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
Image.MAX_IMAGE_PIXELS=400000000
N=Path('viridien_work/bakeoff/notes/round6_local/r6_partial_bulk');O=Path('viridien_work/external_unlabelled/round6/r6_partial_bulk/images')
docs=json.loads((N/'review_index.json').read_text(encoding='utf-8'));decs=json.loads((N/'decisions.json').read_text(encoding='utf-8'))
def convert(item):
 i,x=item;dec=decs.get(x['url'],decs.get(str(i),{}))
 if dec.get('similarity',0)<2:return 0
 p=Path(x['path']);sha=hashlib.sha256(p.read_bytes()).hexdigest();count=0
 with fitz.open(p) as d:
  for pn in dec.get('accepted_pages',[1]):
   out=O/('r6_partial_'+sha[:12]+f'_p{pn:03}.png')
   if out.exists():continue
   page=d[pn-1];ims=page.get_images(full=True)
   covers='oregon' not in x['path'] and bool(ims) and any(rect.get_area()>.75*page.rect.get_area() for rect in page.get_image_rects(ims[0][0]))
   if len(ims)==1 and ('oregon' in x['path'] or covers):
    data=d.extract_image(ims[0][0]);im=Image.open(io.BytesIO(data['image']));im.load()
   else:
    z=min(200/72,1800/page.rect.width,math.sqrt(24000000/(page.rect.width*page.rect.height)));pix=page.get_pixmap(matrix=fitz.Matrix(z,z),alpha=False);im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
   if im.width*im.height>40000000:im.thumbnail((2000,20000))
   if im.mode not in ('1','L','LA','P','RGB','RGBA'):im=im.convert('RGB')
   temp=out.with_suffix('.tmp');im.save(temp,format='PNG',compress_level=1);os.replace(temp,out);count+=1
 return count
with ThreadPoolExecutor(max_workers=2) as pool:
 counts=list(pool.map(convert,reversed(list(enumerate(docs)))))
print('PRECONVERT COMPLETE',sum(counts),flush=True)

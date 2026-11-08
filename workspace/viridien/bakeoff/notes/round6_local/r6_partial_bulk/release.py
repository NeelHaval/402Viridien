from pathlib import Path
import json,csv,hashlib,fitz,io,math
from PIL import Image,ImageDraw
Image.MAX_IMAGE_PIXELS=400000000
N=Path('viridien_work/bakeoff/notes/round6_local/r6_partial_bulk');R=Path('viridien_work/external_untrusted/round6/r6_partial_bulk');O=Path('viridien_work/external_unlabelled/round6/r6_partial_bulk')
docs=json.loads((N/'review_index.json').read_text(encoding='utf-8'));decisions=json.loads((N/'decisions.json').read_text(encoding='utf-8'))
rows=[];shas=set();dups=[];total=0
raw_size=sum(z.stat().st_size for z in R.rglob('*') if z.is_file())
for i,x in enumerate(docs):
 dec=decisions.get(x['url'],decisions.get(str(i),{}));x.update(dec)
 if dec.get('similarity',0)<2:continue
 p=Path(x['path']);d=fitz.open(p);srcsha=hashlib.sha256(p.read_bytes()).hexdigest()
 for pn in dec.get('accepted_pages',[1]):
  page=d[pn-1];ident='r6_partial_'+srcsha[:12]+f'_p{pn:03}'
  cached=O/'images'/f'{ident}.png'
  if cached.exists():
   data=cached.read_bytes();im=Image.open(io.BytesIO(data))
  else:
   ims=page.get_images(full=True)
   image_covers_page='oregon' not in x['path'] and bool(ims) and any(rect.get_area()>.75*page.rect.get_area() for rect in page.get_image_rects(ims[0][0]))
   if len(ims)==1 and ('oregon' in x['path'] or image_covers_page):
    dat=d.extract_image(ims[0][0]);im=Image.open(io.BytesIO(dat['image']));im.load()
   else:
    z=min(200/72,1800/page.rect.width,math.sqrt(24000000/(page.rect.width*page.rect.height)));pix=page.get_pixmap(matrix=fitz.Matrix(z,z),alpha=False);im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
   if im.width*im.height>40000000:im.thumbnail((2000,20000))
   if im.mode not in ('1','L','LA','P','RGB','RGBA'):im=im.convert('RGB')
   buf=io.BytesIO();im.save(buf,format='PNG',compress_level=1);data=buf.getvalue()
  sha=hashlib.sha256(data).hexdigest()
  if sha in shas:dups.append({'url':x['url'],'page':pn,'sha256':sha});continue
  if total+len(data)+raw_size>1950000000:raise RuntimeError('Combined lane budget exceeded')
  shas.add(sha);(O/'images'/f'{ident}.png').write_bytes(data);total+=len(data)
  rows.append(dict(id=ident,source='r6_partial_'+('oregon' if 'oregon' in x['url'] else 'nwt' if 'orogo' in x['url'] else 'gdr'),well=x['url'].split('/')[-2],page=pn,source_url=x['url'],family=x.get('family_match','colour_grid'),layout_family_guess=x.get('family_match','colour_grid'),has_legend=x.get('has_legend'),legend_source_page='',similarity=x['similarity'],column_semantics='mud_log_lithology',site_group=x['url'].split('/')[-2],api_as_printed='',source_sha256=srcsha,image_sha256=sha,width=im.width,height=im.height,window_y0=0,window_y1=im.height,truth_status='absent',notes='Visually reviewed. Full page; no rock labels or boxes. Public access; reuse terms unverified.'))
 if 'pages' not in x:x['pages']=len(d)
 x['bytes']=p.stat().st_size
 x.setdefault('raster_or_vector','mixed');x.setdefault('notes','Visual comparison completed.')
fields=['id','source','well','page','source_url','family','layout_family_guess','has_legend','legend_source_page','similarity','column_semantics','site_group','api_as_printed','source_sha256','image_sha256','width','height','window_y0','window_y1','truth_status','notes']
with (O/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
sources=[]
for key,name in [('oregon','Oregon DOGAMI'),('nwt','NWT OROGO'),('gdr','OpenEI GDR'),('alaska','Alaska DGGS')]:
 ds=[x for x in docs if key in x['path']]
 sources.append(dict(name=name,urls_seen=[x['url'] for x in ds],bulk_route='Public index links; exact URLs retained in notes.',n_available_estimate=len(ds),access='Public; no login or forms submitted.',license_terms='Public access does not establish reuse rights; file-specific license not verified.',documents=ds))
events=[]
for f in R.glob('*events.json'):events.extend(json.loads(f.read_text(encoding='utf-8')))
triage=dict(family='partial_bulk: bw_hatch, colour_grid, geoservices_grid',sources=sources,dead_ends=events)
(R/'triage.json').write_text(json.dumps(triage,indent=2),encoding='utf-8')
(O/'deduplication.json').write_text(json.dumps(dups,indent=2),encoding='utf-8')
summary=dict(documents_downloaded=len(docs),documents_released=len(set(r['source_sha256'] for r in rows)),pages_released=len(rows),labelled_windows=0,best_similarity=max((r['similarity'] for r in rows),default=0),output_bytes=total,raw_bytes=sum(p.stat().st_size for p in R.rglob('*') if p.is_file()))
(N/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))

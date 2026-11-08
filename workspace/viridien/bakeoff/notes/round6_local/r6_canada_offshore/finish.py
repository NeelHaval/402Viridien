import os,sys,json,csv,hashlib
os.environ['OMP_NUM_THREADS']='3'
from pathlib import Path
import fitz
from PIL import Image,ImageDraw
base=Path(sys.argv[1]);lane='r6_canada_offshore';raw=base/'external_untrusted/round6'/lane;notes=base/'bakeoff/notes/round6_local'/lane;out=base/'external_unlabelled/round6'/lane
for d in [out/'images',out/'bbox_annotation']:d.mkdir(parents=True,exist_ok=True)
items=json.loads((raw/'cnlopb/downloads.json').read_text(encoding='utf-8'))+json.loads((raw/'cnlopb/composite_downloads.json').read_text(encoding='utf-8'))
docs=[];seen=set();best=None
for x in items:
 f=Path(x['path']).resolve();sha=hashlib.sha256(f.read_bytes()).hexdigest()
 if sha in seen:continue
 seen.add(sha);d=fitz.open(f);use='ee44b036' in f.name
 if use:best=(x,f,sha)
 docs.append(dict(url=x['url'],path=str(f.relative_to(base)),bytes=f.stat().st_size,pages=len(d),raster_or_vector='vector text and graphics' if use else 'mixed PDF; not fully classified',has_lith_column=bool(use),has_legend=bool(use),similarity=2 if use else 0,family_match='Gulf of Mexico 3-panel deepwater mud/master logs (column style only)' if use else None,notes=x['title']+ ('; Datalog Canada Ltd; grey hatch interpreted and percentage columns. Different layout.' if use else '; summary report or drilling engineering composite. Not accepted for conversion.')))
x,f,sha=best;d=fitz.open(f);manifest=[]
fields=['id','source','well','page','source_url','family','layout_family_guess','has_legend','legend_source_page','similarity','column_semantics','site_group','api_as_printed','source_sha256','image_sha256','width','height','window_y0','window_y1','truth_status','notes']
for i in range(1,len(d)):
 pg=d[i];key='r6_canada_mizzen_l11_p%03d'%(i+1);pix=pg.get_pixmap(matrix=fitz.Matrix(200/72,200/72),alpha=False);dest=out/'images'/(key+'.png');pix.save(dest);w,h=pix.width,pix.height
 boxes={}
 if i>=2:
  boxes={'interpreted_lithology_column':[round(w*.248),0,round(w*.324),h],'percentage_lithology_column':[round(w*.708),0,round(w*.809),h],'depth_column':[round(w*.207),0,round(w*.248),h]}
 if boxes:
  (out/'bbox_annotation'/(key+'.json')).write_text(json.dumps(dict(image=dest.name,image_size=[w,h],bounding_boxes=boxes,source_url=x['url'],source_sha256=sha,source_page=i+1,source_key='r6_canada_mizzen_l11',well='Mizzen L-11',family='datalog_canada_mudlog',column_semantics='cuttings_mud_log',truth_status='absent',reviewed_similarity=2,notes='Column bounds from visual review. No interval labels. Empty shallow intervals retained.'),indent=2),encoding='utf-8')
 manifest.append(dict(zip(fields,[key,lane,'Mizzen L-11',i+1,x['url'],'datalog_canada_mudlog','Gulf column-style analogue',True,2,2,'cuttings_mud_log','Mizzen L-11','',sha,hashlib.sha256(dest.read_bytes()).hexdigest(),w,h,0,h,'absent','Full PDF page at 200 dpi. Page 2 contains legend. No labelled intervals.'])))
with (out/'manifest.csv').open('w',encoding='utf-8',newline='') as stream:
 wr=csv.DictWriter(stream,fieldnames=fields);wr.writeheader();wr.writerows(manifest)
triage=dict(family='Gulf of Mexico 3-panel deepwater mud/master logs',sources=[dict(name='C-NLOPB / C-NLOER public ArcGIS hub',urls_seen=['https://home-cnlopb.hub.arcgis.com/'],bulk_route='Hub site JSON -> well Hub Page JSON -> inventory table View/Download item data URL. candidate_rows.json retains discovered links.',n_available_estimate=len(json.loads((raw/'cnlopb/candidate_rows.json').read_text(encoding='utf-8'))),access='No login required for sampled PDF files',license_terms='Public access verified. No explicit reuse licence verified. Some reports show contractor confidentiality marks; retain for review.',documents=docs),dict(name='CNSOPB / CNSOER DMC',urls_seen=['https://cnsopbdigitaldata.ca/geoviewer/dmc','https://cnsopbdigitaldata.ca/geoviewer/dmc/public/query/1/getData'],bulk_route=None,n_available_estimate=235,access='Public inventory: 235 wells. Main DMC has login. Public inventory has counts but no direct document links in returned records.',license_terms='CNSOER copyright; reuse not verified',documents=[]),dict(name='Canada Energy Regulator Frontier Information Office',urls_seen=['https://www.cer-rec.gc.ca/en/about/library/north-offshore-data.html','https://rec-cer.gc.ca/en/about/north-offshore/access-information.html'],bulk_route=None,n_available_estimate=None,access='Library and request route; no direct full mud-log download found',license_terms='Not checked for individual documents',documents=[])],dead_ends=['CNSOER public inventory lists Cheshire D404 (133 linked items), Monterey Jack D405 (180), Aspy D406 (127), but no unauthenticated document route was established. No account created.', 'CER frontier site points to library/request access. No email sent.', 'Most Mudloggers Reports omit full log attachments. Titles alone do not establish useful logs.', 'Cappahayden K-67 and K-67Z pages link the same PDF; one unique document counted. Duplicate retained as downloaded.'])
(raw/'triage.json').write_text(json.dumps(triage,indent=2),encoding='utf-8')
total=sum(q['pages'] for q in docs)
report=f'''# Canada offshore search

Best result: Mizzen L-11, Datalog Canada Ltd. Similarity 2 of 3 for the Gulf family by column style. It has a full mud log and a colour symbol legend. Its layout differs from the Gulf 3-panel layout. No score-3 match found. No confirmed match for the other six families.

## Counts

- 17 PDF downloads; 16 unique documents; {total} unique source pages.
- 15-document first sample, followed by 2 drilling composite files.
- 54 converted pages from one full mud log, including its legend.
- 53 pages have column boxes. Labelled windows: 0. These are unlabelled data.
- 18 candidate inventory entries found across 17 public well pages. This is a bounded sample, not the full hub count.

## Ranked routes

1. [C-NLOER public hub](https://home-cnlopb.hub.arcgis.com/): direct public PDFs. One full log accepted, 55 source pages. Use the Mizzen L-11 Mud Log entry. The adjacent Mudloggers Report is a summary. Geoservices reports for Bay du Nord, Baccalieu and Cappahayden and Weatherford reports for Mizzen are mainly summaries. The sampled drilling composites show engineering curves; they were not accepted.
2. [CNSOER public inventory](https://cnsopbdigitaldata.ca/geoviewer/dmc/public/query/1): 235 well records obtained through the page's getData endpoint. Cheshire, Monterey Jack and Aspy are present. Their linked-item counts are 133, 180 and 127. Counts do not prove that full mud logs are available without login. The [main DMC](https://cnsopbdigitaldata.ca/geoviewer/dmc) has a login form. No account was used.
3. [CER Frontier Information Office](https://www.cer-rec.gc.ca/en/about/library/north-offshore-data.html): records are held by the library. No direct full mud-log download found. The [access page](https://rec-cer.gc.ca/en/about/north-offshore/access-information.html) explains the request route. No request was sent.

## Files and checks

Sources: external_untrusted/round6/{lane}/. Converted data: external_unlabelled/round6/{lane}/. Scripts and contact sheet: bakeoff/notes/round6_local/{lane}/.

Downloads used the required User-Agent, no email, no account, and pauses of at least 1.1 seconds. Source files were read by isolated Python. No outside test data or eval images were used. The Gulf exemplar is recorded in exemplar.json and is outside the eval list. No training was run.

Public download access does not establish a reuse licence. Individual report reuse terms need review. Full-page rendering preserves the source layout. Column boxes are approximate visual bounds; no interval truth was added.
'''
(notes/'REPORT.md').write_text(report,encoding='utf-8')
# Compare a useful body crop from the non-eval exemplar to the candidate.
ex=json.loads((notes/'exemplar.json').read_text(encoding='utf-8'));Image.MAX_IMAGE_PIXELS=None;im=Image.open(ex['path']);w,h=im.size;im=im.crop((0,min(6000,h//3),w,min(6000,h//3)+min(6500,h//2)));im.thumbnail((490,820))
sheet=Image.new('RGB',(1500,900),'white');dr=ImageDraw.Draw(sheet);dr.text((8,8),'Non-eval Gulf exemplar',fill='black');sheet.paste(im,(0,40))
for col,idx,title in [(1,1,'Mizzen L-11 legend - score 2'),(2,24,'Mizzen L-11 body - score 2')]:
 pg=d[idx];sc=min(490/pg.rect.width,820/pg.rect.height);pix=pg.get_pixmap(matrix=fitz.Matrix(sc,sc),alpha=False);sheet.paste(Image.frombytes('RGB',[pix.width,pix.height],pix.samples),(col*500,40));dr.text((col*500+8,8),title,fill='black')
sheet.save(notes/'contact_sheet.png');sheet.thumbnail((1024,1024));sheet.save(notes/'contact_sheet_small.png')
print(f'COMPLETE: {len(docs)} unique documents; {total} source pages; {len(manifest)} converted pages; 53 column boxes; 0 labelled windows')

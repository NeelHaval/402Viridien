from pathlib import Path
import json,csv,hashlib,fitz
from PIL import Image,ImageDraw
Image.MAX_IMAGE_PIXELS=None
r=Path('C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work');n=r/'bakeoff/notes/round6_local/r6_europe_africa';u=r/'external_untrusted/round6/r6_europe_africa';out=r/'external_unlabelled/round6/r6_europe_africa';(out/'images').mkdir(parents=True,exist_ok=True);(out/'bbox_annotation').mkdir(exist_ok=True)
records=[]
for fn in ['download_records.json','target_records.json','more_records.json']:records+=json.loads((u/'sodir'/fn).read_text(encoding='utf-8'))
records=list({x['url']:x for x in records}.values())
for rec in records:
 p=r/rec['path'];d=fitz.open(p);best='5093_' in p.name
 rec.update(raster_or_vector='vector/text with images' if d[0].get_text().strip() else 'raster scan',has_lith_column=best or 'COMPLETION_LOG' in p.name.upper(),has_legend=best or '4910_' in p.name,similarity=2 if best else 1,family_match='Geoservices grid' if best else 'different completion/mud-log layout',notes='Pages 109-112 have coloured rock intervals; page 106 supplies the legend. Pages 113-116 are pressure logs, excluded.' if best else 'Screened as a weaker route; no pages converted. Report titles can mention enclosures that are absent.');rec.pop('sizes',None)
sources=[dict(name='Norway Sodir FactPages',urls_seen=['https://factpages.sodir.no/en/wellbore/TableView/With/Documents'],bulk_route='Extract published PDF addresses from the public document table. The first 300 table rows supplied 37 completion/mud-log URLs; export is needed for the full national list.',n_available_estimate=37,access='Direct HTTPS PDFs; no login used.',license_terms='Public release after secrecy expires. No explicit reuse licence was verified. Public access does not establish unrestricted reuse rights.',documents=records)]
others=[('Denmark GEUS',['https://eng.geus.dk/about/news/news-archive/2022/june/free-data','https://frisbee.geus.dk/'],'GEUS states that released well packages, including logs, are free from July 2022. Local request outcome is in route_checks.json; no log files acquired.'),('Netherlands NLOG',['https://www.nlog.nl/en/data','https://www.nlog.nl/datacenter/brh-overview'],'Public borehole portal found. No new well download verified in this lane. No existing or excluded wells added.'),('UK NSTA',['https://www.nstauthority.co.uk/data-and-insights/about-nsta-data/how-to-access-our-data/','https://www.nstauthority.co.uk/data-and-insights/data/themes/wells/'],'Open GIS and stratigraphy downloads found; no no-login full mud-log release verified outside NDR.'),('Ireland PAD / DCEE',['https://www.gov.ie/en/department-of-climate-energy-and-the-environment/publications/oil-and-gas-exploration-production-data/'],'Official well-data page requires a contact request. No request sent.'),('South Africa PASA',['https://www.petroleumagencysa.com/viewing-our-technical-data/'],'Data ordering requires entity registration and issued login. Skipped.'),('Namibia NAMCOR',['https://www.namcor.com.na/technical-data/'],'Mud logging reports and scanned logs are listed, but access is by contact/purchase. Skipped.'),('Morocco ONHYM',['https://www.onhym.com/sites/default/files/rapport_onhym_vf2020.pdf'],'Official annual report mentions organised data rooms; no direct public full-log download verified.')]
for name,urls,note in others:sources.append(dict(name=name,urls_seen=urls,bulk_route=note,n_available_estimate=None,access='No qualifying no-login PDF downloaded',license_terms='Not verified',documents=[]))
triage=dict(family='Gulf 3-panel and Geoservices grid',sources=sources,dead_ends=[x[2] for x in others]+['4423/34-7-32 lists log enclosures on its final page but does not include them.','4939/6608-11-4 lists a Geoservices enclosure, but inspected enclosure pages are sample descriptions.','No similarity-3 Gulf three-panel match found. No deepwater match confirmed; 7131/4-1 states 331 m water depth.'])
(u/'triage.json').write_text(json.dumps(triage,indent=2),encoding='utf-8')
# Convert only visually reviewed formation-evaluation pages with coloured rock intervals.
p=u/'sodir/5093_7131_4_1_COMPLETION_GEOLOGICAL_REPORT.PDF';d=fitz.open(p);sha=hashlib.sha256(p.read_bytes()).hexdigest();manifest=[]
for page in [109,110,111,112]:
 pg=d[page-1];ident=f'r6_europe_africa_7131_4_1_p{page:03}';f=out/'images'/f'{ident}.png';px=pg.get_pixmap(matrix=fitz.Matrix(3,3),alpha=False);px.save(f)
 manifest.append(dict(id=ident,source='sodir',well='7131/4-1',page=page,source_url=next(x['url'] for x in records if '5093_' in x['url']),family='Geoservices grid',layout_family_guess='Geoservices grid',has_legend=True,legend_source_page=106,similarity=2,column_semantics='cuttings percentage and interpreted lithology',site_group='7131/4-1',api_as_printed='',source_sha256=sha,image_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),width=px.width,height=px.height,window_y0=0,window_y1=px.height,truth_status='absent',notes='Full page. No rock labels or bounding boxes supplied. Legend retained in source PDF page 106.'))
with (out/'manifest.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=manifest[0].keys());w.writeheader();w.writerows(manifest)
# Flat comparison sheet, with non-evaluation source exemplars and all screened documents.
tiles=[];ev={x['log'] for x in json.loads((r/'bakeoff/eval_set.json').read_text(encoding='utf-8'))};rows=list(csv.DictReader((r/'analysis/label_shares_per_log.csv').open(encoding='utf-8')));seen=set();imgs=r.parent/'viridien_challenge/Viridien_HTB_challenge/hack27/project/dataset/images'
for row in rows:
 fam=row['family']
 if fam not in ['Gulf 3-panel','Geoservices grid'] or fam in seen or row['log'] in ev:continue
 ff=next((x for x in imgs.iterdir() if x.stem==row['log'] or x.name==row['log']),None)
 if ff:
  im=Image.open(ff);y=int(im.height*.35);im=im.crop((0,y,im.width,min(im.height,y+im.width*1.35)));im.thumbnail((300,400));tiles.append((fam+' non-eval exemplar',im.copy()));seen.add(fam)
for rec in records:
 f=r/rec['path'];doc=fitz.open(f);pi=109-1 if '5093_' in f.name else (1 if '4910_' in f.name else 0);pg=doc[pi];rect=pg.rect
 if rect.height>rect.width*3:clip=fitz.Rect(0,rect.height*.3,rect.width,min(rect.height,rect.height*.3+rect.width*1.35))
 else:clip=rect
 z=min(300/clip.width,400/clip.height);px=pg.get_pixmap(matrix=fitz.Matrix(z,z),clip=clip);im=Image.frombytes('RGB',(px.width,px.height),px.samples);tiles.append((f.name[:35]+f' score {rec["similarity"]}',im))
can=Image.new('RGB',(1500,450*((len(tiles)+4)//5)),'white');dr=ImageDraw.Draw(can)
for i,(lab,im) in enumerate(tiles):x=i%5*300;y=i//5*450;can.paste(im,(x,y+35));dr.text((x+3,y+3),lab[:46],fill='black')
can.save(n/'contact_sheet.png');can.thumbnail((1024,1024));can.save(n/'contact_sheet_preview.png')
summary=dict(documents=len(records),pdf_pages=sum(x['pages'] for x in records),converted_pages=len(manifest),labelled_windows=0,best_similarity=2,bytes=sum(x['bytes'] for x in records))
(n/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))

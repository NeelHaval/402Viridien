import json, sys
from pathlib import Path
import fitz
from PIL import Image,ImageDraw
root=Path(sys.argv[1]); out=root/'external_untrusted/round4/bw_hatch'; dest=out/'oregon_dogami'
meta=json.loads((dest/'metadata.json').read_text()); docs=[]
for item in meta['documents']:
 p=root/item['path']; d=fitz.open(p); page=d[0]; scale=1800/page.rect.height
 page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).save(str(dest/(p.stem+'_full.png')))
 stem=p.stem; exceptional=stem.startswith('CC12'); coos=stem.startswith('Coos'); info=(1894,13198,1,1200) if exceptional else (*item['image_info'][0][:3],220 if coos else 200)
 info_note=f'Embedded image {info[0]} x {info[1]} pixels, {info[2]} bit, {info[3]} ppi as placed in PDF.'
 if exceptional: info_note+=' 1200 ppi is PDF placement metadata, not verified optical scan resolution; log is a narrow image on a letter page.'
 contractor='Epoch Well Logging' if exceptional else ('Exploration Logging' if coos else 'Western Geo-Engineers')
 docs.append({k:item[k] for k in ('url','path','bytes','pages','raster_or_vector','producer')} | {'has_lith_column':True,'has_legend':True,'similarity':2,'notes':f'{contractor}; scanned mud log with patterned sample lithology and printed lithology key, different track ordering and header from INTEQ. {info_note}'})
 d.close()
sheet=Image.new('RGB',(1800,900*((len(docs)+2)//3)),'white'); draw=ImageDraw.Draw(sheet)
for row in range((len(docs)+2)//3):
 y=row*900; draw.text((6,y+5),'D-582 exemplar: header and lithology slice',fill='grey')
 for fn,top,box in [('01_header.png',y+25,(430,330)),('01_slice_mid.png',y+370,(430,510))]:
  im=Image.open(out/'_exemplars'/fn).convert('RGB'); im.thumbnail(box);sheet.paste(im,(5,top))
 for col,doc in enumerate(docs[row*3:row*3+3]):
  x=450+col*450; stem=Path(doc['path']).stem; draw.text((x,y+5),stem+' | similarity 2',fill='grey')
  fn=dest/('CC12_detail.png' if stem.startswith('CC12') else stem+'_p1.png'); im=Image.open(fn).convert('RGB'); im.thumbnail((440,870));sheet.paste(im,(x,y+25))
sheet.save(out/'_sheets/oregon_dogami.png')
triage={'family':'bw_hatch','exemplar_findings':{'contractors':['Baker Hughes INTEQ (D-582)','Weatherford (MC761)'],'software':[],'scales':['D-582 header: formation evaluation 1 in/100 ft (1:1200) and 5 in/100 ft (1:240); gas ratio 1:240','MC761: 1:1200 measured depth or true vertical depth'],'layout':'D-582: 400 dpi 1-bit scans; three-panel header with lithology-symbol swatches; gridded curve tracks, depth and a narrow interpreted hatch column next to descriptions. Chromatograph pages share the header but differ in track content. MC761: Weatherford colour mineral and lithology keys; 150 dpi RGB or 203 dpi palette, percentage lithology near depth, gas curves, interpreted lithology and comments. No software name was legible in inspected headers.'},'sources':[{'name':'Oregon DOGAMI oil and gas well log archive','urls_seen':['https://www.oregon.gov/dogami/mlrr/Pages/oilgas-logs.aspx']+[x['url'] for x in docs],'bulk_route':'Open public index, select mud links per well; direct individual PDF downloads. 319 unique mud-path links observed, including these 8 and 40 sampled in an earlier round. Count is links, not verified unique wells or suitable logs.','n_available_estimate':'311 other indexed mud links beyond these eight; 271 beyond these eight and the 40 previously sampled; accessibility and visual suitability of the remaining links not individually verified.','access':'Public direct downloads; no account observed.','license_terms':'Index states data distributed without warranty and should not be used for site-specific studies; no explicit reuse license identified. Check rights for individual operator-submitted scans.','documents':docs}], 'dead_ends':['OpenEI Coso 52B-7 direct PDF https://gdr.openei.org/files/758/52B-7%20Lithology%20Log.pdf returned HTTP 429 on this attempt; stopped after rate limit, no sample downloaded.','Alaska DGGS https://dggs.alaska.gov/gmc/seismic-order.html provides a seismic/well data request form and download/purchase table; no suitable immediate mud-log file was verified. This does not establish whether AOGCC has another public route.','Search-engine responses for targeted contractor and archive queries were irrelevant or bot gated; Kansas KGS https://www.kgs.ku.edu/PRS/petroDB.html links scanned wireline logs and well searches, but an older patterned mud-log download was not verified.']}
(out/'triage.json').write_text(json.dumps(triage,indent=2)+'\n')
print('documents',len(docs),'bytes',sum(x['bytes'] for x in docs),'exemplar crops',len(list((out/'_exemplars').glob('*.png'))))

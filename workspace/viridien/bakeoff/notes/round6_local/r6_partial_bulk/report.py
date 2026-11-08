from pathlib import Path
import json,csv,collections,fitz
from urllib.parse import unquote
N=Path('viridien_work/bakeoff/notes/round6_local/r6_partial_bulk');R=Path('viridien_work/external_untrusted/round6/r6_partial_bulk');O=Path('viridien_work/external_unlabelled/round6/r6_partial_bulk')
s=json.loads((N/'summary.json').read_text(encoding='utf-8'));tri=json.loads((R/'triage.json').read_text(encoding='utf-8'));rows=list(csv.DictReader((O/'manifest.csv').open(encoding='utf-8')))
decisions=json.loads((N/'decisions.json').read_text(encoding='utf-8'))
well_pages={x['url']:x.get('well_url','') for x in json.loads((N/'survey_candidates.json').read_text(encoding='utf-8'))}
for row in rows:
 if row['source']=='r6_partial_nwt':
  row['well']=well_pages[row['source_url']].rstrip('/').split('/')[-1]
  row['site_group']=row['well']
 if row['source']=='r6_partial_gdr':row['well']='Wister 85-20';row['site_group']='Wister 85-20'
 if row['has_legend']=='True':
  row['legend_source_page']='2' if 'loon' in row['well'].lower() else '1'
  row['notes']+=' Legend reference is the source PDF page; it may not be included among the converted pages.'
with (O/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
routes={'Oregon DOGAMI':('https://www.oregon.gov/dogami/mlrr/Pages/oilgas-logs.aspx','272 unseen indexed mud-PDF URLs; 15 sampled before bulk collection.'),'NWT OROGO':('https://www.orogo.gov.nt.ca/en/well-data','35 recent well pages checked from a 100-record index. 24 candidate links; 20 downloaded. This is not a survey of all 664 earlier unexamined records.'),'OpenEI GDR':('https://gdr.openei.org/submissions/904','Second Wister 85-20 continuous log. Four further submission pages checked; no matching direct PDF downloaded from them.'),'Alaska DGGS':('https://dggs.alaska.gov/gmc/seismic-order.html','Public page uses request/order forms. No form submitted. No direct public mud-log PDF found in this check.')}
for src in tri['sources']:
 u,route=routes[src['name']];src['urls_seen'].insert(0,u);src['bulk_route']=route
 if src['name']=='NWT OROGO':src['n_available_estimate']='Unknown matching volume; 35 well pages checked, 24 candidate URLs.'
 if src['name']=='Oregon DOGAMI':src['n_available_estimate']=272
 if src['name']=='Alaska DGGS':src['n_available_estimate']='Unknown; no direct public mud-log file found in this check.'
 for d in src['documents']:
  if 'pages' not in d:
   with fitz.open(d['path']) as pdf:d['pages']=len(pdf)
  d.setdefault('similarity',None);d.setdefault('has_lith_column',None);d.setdefault('has_legend',None);d.setdefault('family_match','unscored');d.setdefault('raster_or_vector','not checked');d.setdefault('bytes',Path(d['path']).stat().st_size)
tri['dead_ends'].append('Alaska request forms skipped. NWT wireline density/neutron logs are not lithology mud logs. Sparse striplogs retained raw only. No 3/3 match established.')
(R/'triage.json').write_text(json.dumps(tri,indent=2),encoding='utf-8')
counts=collections.Counter(r['source'] for r in rows)
families=collections.Counter(r['family'] for r in rows)
text=f'''# r6_partial_bulk results

Best similarity: 2/3. {s['documents_downloaded']} PDF documents were downloaded. {s['documents_released']} source documents supplied {s['pages_released']} unique converted pages. Labelled windows: 0. No rock labels or boxes were inferred.

## Ranked public routes

1. Oregon DOGAMI: patterned black-and-white lithology columns, score 2. Highest usable volume. {counts['r6_partial_oregon']} pages converted. Layout differs from D-582 and MC761. The 272 URLs were absent from both existing manifests. File and image duplicates were checked within the new collection.
2. NWT OROGO: colour/pattern strips, score 2; layout differs from MC765. {counts['r6_partial_nwt']} pages converted. Only the first ten pages of each downloaded file were visually checked. Selected pages contain populated lithology tracks. Header legends remain in the source PDFs. Recent registry records were prioritised; the full registry remains unchecked.
3. OpenEI GDR: Wister 85-20 continuous log, score 2. {counts['r6_partial_gdr']} page converted. Similar column style; not a Geoservices contractor match. Other checked submissions gave no new matching direct PDF.
4. Alaska DGGS: no new direct public mud log found. Request forms were skipped.

## Files and checks

- Raw files and triage: external_untrusted/round6/r6_partial_bulk/
- Converted images and manifest: external_unlabelled/round6/r6_partial_bulk/
- Scripts, labelled review sheets, decisions and comparison: bakeoff/notes/round6_local/r6_partial_bulk/
- contact_sheet.jpg places a non-eval D-582 crop beside a new Oregon candidate. review_*.jpg and survey_review_*.jpg retain the visual checks.
- bbox_annotation/ is empty. Column limits were not traced. Pages are unlabelled.
- See validation.json for the final hash, dimensions, duplicate, exclusion and size checks.
- Source links were taken from public indexes. No login, signup, email or order request was sent. Reuse rights are not established by public access.
- No training, tuning, commit or push was done.

Raw bytes: {s['raw_bytes']}. Converted PNG bytes: {s['output_bytes']}. The final check includes notes and review sheets in the 2 GB lane limit.

## Access details

'''
text+='\nConverted pages by family: '+str(dict(families))+'\n\nOregon review covered every first page and all later pages in its six multi-page files. Three Geolog files supplied additional colour/pattern pages. Five empty Oregon downloads remained empty on one retry.\n\n'
for name,(u,route) in routes.items():text+=f'- {name}: {u} — {route}\n'
for p in [N/'REPORT.md',R/'REPORT.md']:p.write_text(text,encoding='utf-8')
print('REPORT written',dict(counts))

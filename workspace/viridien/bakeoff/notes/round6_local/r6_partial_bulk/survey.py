from pathlib import Path
exec(Path('viridien_work/bakeoff/notes/round6_local/r6_partial_bulk/common.py').read_text(encoding='utf-8'))
idx=json.loads((NOTE/'index_links.json').read_text(encoding='utf-8'));discovered=[]
u='https://www.orogo.gov.nt.ca/en/well-data?items_per_page=100&order=field_well_id&sort=desc'
p=fetch(u,'nwt','recent_index.html'); ll=links(p,u) if p else []
wells=list(dict.fromkeys(u for u,t in ll if '/en/well-data/' in u and not excluded(u)))
print('NWT wells',len(wells),flush=True)
for i,u in enumerate(wells[:35]):
 p=fetch(u,'nwt','well_'+str(i)+'.html')
 if p:
  for v,t in links(p,u):
   if re.search(r'strip.?log|mud.?log|litholog',unquote(v)+' '+t,re.I) and '.pdf' in v.lower() and unquote(v).lower() not in seen and not excluded(v):discovered.append({'source':'nwt','url':v,'title':t,'well_url':u})
 print('NWT',i,len(discovered),flush=True)
 (NOTE/'survey_candidates.json').write_text(json.dumps(discovered,indent=2),encoding='utf-8')
for source,u in [('alaska','https://dggs.alaska.gov/gmc/seismic-order.html'),('gdr','https://gdr.openei.org/search?q=mud')]:
 p=fetch(u,source,'detail.html')
 if p:
  ll=links(p,u);(NOTE/(source+'_detail_links.json')).write_text(json.dumps(ll,indent=2),encoding='utf-8')
  if source=='alaska':print(BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser').get_text(' ',strip=True)[-11000:],flush=True)
for u,t in idx['gdr']:
 if '85-20 Final Continuous.pdf' in u and unquote(u).lower() not in seen:discovered.append({'source':'gdr','url':u,'title':t})
for i,item in enumerate(list({x['url']:x for x in discovered}.values())[:20]):
 p=fetch(item['url'],item['source'],hashlib.sha256(item['url'].encode()).hexdigest()[:12]+'.pdf')
 if p:item['path']=str(p)
(NOTE/'survey_candidates.json').write_text(json.dumps(discovered,indent=2),encoding='utf-8')
(RAW/'survey_events.json').write_text(json.dumps(events,indent=2),encoding='utf-8');print('SURVEY COMPLETE',flush=True)

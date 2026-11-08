from pathlib import Path
exec(Path('viridien_work/bakeoff/notes/round6_local/r6_partial_bulk/common.py').read_text(encoding='utf-8'))
ll=json.loads((NOTE/'gdr_detail_links.json').read_text(encoding='utf-8'));docs=[]
for u,t in ll:
 if not any('/submissions/'+n==urlparse(u).path for n in ['1736','1737','1751','1292']):continue
 p=fetch(u,'gdr',urlparse(u).path.split('/')[-1]+'.html')
 if p:
  for v,title in links(p,u):
   if '/files/' in v and '.pdf' in v.lower() and re.search(r'mud|lith|composite',unquote(v),re.I) and unquote(v).lower() not in seen:
    q=fetch(v,'gdr',hashlib.sha256(v.encode()).hexdigest()[:12]+'.pdf')
    if q:docs.append({'source':'gdr','url':v,'path':str(q),'title':title})
  print(u,len(docs),flush=True)
u='https://gdr.openei.org/submissions/904';p=RAW/'gdr/index.html'
for v,t in links(p,u):
 if '85-20 Final Continuous.pdf' in v and unquote(v).lower() not in seen:
  q=fetch(v,'gdr',hashlib.sha256(v.encode()).hexdigest()[:12]+'.pdf')
  if q:docs.append({'source':'gdr','url':v,'path':str(q),'title':t})
(NOTE/'gdr_candidates.json').write_text(json.dumps(docs,indent=2),encoding='utf-8');(RAW/'gdr_events.json').write_text(json.dumps(events,indent=2),encoding='utf-8');print('GDR COMPLETE',len(docs),flush=True)

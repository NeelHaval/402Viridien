from pathlib import Path
exec(Path('viridien_work/bakeoff/notes/round6_local/r6_partial_bulk/common.py').read_text(encoding='utf-8'))
queue=json.loads((NOTE/'oregon_queue.json').read_text(encoding='utf-8'))
records=[]
for i,u in enumerate(queue):
 p=fetch(u,'oregon',hashlib.sha256(u.encode()).hexdigest()[:12]+'.pdf')
 if p:
  try:
   d=fitz.open(p);records.append({'url':u,'path':str(p),'bytes':p.stat().st_size,'pages':len(d),'raster_or_vector':'raster' if d[0].get_images() else 'vector','has_lith_column':None,'has_legend':None,'similarity':None,'family_match':'bw_hatch','notes':'Awaiting visual review; not released.'})
  except Exception as e:events.append({'url':u,'error':str(e)})
 if i%10==0:print(i+1,len(records),flush=True)
 (RAW/'oregon_documents.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
(RAW/'bulk_events.json').write_text(json.dumps(events,indent=2),encoding='utf-8');print('BULK COMPLETE',len(records),flush=True)

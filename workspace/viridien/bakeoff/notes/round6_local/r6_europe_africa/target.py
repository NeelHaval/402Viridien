import requests,time,json,fitz
from pathlib import Path
root=Path('C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work')
out=root/'external_untrusted/round6/r6_europe_africa/sodir'
s=requests.Session();s.headers['User-Agent']='Mozilla/5.0 (research crawler)'
urls=['https://factpages.sodir.no/pbl/wellbore_documents/4408_16_2_2_COMPLETION_REPORT.pdf','https://factpages.sodir.no/pbl/wellbore_documents/5093_7131_4_1_COMPLETION_GEOLOGICAL_REPORT.PDF','https://factpages.sodir.no/pbl/wellbore_documents/70_02_31_2_7_Mud_logging_report.pdf','https://factpages.sodir.no/pbl/wellbore_documents/4910_24_9_7_COMPLETION_LOG.pdf']
records=[]
for u in urls:
 time.sleep(1.1)
 try:
  r=s.get(u,timeout=45);r.raise_for_status();p=out/u.split('/')[-1];p.write_bytes(r.content)
  d=fitz.open(p);hits=[]
  for i,pg in enumerate(d):
   t=pg.get_text()
   if 'FORMATION EVALUATION LOG' in t.upper() or pg.rect.height>pg.rect.width*3:
    hits.append(i)
  rec={'url':u,'path':str(p.relative_to(root)),'bytes':len(r.content),'pages':len(d),'hits':hits};records.append(rec)
  for i in (hits[:3] or [0]):
   pg=d[i];z=min(900/pg.rect.width,1700/pg.rect.height);pg.get_pixmap(matrix=fitz.Matrix(z,z)).save(out/(p.stem+f'_p{i+1}_preview.png'))
  print(p.name,len(d),hits,flush=True)
 except Exception as e:print(type(e).__name__,str(e)[:180],flush=True)
 (out/'target_records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

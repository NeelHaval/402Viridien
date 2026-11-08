import os,json,time,re,requests,fitz
from pathlib import Path
from bs4 import BeautifulSoup
os.environ['OMP_NUM_THREADS']='3'
root=Path('C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work')
out=root/'external_untrusted/round6/r6_europe_africa/sodir'
s=requests.Session();s.headers['User-Agent']='Mozilla/5.0 (research crawler)'
u='https://factpages.sodir.no/en/wellbore/TableView/With/Documents'
r=s.get(u,timeout=60);r.raise_for_status();(out/'index.html').write_text(r.text,encoding='utf-8')
soup=BeautifulSoup(r.text,'html.parser')
urls=list(dict.fromkeys(u for u in re.findall(r"https://factpages[.]sodir[.]no/pbl/[^<> ]+?[.][pP][dD][fF]",soup.get_text(" ")) if "completion_log" in u.lower() or "mud_log" in u.lower()))
(out/'discovered.json').write_text(json.dumps(urls,indent=2),encoding='utf-8')
print('discovered',len(urls),flush=True)
records=[]
for u in urls[:15]:
 time.sleep(1.1)
 try:
  r=s.get(u,timeout=45)
  if r.status_code in (429,503): print('backoff',r.status_code,flush=True);break
  r.raise_for_status();p=out/u.split('/')[-1];p.write_bytes(r.content)
  d=fitz.open(p);rec={'url':u,'path':str(p.relative_to(root)),'bytes':len(r.content),'pages':len(d),'sizes':[[round(x.rect.width),round(x.rect.height)] for x in d]}
  records.append(rec)
  pg=d[0];scale=min(1000/pg.rect.width,1800/pg.rect.height);pg.get_pixmap(matrix=fitz.Matrix(scale,scale)).save(out/(p.stem+'_preview.png'))
  print(p.name,len(d),rec['sizes'][:3],flush=True)
 except Exception as e: print(type(e).__name__,str(e)[:160],flush=True)
 (out/'download_records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

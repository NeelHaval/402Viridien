import pathlib, requests, json, time, sys
from bs4 import BeautifulSoup
root=pathlib.Path(sys.argv[1]); base=root/'external_untrusted/round6/r6_americas'
items=[
('brazil','https://reate.cprm.gov.br/anp/TERRESTREen'),
('mexico','https://rondasmexico.energia.gob.mx/media/2073/trion-ficha-final.pdf'),
('mexico','https://rondasmexico.energia.gob.mx/media/2115/lista-de-estudios-y-pozos-trion_2.pdf'),
('mexico','https://rondasmexico.energia.gob.mx/media/2057/shcp-sener-cnh.pdf'),
('trinidad','https://www.energy.gov.tt/wp-content/uploads/2013/11/DATA_ROOM_GUIDELINES.pdf'),
('trinidad','https://www.energy.gov.tt/wp-content/uploads/2018/06/Virtual-Data-Room.pdf'),
('colombia','https://www.anh.gov.co/documents/25484/ESTRATEGIA_CONJUNTA_ANH-SGC_2024.pdf'),
('suriname','https://www.staatsolie.com/media/naehgsfs/confidentiality-agreement-registration-sho-bid-round-2023-2024.pdf'),
('guyana','https://petroleum.gov.gy/well-activities/')]
records=[];last={}
for source,url in items:
 host=requests.compat.urlparse(url).netloc;time.sleep(max(0,1.1-(time.monotonic()-last.get(host,0))))
 folder=base/source;folder.mkdir(parents=True,exist_ok=True)
 try:
  r=requests.get(url,headers={'User-Agent':'Mozilla/5.0 (research crawler)'},timeout=14);last[host]=time.monotonic()
  name=url.split('/')[-1] or 'index';name=name if '.' in name else name+'.html';p=folder/name;p.write_bytes(r.content)
  rec=dict(source=source,url=url,status=r.status_code,path=str(p.relative_to(root)),bytes=len(r.content),pages=None)
  if r.content.startswith(b'%PDF'):
   import fitz
   with fitz.open(p) as d:
    rec['pages']=len(d); texts=[p.get_text() for p in d];(folder/(name+'.txt')).write_text('\n'.join(texts),encoding='utf-8')
    rec['hits']=[i+1 for i,t in enumerate(texts) if any(w in t.lower() for w in ['mud log','master log','litolog','litholog','registros'])]
  else:
   s=BeautifulSoup(r.text,'html.parser');(folder/(name+'.links.json')).write_text(json.dumps([dict(text=a.get_text(' ',strip=True),url=requests.compat.urljoin(r.url,a['href'])) for a in s.find_all('a',href=True)],indent=2),encoding='utf-8')
   rec['embeds']=[str(a)[:1200] for a in s.find_all(['iframe','object'])]
  records.append(rec);print(json.dumps(rec,ensure_ascii=False),flush=True)
 except Exception as e: records.append(dict(source=source,url=url,error=str(e)));print(source,str(e),flush=True)
 (base/'sample_index.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')

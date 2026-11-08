import requests,pathlib,json,time
from bs4 import BeautifulSoup
ROOT=pathlib.Path('C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work')
base=ROOT/'external_untrusted/round6/r6_americas'
urls=[('brazil','https://reate.cprm.gov.br/anp/TERRESTRE'),('mexico','https://rondasmexico.energia.gob.mx/esp/asociaciones/pemex-2016/cnh-a1-trion2016/'),('trion','https://rondasmexico.gob.mx/media/2073/trion-ficha-final.pdf')]
records=[]
for name,url in urls:
 p=base/name;p.mkdir(parents=True,exist_ok=True)
 try:
  r=requests.get(url,headers={'User-Agent':'Mozilla/5.0 (research crawler)'},timeout=22)
  dest=p/('source.pdf' if r.content.startswith(b'%PDF') else 'source.html');dest.write_bytes(r.content)
  links=[{'text':a.get_text(' ',strip=True)[:120],'url':requests.compat.urljoin(r.url,a['href'])} for a in BeautifulSoup(r.text,'html.parser').find_all('a',href=True)] if dest.suffix=='.html' else []
  records.append(dict(name=name,url=url,status=r.status_code,path=str(dest),bytes=len(r.content),links=links))
  print(name,r.status_code,len(r.content),json.dumps(links,ensure_ascii=False)[:13000],flush=True)
 except Exception as e:records.append(dict(name=name,url=url,error=str(e)));print(name,str(e),flush=True)
(base/'fetch_index.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

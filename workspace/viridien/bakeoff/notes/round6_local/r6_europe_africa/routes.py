import requests,json,time
from pathlib import Path
from bs4 import BeautifulSoup
r=Path('C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/external_untrusted/round6/r6_europe_africa');records=[]
s=requests.Session();s.headers['User-Agent']='Mozilla/5.0 (research crawler)'
for name,u in [('geus','https://frisbee.geus.dk/'),('nlog','https://www.nlog.nl/datacenter/brh-overview')]:
 p=r/name;p.mkdir(exist_ok=True)
 try:
  x=s.get(u,timeout=30);(p/'index.html').write_text(x.text,encoding='utf-8');bs=BeautifulSoup(x.text,'html.parser');record={'name':name,'url':u,'status':x.status_code,'final_url':x.url,'title':bs.title.get_text() if bs.title else '', 'links':[a.get('href') for a in bs.select('a[href]')][:40],'scripts':[a.get('src') for a in bs.select('script[src]')]};print(json.dumps(record)[:2000],flush=True);records.append(record)
 except Exception as e:records.append({'name':name,'url':u,'error':str(e)});print(name,str(e)[:180],flush=True)
 time.sleep(1.1)
(r/'route_checks.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

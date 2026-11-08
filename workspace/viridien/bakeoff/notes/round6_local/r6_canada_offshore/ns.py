import sys,requests,re,time
from pathlib import Path
p=Path(sys.argv[1]);p.mkdir(exist_ok=True,parents=True);s=requests.Session();s.headers['User-Agent']='Mozilla/5.0 (research crawler)'
u='https://cnsopbdigitaldata.ca/geoviewer/dmc/public/query/1';r=s.get(u,timeout=25);(p/'boreholes.html').write_text(r.text,encoding='utf-8')
for l in r.text.splitlines():
 if any(k in l.lower() for k in ['ajax','url:','script src','datatable','cheshire']):print(l[:400])

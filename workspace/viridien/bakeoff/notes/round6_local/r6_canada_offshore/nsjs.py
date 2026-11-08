import requests,sys,time
from pathlib import Path
p=Path(sys.argv[1]);s=requests.Session();s.headers['User-Agent']='Mozilla/5.0 (research crawler)'
for name,u in [('custom','https://cnsopbdigitaldata.ca/geoviewer/dmc/public/script/cnsoer-V6.2.js'),('main','https://cnsopbdigitaldata.ca/geoviewer/script/query-summary/main.js')]:
 time.sleep(1.1);r=s.get(u,timeout=25);(p/(name+'.js')).write_text(r.text,encoding='utf-8');print(name,r.status_code)
 for l in r.text.splitlines():
  if any(k in l for k in ['url:', 'ajax','getJSON','query/','json/']):print(l[:250])

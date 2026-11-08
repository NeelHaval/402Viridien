import requests,time,json,sys
from pathlib import Path
root=Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=True)
s=requests.Session();s.headers['User-Agent']='Mozilla/5.0 (research crawler)'
u='https://home-cnlopb.hub.arcgis.com/'
r=s.get(u,timeout=30);(root/'hub.html').write_text(r.text,encoding='utf-8')
import re
print('hub',r.status_code,len(r.content));print('\n'.join(re.findall(r'.{0,100}(?:itemId|siteId|[a-f0-9]{32}).{0,100}',r.text)[:20]))
time.sleep(1.1)
u='https://www.arcgis.com/sharing/rest/search'
r=s.get(u,params={'q':'cnlopb well','num':100,'f':'json'},timeout=30);(root/'arcgis_search.json').write_text(r.text,encoding='utf-8')
for x in r.json().get('results',[]):print(x['id'],x['title'],x['type'],x.get('url',''))

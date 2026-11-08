import requests,time,json,sys,re
from pathlib import Path
from bs4 import BeautifulSoup
p=Path(sys.argv[1]);s=requests.Session();s.headers['User-Agent']='Mozilla/5.0 (research crawler)'
def get(u):
 time.sleep(1.1);r=s.get(u,timeout=35);r.raise_for_status();return r
site=json.loads((p/'site.json').read_text(encoding='utf-8'));pages=[];rows=[]
def walk(x):
 if isinstance(x,dict):
  if 'slug' in x and 'id' in x and any(k in x['slug'] for k in ['mizzen','harpoon','bay-du-nord','great-barasway','lona','blue-jay','cappahayden','baccalieu']):pages.append(x)
  for v in x.values():walk(v)
 elif isinstance(x,list):
  for v in x:walk(v)
def scan(x,well):
 if isinstance(x,str) and '<' in x:
  for tr in BeautifulSoup(x,'html.parser').find_all('tr'):
   t=tr.get_text(' ',strip=True)
   if re.search(r'Mudloggers|Mud Log|Master Log|Drilling Composite|Geological Report|Well History',t,re.I):
    for a in tr.find_all('a',href=True):rows.append({'well':well,'title':t,'url':a['href']})
 elif isinstance(x,dict):
  for v in x.values():scan(v,well)
 elif isinstance(x,list):
  for v in x:scan(v,well)
walk(site);print('well pages',len(pages),flush=True)
for x in pages[:18]:
 f=p/(x['id']+'.json')
 if not f.exists():f.write_bytes(get('https://www.arcgis.com/sharing/rest/content/items/'+x['id']+'/data?f=json').content)
 scan(json.loads(f.read_text(encoding='utf-8')),x['slug'])
(p/'candidate_rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print('candidates',len(rows),flush=True)
rows.sort(key=lambda x:(not bool(re.search('Mudloggers|Master Log|Mud Log',x['title'],re.I)),x['well']))
for i,x in enumerate(rows[:15]):
 try:
  r=get(x['url']);f=p/(x['well']+'_'+re.search('/items/([^/]+)',x['url'])[1]+'.pdf');f.write_bytes(r.content);x.update(path=str(f),bytes=len(r.content),content_type=r.headers.get('Content-Type'));print(i,x['title'][:100],len(r.content),r.content[:10],flush=True)
 except Exception as e:x['error']=str(e);print('error',str(e),flush=True)
(p/'downloads.json').write_text(json.dumps(rows[:15],indent=2),encoding='utf-8')

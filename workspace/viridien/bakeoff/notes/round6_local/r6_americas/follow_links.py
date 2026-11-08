import pathlib,json,requests,time
from bs4 import BeautifulSoup
root=pathlib.Path('C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work');base=root/'external_untrusted/round6/r6_americas'
idx=json.loads((base/'fetch_index.json').read_text(encoding='utf-8'))
links=idx[1]['links']; urls=[a['url'] for a in links if a['text'] in ['Acceso a datos','Información','Presentaciones']]
for i,u in enumerate(urls):
 time.sleep(1.1)
 try:
  r=requests.get(u,headers={'User-Agent':'Mozilla/5.0 (research crawler)'},timeout=18);p=base/'mexico'/f'linked_{i}.html';p.write_bytes(r.content)
  s=BeautifulSoup(r.text,'html.parser'); out=[{'text':a.get_text(' ',strip=True)[:100],'url':requests.compat.urljoin(r.url,a['href'])} for a in s.find_all('a',href=True) if '/media/' in a['href'] or 'cnih' in a['href']]
  (p.with_suffix('.links.json')).write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8');print(i,r.status_code,s.get_text(' ',strip=True)[-10000:],json.dumps(out,ensure_ascii=False),flush=True)
 except Exception as e: print(i,str(e),flush=True)

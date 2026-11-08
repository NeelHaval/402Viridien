import requests,json,time,sys,re
from pathlib import Path
from urllib.parse import unquote
p=Path(sys.argv[1]);s=requests.Session();s.headers['User-Agent']='Mozilla/5.0 (research crawler)'
h=(p/'hub.html').read_text(encoding='utf-8');d=json.loads(unquote(re.search(r'window.__SITE="([^"]+)',h)[1]));(p/'site.json').write_text(json.dumps(d),encoding='utf-8')
text=json.dumps(d)
print('site links', '\n'.join(re.findall(r'https?[^"<> ]+',text)[:25]))
for id in ['d0f35f9152334fde89b05ffeaee38275','0e37affa1b4042829a7eceebeeb66358','b0309e160b184feba1035eee4c929d85']:
 time.sleep(1.1);r=s.get('https://www.arcgis.com/sharing/rest/content/items/'+id+'/data',params={'f':'json'},timeout=25);(p/(id+'.json')).write_text(r.text,encoding='utf-8');print(id,'\n'.join(re.findall(r'https?[^"<>\s\\]+',r.text)[:50]))

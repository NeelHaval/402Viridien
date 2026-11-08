import json,sys
from pathlib import Path
from bs4 import BeautifulSoup
p=Path(sys.argv[1])
for name in ['d0f35f9152334fde89b05ffeaee38275','0e37affa1b4042829a7eceebeeb66358']:
 d=json.loads((p/(name+'.json')).read_text(encoding='utf-8'))
 def walk(x):
  if isinstance(x,str) and ('<a ' in x or 'mud' in x.lower()):
   soup=BeautifulSoup(x,'html.parser')
   for a in soup.find_all('a'):
    t=a.get_text(' ',strip=True);u=a.get('href','')
    if t:print(t[:130],u)
  elif isinstance(x,dict):
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 print(name);walk(d)

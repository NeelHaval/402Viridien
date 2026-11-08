import requests,sys,json
from pathlib import Path
p=Path(sys.argv[1]);r=requests.get('https://cnsopbdigitaldata.ca/geoviewer/dmc/public/query/1/getData',headers={'User-Agent':'Mozilla/5.0 (research crawler)'},timeout=25);(p/'getData.json').write_text(r.text,encoding='utf-8');print(r.status_code,r.text[:600])

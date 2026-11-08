from pathlib import Path
import sys,re,json
p=Path(sys.argv[1]);d=json.loads((p/'getData.json').read_text(encoding='utf-8'));print({k:v for k,v in d.items() if k!='data'});print('records',len(d['data']['records']));print([r for r in d['data']['records'] if any(k in str(r).lower() for k in ['cheshire','aspy','monterey'])]);s=(p/'main.js').read_text(encoding='utf-8')
for pat in ['tablePaged','getRows','getPage']:
 for m in list(re.finditer(pat,s))[-3:]:print(s[max(0,m.start()-150):m.end()+800])

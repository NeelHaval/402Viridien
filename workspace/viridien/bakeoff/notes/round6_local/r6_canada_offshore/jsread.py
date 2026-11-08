from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text(encoding='utf-8')
for pat in ['getData','serviceName','/query/','getJSON']:
 print(pat,[s[max(0,m.start()-80):m.end()+150] for m in list(re.finditer(re.escape(pat),s))[-8:]])

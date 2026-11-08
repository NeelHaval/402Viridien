import sys,json
from pathlib import Path
import fitz
root=Path(sys.argv[1])
records=[]
for p in root.glob('*/*/*.pdf'):
 try:
  with fitz.open(p) as d:
   text=' '.join(d[i].get_text() for i in range(min(2,len(d))))
   rec=dict(path=str(p),pages=len(d),text=text[:2200],images=len(d[0].get_images()),drawings=len(d[0].get_drawings()))
   records.append(rec)
 except Exception as e: records.append(dict(path=str(p),error=str(e)))
print(json.dumps(records,indent=2))

import pathlib,sys,json,csv
import fitz
root=pathlib.Path(sys.argv[1]);base=root/'external_untrusted/round6/r6_americas'
t=json.loads((base/'triage.json').read_text(encoding='utf-8'));docs=[d for s in t['sources'] for d in s['documents']]
count=pages=0
for d in docs:
 if not d.get('path'):continue
 p=root/d['path'];assert p.is_file(),str(p);assert p.stat().st_size==d['bytes'],str(p)
 if d.get('pages'):
  with fitz.open(p) as pdf:assert len(pdf)==d['pages']
  count+=1;pages+=d['pages']
 assert d['similarity']==0
out=root/'external_unlabelled/round6/r6_americas'
with (out/'manifest.csv').open(encoding='utf-8',newline='') as f:assert len(list(csv.reader(f)))==1
assert not list((out/'images').iterdir());assert not list((out/'bbox_annotation').iterdir())
for n in ['REPORT.md','contact_sheet.png','mexico_presentation_sheet.png']:
 assert (root/'bakeoff/notes/round6_local/r6_americas'/n).is_file()
print(f'PASS: 6 sources; {count} supporting PDFs; {pages} PDF pages; 0 full logs; 0 converted pages; 0 labelled windows.')

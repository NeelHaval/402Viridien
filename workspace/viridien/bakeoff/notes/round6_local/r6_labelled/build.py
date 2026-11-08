import os,sys,json,importlib.util,csv,hashlib
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='3'
sys.dont_write_bytecode=True
ROOT=Path(sys.argv[1]).resolve()
OUT=ROOT/'external_tests_round6'
NOTES=ROOT/'bakeoff/notes/round6_local/r6_labelled'
def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def dump(path,obj): path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def main():
 import requests,fitz
 inputs=ROOT/'external_untrusted/round6/r6_labelled'; source=inputs/'idaho'; source.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
 url='https://ogcc.idaho.gov/wp-content/uploads/2017/12/1107520023_DJS2-14_MUD_20140916_PTS.pdf'
 pdf=source/'DJS_Properties_2-14.pdf'
 if not pdf.exists():
  r=requests.get(url,headers={'User-Agent':'Mozilla/5.0 (research crawler)'},timeout=90); r.raise_for_status(); pdf.write_bytes(r.content)
 kgs=mod('r6kgs',ROOT/'bakeoff/external/kgs_extract.py'); kgs.OUTPUT_ROOT=OUT/'extraction'
 pages=kgs.extract_pdf(pdf,kgs.OUTPUT_ROOT/'idaho'/pdf.stem,ft_per_px=.1)
 print('extraction',[(p['page'],p['status'],p.get('reason','')) for p in pages],flush=True)
 doc=dict(path=str(pdf.relative_to(ROOT)),url=url,pages=len(pages),raster_or_vector='vector',has_lith_column=True,has_legend=True)
 dump(inputs/'triage.json',dict(family='texas_rock_types',sources=[dict(name='Idaho DJS Properties 2-14',urls_seen=[url],access='public',license_terms='No explicit reuse licence verified',documents=[doc])],dead_ends=['Twin Peaks 1W: mud log listed without a download link']))
 # Adapter retains the established exact-tile conversion and packaging rules.
 adapter=mod('r6convert',ROOT/'bakeoff/external/round3_tools/convert_round3.py'); adapter.ROOT=ROOT
 adapter.documents=lambda _: iter([('idaho',doc,pdf)])
 dump(OUT/'extraction_audit.json',[dict(source='idaho',path=doc['path'],extraction=str((kgs.OUTPUT_ROOT/'idaho'/pdf.stem).relative_to(ROOT)),pages=[{k:v for k,v in p.items() if k!='meta'} for p in pages])])
 adapter.package_truth(inputs,OUT)
 # IDs are opaque identifiers. Rename their round prefix in all package records.
 for sub in ('images','bbox_annotation','truth','gallery'):
  for p in (OUT/sub).glob('r3_*'): p.rename(p.with_name(p.name.replace('r3_','r6_',1)))
 for p in list(OUT.glob('*.json'))+list((OUT/'bbox_annotation').glob('*.json'))+[OUT/'manifest.csv']:
  p.write_text(p.read_text(encoding='utf-8').replace('r3_idaho','r6_idaho').replace('external_round3','external_round6'),encoding='utf-8')
 print((OUT/'build_summary.json').read_text(encoding='utf-8'),flush=True)
if __name__=='__main__': main()

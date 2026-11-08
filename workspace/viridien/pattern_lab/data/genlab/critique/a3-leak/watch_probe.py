"""Simulate input stat changes in memory; never modify live inputs."""
import sys, os, json
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1'
from genlab.server import Lab
from genlab.dataset_loop import snapshot
from genlab.assemble.stats import watch_signature
state=json.loads((ROOT/'data/genlab/state.json').read_text(encoding='utf-8'))
card=next(c for c in state['cards'] if c['source']=='agent-made')
from genlab import proposals
p=next(p for p in proposals.inventory(card['doc_id']) if p['legend_id']==card['legend_id'])
targets={
 'split-policy':ROOT/'data/normalized/split-policy.json',
 'current-labels':ROOT/'data/normalized/current.json',
 'machine-prefill':ROOT.parent/'bakeoff/labeltool/data/prefill.jsonl',
 'proposal-png':Path(p['tile_path']),
 'candidate-json':ROOT/f"data/genlab/candidates/{card['doc_id']}.json",
 'mix-cache':ROOT/'data/genlab/assemble/stats/mix-metadata.json',
}
original=Path.stat
class Stat:
 def __init__(self,value): self.value=value
 def __getattr__(self,key):
  value=getattr(self.value,key)
  return value+1000000 if key in ('st_mtime_ns','st_size') else value
for name,fn in [('lab',lambda:Lab.fingerprint(None)),('dataset',snapshot),('stats',watch_signature)]:
 baseline=fn(); result={}
 for label,target in targets.items():
  def change(path,*a,**kw):
   value=original(path,*a,**kw)
   return Stat(value) if os.path.normcase(os.path.abspath(path))==os.path.normcase(os.path.abspath(target)) else value
  with patch.object(Path,'stat',change): result[label]=fn()!=baseline
 print(name,json.dumps(result),flush=True)

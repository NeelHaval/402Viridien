import sys, os, json, hashlib, urllib.request
from pathlib import Path
from collections import Counter, defaultdict
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1'
from genlab.dataset import well_split, DatasetBuilder
from genlab import learn, paths
import model
def read(p): return json.loads((ROOT/p).read_text(encoding='utf-8'))
def out(k,v): print(k,json.dumps(v,ensure_ascii=False),flush=True)
store=model.Store(); docs={d['id']:store.get(d['id']) for d in store.catalog}
policy=read('data/normalized/split-policy.json')
splits={d:well_split(doc,policy)[1] for d,doc in docs.items()}
names={doc['name']:did for did,doc in docs.items()}
idx=read('data/normalized/current.json'); by_ann={(r['document_id'],r['annotation_id']):r for r in idx['records']}
builder=DatasetBuilder.__new__(DatasetBuilder); builder.docs=docs; builder.baseline=None
builder.doc_info={d:dict(well=well_split(doc,policy)[0],split=splits[d],family='audit') for d,doc in docs.items()}
rows=builder.human_rows()
bad=[r for r in rows if by_ann.get((r['doc'],r['annotation_id']),{}).get('quality_issues')]
out('live_human_rows',dict(total=len(rows),quality_bypass=len(bad),counts=Counter(r['split'] for r in bad)))
tests=defaultdict(list)
for r in idx['records']:
 if r['split']=='test': tests[r['pixel_sha256']].append(r)
out('duplicate_links',[dict(doc=r['document_id'],ann=r['annotation_id'],targets=r['targets'],box=r['box_xyxy'],test_docs=[t['document_id'] for t in tests[r['pixel_sha256']]],sha=r['pixel_sha256']) for r in idx['records'] if 'duplicate_of_test' in r['quality_issues']])
folder=ROOT/'data/genlab/datasets/v5'
real=json.loads((folder/'real.json').read_text(encoding='utf-8'))
measure=json.loads((folder/'realism.json').read_text(encoding='utf-8'))['measurements']
bad_keys={(r['doc'],tuple(r['box'])) for r in bad}
out('bad_measurements',dict(n=sum((m['doc'],tuple(m['box'])) in bad_keys for m in measure),classes=sorted({m['class_id'] for m in measure if (m['doc'],tuple(m['box'])) in bad_keys})))
out('saved_test_pixel_matches',[dict(doc=r['doc'],ann=r['annotation_id'],split=r['split'],sha=r['pixel_sha256']) for r in real if r['pixel_sha256'] in tests])
with urllib.request.urlopen('http://127.0.0.1:8771/api/mix',timeout=12) as f: mix=json.load(f)
state=read('data/genlab/state.json'); stats=read('data/genlab/assemble/stats.json')
supported={c['key'] for c in state['cards']} & set(stats['classes'])
by_doc=defaultdict(list)
for cid in sorted(supported): by_doc[stats['classes'][cid]['doc_id']].append(cid)
emitted=[]; exclusive=[]
for pair in mix['pairs']:
 logs=pair.get('logs',[pair['log']]); origin=[splits[names[n]] for n in logs if n in names]
 wanted={learn.norm(pair['a']),learn.norm(pair['b'])}
 if len(wanted)!=2: continue
 for did,cids in by_doc.items():
  found={cid for cid in cids if learn.norm(stats['classes'][cid]['name']) in wanted}
  if {learn.norm(stats['classes'][cid]['name']) for cid in found}==wanted:
   r=dict(doc=did,a=pair['a'],b=pair['b'],origin=origin,origin_logs=logs,classes=sorted(found))
   emitted.append(r)
   if origin and set(origin)=={'test'}: exclusive.append(r)
out('mix_pair_origins',Counter('test-only' if set(splits[names[n]] for n in p.get('logs',[p['log']]) if n in names)=={'test'} else 'other' for p in mix['pairs']))
out('suite_pair_leak',dict(emitted=len(emitted),test_only=len(exclusive),unique_test_only=len({(p['doc'],p['a'],p['b']) for p in exclusive}),examples=exclusive[:4],first3=emitted[:3]))
test_reports=[]; test_props=0
for d in docs:
 if splits[d]!='test': continue
 p=ROOT/f'data/genlab/candidates/{d}.json'
 if p.exists():
  r=json.loads(p.read_text(encoding='utf-8')); test_reports.append(dict(doc=d,n=len(r['entries']),tiers=Counter(e['tier'] for e in r['entries'])))
 test_props+=len(list((ROOT/f'data/genlab/proposals/{d}').glob('legend-[0-9]*.json')))
out('test_exposure',dict(reports=len(test_reports),entries=sum(d['n'] for d in test_reports),proposal_files=test_props,example=test_reports[:1]))
stale=[]
for m in read('data/genlab/assemble/noise/measurements.json')['measurements']:
 if (m['source_doc'],tuple(m['box'])) in bad_keys: stale.append(m['id'])
out('noise_bad_exact_boxes',stale)
pairkey=lambda p:(p['doc'],tuple(sorted([p['a'],p['b']])))
testkeys={pairkey(p) for p in exclusive}
cleankeys={pairkey(p) for p in emitted if any(s!='test' for s in p['origin'])}
out('test_only_new_pairs',dict(n=len(testkeys-cleankeys),examples=sorted(testkeys-cleankeys)[:4]))
val={did for did,s in splits.items() if s=='validation'}
out('validation_stats_rows',dict(human=sum(stats['levels']['doc:'+d]['evidence']['human_rows'] for d in val),machine=sum(stats['levels']['doc:'+d]['evidence']['machine_rows'] for d in val)))
held=[c for c in state['cards'] if c.get('realism') and c['realism'].get('held_out',0)]
out('claimed_holdout_totals',dict(cards=len(held),crop_scores=sum(c['realism']['held_out'] for c in held),all_have_selection=all(c.get('tile_scores') for c in held)))
from genlab.assemble.bench import regression_values
results={}
for p in (ROOT/'data/genlab/assemble/bench/runs').glob('*/*.json'):
 if p.stem not in ('fidelity','realism','detector'):continue
 value=json.loads(p.read_text(encoding='utf-8'))
 if 'fixtures' not in Path(value.get('run','')).parts:results.setdefault(p.parent.name,{})[p.stem]=value
values=regression_values(results)
out('regression_inputs',dict(total=len(values),test=sum('/test/' in k for k in values)))
bad_ids={m['id']:by_ann[(r['doc'],r['annotation_id'])]['quality_issues'] for m in measure for r in bad if m['doc']==r['doc'] and m['box']==r['box']}
counts=Counter()
for line in (folder/'manifest.jsonl').open(encoding='utf-8'):
 r=json.loads(line)
 if r.get('measurement_id') in bad_ids:counts[r['variant']+':'+','.join(bad_ids[r['measurement_id']])]+=1
out('bad_measurement_sample_uses',counts)
try:
 import psutil
 out('peak_working_set_MiB',round(psutil.Process().memory_info().peak_wset/1024**2,1))
except ImportError:pass

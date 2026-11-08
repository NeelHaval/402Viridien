"""Read-only end-to-end audit; run with python -B -X utf8."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

policy = read(ROOT/'data/normalized/split-policy.json')
def is_test(d):
    g = policy['document_groups'][d]
    return int(hashlib.sha256((policy['seed']+':'+g).encode()).hexdigest()[:8],16)%5 == 0

stats = read(ROOT/'data/genlab/assemble/stats.json')
print('STATS', stats['inputs'])
print('TEST DOCS in stats', [k for k in stats['levels'] if k.startswith('doc:') and is_test(k[4:])])
for level in ['global', "family:Texas 'Rock Types' box", 'family:MC765 colour grid', 'doc:d368760651ded938']:
    b=stats['levels'][level]
    fits=list(b['thickness'].values())
    print('LEVEL',level, 'n',b['n_layers'], 'evidence',b['evidence'], 'gaps', b['gaps'],
          'censored',sum(f.get('censored_n',0) for f in fits), 'thickness_n',sum(f['n'] for f in fits))

for run in ['starter','iteration-2']:
    counts=Counter(); subs=Counter(); examples=[]; docs=set(); page=[]; leaked=[]
    for f in sorted((ROOT/'data/genlab/assemble/out'/run).glob('*/truth.json')):
        t=read(f); counts['images']+=1; docs.add(t['base_doc']); counts['rows']+=t['height']
        if t['mode']=='page': page.append((str(f.relative_to(ROOT)),t['base_doc'],t['width'],t['height'],t.get('page')))
        for d in t.get('source_docs',[]):
            if is_test(d): leaked.append((str(f),d))
        gap=sum(l['bottom']-l['top'] for l in t['layers'] if l['class_id']=='__gap__')
        counts['gaps']+=gap
        for l in t['layers']:
            counts['layers']+=1
            if l.get('pattern_source')=='family-substitute':
                counts['substituted_rows']+=l['bottom']-l['top']; subs[(l['class_id'],l.get('pattern_id'))]+=l['bottom']-l['top']
        if gap/t['height']>.25: examples.append((f.parent.name,gap,t['height'],round(gap/t['height'],4)))
    print('RUN',run,dict(counts),'docs',len(docs),'test_sources',leaked,'GAP EXAMPLES',examples,'SUBSTITUTES',subs.most_common(20),'PAGES',page)

for f in sorted((ROOT/'data/genlab/assemble/suites').glob('*/suite.json')):
    t=read(f)
    print('SUITE',f.parent.name,'keys',list(t),'summary',str(t)[:900])

for f in (ROOT/'data/genlab/assemble/bench/runs/starter').glob('*.json'):
    t=read(f)
    print('BENCH',f.name, 'keys',list(t),'summary',str(t)[:2600])

from genlab.assemble.bench import score_items
t={'synth_id':'proof','height':100,'layers':[{'top':0,'bottom':50,'class_id':'A'},{'top':50,'bottom':100,'class_id':'B'}]}
p={'proof':[dict(l,bbox=[999,999,1000,1000]) for l in t['layers']]}
print('WRONG_BBOX_PERFECT_ROWS',score_items([t],p))

t={'synth_id':'partial','height':100,'layers':[{'top':40,'bottom':60,'class_id':'A'}]}
p={'partial':[{'top':0,'bottom':100,'class_id':'A'}]}
print('PARTIAL_CROP_CORRECT_BED',score_items([t],p,labelled_only=True))

from genlab.assemble.compose import Sampler
block={'confidence':{'composition':'high'},'composition':{'all':{'d:A':.5,'d:B':.5}},
       'thickness':{f'd:{c}':{'dist':'empirical','quantiles_px':[h]*5,'n':100,'effective_n':100} for c,h in [('A',10),('B',100)]}}
synthetic_stats={'classes':{'d:A':{'norm':'a'},'d:B':{'norm':'b'}},'levels':{'doc:d':block}}
choices={c:{'pattern_source':'document'} for c in synthetic_stats['classes']}
for seed in [0,1,2]:
    sequence=Sampler(synthetic_stats,'d','f',choices,seed).sequence(100000)
    shares=Counter()
    for l in sequence:shares[l['class_id']]+=l['bottom']-l['top']
    print('COMPOSITION_FALLBACK_TARGET_50_50',seed,{k:v/100000 for k,v in shares.items()},'beds',len(sequence))

for version in ['v4','v5']:
    folder=ROOT/'data/genlab/datasets'/version
    print('DATASET',version,'top_files',[(p.name,p.stat().st_size) for p in folder.iterdir() if p.is_file()])

try:
    import psutil
    print('PROCESS_MEMORY',psutil.Process().memory_info())
except ImportError:
    pass

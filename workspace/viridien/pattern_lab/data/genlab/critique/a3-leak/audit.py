"""Read-only leakage audit; any artifacts stay beside this script."""
import sys, os, json, hashlib, time
from pathlib import Path
from collections import Counter, defaultdict
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True
def read(p): return json.loads((ROOT/p).read_text(encoding='utf-8'))
def out(key, value): print(key, json.dumps(value, ensure_ascii=False), flush=True)
policy = read('data/normalized/split-policy.json')
groups = policy['document_groups']
def split(d):
    g = groups[d]
    def fifth(seed): return int(hashlib.sha256((seed+':'+g).encode()).hexdigest()[:8],16)%5==0
    return 'test' if fifth(policy['seed']) else 'validation' if fifth('genlab-validation-v1') else 'train'
index = read('data/normalized/current.json')
state = read('data/genlab/state.json')
manifest = read('data/genlab/generated/manifest.json')
out('snapshot', dict(snapshot=index['snapshot'], records=len(index['records']), state_built=state.get('built_at'), at=time.time()))
out('doc_splits', Counter(split(d) for d in groups))
out('well_splits', Counter(split(next(d for d in groups if groups[d]==g)) for g in set(groups.values())))
out('test_previously_seen', sorted({groups[d] for d in groups if split(d)=='test'} & set(policy['previously_seen_groups'])))
out('normalized_quality', Counter(q for r in index['records'] for q in r['quality_issues']))
out('cards', dict(total=len(state['cards']), by_split=Counter(split(c['doc_id']) for c in state['cards']), sources=Counter(c['source'] for c in state['cards'])))
out('manifest', dict(patterns=len(manifest['patterns']),test_docs=sum(split(r['doc_id'])=='test' for r in manifest['patterns']),test_donors=sum(split(r['donor']['doc_id'])=='test' for r in manifest['patterns'] if r.get('donor'))))
out('claimed_holdouts', [dict(key=c['key'],realism=c['realism'],selection=c.get('tile_scores')) for c in state['cards'] if c.get('realism',{} ) and c['realism'].get('held_out',0)][:5])
stats = read('data/genlab/assemble/stats.json')
noise = read('data/genlab/assemble/noise/measurements.json')
out('asm_stats',dict(inputs=stats['inputs'], keys=list(stats), global_docs=stats['levels']['global'].get('document_ids'),doc_splits=Counter(split(k[4:]) for k in stats['levels'] if k.startswith('doc:'))))
out('asm_noise',dict(keys=list(noise),built=noise.get('built_at'),measurements=len(noise['measurements']),source_splits=Counter(split(m['source_doc']) for m in noise['measurements']),rail_splits=Counter(split(d) for d in noise['rails']),boundary_splits=Counter(split(b['doc']) for bb in noise['boundaries'].values() for b in bb)))
by_ann = {(r['document_id'],r['annotation_id']):r for r in index['records']}
for folder in sorted((ROOT/'data/genlab/datasets').glob('v[0-9]*')):
    if not (folder/'stats.json').exists(): continue
    st=json.loads((folder/'stats.json').read_text(encoding='utf-8'))
    real=json.loads((folder/'real.json').read_text(encoding='utf-8'))
    meas=json.loads((folder/'realism.json').read_text(encoding='utf-8'))['measurements']
    ev=json.loads((folder/'eval.json').read_text(encoding='utf-8')) if (folder/'eval.json').exists() else {}
    bad=[dict(doc=r['doc'],id=r['annotation_id'],split=r['split'],issues=by_ann.get((r['doc'],r['annotation_id']),{}).get('quality_issues')) for r in real if by_ann.get((r['doc'],r['annotation_id']),{}).get('quality_issues')]
    groups_hash=defaultdict(list)
    for c in st['classes']:
        p=folder/c['model_file']
        if p.exists(): groups_hash[hashlib.sha256(p.read_bytes()).hexdigest()].append((c['doc'],c['class_id'],c['split']))
    cross=[v for v in groups_hash.values() if len({x[2] for x in v})>1]
    out('dataset_'+folder.name,dict(patterns=st['patterns'],real=Counter(r['split'] for r in real),measurements=Counter(r['split'] for r in meas),quality_bypass_n=len(bad),quality_bypass=bad[:6],cross_split_identical_models=cross[:8],cross_split_identical_groups=len(cross),test_real=sum(split(r['doc'])=='test' for r in real),truth_n=ev.get('truth_n'),truth_fingerprint=ev.get('truth_fingerprint')))
out('current_vs_stats',dict(current=index['snapshot'],asm=stats['inputs'].get('normalized_snapshot')))

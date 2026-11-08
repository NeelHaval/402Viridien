"""Read-only cross-owner leak checks; tolerate changing assembler schemas."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from collections import Counter
from genlab.dataset import DatasetBuilder,well_split
from genlab.assemble.bench import regression_values
root = Path('data/genlab')
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
b = DatasetBuilder()
splits = {d:i['split'] for d,i in b.doc_info.items()}
val = {d for d,s in splits.items() if s=='validation'}
stats = read(root/'assemble/stats.json')
global_level = stats.get('levels',{}).get('global',{})
stats_docs = set(global_level.get('document_ids',stats.get('document_ids',[])))
levels = stats.get('levels',{})
validation_rows = Counter()
for d in val:
    evidence = levels.get('doc:'+d,{}).get('evidence',{})
    validation_rows.update({k:evidence.get(k,0) for k in ('human_rows','machine_rows')})
results = {}
for p in (root/'assemble/bench/runs').glob('*/*.json'):
    if p.stem in ('fidelity','realism','detector'):
        value = read(p)
        if 'fixtures' not in Path(value.get('run','')).parts:
            results.setdefault(p.parent.name,{})[p.stem] = value
values = regression_values(results)
noise = read(root/'assemble/noise/measurements.json')
measurements = noise['measurements']
print(json.dumps(dict(validation_stats_documents=len(stats_docs&val),validation_stats_rows=dict(validation_rows),
    development_regression_values=len(values),test_regression_values=sum('/test/' in k.lower() for k in values),
    noise_measurements=len(measurements),held_noise_sources=sum(splits.get(m.get('source_doc',m.get('source_doc_id')))!='train' or splits.get(m.get('doc_id',m.get('doc')))!='train' for m in measurements),
    noise_with_snapshot=sum(bool(m.get('normalized_snapshot') or m.get('snapshot')) for m in measurements),
    dataset_live_rows=len(b.human_rows()),dataset_live_quality_issues=sum(bool(r['quality_issues']) for r in b.human_rows())),indent=2))

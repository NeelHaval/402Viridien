"""One bounded CPU recheck. All new writes stay in this review directory."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
import sys, json, hashlib, importlib.util, copy
from pathlib import Path
from unittest.mock import patch
R = Path(__file__).resolve().parent
B = R.parents[1]
sys.path[:0] = [str(B), str(R)]
os.chdir(B)
T = R/'astra_recheck_sandbox'
T.mkdir(exist_ok=True)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def module(name):
    spec=importlib.util.spec_from_file_location('recheck_'+name,R/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
import pytest
# Existing scorer CSV test uses ROOT/out/_cache. Isolate that write too.
from harness import io, legend, legend_v3, score
(T/'out'/'_cache').mkdir(parents=True,exist_ok=True)
with patch.object(io,'ROOT',T):
    result=pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(T/'pytest'),
        'strategies/test_s23_identity.py','strategies/test_scoreboard_v4.py',
        'report/scoreboard_1108/test_scoreboard.py',
        'report/scoreboard_1108/test_report.py','report/scoreboard_1108/test_report_gate.py',
        'harness/test_score.py'])
e={'pytest_exit':int(result)}
from strategies.test_s21_fusion import main as fusion
fusion();e['fusion_cpu']='passed'
runner=module('score_all');report=module('build_report')
import report_inputs as inputs
m=read(R/'manifest_1108.json')
e['manifest']={'sha256':sha(R/'manifest_1108.json'),'digest':runner.manifest_digest(m),'cases':len(m['cases']),'candidates':m['candidates'],'expected':len(m['cases'])*len(m['candidates']),'freeze':m['freeze']}
old=read(R/'astra_review_hashes.json')['reviewed_file_sha256']
paths=[p for p in old if p.startswith(('strategies/','harness/','ml/')) and Path(p).is_file()]
e['baseline_hash_checks']={p:{'before':old[p],'now':sha(B/p),'same':old[p]==sha(B/p)} for p in paths}
e['source_hash_checks']={'count':len(m['source_hashes']),'mismatches':[p for p,h in m['source_hashes'].items() if sha(B/p)!=h]}
assets={}
for c in m['cases']:
    for pk,hk in [('image','image_sha256'),('annotation','annotation_sha256'),('truth_csv','truth_sha256')]:assets[(c[pk],c[hk])]=None
    for s in c['truth_sources']:assets[(s['path'],s['sha256'])]=None
e['input_hash_checks']={'count':len(assets),'mismatches':[p for p,h in assets if sha(p)!=h]}
e['image_suffixes']=sorted({Path(c['image']).suffix.lower() for c in m['cases']})
mapping=inputs.build_dev_map(m)
e['dev_map']={'matches':mapping==read(R/'reviewed_dev_wells.json'),'counts':mapping['group_case_counts'],'logs':len(mapping['log_to_group'])}
report.DEV_GROUPS=mapping['log_to_group']
e['report_source_freeze']={p:sha(R/p)==h for p,h in read(R/'report_source_freeze.json').items()}
history=[]
for p in sorted(R.glob('checkpoint_pre_m2_*.json')):
    stamp=p.stem.split('_')[-1];mp=R/('manifest_pre_m2_'+stamp+'.json')
    if not mp.exists():continue
    rm=read(mp);rr=read(p)['records']
    history.append({'checkpoint':p.name,'sha256':sha(p),'manifest':mp.name,'manifest_sha256':sha(mp),'records':len(rr),'strategies':sorted({r['strategy'] for r in rr}),'migration_notes':sorted({r.get('freeze_migration','absent') for r in rr}),'old_digests_match':all(r.get('manifest_digest') in (None,runner.manifest_digest(rm)) for r in rr),'current_case_identity_match':all(all(r.get(f)==next(c for c in m['cases'] if c['case_id']==r['case_id']).get(f) for f in ('log','suite','window','image_sha256','annotation_sha256','truth_sha256','truth_sources')) for r in rr)})
e['preserved_history']=history
# Same input pixels can return different precomputed OCR evidence. No OCR or GPU call.
import numpy as np
from types import SimpleNamespace
rgb=np.zeros((12,12,3),dtype=np.uint8)
digest=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
home=T/'v3_cache';(home/'cache').mkdir(parents=True,exist_ok=True)
cache=home/'cache'/f'v{legend_v3.VERSION}_{digest}.json'
with patch.object(legend_v3,'HOME',home),patch.object(io,'read_rgb',return_value=rgb):
    def cached_label(name):
        write(cache,{'layout':'flat','items':[{'box':[0,0,12,12],'candidates':[{'text':name,'confidence':1.0,'engine':'page','direction':'right'}]}]})
        return [x.label for x in legend_v3.parse(SimpleNamespace(boxes={'lithology_legend':[0,0,12,12]}),config={'easy':True,'cleanup':True,'easy_weight':.9})]
    a=cached_label('Clay');b=cached_label('Shale')
e['unfrozen_cache_probe']={'input_pixel_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'first':a,'second':b,'different':a!=b,'cache_paths_in_manifest':[p for p in m['source_hashes'] if 'legend_cache' in p or 'legend_names' in p or 'out' in p]}
# Demonstrate a complete replay can pass despite an invalid run digest and model pin.
audit=module('audit_saved_predictions');d=T/'audit_gate';d.mkdir(exist_ok=True)
asset=d/'asset';asset.write_text('synthetic asset',encoding='utf-8')
truth=d/'truth.csv';pred=d/'pred.csv'
rows=[dict(top_y=0,bottom_y=10,lithology='Shale',cluster_id=0)]
io.save_rows(truth,rows);io.save_rows(pred,rows)
c=dict(case_id='synthetic',suite='dev',log='synthetic',y0=0,y1=10,window=[0,10],family='f',well='w',dataset=str(d),column_box=[0,0,10,10],image=str(asset),annotation=str(asset),truth_csv=str(truth),image_sha256=sha(asset),annotation_sha256=sha(asset),truth_sha256=sha(truth),truth_sources=[dict(path=str(truth),sha256=sha(truth))])
fm=dict(cases=[c],candidates=list(m['candidates']),source_hashes={},freeze=m['freeze'])
fr=[dict(c,strategy=s,prediction_csv=str(pred),manifest_digest='stale',strategy_diagnostics={'model':{'weights_sha256':'wrong','config_sha256':'wrong'}},**score.score(rows,rows,(0,10),min_layer_px=0)) for s in fm['candidates']]
raw=dict(complete=True,records=fr,summaries=[],manifest_digest='stale')
write(d/'manifest_1108.json',fm);write(d/'scores.json',raw)
with patch.object(audit,'R',d): audit_status=audit.main()
gate=inputs.validate_records(raw,fm,expected_case_count=1)
binding=inputs.validate_audit(read(d/'audit_saved_predictions.json'),fr,d/'manifest_1108.json')
try:runner.validate_records(fr,fm)
except RuntimeError as exc:strict=str(exc)
else:strict='unexpected acceptance'
e['integrity_gate_probe']={'replay_exit':int(audit_status),'report_record_gate':gate['status'],'report_audit_binding':binding['status'],'runner_rejection':strict,'model_pin':'wrong in every synthetic record'}
# Sign and grouping agreement on the real case identities, with synthetic scores.
rr=[]
for i,c in enumerate(m['cases']):
    if c['suite']!='ext_dev':continue
    for s,v in [('s19_combine',0.),('s23_v4dino_identity',(i%7)/10)]:
        rr.append(dict(c,strategy=s,score=v,row_acc=v,boundary_f1=1.,cluster_ari=0.))
_,rp=runner.summarize(rr);rp=next(x for x in rp if x['suite']=='ext_dev')
bp=report.pairs(rr)[0]
e['pairing_agreement']={'runner_units':rp['metrics']['score']['well_ci']['n_units'],'report_units':bp['wells'],'paired_cases':bp['paired_cases'],'runner_delta':rp['metrics']['score']['log_ci']['mean'],'report_delta':bp['mean_delta'],'opposite_declared_direction_agrees':abs(rp['metrics']['score']['log_ci']['mean']+bp['mean_delta'])<1e-12}
write(R/'astra_recheck_checks.json',e)
print('ASTRA RECHECK EXIT',int(result),flush=True)
sys.exit(int(result))

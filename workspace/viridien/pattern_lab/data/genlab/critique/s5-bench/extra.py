import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,hashlib,urllib.request
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from genlab.assemble import bench as b
from genlab.dataset import closed_summary
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
results={}

# All writes in this harness are redirected to the allowed scratch directory.
old_read=b.read
old_suites=b.SUITES
fake=OUT/'cache-repro';fake.mkdir(exist_ok=True)
b.SUITES=fake/'suites'
b.safe_path=lambda p: Path(p) if Path(p).resolve().is_relative_to(OUT) else (_ for _ in ()).throw(ValueError('outside scratch'))
b.report=lambda: None
stats={'schema':'asm-1','built_at':1,'classes':{'doc:1':dict(doc_id='doc',norm='shale',name='Shale')},'levels':{'global':dict(composition={'all':{'shale':1}})}}
state={'cards':[{'key':'doc:1'}]}; manifests={}; calls=[]
def reader(p):
    p=Path(p)
    if p==b.ASSEMBLE/'stats.json':return stats
    if p==b.ROOT/'data/genlab/state.json':return state
    if p.name=='manifest.json' and p.parent.name in manifests:return manifests[p.parent.name]
    return old_read(p)
b.read=reader
def cli(cmd,log):
    calls.append(cmd)
    run=cmd[cmd.index('--run')+1]; seed=int(cmd[cmd.index('--seed')+1])
    source=fake/'mock-compose'/str(seed);source.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.full((16,8,3),120,np.uint8)).save(source/'column.png')
    t=dict(synth_id='local',height=16,width=8,seed=seed,mode='column',base_doc='doc',layers=[dict(top=0,bottom=16,class_id='doc:1',norm='shale')])
    b.write_json(source/'truth.json',t);manifests[run]={'outputs':[{'path':str(source)}]}
b.execute_compose=cli
previous=b.SUITES/'typical/suite.json'
if previous.exists():
    stale=read(previous);stale['status']='building';b.write_json(previous,stale)
m=b.build_suite('typical',seed=93817,n=1,height=16)
p=b.SUITES/'typical'/m['samples'][0]['path']/'column.png'
before=hashlib.sha256(p.read_bytes()).hexdigest()
Image.fromarray(np.full((16,8,3),255,np.uint8)).save(p)
tampered=hashlib.sha256(p.read_bytes()).hexdigest()
state['cards']=[]  # The only generated class is now withdrawn.
again=b.build_suite('typical',seed=93817,n=1,height=16)
results['cache_repro']=dict(version_before=m['version'],version_after=again['version'],
    compose_calls=len(calls),pixel_hash_changed=before!=tampered,
    corrupted_image_retained=hashlib.sha256(p.read_bytes()).hexdigest()==tampered,
    returned_status=again['status'],withdrawn_inventory_accepted=not state['cards'])

# Owner look-alike endpoint aggregates evidence from all wells. Count provenance.
import model
store=model.Store()
current=read(ROOT/'data/normalized/current.json')
policy=read(ROOT/'data/normalized/split-policy.json')
test_docs={did for did,g in policy['document_groups'].items()
           if int(hashlib.sha256((policy['seed']+':'+g).encode()).hexdigest()[:8],16)%5==0}
docs={x['id']:store.get(x['id']) for x in store.catalog}
test_names={d['name'] for did,d in docs.items() if did in test_docs}
with urllib.request.urlopen('http://127.0.0.1:8771/api/mix',timeout=15) as response: mix=json.load(response)
test_pairs=[p for p in mix['pairs'] if set(p.get('logs',[]))&test_names]
stats=read(b.ASSEMBLE/'stats.json');state=read(ROOT/'data/genlab/state.json')
supported=set(stats['classes'])&{c['key'] for c in state['cards']}
by_doc={}
for cid in sorted(supported):by_doc.setdefault(stats['classes'][cid]['doc_id'],[]).append(cid)
expanded=[]
for pair in mix['pairs']:
    wanted={b.norm(pair['a']),b.norm(pair['b'])}
    if len(wanted)!=2:continue
    for did,cids in by_doc.items():
        found={cid for cid in cids if b.norm(stats['classes'][cid]['name']) in wanted}
        if {b.norm(stats['classes'][cid]['name']) for cid in found}==wanted:
            expanded.append(dict(doc=did,a=pair['a'],b=pair['b'],source_logs=pair.get('logs'),
                test_source=sorted(set(pair.get('logs',[]))&test_names),rows_at_stake=pair.get('rows_at_stake')))
results['pair_provenance']=dict(all_pairs=len(mix['pairs']),pairs_with_test_source=len(test_pairs),
    expanded_count=len(expanded),first_three=expanded[:3],
    expanded_from_test_source=sum(bool(p['test_source']) for p in expanded),
    test_only_pairs=sum(bool(p.get('logs')) and set(p['logs']).issubset(test_names) for p in mix['pairs']))

# Fresh native-size, same-class comparisons from saved real RGB and generated RGB.
folder=ROOT/'data/genlab/datasets/v4';reals=read(folder/'real.json');candidates=read(folder/'eval-swatches.json')
chosen=[];seen=set()
for r in reals:
    if r['split']!='validation' or r['name'] in seen:continue
    cs=[c for c in candidates if c['class_id']==r['class_id'] and c['variant']=='clean']
    if not cs:continue
    with Image.open(folder/r['file']) as im:
        if im.height<70:continue
        real=im.convert('RGB').copy()
    fake_c=cs[-1]
    with Image.open(folder/fake_c['file']) as im:generated=im.convert('RGB').copy()
    w=min(64,real.width,generated.width);h=min(128,real.height,generated.height)
    def center(im):
        x=(im.width-w)//2;y=(im.height-h)//2;return im.crop((x,y,x+w,y+h))
    chosen.append((r,center(real),center(generated),fake_c));seen.add(r['name'])
    if len(chosen)==4:break
sheet=Image.new('RGB',(520,len(chosen)*175),'white');draw=ImageDraw.Draw(sheet)
for i,(r,a,z,c) in enumerate(chosen):
    y=i*175;draw.text((8,y+4),r['class_id']+' '+r['name'],fill='black')
    draw.text((8,y+24),'REAL (left) / GENERATED (right), native pixels',fill='black')
    sheet.paste(a,(100,y+43));sheet.paste(z,(240,y+43))
sheet.save(OUT/'native-comparison.png')
results['native_comparison']=[dict(class_id=r['class_id'],name=r['name'],real=r['file'],generated=c['file'],size=list(a.size)) for r,a,z,c in chosen]
results['closed_summary_truncated']=closed_summary([dict(class_id='a',name='a')]*4,['a'],[2]*4)

results['suite_baselines']={}
for name in b.NAMES:
    folder=old_suites/name
    if not (folder/'suite.json').exists():continue
    m=read(folder/'suite.json');ts=[];ps={};cs=Counter();per=[]
    for s in m['samples']:
        t=read(folder/s['path']/'truth.json');ts.append(t)
        rc=Counter()
        for l in t['layers']:rc[l['class_id']]+=l['bottom']-l['top']
        cs.update(rc);winner,n=rc.most_common(1)[0]
        ps[t['synth_id']]=[dict(top=0,bottom=t['height'],class_id=winner)]
        per.append(dict(doc=t['base_doc'],majority_class=winner,share=n/t['height'],classes=len(rc)))
    scores=b.score_items(ts,ps)
    results['suite_baselines'][name]=dict(majority_oracle_row_accuracy=scores['row_accuracy'],
        boundary_f1=scores['boundary']['f1'],per_image=per)
results['training_inventory']=dict(classes=len(supported),documents=len(by_doc))
results['noise_test_docs']=sorted({m['doc_id'] for m in read(b.ASSEMBLE/'noise/measurements.json')['measurements']}&test_docs)

# Execute the utility status path with a detector wrong on every row in both domains.
from types import SimpleNamespace
image_dir=OUT/'utility-repro';image_dir.mkdir(exist_ok=True)
Image.new('RGB',(8,16),'white').save(image_dir/'column.png')
ls=[dict(top=0,bottom=16,class_id='doc:1',norm='right')]
t=dict(synth_id='synthetic',base_doc='doc',height=16,width=8,layers=ls,_dir=str(image_dir),mode='column')
doc=dict(id='doc',boxes={'interpreted_lithology_column':[0,0,8,16]})
real=dict(synth_id='doc',base_doc='doc',height=16,width=8,layers=ls,doc=doc,
          well='well',split='test',fully_labelled=True)
b.load_run=lambda run:[t]
b.real_data=lambda:([real],{'document_groups':{'doc':'well'}},'scratch')
model.Store=lambda:SimpleNamespace(get=lambda did:doc)
b.RealCrops=lambda:SimpleNamespace(get=lambda doc,box:np.full((box[3]-box[1],box[2]-box[0],3),255,np.uint8),close=lambda:None)
b.detector_entries=lambda doc,crops:[]
b.matcher_predict=lambda rgb,entries:[dict(top=0,bottom=len(rgb),norm='wrong')]
b.save_result=lambda *args:None
utility=b.detector('scratch',force_matcher=True)
results['zero_utility_repro']={k:utility[k] for k in ('status','human_all','test','mode_results')}
try:
    import psutil
    results['peak_memory_MiB']=round(psutil.Process().memory_info().peak_wset/(1024**2),2)
except (ImportError,AttributeError):pass
(OUT/'extra-results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))

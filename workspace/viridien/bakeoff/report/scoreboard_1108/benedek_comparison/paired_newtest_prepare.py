"""Stage three authorized new inputs; no interval truth or old caches."""
import hashlib,json,shutil,tarfile
from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parents[2];ROOT=B.parents[1]
O=B/'out/benedek_compare_newtest';O.mkdir(parents=True,exist_ok=True)
I=ROOT/'bakeoff/out/drive_test_1VovAwov_20261108/inference'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,t):t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
for p in (B/'out/benedek_compare_first3/source').rglob('*.py'):copy(p,O/'source'/p.relative_to(B/'out/benedek_compare_first3/source'))
for folder in ('combine','boundary','match','integrate'):
    for p in (B/'ml'/folder).glob('*.py'):copy(p,O/'source'/p.relative_to(B))
for name in ('boundary_net.pt','boundary_real.pt','matcher.npz','matcher_real.npz','selected.json'):
    p=B/'ml/combine'/name;copy(p,O/'source'/p.relative_to(B))
for name in ('s19_combine.py','s14_learned_match.py','s11_fft_legend.py','s02_features_pelt.py'):
    p=B/'strategies'/name;copy(p,O/'source'/p.relative_to(B))
text=(H/'paired_worker.py').read_text(encoding='utf-8')
text=text.replace("for case in m['cases']:","for case in m['cases']:")
text=text.replace("rows=s24_benedek.run(log,tuple(case['window']),{})","module=__import__('strategies.'+case['strategy'],fromlist=['run']); rows=module.run(log,tuple(case['window']),{'device':'cpu','cache_dir':P/'cache'})")
text=text.replace("case['case_id']+'.csv'","case['case_id']+'_'+case['strategy']+'.csv'")
text=text.replace("r['case_id'] for r in done['records']","(r['case_id'],r['strategy']) for r in done['records']")
text=text.replace("if case['case_id'] in finished:","if (case['case_id'],case['strategy']) in finished:")
(O/'paired_worker.py').write_text(text,encoding='utf-8')
cases=[]
for data in json.loads((I/'boxes_only.json').read_text(encoding='utf-8')):
    img=I/data['image_name'];cid=sha(img)[:16];ann=O/'inputs'/(cid+'.json')
    ann.parent.mkdir(parents=True,exist_ok=True);ann.write_text(json.dumps({'bounding_boxes':data['bounding_boxes']}),encoding='utf-8')
    dest=O/'inputs'/(cid+img.suffix);copy(img,dest);box=data['bounding_boxes']['interpreted_lithology_column']
    for strategy in ('s19_combine','s24_benedek'):
        cases.append(dict(case_id=cid,strategy=strategy,log=img.stem,window=[box[1],box[3]],image=dest.relative_to(O).as_posix(),annotation=ann.relative_to(O).as_posix(),image_sha256=sha(dest),annotation_sha256=sha(ann),truth_available=False))
files={p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file() and p.name not in ('manifest.json','packet.tar.gz')}
(O/'manifest.json').write_text(json.dumps(dict(cases=cases,file_hashes=files,seed=1108,truth_supplied_to_worker=False,compute_timeout_seconds=300),indent=2),encoding='utf-8')
with tarfile.open(O/'packet.tar.gz','w:gz') as t:
    for p in O.rglob('*'):
        if p.is_file() and p.name!='packet.tar.gz':t.add(p,arcname=p.relative_to(O).as_posix())
print('NEWTEST_PACKET_OK inputs=3 model_cases=6 archive_sha256='+sha(O/'packet.tar.gz'))

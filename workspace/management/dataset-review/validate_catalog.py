"""Check saved dataset review records without running inference."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def main():
    c=json.loads((OUT/'catalog.json').read_text(encoding='utf-8'))
    errors=[];ids=[];ready=0
    for r in c['records']:
        ids.append(r['id'])
        if r.get('split') in ['sealed','verify']:errors.append(r['id']+': sealed record exposed')
        if r.get('split') in ['test','shared_verify'] and (r.get('truth') or r.get('intervals')):errors.append(r['id']+': held-out labels exposed')
        if r.get('ready'):
            ready+=1
            for key in ['image','annotation']:
                p=r.get('paths',{}).get(key)
                if not p or not Path(p).is_file():errors.append(r['id']+': ready but missing '+key)
        if r.get('image') and not (OUT/r['image']).is_file():errors.append(r['id']+': preview missing')
        if r['id'].startswith('synth-') and not r['id'].startswith('synth-v1-') and r.get('ready'):errors.append(r['id']+': defective v0 promoted')
        for s in r.get('reviewSuggestions',[]):
            if s.get('humanConfirmed'):errors.append(r['id']+': machine suggestion silently confirmed')
    if len(ids)!=len(set(ids)):errors.append('Duplicate record IDs')
    for p in (OUT/'packs').glob('*.json'):
        j=json.loads(p.read_text(encoding='utf-8'))
        if p.name=='current-human-dev.json':
            for x in j['cases']:
                for k in ['image','annotation','truth_csv']:
                    if not Path(x[k]).is_file():errors.append(p.name+': missing '+k)
    report={'records':len(ids),'ready':ready,'errors':errors,'checks':'Existing inputs, sealed isolation, held-out label hiding, v0 gate and preview links'}
    (OUT/'catalog-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('CATALOG_CHECK '+('PASS' if not errors else 'FAIL')+' records='+str(len(ids))+' ready='+str(ready)+' errors='+str(len(errors)))
    for e in errors[:12]:print(e)
    return bool(errors)
if __name__=='__main__':raise SystemExit(main())

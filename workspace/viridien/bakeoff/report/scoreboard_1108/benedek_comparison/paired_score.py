"""Score saved Benedek predictions with the frozen shared scoring policy."""
import argparse, copy, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
B=HERE.parents[2]
sys.path.insert(0,str(B))
from harness import io,score
from strategies import grading_availability_v2 as av, grading_policy

def build(predictions, output):
    if av.sha(B/'harness/score.py')!='50ee28c914bc0dc9f9f8a255d5d200e5d9e61ff9ef1f9c7187e2bdc4e5c59de6':
        raise ValueError('Shared scorer hash changed')
    manifest=av.read(HERE.parent/'manifest_1108.json')
    old=av.read(HERE.parent/'availability_v2/derived_scores.json')
    cases={r['case_id']:r for r in manifest['cases']}
    templates={r['case_id']:r for r in old['records'] if r['strategy']=='s19_combine'}
    incoming=av.read(predictions)
    records=[]; seen=set(); pins={str(predictions):av.sha(predictions)}
    for item in incoming['records']:
        cid=item['case_id']
        if cid in seen or cid not in cases:raise ValueError('Duplicate or unknown case')
        seen.add(cid);c=cases[cid]
        for key in ('image_sha256','annotation_sha256','window'):
            if item[key]!=c[key]:raise ValueError('Prediction identity changed: '+key)
        pred=Path(item['prediction_csv']); pins[str(pred)]=av.sha(pred)
        if pins[str(pred)]!=item['prediction_sha256']:raise ValueError('Prediction hash changed')
        truth=Path(c['truth_csv']);missing=not truth.is_file()
        gt=[] if missing else io.load_rows(truth)
        if not missing and av.sha(truth)!=c['truth_sha256']:raise ValueError('Truth hash changed')
        rows=io.load_rows(pred);r=copy.deepcopy(templates[cid])
        r.update(strategy='s24_benedek',views={},raw_metrics={},prediction_failed={},prediction_references={})
        failed=bool(item.get('error'))
        raw={'strict':score.score(rows,gt,tuple(c['window']),min_layer_px=0),
             'owner_equivalent':grading_policy.score_equivalent(rows,gt,tuple(c['window']),min_layer_px=0)}
        if failed:
            for values in raw.values():values.update({k:0. for k in av.METRICS if k!='depth_mae_ft'},depth_mae_ft=None)
        r['raw_metrics']['primary']=raw;r['prediction_failed']['primary']=failed
        r['prediction_references']['primary']={'path':str(pred),'sha256':pins[str(pred)]}
        r['views']['primary']={s:{p:av.metric_view(raw[p],r['structural_supervision'],series=s,truth_status=r['truth_status'],legend_invalid=r['legend_vocabulary_invalid'],prediction_failed=failed) for p in av.POLICIES} for s in av.SERIES}
        records.append(r)
    records.extend(copy.deepcopy(r) for r in old['records'] if r['case_id'] in seen and r['strategy'] in ('s19_combine','s21_v4dino','s23_v4dino_identity'))
    candidates=['s19_combine','s24_benedek','s21_v4dino','s23_v4dino_identity']
    summaries,paired=av.aggregate(records,candidates)
    av.write(output,dict(complete=len(seen)==len(cases),completed_cases=len(seen),expected_cases=len(cases),records=records,summaries=summaries,paired=paired,pins=pins,seed=1108,development_only=True))
    print('PAIRED_SCORE_OK cases='+str(len(seen))+' expected='+str(len(cases)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.predictions,a.output)

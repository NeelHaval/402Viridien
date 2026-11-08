"""Derived metric availability. Frozen inference and the shared scorer stay intact."""
from __future__ import annotations
import argparse,copy,csv,hashlib,itertools,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from harness import io,score
from strategies import grading_policy

METRICS=('score','row_acc','boundary_f1','cluster_ari','depth_mae_ft')
LITHOLOGY={'score','row_acc','cluster_ari'}
SERIES=('reference_diagnostic','qualified_development')
POLICIES=('strict','owner_equivalent')
SILVER_STATUS='unreviewed-two-source-A-tie-break'
TRUTH_STATUSES={SILVER_STATUS,'human-unqa','QA-human','gold-development','external-DEV-development'}
INVALID_IDS={'6fb4e860a31a7c9b','dc6d9c7cccfe7e79'}
NEPTUNE='df889653b4852560'
B=Path(__file__).resolve().parents[1]
ROOT=B.parents[1]
H=B/'report'/'scoreboard_1108'
REVIEW=ROOT/'management'/'offload'/'outside-pack'/'review'/'sol5-current'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def truth_structure(rows,window,missing=False):
    """Count supervision only. Do not change or print labels or contacts."""
    if missing:return dict(missing_truth=True,filled_rows=0,intervals=0,finite_depth_rows=0,finite_depth_boundaries=0)
    filled=score.raster(rows,tuple(window))!=''
    positive=[r for r in rows if min(int(r['bottom_y']),window[1])>max(int(r['top_y']),window[0]) and score.normalize(r.get('lithology',''))]
    finite_rows=0
    for r in positive:
        if all(r.get(k) not in (None,'') and math.isfinite(float(r[k])) for k in ('top_depth','bottom_depth')):finite_rows+=1
    finite_edges=sum(v is not None and math.isfinite(v) for v in score.boundaries(positive,tuple(window)).values())
    return dict(missing_truth=False,filled_rows=int(filled.sum()),intervals=len(positive),finite_depth_rows=finite_rows,finite_depth_boundaries=finite_edges)

def metric_view(raw,structure,*,series,truth_status,legend_invalid=False,prediction_failed=False):
    """An unavailable metric is null. Failure zeros retain valid supervision."""
    if series not in SERIES:raise ValueError('Unknown availability series')
    if truth_status not in TRUTH_STATUSES:raise ValueError('Unknown truth status')
    result={}
    for metric in METRICS:
        reasons=[]
        if structure['missing_truth']:reasons.append('missing-truth')
        elif not structure['filled_rows']:reasons.append('zero-supervision')
        if metric in LITHOLOGY and legend_invalid:reasons.append('legend-vocabulary-invalid')
        if series=='qualified_development' and (truth_status=='human-unqa' or truth_status==SILVER_STATUS and metric in LITHOLOGY):
            reasons.append('truth-not-qualified-for-lithology' if truth_status==SILVER_STATUS else 'human-truth-not-QA')
        if metric=='depth_mae_ft' and not structure['finite_depth_boundaries']:reasons.append('no-depth-truth')
        eligible=not reasons
        value=raw.get(metric)
        if eligible and prediction_failed and metric!='depth_mae_ft':value=0.
        if eligible and value is None:reasons.append('prediction-depth-unavailable' if metric=='depth_mae_ft' else 'metric-output-unavailable')
        available=eligible and value is not None
        if available and (not isinstance(value,(int,float)) or not math.isfinite(value)):raise ValueError('Nonfinite available metric')
        result[metric]=dict(value=float(value) if available else None,available=available,eligible=eligible,reasons=reasons)
    # Product cannot survive an unavailable accuracy component.
    if not result['row_acc']['available']:
        result['score']['value']=None;result['score']['available']=False
    return result

def select_overall(records):
    rank={'human_qa':0,'human_gold':1,'viridien_eval':2,'ext_dev':3}
    by=defaultdict(list)
    for r in records:
        if r['suite']!='hadrian_unqa':by[(r['strategy'],r['image_sha256'])].append(r)
    selected=[]
    for (_,image_sha256),rows in by.items():
        chosen=[r for r in rows if rank[r['suite']]==min(rank[x['suite']] for x in rows)]
        canonical_log=min(r['log'] for r in chosen)
        selected.extend(dict(r,log=canonical_log,original_log=r['log'],overall_image_sha256=image_sha256) for r in chosen)
    return selected

def denominator(rows,metric,series,policy,view):
    total_logs={r['log'] for r in rows}
    cells=[(r,r['views'][view][series][policy][metric]) for r in rows]
    available=[(r,c) for r,c in cells if c['available']]
    eligible=[r for r,c in cells if c['eligible']]
    log_values=defaultdict(list)
    for r,c in available:log_values[r['log']].append(c['value'])
    reason_windows=Counter(reason for _,c in cells for reason in c['reasons'])
    reason_logs={reason:len({r['log'] for r,c in cells if reason in c['reasons']}) for reason in reason_windows}
    failure_rows=[r for r in rows if r['prediction_failed'].get(view,False)]
    supervised_failures=[r for r,c in cells if c['eligible'] and r['prediction_failed'].get(view,False)]
    return dict(mean=float(np.mean([np.mean(v) for v in log_values.values()])) if log_values else None,
                window_mean=float(np.mean([c['value'] for _,c in available])) if available else None,
                denominator=dict(total_windows=len(rows),available_windows=len(available),excluded_windows=len(rows)-len(available),eligible_windows=len(eligible),
                  total_logs=len(total_logs),available_logs=len(log_values),excluded_logs=len(total_logs-set(log_values)),eligible_logs=len({r['log'] for r in eligible}),
                  failed_prediction_windows=len(failure_rows),supervised_failure_windows=len(supervised_failures),
                  zero_supervision_windows=sum(not r['structural_supervision']['filled_rows'] and not r['structural_supervision']['missing_truth'] for r in rows),
                  missing_truth_windows=sum(r['structural_supervision']['missing_truth'] for r in rows),legend_invalid_windows=sum(r['legend_vocabulary_invalid'] for r in rows),
                  no_depth_truth_windows=sum(not r['structural_supervision']['finite_depth_boundaries'] for r in rows),reason_windows=dict(reason_windows),reason_logs=reason_logs))

def ci(values,draws=10000):
    if not values:return dict(n_logs=0,mean=None,low=None,high=None)
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(1108)
    means=a[rng.integers(0,len(a),(draws,len(a)))].mean(axis=1)
    return dict(n_logs=len(a),mean=float(a.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),draws=draws,seed=1108)

def aggregate(records,candidates):
    summaries=[];paired=[]
    cohorts={s:[r for r in records if r['suite']==s] for s in sorted({r['suite'] for r in records})}
    cohorts['overall_unique_log']=select_overall(records)
    cohorts['qa_gold_only']=select_overall([r for r in records if r['truth_status'] in ('QA-human','gold-development')])
    for suite,base in cohorts.items():
        for family in ['all']+sorted({r['family'] for r in base}):
            group=base if family=='all' else [r for r in base if r['family']==family]
            for view in ('primary','pixel_only'):
                rr=[r for r in group if view in r['views']]
                if not rr:continue
                for series,policy in itertools.product(SERIES,POLICIES):
                    for strategy in candidates:
                        rows=[r for r in rr if r['strategy']==strategy]
                        if rows:summaries.append(dict(suite=suite,family=family,strategy=strategy,view=view,series=series,policy=policy,metrics={k:denominator(rows,k,series,policy,view) for k in METRICS}))
                    for a,b in itertools.combinations(candidates,2):
                        aa={r['case_id']:r for r in rr if r['strategy']==a};bb={r['case_id']:r for r in rr if r['strategy']==b}
                        metrics={}
                        for metric in METRICS:
                            differences=defaultdict(list);case_ids=[]
                            for case_id in sorted(aa.keys()&bb.keys()):
                                x,y=aa[case_id],bb[case_id]
                                if any(x[k]!=y[k] for k in ('log','window','truth_sha256','image_sha256','annotation_sha256')):raise ValueError('Paired identity mismatch')
                                u=x['views'][view][series][policy][metric];v=y['views'][view][series][policy][metric]
                                if u['available'] and v['available']:differences[x['log']].append(u['value']-v['value']);case_ids.append(case_id)
                            logs=sorted(differences);metrics[metric]=dict(ci=ci([float(np.mean(differences[k])) for k in logs]),paired_logs=logs,paired_case_ids=case_ids,paired_windows=len(case_ids))
                        paired.append(dict(suite=suite,family=family,a=a,b=b,view=view,series=series,policy=policy,metrics=metrics))
    return summaries,paired

def build(output):
    originals=[H/'scores.json',H/'raw_results.json',H/'manifest_1108.json',B/'harness'/'score.py',B/'harness'/'run.py',B/'strategies'/'grading_policy.py']
    reviews=[REVIEW/n for n in ('findings.json','saved-consensus-check.json','deep-results.json','audit-results.json')]
    containment=ROOT/'management'/'dataset-review'/'packs'/'root-containment-v1.json'
    if sha(containment)!='b5bcffb817cf137264cfd63494e092a76992b629c43e303bfba93148d8af029e':raise ValueError('Containment status evidence hash mismatch')
    reviews.append(containment)  # Byte pin only. Never follow listed sealed paths.
    pins={str(p):sha(p) for p in originals+reviews+[Path(__file__)]}
    baseline=read(H/'scores.json');manifest=read(H/'manifest_1108.json');tie=read(REVIEW/'saved-consensus-check.json');silver={r['id'] for r in tie}
    if len(silver)!=12 or any(r['saved_differs_from_A_rows']!=0 for r in tie):raise ValueError('Expected twelve reviewed A-tie rasters')
    if set(grading_policy.EQUIVALENT_LABELS)!={'Clay','Shale','Clay/Shale','Clay-Shale'}:raise ValueError('Owner equivalence changed')
    cases={c['case_id']:c for c in manifest['cases']};expected={(cid,s) for cid in cases for s in manifest['candidates']}
    actual=[(r['case_id'],r['strategy']) for r in baseline['records']]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Baseline matrix is not exact')
    structures={};csvpins={};records=[];replay_checks=0
    for cid,c in cases.items():
        path=Path(c['truth_csv']);missing=not path.is_file();rows=[] if missing else io.load_rows(path)
        if not missing:
            csvpins[str(path)]=sha(path)
            if csvpins[str(path)]!=c['truth_sha256']:raise ValueError('Frozen truth hash mismatch')
        structures[cid]=truth_structure(rows,c['window'],missing)
    if structures[NEPTUNE]['filled_rows']!=0 or cases[NEPTUNE]['window']!=[2699,8699]:raise ValueError('Reviewed Neptune zero-supervision fact changed')
    depth_review=read(REVIEW/'deep-results.json')['catalog_truth_errors']
    if {r['id'] for r in depth_review}!=silver or any(r['truth_rows_with_depth'] for r in depth_review):raise ValueError('Review depth scope changed')
    if any(structures[c]['finite_depth_boundaries'] for c in silver):raise ValueError('Reviewed silver no-depth fact changed')
    for index,old in enumerate(baseline['records']):
        c=cases[old['case_id']];cid=c['case_id'];status=SILVER_STATUS if cid in silver else 'human-unqa' if c['suite']=='hadrian_unqa' else 'QA-human' if c['suite']=='human_qa' else 'gold-development' if c['suite'] in ('human_gold','viridien_eval') else 'external-DEV-development'
        record={k:copy.deepcopy(c[k]) for k in ('case_id','log','suite','family','well','window','truth_csv','truth_sha256','image_sha256','annotation_sha256')}
        record.update(strategy=old['strategy'],truth_status=status,legend_vocabulary_invalid=cid in INVALID_IDS,structural_supervision=structures[cid],raw_record_index=index,raw_metrics={},views={},prediction_failed={},prediction_references={})
        for view,target in [('primary',old)]+([('pixel_only',old['pixel_only'])] if old.get('pixel_only') else []):
            pred=Path(target['prediction_csv']);csvpins[str(pred)]=sha(pred);record['prediction_references'][view]=dict(path=str(pred),sha256=csvpins[str(pred)],labels_intervals_clusters_depths_unchanged=True)
            raw_strict={k:target.get(k) for k in METRICS};failed=bool(target.get('error')) if view=='primary' else False
            predrows=io.load_rows(pred);truthrows=[] if structures[cid]['missing_truth'] else io.load_rows(c['truth_csv'])
            if not structures[cid]['missing_truth']:
                computed_strict=score.score(predrows,truthrows,tuple(c['window']),min_layer_px=0)
                computed_owner=grading_policy.score_equivalent(predrows,truthrows,tuple(c['window']),min_layer_px=0)
                if failed:
                    computed_owner.update({k:0. for k in METRICS if k!='depth_mae_ft'});computed_owner['depth_mae_ft']=None
                else:
                    for k in METRICS:
                        x,y=raw_strict[k],computed_strict[k]
                        if (x is None)!=(y is None) or x is not None and abs(x-y)>1e-9:raise ValueError('Cached strict replay mismatch '+cid+' '+k)
                    replay_checks+=1
                accepted={k:target.get('accepted_metrics',computed_owner).get(k) for k in METRICS}
                if any((accepted[k] is None)!=(computed_owner[k] is None) or accepted[k] is not None and abs(accepted[k]-computed_owner[k])>1e-9 for k in METRICS):raise ValueError('Owner-equivalent replay mismatch')
            else:accepted={k:target.get('accepted_metrics',{}).get(k) for k in METRICS}
            record['raw_metrics'][view]=dict(strict=raw_strict,owner_equivalent=accepted);record['prediction_failed'][view]=failed;record['views'][view]={}
            for series in SERIES:record['views'][view][series]={policy:metric_view(raw,structures[cid],series=series,truth_status=status,legend_invalid=cid in INVALID_IDS,prediction_failed=failed) for policy,raw in [('strict',raw_strict),('owner_equivalent',accepted)]}
        records.append(record)
    if {(r['case_id'],r['strategy']) for r in records}!=expected:raise ValueError('Derived matrix identity mismatch')
    summaries,paired=aggregate(records,manifest['candidates'])
    data=dict(version='availability_v2',complete=True,generalization_verified=False,selection_promoted=False,selection_status='Champion claim withdrawn; no automatic promotion. Availability audit is a derived diagnostic, not an official grade.',
      series_definitions=dict(reference_diagnostic='Available supervision with explicit review exclusions. Unreviewed silver remains a source-A diagnostic, never independent truth.',qualified_development='Exclude unreviewed silver lithology/ARI/product and Hadrian unQA. Silver contacts remain source-A contact diagnostics. QA/gold and supplied DEV retain development/exposure caveats; no unexposed-well or human-truth certification is implied for DEV.'),
      method=dict(mean='Equal physical-log mean of available window metrics; overall precedence QA > gold > eval > DEV; Hadrian unQA separate.',product='Per-window row accuracy times boundary F1. Null if row accuracy unavailable. Failed predictions with valid supervision stay zero.',paired='Exact same cases, then common available log means; deterministic10,000 log bootstrap draws, seed1108. No inference or truth correction.',depth='Finite in-window GT boundary depth required; no matched finite prediction depth is unavailable, never zero MAE.',silver_truth_status=SILVER_STATUS),
      candidates=manifest['candidates'],records=records,summaries=summaries,paired=paired,pins=pins,prediction_and_truth_sha256=csvpins,baseline_records_sha256=canonical_sha(baseline['records']),review_ids=['sol5-current-03','sol5-current-04','sol5-current-05','sol5-current-06'])
    # Validate every protected baseline and CSV byte again before publication.
    errors=[p for p,h in {**pins,**csvpins}.items() if sha(p)!=h]
    if errors:raise ValueError('Input changed during derived audit')
    output=Path(output);write(output/'derived_scores.json',data)
    receipt=dict(complete=True,errors=[],records=len(records),expected=len(expected),exact_keys=True,strict_successful_replays=replay_checks,baseline_scores_sha256=pins[str(H/'scores.json')],derived_sha256=sha(output/'derived_scores.json'),baseline_scores_unchanged=sha(H/'scores.json')==pins[str(H/'scores.json')],derived_scores_sha256=sha(output/'derived_scores.json'),source_pins=pins,csv_hash_checks=len(csvpins),zero_supervision_cases=[cid for cid,s in structures.items() if not s['filled_rows'] and not s['missing_truth']],missing_truth_cases=[cid for cid,s in structures.items() if s['missing_truth']],legend_invalid_cases=sorted(INVALID_IDS),unreviewed_silver_cases=sorted(silver),no_depth_truth_cases=[cid for cid,s in structures.items() if not s['finite_depth_boundaries']])
    write(output/'audit.json',receipt);write(output/'metadata.json',dict(pins=pins,review_scope='Existing review JSON plus structural CSV counts only. No image or new human label review. External DEV only.',records_sha256=canonical_sha(records)))
    (output/'README.md').write_text('Availability v2 is a separate derived view. Baseline scores, truth, predictions and shared scorer are unchanged.\n\nUse audit.json to bind derived_scores.json and the pinned inputs. reference_diagnostic retains qualified source-A diagnostics; qualified_development excludes lithology/ARI/product for all twelve unreviewed silver rasters and excludes Hadrian unQA. Silver contact scores remain source-A diagnostics. This means development eligibility, not unexposed validation. DEV and development gold remain development data. No champion is declared or promoted.\n\nA missing metric is null with reasons and explicit denominators. Failed predictions on valid supervision remain zero for score, row accuracy, boundary F1 and ARI. Missing depth is never zero MAE. Labels, interval contacts, clusters and depth values remain in the referenced frozen CSVs. The original strict and four-name owner-equivalent raw metrics are retained per record. No extra spelling or geology mapping was added.\n',encoding='utf-8')
    print(f'AVAILABILITY V2 PASS: {len(records)}/{len(expected)} records; baseline bytes unchanged; {len(silver)} unreviewed silver; {len(receipt["zero_supervision_cases"])} zero-supervision cases; {replay_checks} cached strict replays')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=H/'availability_v2');build(p.parse_args().output)
